"""DAC 16kHz DECODER-ONLY (continuous latent -> audio) -> LiteRT, GPU-clean.

The full decode path (codes -> audio) fails on Mali: the RVQ EMBEDDING_LOOKUP + INT64 code
indices / casts are GPU-rejected ("CAST: Tensor type(INT64) is not supported"). Standard codec
split: do the cheap RVQ lookup (codes -> continuous latent) in app code, run only the heavy conv
DECODER (pure float: Snake1d + Conv1d + zero-stuff upsample) on GPU. Input = [1, dim, T] float.
ConvTranspose1d -> ZeroStuffConvT1d (no TRANSPOSE_CONV). No embedding, no int64.

    ~/clipconv/bin/python scripts/convert_dac_deconly.py [T_frames]
"""
import _stub  # noqa: F401
import os, sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import litert_torch
from ai_edge_litert.interpreter import Interpreter
from transformers.models.dac.modeling_dac import DacModel

BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP"}


class ZeroStuffConvT1d(nn.Module):
    def __init__(self, ct, L):
        super().__init__()
        self.s = ct.stride[0]; self.k = ct.kernel_size[0]
        self.p = ct.padding[0]; self.op = ct.output_padding[0]; self.L = L
        self.register_buffer("w", ct.weight.flip(2).transpose(0, 1).contiguous())
        self.register_buffer("b", ct.bias.detach().clone() if ct.bias is not None
                             else torch.zeros(ct.out_channels))
        mk = np.zeros((L * self.s,), np.float32); mk[::self.s] = 1.0
        self.register_buffer("mask", torch.from_numpy(mk)[None, None])

    def forward(self, x):
        s, k, p, op = self.s, self.k, self.p, self.op
        xn = F.interpolate(x.unsqueeze(2), size=(1, self.L * s), mode="nearest").squeeze(2)
        xn = xn * self.mask
        y = F.conv1d(xn, self.w, bias=self.b, padding=k - 1)
        out_len = (self.L - 1) * s + k - 2 * p + op
        return y[:, :, p:p + out_len]


def swap_convtranspose(decoder, lengths):
    for name, mod in list(decoder.named_modules()):
        if isinstance(mod, nn.ConvTranspose1d):
            parent = decoder; *path, last = name.split(".")
            for q in path:
                parent = getattr(parent, q)
            setattr(parent, last, ZeroStuffConvT1d(mod, lengths[name]))


def fp16_recipe():
    from ai_edge_quantizer import recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    oc = qtyping.OpQuantizationConfig(
        weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
        compute_precision=qtyping.ComputePrecision.FLOAT)
    rm.add_quantization_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                               op_config=oc, algorithm_key=AlgorithmName.FLOAT_CASTING)
    return rm.get_quantization_recipe()


def op_report(path):
    itp = Interpreter(model_path=path); itp.allocate_tensors()
    ops = Counter(o["op_name"] for o in itp._get_ops_details())
    maxnd = max(len(t["shape"]) for t in itp.get_tensor_details())
    banned = sorted(k for k in ops if k.upper() in BANNED)
    flex = sorted(k for k in ops if "flex" in k.lower() or k.lower() == "custom")
    return itp, dict(ops), banned, flex, maxnd


def main():
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    out_dir = "out/dac"; os.makedirs(out_dir, exist_ok=True)
    model = DacModel.from_pretrained("descript/dac_16khz").eval()
    cfg = model.config
    codes = torch.randint(0, cfg.codebook_size, (1, cfg.n_codebooks, T), dtype=torch.long)
    with torch.no_grad():
        cont = model.quantizer.from_codes(codes)[0].detach()   # [1, dim, T] float (RVQ done here)
    dec = model.decoder.eval()
    with torch.no_grad():
        ref = dec(cont).numpy()
    print(f"latent {tuple(cont.shape)} -> audio {ref.shape}  (RVQ lookup runs in app)")

    lengths, hooks = {}, []
    for nm, mod in dec.named_modules():
        if isinstance(mod, nn.ConvTranspose1d):
            def mk(n):
                def h(m, i, o): lengths[n] = i[0].shape[-1]
                return h
            hooks.append(mod.register_forward_hook(mk(nm)))
    with torch.no_grad():
        dec(cont)
    for h in hooks:
        h.remove()

    swap_convtranspose(dec, lengths)
    with torch.no_grad():
        ref2 = dec(cont).numpy()
    corr = float(np.corrcoef(ref.flatten(), ref2.flatten())[0, 1])
    print(f">>> EAGER parity (zero-stuff vs original): corr={corr:.7f} max|diff|={np.abs(ref-ref2).max():.2e}")
    if corr < 0.999:
        print("!! eager parity failed"); return

    fp32 = f"{out_dir}/dac_16khz_deconly_zs.tflite"
    litert_torch.convert(dec, (cont,)).export(fp32)
    print(f"FP32 exported: {round(os.path.getsize(fp32)/1e6, 2)} MB")
    itp, ops, banned, flex, maxnd = op_report(fp32)
    print(f"  ops={ops}")
    print(f"  >>> BANNED={banned or 'NONE'} FLEX/CUSTOM={flex or 'NONE'} max_ndim={maxnd}")
    ins, outs = itp.get_input_details(), itp.get_output_details()
    itp.set_tensor(ins[0]["index"], cont.numpy().astype(ins[0]["dtype"])); itp.invoke()
    got = itp.get_tensor(outs[0]["index"])
    print(f"  >>> parity tflite vs eager: corr={float(np.corrcoef(ref2.flatten(), got.flatten())[0,1]):.6f}")

    from ai_edge_quantizer import quantizer
    fp16 = f"{out_dir}/dac_16khz_deconly_zs_fp16.tflite"
    if os.path.exists(fp16):
        os.remove(fp16)
    qt = quantizer.Quantizer(float_model=fp32); qt.load_quantization_recipe(fp16_recipe())
    qt.quantize().export_model(fp16)
    print(f"FP16 exported: {round(os.path.getsize(fp16)/1e6, 2)} MB")
    _, _, b16, f16, nd16 = op_report(fp16)
    print(f"  >>> FP16 BANNED={b16 or 'NONE'} FLEX/CUSTOM={f16 or 'NONE'} max_ndim={nd16} (input dim={cfg.hidden_size})")


if __name__ == "__main__":
    main()
