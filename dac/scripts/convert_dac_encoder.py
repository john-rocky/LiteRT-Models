"""DAC 16kHz ENCODER (audio -> continuous latent) -> LiteRT, GPU-clean check.

Encoder = strided Conv1d down-stack + Snake1d (no ConvTranspose, no embedding, no int64). The RVQ
quantize (continuous -> codes, an argmin over codebooks) is GPU-hostile (TOPK/argmin) -> runs in
app, like the decoder's RVQ lookup. So the encoder tflite stops at the continuous latent. This RUN
just checks the encoder convs convert + run GPU-clean (expected trivially, Descript static padding).

    ~/clipconv/bin/python scripts/convert_dac_encoder.py [S_samples]
"""
import _stub  # noqa: F401
import os, sys
from collections import Counter

import numpy as np
import torch
import litert_torch
from ai_edge_litert.interpreter import Interpreter
from transformers.models.dac.modeling_dac import DacModel

BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP"}


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
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
    out_dir = "out/dac"; os.makedirs(out_dir, exist_ok=True)
    model = DacModel.from_pretrained("descript/dac_16khz").eval()
    enc = model.encoder.eval()
    audio = torch.randn(1, 1, S)
    with torch.no_grad():
        ref = enc(audio).numpy()
    print(f"audio {tuple(audio.shape)} -> latent {ref.shape}")

    fp32 = f"{out_dir}/dac_16khz_encoder.tflite"
    litert_torch.convert(enc, (audio,)).export(fp32)
    print(f"FP32 exported: {round(os.path.getsize(fp32)/1e6, 2)} MB")
    itp, ops, banned, flex, maxnd = op_report(fp32)
    print(f"  ops={ops}")
    print(f"  >>> BANNED={banned or 'NONE'} FLEX/CUSTOM={flex or 'NONE'} max_ndim={maxnd}")
    ins, outs = itp.get_input_details(), itp.get_output_details()
    itp.set_tensor(ins[0]["index"], audio.numpy().astype(ins[0]["dtype"])); itp.invoke()
    got = itp.get_tensor(outs[0]["index"])
    print(f"  >>> parity tflite vs eager: corr={float(np.corrcoef(ref.flatten(), got.flatten())[0,1]):.6f}")

    from ai_edge_quantizer import quantizer
    fp16 = f"{out_dir}/dac_16khz_encoder_fp16.tflite"
    if os.path.exists(fp16):
        os.remove(fp16)
    qt = quantizer.Quantizer(float_model=fp32); qt.load_quantization_recipe(fp16_recipe())
    qt.quantize().export_model(fp16)
    print(f"FP16 exported: {round(os.path.getsize(fp16)/1e6, 2)} MB")
    _, _, b16, f16, nd16 = op_report(fp16)
    print(f"  >>> FP16 BANNED={b16 or 'NONE'} FLEX/CUSTOM={f16 or 'NONE'} max_ndim={nd16}")


if __name__ == "__main__":
    main()
