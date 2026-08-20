#!/usr/bin/env python3
"""japanese-zipformer-base (reazon-research, Apache-2.0) -> LiteRT CompiledModel GPU.

96.5M ZipformerForCTC: wav2vec2-style raw-waveform conv frontend (stride 320, NO FFT anywhere)
+ Zipformer2 (6 stacks, output_downsampling_factor=1 -> 50 Hz) + CTC Linear (BPE-3004,
blank <blk>=3001). First Japanese ASR in the zoo.

Same GPU re-authoring set as the English Zipformer (build_zipformer_ctc.py) applied to the
repo's bundled zipformer.py/scaling.py, plus the wav2vec2-frontend recipe (GELU->tanh-GELU,
Fp32GroupNorm->GN4D). Fixed 16 s window: waveform [1,256000] -> T50=799; additive bias masks
[1,799],[1,400],[1,200],[1,100].

Run: ~/clipconv/bin/python build_ja_zipformer.py {parity,all}
"""
import os, sys, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
open(os.path.join(HERE, "model", "__init__.py"), "a").close()

from model.configuration_zipformer import ZipformerConfig   # noqa: E402
from model.modeling_zipformer import ZipformerForCTC        # noqa: E402
import model.zipformer as jzf                               # noqa: E402
import model.scaling as jsc                                 # noqa: E402
import model.wav2vec2_module as jwm                         # noqa: E402
import model.utils as jut                                   # noqa: E402

# Fine-tune overrides (defaults reproduce the official ship exactly):
#   ZIPJA_MODEL_DIR=dir  a fine-tune of reazon-research/japanese-zipformer-base saved
#                        with save_pretrained() (config.json + model.safetensors); the
#                        modeling CODE still comes from the base snapshot in HERE/model
#   ZIPJA_VOCAB=N        CTC width when the fine-tune changed the vocab (default 3004)
#   ZIPJA_WAV=path       16 kHz mono wav for the parity check
MODEL_DIR = os.environ.get("ZIPJA_MODEL_DIR", os.path.join(HERE, "model"))
WAV = os.environ.get("ZIPJA_WAV", os.path.join(HERE, "ja_test.wav"))
SR = 16000
N_IN = 256000            # 16 s
T50 = 799
VOCAB = int(os.environ.get("ZIPJA_VOCAB", "3004"))
BLANK = 0                # id 0 ('<unk>' label) behaves as the CTC blank (icefall convention)
BIAS_LENS = (799, 400, 200, 100)
torch.manual_seed(0)

BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP",
          "EQUAL", "NOT_EQUAL", "GREATER", "GREATER_EQUAL", "LESS", "LOGICAL_AND",
          "PACK", "SPLIT", "SPLIT_V",
          "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM"}


def build_model():
    cfg = ZipformerConfig.from_pretrained(MODEL_DIR)
    m = ZipformerForCTC(cfg).eval()
    from safetensors.torch import load_file
    sd = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f"[build] load: missing {len(missing)} unexpected {len(unexpected)}")
    assert not unexpected, unexpected[:5]
    for k in missing:
        print("  missing:", k)
    return m


# ---------------------------------------------------------------- GPU patches (JA copy)
class TanhGELU(nn.Module):
    C = math.sqrt(2.0 / math.pi)
    def forward(s, x):
        return 0.5 * x * (1.0 + torch.tanh(s.C * (x + 0.044715 * x * x * x)))


class GN4D(nn.Module):
    def __init__(s, gn):
        super().__init__(); s.g = gn.num_groups; s.eps = gn.eps
        s.register_buffer("w", gn.weight.detach().reshape(1, -1, 1))
        s.register_buffer("b", gn.bias.detach().reshape(1, -1, 1))
    def forward(s, x):
        b, c, t = x.shape; x = x.reshape(b, s.g, c // s.g, t)
        m = x.mean((2, 3), keepdim=True); v = (x - m).pow(2).mean((2, 3), keepdim=True)
        return (((x - m) * torch.rsqrt(v + s.eps)).reshape(b, c, t)) * s.w + s.b


def swap_frontend(m):
    for n, c in list(m.named_children()):
        if isinstance(c, nn.GELU):
            setattr(m, n, TanhGELU())
        elif isinstance(c, nn.GroupNorm):
            setattr(m, n, GN4D(c))
        else:
            swap_frontend(c)


def apply_gpu_patches():
    def softplus_stable(z):
        return torch.relu(z) + torch.log1p(torch.exp(-torch.abs(z)))
    def swoosh_l(x): return softplus_stable(x - 4.0) - 0.08 * x - 0.035
    def swoosh_r(x): return softplus_stable(x - 1.0) - 0.08 * x - 0.313261687

    jsc.SwooshL.forward = lambda self, x: swoosh_l(x)
    jsc.SwooshR.forward = lambda self, x: swoosh_r(x)
    if hasattr(jsc, "SwooshLForward"):
        jsc.SwooshLForward = swoosh_l
        jsc.SwooshRForward = swoosh_r
    jsc._no_op = lambda x: x
    jsc.Identity.forward = lambda self, x: x
    jsc.Balancer.forward = lambda self, x: x
    jsc.Whiten.forward = lambda self, x: x
    jsc.ScaleGrad.forward = lambda self, x: x

    def biasnorm_forward(self, x):
        channel_dim = self.channel_dim
        if channel_dim < 0:
            channel_dim += x.ndim
        bias = self.bias
        for _ in range(channel_dim + 1, x.ndim):
            bias = bias.unsqueeze(-1)
        scales = (
            torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5
        ) * self.log_scale.exp()
        return x * scales
    jsc.BiasNorm.forward = biasnorm_forward

    jzf.softmax = lambda x, dim: torch.softmax(x, dim=dim)

    def attn_forward(self, x, pos_emb, key_padding_mask=None, attn_mask=None):
        assert attn_mask is None
        x = self.in_proj(x)
        qhd, phd, H = self.query_head_dim, self.pos_head_dim, self.num_heads
        T, B, _ = x.shape
        qd = qhd * H
        q = x[..., 0:qd].reshape(T, B, H, qhd).permute(2, 1, 0, 3)
        k = x[..., qd:2 * qd].reshape(T, B, H, qhd).permute(2, 1, 3, 0)
        p = x[..., 2 * qd:].reshape(T, B, H, phd).permute(2, 1, 0, 3)
        attn_scores = torch.matmul(q, k)
        pos_emb = self.linear_pos(pos_emb)
        L = 2 * T - 1
        pos_emb = pos_emb.reshape(-1, L, H, phd).permute(2, 0, 3, 1)
        pos_scores = torch.matmul(p, pos_emb)
        pos_scores = F.pad(pos_scores, (0, 1)).reshape(H, B, T * 2 * T)
        pos_scores = pos_scores[..., :T * 2 * T - T].reshape(H, B, T, 2 * T - 1)
        attn_scores = attn_scores + pos_scores[..., T - 1:]
        if key_padding_mask is not None:
            attn_scores = attn_scores + key_padding_mask.unsqueeze(1)
        return torch.softmax(attn_scores, dim=-1)
    jzf.RelPositionMultiheadAttentionWeights.forward = attn_forward

    def nonlin_forward(self, x, attn_weights):
        x = self.in_proj(x)
        T, B, _ = x.shape
        h = self.hidden_channels
        s, x, y = x[..., :h], x[..., h:2 * h], x[..., 2 * h:]
        s = self.tanh(self.balancer(s))
        x = self.whiten1(x) * s
        x = self.identity1(x)
        H = attn_weights.shape[0]
        x = x.reshape(T, B, H, -1).permute(2, 1, 0, 3)
        x = torch.matmul(attn_weights, x)
        x = x.permute(2, 1, 0, 3).reshape(T, B, -1)
        x = x * self.identity2(y)
        x = self.identity3(x)
        return self.whiten2(self.out_proj(x))
    jzf.NonlinAttention.forward = nonlin_forward

    def conv_forward(self, x, src_key_padding_mask=None, chunk_size=-1):
        x = self.in_proj(x)
        C = x.shape[-1] // 2
        x, s = x[..., :C], x[..., C:]
        s = self.sigmoid(self.balancer1(s))
        x = self.activation1(x) * s
        x = self.activation2(x)
        x = x.permute(1, 2, 0)
        if src_key_padding_mask is not None:
            x = x * (1.0 + src_key_padding_mask.unsqueeze(1) / 1000.0)
        x = self.depthwise_conv(x)
        x = self.balancer2(x)
        x = x.permute(2, 0, 1)
        return self.out_proj(self.whiten(x))
    jzf.ConvolutionModule.forward = conv_forward

    def upsample_forward(self, src):
        upsample = self.upsample
        T, B, C = src.shape
        src = torch.cat([src.unsqueeze(1)] * upsample, dim=1)
        return src.reshape(T * upsample, B, C)
    jzf.SimpleUpsample.forward = upsample_forward

    def downsample_forward(self, src):
        T, B, C = src.shape
        ds = self.downsample
        d_seq_len = (T + ds - 1) // ds
        pad = d_seq_len * ds - T
        if pad > 0:
            src = torch.cat([src] + [src[T - 1:]] * pad, dim=0)
        src = src.reshape(d_seq_len, ds, B, C)
        if not hasattr(self, "_w"):
            self._w = self.bias.detach().softmax(dim=0).reshape(1, ds, 1, 1)
        return (src * self._w).sum(dim=1)
    jzf.SimpleDownsample.forward = downsample_forward

    def cpe_forward(self, x, left_context_len=0):
        assert left_context_len == 0
        if self.pe is None or self.pe.size(0) < 2 * x.size(0) - 1:
            self.extend_pe(x, left_context_len)
        pe = self.pe
        T = x.size(0)
        start = pe.size(0) // 2 - T + 1
        end = pe.size(0) // 2 + T
        return pe[start:end].unsqueeze(0)
    jzf.CompactRelPositionalEncoding.forward = cpe_forward

    def zip2_forward(self, x, x_lens, src_key_padding_mask=None):
        ds2idx = {1: 0, 2: 1, 4: 2, 8: 3}
        outputs = []
        for i, module in enumerate(self.encoders):
            ds = self.downsampling_factor[i]
            x = jzf.convert_num_channels(x, self.encoder_dim[i])
            m = None if src_key_padding_mask is None else src_key_padding_mask[ds2idx[ds]]
            x = module(x, chunk_size=-1, feature_mask=1.0,
                       src_key_padding_mask=m, attn_mask=None)
            outputs.append(x)
        x = self._get_full_dim_output(outputs)   # no output downsampling in this repo (50 Hz)
        return x, x_lens
    jzf.Zipformer2.forward = zip2_forward

    # frontend Fp32 casts: plain float32 math (input is already fp32; .float() emits CASTs)
    jut.Fp32LayerNorm.forward = lambda self, x: F.layer_norm(
        x, self.normalized_shape, self.weight, self.bias, self.eps)


# ---------------------------------------------------------------- wrapper + helpers
class JaZipCtc(nn.Module):
    def __init__(self, m):
        super().__init__()
        enc = m.encoder
        self.feature_extractor = enc.feature_extractor
        self.layer_norm = enc.layer_norm
        self.post_extract_proj = enc.post_extract_proj
        self.encoder = enc.encoder
        self.ctc_linear = m.ctc_output[1]
        self.register_buffer("x_lens", torch.tensor([T50], dtype=torch.int64))

    def forward(self, x, b1, b2, b4, b8):     # x [1, 256000]
        f = self.feature_extractor(x)         # (B, 512, T50)
        f = f.transpose(1, 2)
        f = self.layer_norm(f)
        f = self.post_extract_proj(f)
        f = f.transpose(0, 1)                 # (T, B, C)
        out, _ = self.encoder(f, self.x_lens, (b1, b2, b4, b8))
        out = out.transpose(0, 1)
        return self.ctc_linear(out)           # [1, 799, 3004] raw logits


def frames50(n):
    L = n
    for k, s in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]:
        L = (L - k) // s + 1
    return L


def load_vocab():
    v = json.load(open(os.path.join(HERE, "model", "vocab.json")))
    return {i: t for t, i in v.items()}


def greedy(logits, id2tok, tv):
    out, prev = [], -1
    for i in logits[:tv].argmax(-1).tolist():
        if i != prev and i != BLANK:
            out.append(id2tok.get(i, ""))
        prev = i
    return "".join(out).replace("▁", " ").strip()


def make_biases(v50):
    b = torch.full((1, T50), -1000.0)
    b[0, :v50] = 0.0
    return b, b[:, ::2].contiguous(), b[:, ::4].contiguous(), b[:, ::8].contiguous()


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    id2tok = load_vocab()
    import torchaudio
    wave, sr = torchaudio.load(WAV)
    a = wave.mean(0)
    n = a.shape[0]

    m = build_model()
    # gold: original classes, native length + the model-card 0.5 s pad on both sides
    lead = 8000
    ga = torch.cat([torch.zeros(lead), a, torch.zeros(lead)])
    with torch.no_grad():
        gold = m(ga.unsqueeze(0)).logits
    gold_text = greedy(gold[0].numpy(), id2tok, gold.shape[1])
    print(f"[gold] logits {tuple(gold.shape)} TEXT: {gold_text}")

    apply_gpu_patches()
    swap_frontend(m.encoder.feature_extractor)
    g = JaZipCtc(m).eval()

    x = torch.zeros(1, N_IN)
    ncl = min(n, N_IN - 2 * lead)
    x[0, lead:lead + ncl] = a[:ncl]
    v50 = frames50(ncl + 2 * lead)
    biases = make_biases(v50)
    with torch.no_grad():
        lp = g(x, *biases)
    text = greedy(lp[0].numpy(), id2tok, v50)
    nv = min(v50, gold.shape[1])
    corr = np.corrcoef(lp[0, :nv].numpy().ravel(), gold[0, :nv].numpy().ravel())[0, 1]
    print(f"[patched] vs gold valid corr {corr:.6f} TEXT: {text}")
    print(f"[patched] text match gold: {text == gold_text}")
    np.save(os.path.join(HERE, "ja_ref_in.npy"), x.numpy())
    for i, b in enumerate(biases):
        np.save(os.path.join(HERE, f"ja_ref_bias{i}.npy"), b.numpy())
    np.save(os.path.join(HERE, "ja_ref_logits.npy"), lp.numpy())
    if stage == "parity":
        os._exit(0)

    import litert_torch
    out = os.path.join(HERE, "ja_zipformer_ctc.tflite")
    litert_torch.convert(g, (x, *biases)).export(out)
    print(f"[convert] wrote {out}")
    it, clean = opcheck(out, "ja")
    got = tfl_run(it, x.numpy(), [b.numpy() for b in biases])
    corr = np.corrcoef(got.ravel(), lp.numpy().ravel())[0, 1]
    print(f"[convert] tflite vs torch corr {corr:.6f} max|d| {np.abs(got - lp.numpy()).max():.4f}")
    print(f"[convert] TFLITE TEXT: {greedy(got[0], id2tok, v50)}")
    if clean:
        quant_fp16(out, id2tok, v50)
    print("[done]", "GPU-CLEAN" if clean else "blockers")


def quant_fp16(path, id2tok, v50):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT),
        algorithm_key=AlgorithmName.FLOAT_CASTING)
    out = path.replace(".tflite", "_fp16.tflite")
    if os.path.exists(out):
        os.remove(out)
    qt = quantizer.Quantizer(float_model=path)
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(out)
    it, _ = opcheck(out, "ja_fp16")
    x = np.load(os.path.join(HERE, "ja_ref_in.npy"))
    biases = [np.load(os.path.join(HERE, f"ja_ref_bias{i}.npy")) for i in range(4)]
    ref = np.load(os.path.join(HERE, "ja_ref_logits.npy"))
    lp = tfl_run(it, x, biases)
    corr = np.corrcoef(lp.ravel(), ref.ravel())[0, 1]
    print(f"[fp16] vs torch corr {corr:.6f} max|d| {np.abs(lp - ref).max():.4f}")
    print(f"[fp16] TEXT: {greedy(lp[0], id2tok, v50)}")


def opcheck(path, label):
    import collections
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    ops = collections.Counter(d.get("op_name", "?") for d in it._get_ops_details())
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    over = sum(1 for d in it.get_tensor_details() if len(d.get("shape", [])) > 4)
    print(f"[{label}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(path)/1e6:.1f}MB")
    print(f"[{label}] VERDICT:", "GPU-CLEAN" if not bad and not over else f"BLOCKERS {bad}")
    return it, (not bad and not over)


def tfl_run(it, x, biases):
    by = {b.shape[1]: b for b in biases}
    for d in it.get_input_details():
        s = list(d["shape"])
        it.set_tensor(d["index"], x.astype(np.float32) if len(s) == 2 and s[1] == N_IN
                      else by[s[1]].astype(np.float32))
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"])


if __name__ == "__main__":
    main()
    os._exit(0)
