#!/usr/bin/env python3
"""Zipformer-medium CR-CTC (icefall, LibriSpeech, Apache-2.0) -> LiteRT CompiledModel GPU.

Checkpoint: Zengwei/icefall-asr-librispeech-zipformer-medium-cr-ctc-20241018
  64.25M params, pure CTC (--use-ctc 1 --use-transducer 0), WER 2.12/4.62 greedy.
Pipeline: host kaldi-fbank[1,T,80] -> [GPU] Conv2dSubsampling + Zipformer2 + ctc Linear
          -> host greedy-CTC + BPE detok (Parakeet app shape).

Run: ~/clipconv/bin/python build_zipformer_ctc.py {baseline,convert,all}
Env: KMP_DUPLICATE_LIB_OK=TRUE JAX_PLATFORMS=cpu
"""
import sys, os, math, types, collections, contextlib

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ZF_DIR = os.path.join(HERE, "icefall/egs/librispeech/ASR/zipformer")

# CR-CTC 20241018 family; identical downsampling structure -> identical I/O signature
VARIANT = os.environ.get("ZIP_VARIANT", "medium")
CONFIGS = {
    "small": dict(layers=(2, 2, 2, 2, 2, 2), ff=(512, 768, 768, 768, 768, 768),
                  dims=(192, 256, 256, 256, 256, 256), unmasked=(192,) * 6),
    "medium": dict(layers=(2, 2, 3, 4, 3, 2), ff=(512, 768, 1024, 1536, 1024, 768),
                   dims=(192, 256, 384, 512, 384, 256),
                   unmasked=(192, 192, 256, 256, 256, 192)),
    "large": dict(layers=(2, 2, 4, 5, 4, 2), ff=(512, 768, 1536, 2048, 1536, 768),
                  dims=(192, 256, 512, 768, 512, 256),
                  unmasked=(192, 192, 256, 320, 256, 192)),
}
CFG = CONFIGS[VARIANT]
# Fine-tune overrides (defaults reproduce the official ships exactly):
#   ZIP_CKPT=ckpt.pt     your own icefall zipformer CTC checkpoint (export.py output)
#   ZIP_TOKENS=tokens.txt  its lang dir tokens file (vocab may differ from 500)
#   ZIP_VOCAB=N          CTC output width when not 500 (must match the checkpoint)
#   ZIP_WAV=sample.wav   16 kHz mono wav for the parity check
CKPT = os.environ.get("ZIP_CKPT", os.path.join(HERE, f"en_{VARIANT}/exp/pretrained.pt"))
TOKENS = os.environ.get("ZIP_TOKENS", os.path.join(HERE, f"en_{VARIANT}/data/lang_bpe_500/tokens.txt"))
VOCAB = int(os.environ.get("ZIP_VOCAB", "500"))
OUT_NAME = "zipformer_ctc.tflite" if VARIANT == "medium" else f"zipformer_ctc_{VARIANT}.tflite"
WAV = os.environ.get("ZIP_WAV", os.path.expanduser(
    "~/Downloads/depthanything-android/moonshine_card_work/wavs/libri_1272_128104_0001.wav"))

SR = 16000
T_IN = 1600            # 16 s of 10ms-hop fbank frames (snip_edges=False)
torch.manual_seed(0)

BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP",
          "EQUAL", "NOT_EQUAL", "GREATER", "GREATER_EQUAL", "LESS", "LOGICAL_AND",
          "PACK", "SPLIT", "SPLIT_V",
          "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM"}

# ---------------------------------------------------------------- stubs (BEFORE model import)
def install_stubs():
    k2 = types.ModuleType("k2")
    def _sp(z):
        return torch.relu(z) + torch.log1p(torch.exp(-torch.abs(z)))
    def _swoosh_l(x):
        return _sp(x - 4.0) - 0.08 * x - 0.035
    def _swoosh_r(x):
        return _sp(x - 1.0) - 0.08 * x - 0.313261687
    k2.swoosh_l = _swoosh_l
    k2.swoosh_l_forward = _swoosh_l
    k2.swoosh_r = _swoosh_r
    k2.swoosh_r_forward = _swoosh_r
    def _no_deriv(*a, **kw):
        raise RuntimeError("k2 deriv stub hit at inference — should not happen")
    k2.swoosh_l_forward_and_deriv = _no_deriv
    k2.swoosh_r_forward_and_deriv = _no_deriv
    sys.modules["k2"] = k2

    icefall = types.ModuleType("icefall")
    utils = types.ModuleType("icefall.utils")
    @contextlib.contextmanager
    def torch_autocast(*a, **kw):
        yield
    utils.torch_autocast = torch_autocast
    def make_pad_mask(lengths, max_len=0):
        assert lengths.ndim == 1
        max_len = max(max_len, int(lengths.max()))
        n = lengths.size(0)
        rng = torch.arange(max_len, device=lengths.device)
        return rng.unsqueeze(0).expand(n, max_len) >= lengths.unsqueeze(1)
    utils.make_pad_mask = make_pad_mask
    icefall.utils = utils
    sys.modules["icefall"] = icefall
    sys.modules["icefall.utils"] = utils


install_stubs()
sys.path.insert(0, ZF_DIR)

from scaling import ScheduledFloat                      # noqa: E402
from subsampling import Conv2dSubsampling               # noqa: E402
from zipformer import Zipformer2                        # noqa: E402
from scaling_converter import convert_scaled_to_non_scaled  # noqa: E402


# ---------------------------------------------------------------- GPU re-authoring patches
# All numerically equivalent to the icefall eval path:
#  * masked_fill(mask, -1000)      -> additive float bias {0,-1000} (icefall itself uses -1000)
#  * rel-shift as_strided/gather   -> pad+reshape+slice (verified exact, diff 0.0)
#  * .chunk()                      -> slices (SPLIT is GPU-banned)
#  * scaling.softmax custom fn     -> torch.softmax
#  * BiasNorm / SwooshL/R custom autograd -> plain eval math
def apply_gpu_patches():
    import torch.nn.functional as F
    import scaling as sc
    import zipformer as zf

    def softplus_stable(z):
        # guard-free stable softplus: relu(z) + log1p(exp(-|z|)); the jax logaddexp
        # lowering emits inf-guard SELECT_V2/EQUAL/LOGICAL_AND chains (GPU-banned)
        return torch.relu(z) + torch.log1p(torch.exp(-torch.abs(z)))

    def swoosh_l(x):
        return softplus_stable(x - 4.0) - 0.08 * x - 0.035

    def swoosh_r(x):
        return softplus_stable(x - 1.0) - 0.08 * x - 0.313261687

    sc.SwooshL.forward = lambda self, x: swoosh_l(x)
    sc.SwooshR.forward = lambda self, x: swoosh_r(x)
    sc.SwooshLForward = swoosh_l
    sc.SwooshRForward = swoosh_r

    sc._no_op = lambda x: x                 # was x.chunk(1)[0] -> split_with_sizes, unlowerable
    sc.Identity.forward = lambda self, x: x

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
    sc.BiasNorm.forward = biasnorm_forward

    zf.softmax = lambda x, dim: torch.softmax(x, dim=dim)

    def attn_forward(self, x, pos_emb, key_padding_mask=None, attn_mask=None):
        assert attn_mask is None
        x = self.in_proj(x)
        qhd, phd, H = self.query_head_dim, self.pos_head_dim, self.num_heads
        T, B, _ = x.shape
        qd = qhd * H
        q = x[..., 0:qd].reshape(T, B, H, qhd).permute(2, 1, 0, 3)
        k = x[..., qd:2 * qd].reshape(T, B, H, qhd).permute(2, 1, 3, 0)
        p = x[..., 2 * qd:].reshape(T, B, H, phd).permute(2, 1, 0, 3)
        attn_scores = torch.matmul(q, k)                    # (H,B,T,T)
        pos_emb = self.linear_pos(pos_emb)
        L = 2 * T - 1
        pos_emb = pos_emb.reshape(-1, L, H, phd).permute(2, 0, 3, 1)
        pos_scores = torch.matmul(p, pos_emb)               # (H,B,T,2T-1)
        # rel->abs shift, pad+reshape+slice (== as_strided path, verified exact)
        pos_scores = F.pad(pos_scores, (0, 1)).reshape(H, B, T * 2 * T)
        pos_scores = pos_scores[..., :T * 2 * T - T].reshape(H, B, T, 2 * T - 1)
        attn_scores = attn_scores + pos_scores[..., T - 1:]
        if key_padding_mask is not None:                    # float (B,T): 0 valid / -1000 pad
            attn_scores = attn_scores + key_padding_mask.unsqueeze(1)
        return torch.softmax(attn_scores, dim=-1)
    zf.RelPositionMultiheadAttentionWeights.forward = attn_forward

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
    zf.NonlinAttention.forward = nonlin_forward

    def conv_forward(self, x, src_key_padding_mask=None, chunk_size=-1):
        x = self.in_proj(x)                                 # (T,B,2C)
        C = x.shape[-1] // 2
        x, s = x[..., :C], x[..., C:]
        s = self.sigmoid(self.balancer1(s))
        x = self.activation1(x) * s
        x = self.activation2(x)
        x = x.permute(1, 2, 0)                              # (B,C,T)
        if src_key_padding_mask is not None:                # gate {1,0} == masked_fill(...,0)
            x = x * (1.0 + src_key_padding_mask.unsqueeze(1) / 1000.0)
        x = self.depthwise_conv(x)
        x = self.balancer2(x)
        x = x.permute(2, 0, 1)
        return self.out_proj(self.whiten(x))
    zf.ConvolutionModule.forward = conv_forward

    import subsampling as sub

    def embed_forward(self, x, x_lens):
        # icefall body minus the `assert ... == x_lens.max().item()` data-dependent guard
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.convnext(x)
        b, c, t, f = x.size()
        x = x.transpose(1, 2).reshape(b, t, c * f)
        x = self.out(x)
        x = self.out_whiten(x)
        x = self.out_norm(x)
        x = self.dropout(x)
        return x, (x_lens - 7) // 2
    sub.Conv2dSubsampling.forward = embed_forward

    def cpe_forward(self, x, left_context_len=0):
        assert left_context_len == 0
        if self.pe is None or self.pe.size(0) < 2 * x.size(0) - 1:
            self.extend_pe(x, left_context_len)     # eager pre-warm only; no-op at export
        pe = self.pe
        T = x.size(0)
        start = pe.size(0) // 2 - T + 1
        end = pe.size(0) // 2 + T
        return pe[start:end].unsqueeze(0)           # dropout is a no-op in eval
    zf.CompactRelPositionalEncoding.forward = cpe_forward

    # expand -> cat-repeat (expand lowers to BROADCAST_TO, GPU-banned)
    def upsample_forward(self, src):
        upsample = self.upsample
        T, B, C = src.shape
        src = torch.cat([src.unsqueeze(1)] * upsample, dim=1)
        return src.reshape(T * upsample, B, C)
    zf.SimpleUpsample.forward = upsample_forward

    def downsample_forward(self, src):
        T, B, C = src.shape
        ds = self.downsample
        d_seq_len = (T + ds - 1) // ds
        pad = d_seq_len * ds - T
        if pad > 0:
            src = torch.cat([src] + [src[T - 1:]] * pad, dim=0)
        src = src.reshape(d_seq_len, ds, B, C)
        if not hasattr(self, "_w"):
            # bake softmax(bias): as a live op it stays a rank-1 SOFTMAX on a
            # constant, which ML Drift rejects (device "Failed to compile model")
            self._w = self.bias.detach().softmax(dim=0).reshape(1, ds, 1, 1)
        return (src * self._w).sum(dim=1)
    zf.SimpleDownsample.forward = downsample_forward

    # masks arrive as a per-rate tuple (ds 1,2,4,8) instead of in-graph [..., ::ds]
    # strided slicing, which lowers to GATHER_ND
    def zip2_forward(self, x, x_lens, src_key_padding_mask=None):
        ds2idx = {1: 0, 2: 1, 4: 2, 8: 3}
        outputs = []
        for i, module in enumerate(self.encoders):
            ds = self.downsampling_factor[i]
            x = zf.convert_num_channels(x, self.encoder_dim[i])
            m = None if src_key_padding_mask is None else src_key_padding_mask[ds2idx[ds]]
            x = module(x, chunk_size=-1, feature_mask=1.0,
                       src_key_padding_mask=m, attn_mask=None)
            outputs.append(x)
        x = self._get_full_dim_output(outputs)
        x = self.downsample_output(x)
        return x, (x_lens + 1) // 2
    zf.Zipformer2.forward = zip2_forward


# ---------------------------------------------------------------- model build (medium defaults)
def build_model():
    encoder_embed = Conv2dSubsampling(
        in_channels=80, out_channels=192,
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)))
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=(1, 2, 4, 8, 4, 2),
        num_encoder_layers=CFG["layers"],
        encoder_dim=CFG["dims"],
        encoder_unmasked_dim=CFG["unmasked"],
        query_head_dim=(32,),
        pos_head_dim=(4,),
        value_head_dim=(12,),
        pos_dim=48,
        num_heads=(4, 4, 4, 8, 4, 4),
        feedforward_dim=CFG["ff"],
        cnn_module_kernel=(31, 31, 15, 15, 15, 31),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=False,
        chunk_size=(-1,),
        left_context_frames=(-1,),
    )
    ctc_output = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(max(CFG["dims"]), VOCAB),
        nn.LogSoftmax(dim=-1),
    )

    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd["model"] if "model" in sd else sd
    ee = {k[len("encoder_embed."):]: v for k, v in sd.items() if k.startswith("encoder_embed.")}
    en = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    ct = {k[len("ctc_output."):]: v for k, v in sd.items() if k.startswith("ctc_output.")}
    encoder_embed.load_state_dict(ee, strict=True)
    encoder.load_state_dict(en, strict=True)
    ctc_output.load_state_dict(ct, strict=True)
    print(f"[build] loaded: embed {len(ee)} keys, encoder {len(en)} keys, ctc {len(ct)} keys")

    class ZipformerCtc(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_embed = encoder_embed
            self.encoder = encoder
            self.ctc_output = ctc_output
            self.register_buffer("x_lens", torch.tensor([T_IN], dtype=torch.int64))

        def forward(self, x, b1, b2, b4, b8):       # x: [1,T_IN,80]; b*: [1,ceil(T50/ds)] 0/-1000
            x, x_lens = self.encoder_embed(x, self.x_lens)
            x = x.permute(1, 0, 2)                  # (T', N, C)
            out, _ = self.encoder(x, x_lens, (b1, b2, b4, b8))
            out = out.permute(1, 0, 2)              # (N, T'', C)
            return self.ctc_output(out)             # log_probs [1, T'', 500]

    return ZipformerCtc().eval()


# ---------------------------------------------------------------- features + decode
def fbank(wav_path):
    import torchaudio
    wave, sr = torchaudio.load(wav_path)
    if sr != SR:
        wave = torchaudio.functional.resample(wave, sr, SR)
    wave = wave.mean(0, keepdim=True)
    feats = torchaudio.compliance.kaldi.fbank(
        wave, num_mel_bins=80, sample_frequency=SR, dither=0.0,
        snip_edges=False, high_freq=-400.0)         # [T, 80], matches icefall kaldifeat opts
    T = feats.shape[0]
    if T < T_IN:
        pad = feats.new_full((T_IN - T, 80), math.log(1e-10))
        feats = torch.cat([feats, pad], 0)
    else:
        feats = feats[:T_IN]
    return feats.unsqueeze(0)                       # [1, T_IN, 80]


def load_tokens():
    id2tok = {}
    for line in open(TOKENS, encoding="utf-8"):
        tok, idx = line.rsplit(maxsplit=1)
        id2tok[int(idx)] = tok
    return id2tok


def greedy_ctc(log_probs, id2tok):
    ids = log_probs[0].argmax(-1)
    out, prev = [], -1
    for i in ids.tolist():
        if i != prev and i != 0:
            out.append(id2tok.get(i, "?"))
        prev = i
    return "".join(out).replace("▁", " ").strip()


# ---------------------------------------------------------------- stages
def valid_frames(wav_path):
    """(fbank_frames, 50Hz_frames_after_embed, 25Hz_output_frames) for the real audio."""
    import torchaudio
    wave, sr = torchaudio.load(wav_path)
    n = wave.shape[1] * SR // sr
    t_fb = n // 160 + 1                              # snip_edges=False, 10ms hop
    t50 = (t_fb - 7) // 2
    return t_fb, t50, (t50 + 1) // 2


def make_bias(t50_valid, t50_total):
    """Per-rate additive biases (0 valid / -1000 pad), host-side [::ds] slicing."""
    b = torch.full((1, t50_total), -1000.0)
    b[0, :t50_valid] = 0.0
    return b, b[:, ::2].contiguous(), b[:, ::4].contiguous(), b[:, ::8].contiguous()


def stage_gold(m):
    """Reference: unpadded input through the ORIGINAL icefall eval path (bool masks)."""
    from icefall.utils import make_pad_mask
    import torchaudio
    wave, sr = torchaudio.load(WAV)
    wave = wave.mean(0, keepdim=True)
    feats = torchaudio.compliance.kaldi.fbank(
        wave, num_mel_bins=80, sample_frequency=SR, dither=0.0,
        snip_edges=False, high_freq=-400.0).unsqueeze(0)
    T = feats.shape[1]
    with torch.no_grad():
        e, elens = m.encoder_embed(feats, torch.tensor([T]))
        mask = make_pad_mask(elens)
        out, _ = m.encoder(e.permute(1, 0, 2), elens, mask)
        lp = m.ctc_output(out.permute(1, 0, 2))
    text = greedy_ctc(lp, load_tokens())
    print(f"[gold] T_fbank {T} -> log_probs {tuple(lp.shape)}")
    print(f"[gold] TEXT: {text}")
    return lp, text


def stage_patched_parity(m, gold_lp, gold_text):
    """After patches: padded window + additive bias must reproduce the gold transcript."""
    x = fbank(WAV)
    t_fb, t50v, t25v = valid_frames(WAV)
    t50_total = (T_IN - 7) // 2
    biases = make_bias(t50v, t50_total)
    with torch.no_grad():
        lp = m(x, *biases)                          # raw logits (LogSoftmax moved host-side)
    text = greedy_ctc(lp, load_tokens())
    lsm = torch.log_softmax(lp, dim=-1)
    nv = min(t25v, gold_lp.shape[1], lp.shape[1])
    corr = np.corrcoef(lsm[0, :nv].numpy().ravel(), gold_lp[0, :nv].numpy().ravel())[0, 1]
    md = np.abs(lsm[0, :nv].numpy() - gold_lp[0, :nv].numpy()).max()
    print(f"[patched] valid-region vs gold: corr {corr:.6f} max|diff| {md:.4f}")
    print(f"[patched] TEXT: {text}")
    print(f"[patched] text match gold: {text == gold_text}")
    np.save(os.path.join(HERE, "ref_in.npy"), x.numpy())
    for i, b in enumerate(biases):
        np.save(os.path.join(HERE, f"ref_bias{i}.npy"), b.numpy())
    np.save(os.path.join(HERE, "ref_logprobs.npy"), lp.numpy())
    return x, biases


def stage_convert(m, x, biases):
    import litert_torch  # import AFTER model construction
    out = os.path.join(HERE, OUT_NAME)
    cm = litert_torch.convert(m, (x, *biases))
    cm.export(out)
    print(f"[convert] wrote {out}")
    it, clean = opcheck(out, "zip")
    lp_t = tfl_run2(it, x.numpy(), [b.numpy() for b in biases])
    ref = np.load(os.path.join(HERE, "ref_logprobs.npy"))
    corr = np.corrcoef(lp_t.ravel(), ref.ravel())[0, 1]
    md = np.abs(lp_t - ref).max()
    print(f"[convert] tflite vs patched-torch: corr {corr:.6f} max|diff| {md:.4f}")
    text = greedy_ctc(torch.from_numpy(lp_t), load_tokens())
    print(f"[convert] TFLITE TEXT: {text}")
    if clean:
        quant_fp16(out)
    return clean


def quant_fp16(path):
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
    print(f"[fp16] wrote {out}")
    it, _ = opcheck(out, "zip_fp16")
    x = np.load(os.path.join(HERE, "ref_in.npy"))
    biases = [np.load(os.path.join(HERE, f"ref_bias{i}.npy")) for i in range(4)]
    ref = np.load(os.path.join(HERE, "ref_logprobs.npy"))
    lp = tfl_run2(it, x, biases)
    corr = np.corrcoef(lp.ravel(), ref.ravel())[0, 1]
    print(f"[fp16] vs torch: corr {corr:.6f} max|diff| {np.abs(lp - ref).max():.4f}")
    text = greedy_ctc(torch.from_numpy(lp), load_tokens())
    print(f"[fp16] TEXT: {text}")


def opcheck(path, label):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    ops = collections.Counter(d.get("op_name", "?") for d in it._get_ops_details())
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    over = sum(1 for d in it.get_tensor_details() if len(d.get("shape", [])) > 4)
    fft = {k: v for k, v in ops.items() if any(t in k.upper() for t in ("FFT", "STFT", "COMPLEX"))}
    print(f"[{label}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} FFT:{fft or 'NONE'} "
          f"size {os.path.getsize(path)/1e6:.1f}MB")
    print(f"[{label}] VERDICT:", "GPU-CLEAN" if not bad and not over and not fft else f"BLOCKERS {bad}")
    return it, (not bad and not over and not fft)


def tfl_run2(it, x, biases):
    by_len = {b.shape[1]: b for b in biases}
    for d in it.get_input_details():
        if len(d["shape"]) == 3:
            it.set_tensor(d["index"], x.astype(np.float32))
        else:
            it.set_tensor(d["index"], by_len[d["shape"][1]].astype(np.float32))
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"])


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    m = build_model()
    gold_lp, gold_text = stage_gold(m)
    apply_gpu_patches()
    convert_scaled_to_non_scaled(m, inplace=True, is_onnx=False)
    # LogSoftmax lowers to REDUCE_MAX (rank-drop) which ML Drift rejects on device;
    # greedy CTC is argmax-invariant, so emit raw logits and log-softmax host-side
    m.ctc_output[2] = nn.Identity()
    x, biases = stage_patched_parity(m, gold_lp, gold_text)
    if stage in ("convert", "all"):
        ok = stage_convert(m, x, biases)
        print("[done] GPU-CLEAN" if ok else "[done] blockers remain")
    os._exit(0)
