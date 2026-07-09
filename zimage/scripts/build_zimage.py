"""Z-Image-Turbo S3-DiT → LiteRT GPU conversion probe.

Builds a fixed-shape, GPU-clean export graph for the Z-Image transformer block
stack and runs the litert_gpu_toolkit compatibility checker. All host-side prep
(patchify, position-id grid, RoPE frequency GATHER, sequence padding, unpatchify)
is moved OUT of the graph; the graph consumes already-prepared runtime tensors:

    x    [1, N, dim]        pre-embedded unified token sequence
    cos  [1, 1, N, hd//2]   RoPE cosines (host-computed from position ids)
    sin  [1, 1, N, hd//2]   RoPE sines
    adaln[1, ADALN_DIM]     timestep embedding for global adaLN modulation

RoPE is applied in a half-split real form (even/odd slice + concat): because
q and k receive the identical channel permutation, the q·k dot product is
unchanged, so this is bit-exact to the reference interleaved (view_as_complex)
RoPE while keeping every tensor 4D.
"""

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/Users/majimadaisuke/Downloads/depthanything-android")
from check_ops import check_ops  # ai_edge_litert-based (no tensorflow dependency)

DIM = 3840
N_HEADS = 30
HEAD_DIM = DIM // N_HEADS          # 128
ADALN_DIM = 256                    # min(dim, ADALN_EMBED_DIM)
FFN_HIDDEN = int(DIM / 3 * 8)      # 10240


MAXNORM_RMS = True  # fp16-safe max-normalized RMS (device residual stream hits ~4400)


def safe_rms(x, weight, eps=1e-5):
    """fp16-safe max-normalized RMSNorm over the last dim.

    The device residual stream reaches ~4400, so plain mean(x^2) overflows the
    delegate's fp16 accumulator (4400^2 = 1.9e7 > 65504). Dividing each row by its
    own max bounds x^2 in [0,1]; mathematically identical to standard RMSNorm
    (device-proven in Qwen3). The amax reduction trips litert-torch's NHWC layout
    pass on 4D tensors, so a 4D input is reshaped to 3D [B, H*N, hd] for the norm
    (amax is over the last dim either way) and reshaped back.
    """
    if not MAXNORM_RMS:
        var = (x * x).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + eps) * weight
    shape = x.shape
    x3 = x.reshape(shape[0], -1, shape[-1]) if x.dim() == 4 else x
    m = x3.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    xs = x3 / m
    var = (xs * xs).mean(dim=-1, keepdim=True)
    y = xs * torch.rsqrt(var + eps / (m * m)) * weight
    return y.reshape(shape)


HALF = HEAD_DIM // 2


def _even_odd_perm():
    """Per-head channel permutation [0,2,..,126, 1,3,..,127] (evens then odds)."""
    return list(range(0, HEAD_DIM, 2)) + list(range(1, HEAD_DIM, 2))


def apply_rope_halfsplit(x, cos, sin):
    """Half-split real RoPE. x: [B, H, N, hd]; cos/sin: [B, 1, N, hd//2].

    Assumes x channels are pre-permuted to [evens(hd/2); odds(hd/2)] (the
    de-interleave is baked into the to_q / to_k weights), so the even/odd halves
    are CONTIGUOUS slices (SLICE, GPU-clean) instead of stride-2 slices (which
    lower to GATHER_ND). q and k share the identical permutation, so attention
    scores are bit-exact to the reference interleaved (view_as_complex) RoPE.
    """
    x_even = x[..., :HALF]        # [B, H, N, hd//2]
    x_odd = x[..., HALF:]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.cat([out_even, out_odd], dim=-1)


class ExportBlock(nn.Module):
    """GPU-clean reimplementation of one ZImageTransformerBlock (modulation=True).

    Weights are lifted from a diffusers ZImageTransformerBlock so the op graph
    matches the real model. Global adaLN modulation only (single-image basic mode,
    no per-token noise mask).
    """

    def __init__(self, block):
        super().__init__()
        attn = block.attention
        # Bake the RoPE de-interleave permutation into to_q / to_k so their outputs
        # arrive in [evens; odds] channel order per head (contiguous-slice RoPE).
        perm = _even_odd_perm()
        full_perm = torch.tensor(
            [h * HEAD_DIM + i for h in range(N_HEADS) for i in perm], dtype=torch.long
        )
        self.to_q = self._permuted_linear(attn.to_q, full_perm)
        self.to_k = self._permuted_linear(attn.to_k, full_perm)
        self.to_v = attn.to_v
        self.to_out = attn.to_out[0]
        # qk RMSNorm weights (over head_dim) — permuted to match to_q / to_k.
        perm_t = torch.tensor(perm, dtype=torch.long)
        self.register_buffer("nq_w", attn.norm_q.weight.detach()[perm_t].clone())
        self.register_buffer("nk_w", attn.norm_k.weight.detach()[perm_t].clone())
        # block RMSNorm weights
        self.register_buffer("an1_w", block.attention_norm1.weight.detach().clone())
        self.register_buffer("an2_w", block.attention_norm2.weight.detach().clone())
        self.register_buffer("fn1_w", block.ffn_norm1.weight.detach().clone())
        self.register_buffer("fn2_w", block.ffn_norm2.weight.detach().clone())
        # feed forward
        self.w1 = block.feed_forward.w1
        self.w2 = block.feed_forward.w2
        self.w3 = block.feed_forward.w3
        # adaLN
        self.adaln = block.adaLN_modulation[0]
        self.scale = HEAD_DIM ** -0.5

    @staticmethod
    def _permuted_linear(lin, row_perm):
        """Clone an nn.Linear with its output rows reordered by row_perm."""
        out_f, in_f = lin.weight.shape
        new = nn.Linear(in_f, out_f, bias=lin.bias is not None)
        new.weight = nn.Parameter(lin.weight.detach()[row_perm].clone())
        if lin.bias is not None:
            new.bias = nn.Parameter(lin.bias.detach()[row_perm].clone())
        return new

    def _attn(self, x, cos, sin):
        B, N, _ = x.shape
        q = self.to_q(x).view(B, N, N_HEADS, HEAD_DIM).transpose(1, 2)  # [B,H,N,hd]
        k = self.to_k(x).view(B, N, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.to_v(x).view(B, N, N_HEADS, HEAD_DIM).transpose(1, 2)
        q = safe_rms(q, self.nq_w)
        k = safe_rms(k, self.nk_w)
        q = apply_rope_halfsplit(q, cos, sin)
        k = apply_rope_halfsplit(k, cos, sin)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                       # [B,H,N,hd]
        out = out.transpose(1, 2).reshape(B, N, DIM)
        return self.to_out(out)

    def _ffn(self, x):
        h = (self.w1(x) * torch.sigmoid(self.w1(x))) * self.w3(x)  # silu(w1)*w3
        return self.w2(h)

    def forward(self, x, cos, sin, adaln):
        mod = self.adaln(adaln).unsqueeze(1)              # [B,1,4*dim]
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=2)
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        attn_out = self._attn(safe_rms(x, self.an1_w) * scale_msa, cos, sin)
        x = x + gate_msa * safe_rms(attn_out, self.an2_w)
        ffn_out = self._ffn(safe_rms(x, self.fn1_w) * scale_mlp)
        x = x + gate_mlp * safe_rms(ffn_out, self.fn2_w)
        return x


class ExportDiT(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList([ExportBlock(b) for b in blocks])

    def forward(self, x, cos, sin, adaln):
        for blk in self.blocks:
            x = blk(x, cos, sin, adaln)
        return x


def build_probe(n_layers=2, seq_len=64, out_path="zdit_probe.tflite"):
    from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock

    torch.manual_seed(0)
    blocks = [
        ZImageTransformerBlock(i, DIM, N_HEADS, N_HEADS, 1e-5, True, modulation=True).eval()
        for i in range(n_layers)
    ]
    model = ExportDiT(blocks).eval()

    x = torch.randn(1, seq_len, DIM)
    cos = torch.randn(1, 1, seq_len, HEAD_DIM // 2)
    sin = torch.randn(1, 1, seq_len, HEAD_DIM // 2)
    adaln = torch.randn(1, ADALN_DIM)

    with torch.no_grad():
        ref = model(x, cos, sin, adaln)
    print(f"[forward] out {tuple(ref.shape)} range [{ref.min():.3f},{ref.max():.3f}]")

    import litert_torch
    print("[convert] litert_torch.convert ...")
    result = litert_torch.convert(model, (x, cos, sin, adaln))
    result.export(out_path)
    import os
    print(f"[convert] saved {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")

    return check_ops(out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--out", default="zdit_probe.tflite")
    args = ap.parse_args()
    build_probe(args.layers, args.seq, args.out)
