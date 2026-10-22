"""Timestral re-authored as one fixed-shape graph for LiteRT export.

One graph per (rows R, context patches L, horizon patches H); S = L + H tokens.

Inputs (all float32, built by the host, see host_forecast.py):
  patches  [R, L, 64]   per context patch: 32 normalized values (0 where missing) + 32 missing flags
  dec      [R, H, 768]  decoder tokens = the checkpoint's forecast_query on every row
  bias     [1, 1, S, S] additive time-attention mask: 0 = may attend, NEG = blocked
  cos/sin  [1, 1, S, 64] RoPE tables for the positions arange(S) - first_observed_patch
Output:
  quantiles [R, H, 32, 21] monotone quantiles in the normalized (asinh) space of the last
  context patch, i.e. the tensor the official model passes to denormalize_quantiles.

Exact rewrites of the official model (timestral a534bdb), each checked against it in
verify_graph.py:
  * wQKV rows are stored as (qkv, head_dim, heads); they are re-ordered to heads-major and
    split into three Linear layers. The 1/sqrt(64) = 2^-3 score scale is folded into Q.
  * Time attention: one masked softmax over [Enc; Dec] with an additive mask equals the
    official two-stream split (the masked weights are exactly 0). Products are einsums so
    the converter keeps rank-4 BATCH_MATMULs.
  * Variate attention over rows of different series (R = 1, or the R = 2 mirror pair) has
    one visible key per query, so softmax = 1 and it reduces to wO(V(x) * sigmoid(G(x))).
  * SwiGLU w12 is split into its gate and up halves (two Linear layers, no SPLIT op).
  * Monotone head: the head's output rows are permuted per step to quantile order
    [11..20, 9..0, 10]; one fixed 42x21 matrix on concat(softplus(raw), raw) then sums the
    increments in the same order as the official cumsum (up: 11 -> j, down: 9 -> j) and adds
    the median last. softplus is written select-free: relu(x) + log(1 + exp(-|x|)).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PATCH = 32
D_MODEL = 768  # Timestral-209M; the modules read the real sizes from the config
N_HEADS = 12
HEAD_DIM = 64
N_Q = 21
MEDIAN = 10
NEG = -1.0e30  # host_forecast.NEG is the one the host uses (logits reach ~2.4e5; -1e4 does not block)
# Head dims RoPE rotates (inv_freq is zero past index 16, rotate_half pairs i with i + 32); the rest pass.
R_DIMS = list(range(0, 16)) + list(range(32, 48))
NR_DIMS = list(range(16, 32)) + list(range(48, 64))
# Head output order per step: up increments ascending, down increments descending, median last.
HEAD_ORDER = list(range(MEDIAN + 1, N_Q)) + list(range(MEDIAN - 1, -1, -1)) + [MEDIAN]


def _linear(weight, bias=None):
    out_f, in_f = weight.shape
    lin = nn.Linear(in_f, out_f, bias=bias is not None)
    with torch.no_grad():
        lin.weight.copy_(weight)
        if bias is not None:
            lin.bias.copy_(bias)
    return lin


def monotone_matrix(levels, width_factor):
    """[42, 21] matrix M with out = concat(softplus(raw_p), raw_p) @ M, raw_p = raw[HEAD_ORDER]."""
    m = levels.index(0.5)
    widths = [0.0] * len(levels)
    for i in range(m + 1, len(levels)):
        widths[i] = levels[i] - levels[i - 1]
    for i in range(m):
        widths[i] = levels[i + 1] - levels[i]
    # Same fp32 coefficients as the official `width_factor * widths_t`.
    coef = (torch.tensor(width_factor, dtype=torch.float32) * torch.tensor(widths, dtype=torch.float32)).tolist()
    mat = torch.zeros(2 * N_Q, N_Q, dtype=torch.float32)
    pos = {q: k for k, q in enumerate(HEAD_ORDER)}
    for j in range(N_Q):
        if j > m:
            for i in range(m + 1, j + 1):
                mat[pos[i], j] = coef[i]
        elif j < m:
            for i in range(j, m):
                mat[pos[i], j] = -coef[i]
        mat[N_Q + pos[m], j] = 1.0
    return mat


class RMSNorm(nn.Module):
    """Official RMSNorm in float32; `pre_scale_exp` k > 0 is the exact 2^-k fp16-safe form.

    mode "exact": x * rsqrt(mean(x^2) + eps) * scale (the official order).
    mode "pow2":  xs = x * 2^-k; xs * rsqrt(mean(xs^2) + eps * 4^-k) * scale. Power-of-two
                  scaling commutes with fp32 rounding, so this is the same number in fp32.
    mode "amax":  m = max(1, max|x|) per token; (x/m) * rsqrt(mean((x/m)^2) + eps/m^2) * scale.
    """

    def __init__(self, scale, eps, mode="exact", k=0):
        super().__init__()
        self.scale = nn.Parameter(scale.detach().clone(), requires_grad=False)
        self.eps = float(eps)
        self.mode = mode
        self.k = int(k)

    def forward(self, x):
        if self.mode == "amax":
            m = x.abs().amax(-1, keepdim=True).clamp_min(1.0)
            xs = x / m
            return xs * torch.rsqrt((xs * xs).mean(-1, keepdim=True) + self.eps / (m * m)) * self.scale
        if self.mode == "pow2" and self.k > 0:
            s = 2.0 ** -self.k
            xs = x * s
            return xs * torch.rsqrt((xs * xs).mean(-1, keepdim=True) + self.eps * s * s) * self.scale
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) * self.scale


def _heads_major(src):
    """wQKV [3d, d] rows (qkv, head_dim, heads) -> three [d, d] blocks with rows (heads, head_dim)."""
    w = src.wQKV.weight.detach()
    d, nh, hd = src.embed_dim, src.num_heads, src.head_dim
    return w.view(3, hd, nh, d).permute(0, 2, 1, 3).reshape(3, d, d)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class TimeAttention(nn.Module):
    """attn_k > 0: the logits are computed 2^-attn_k smaller (folded into Q), the row max is taken
    and subtracted at that scale, and only the non-positive differences are scaled back up before
    the softmax: the same numbers in fp32 (power-of-two scaling), logits within fp16 range."""

    def __init__(self, src, attn_k=0, key_center=None, attn_clamp=False):
        super().__init__()
        self.d, self.nh, self.hd = src.embed_dim, src.num_heads, src.head_dim
        self.attn_k = int(attn_k)
        # attn_clamp: floor the max-subtracted logits at -3e4 before the softmax. exp(-3e4) and exp(-inf)
        # are both 0 (fp32 and fp16), so the numbers do not change; it keeps -inf out of the softmax
        # input, where ML Drift's fp16 kernel turned rows that start with -inf into NaN (desktop GPU, Metal).
        self.attn_clamp = bool(attn_clamp)
        q, k, v = _heads_major(src)
        self.q = _linear(q * (self.hd ** -0.5) * (2.0 ** -self.attn_k))
        if key_center is None:
            self.k = _linear(k)
        else:
            # key_center [heads, 32]: subtracted from the unrotated head dims (16-31, 48-63) of every key.
            # RoPE leaves those dims alone, so each query's logits shift by one constant: softmax unchanged.
            kb = torch.zeros(self.nh, self.hd)
            kb[:, NR_DIMS] = -torch.as_tensor(key_center, dtype=torch.float32)
            self.k = _linear(k, kb.reshape(-1))
        self.v = _linear(v)
        self.g = _linear(src._gate.weight.detach())
        self.o = _linear(src.wO.weight.detach())

    def forward(self, x, bias, cos, sin):
        r, s, _ = x.shape
        q = self.q(x).view(r, s, self.nh, self.hd).permute(0, 2, 1, 3)
        k = self.k(x).view(r, s, self.nh, self.hd).permute(0, 2, 1, 3)
        v = self.v(x).view(r, s, self.nh, self.hd).permute(0, 2, 1, 3)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        w = torch.einsum("bhqd,bhkd->bhqk", q, k) + bias
        if self.attn_k:
            w = (w - w.amax(dim=-1, keepdim=True)) * (2.0 ** self.attn_k)
        if self.attn_clamp:
            w = torch.clamp(w, min=-3.0e4)
        p = torch.softmax(w, dim=-1)
        o = torch.einsum("bhqk,bhkd->bhqd", p, v).permute(0, 2, 1, 3).reshape(r, s, self.d)
        return self.o(o * torch.sigmoid(self.g(x)))


class VariateAttention(nn.Module):
    """Variate attention when every row is its own series: one key per query."""

    def __init__(self, src):
        super().__init__()
        _, _, v = _heads_major(src)
        self.v = _linear(v)
        self.g = _linear(src._gate.weight.detach())
        self.o = _linear(src.wO.weight.detach())

    def forward(self, x):
        return self.o(self.v(x) * torch.sigmoid(self.g(x)))


class SwiGLU(nn.Module):
    def __init__(self, src):
        super().__init__()
        w12 = src.w12.weight.detach()
        hidden = w12.shape[0] // 2
        self.gate = _linear(w12[:hidden])
        self.up = _linear(w12[hidden:])
        self.w3 = _linear(src.w3.weight.detach())

    def forward(self, x):
        return self.w3(self.gate(x) * F.silu(self.up(x)))


class ResidualMLP(nn.Module):
    """output(silu(hidden(x))) + residual(x); `out_rows` optionally permutes the output rows."""

    def __init__(self, src, out_rows=None):
        super().__init__()
        hb = src.hidden_layer.bias
        ob, rb = src.output_layer.bias, src.residual_layer.bias
        ow, rw = src.output_layer.weight.detach(), src.residual_layer.weight.detach()
        if out_rows is not None:
            ow, rw = ow[out_rows], rw[out_rows]
            ob = ob[out_rows] if ob is not None else None
            rb = rb[out_rows] if rb is not None else None
        self.hidden = _linear(src.hidden_layer.weight.detach(), hb.detach() if hb is not None else None)
        self.output = _linear(ow, ob.detach() if ob is not None else None)
        self.residual = _linear(rw, rb.detach() if rb is not None else None)

    def forward(self, x):
        return self.output(F.silu(self.hidden(x))) + self.residual(x)


class Block(nn.Module):
    def __init__(self, src, norm_mode, norm_k, attn_k=0, key_center=None, attn_clamp=False):
        super().__init__()
        self.kinds = [a.is_time for a in src.attns]
        self.attn_norms = nn.ModuleList(
            [RMSNorm(n.scale, n.eps, norm_mode, norm_k.get(f"attn{j}", 0)) for j, n in enumerate(src.attn_norms)])
        self.attns = nn.ModuleList(
            [TimeAttention(a, attn_k, key_center, attn_clamp) if a.is_time else VariateAttention(a)
             for a in src.attns])
        self.mlp_norm = RMSNorm(src.mlp_norm.scale, src.mlp_norm.eps, norm_mode, norm_k.get("mlp", 0))
        self.mlp = SwiGLU(src.mlp)

    def forward(self, x, bias, cos, sin):
        for is_time, norm, attn in zip(self.kinds, self.attn_norms, self.attns):
            x = x + (attn(norm(x), bias, cos, sin) if is_time else attn(norm(x)))
        return x + self.mlp(self.mlp_norm(x))


class TimestralGraph(nn.Module):
    """Patch embedding + 24 blocks + out_norm + monotone quantile head on the decoder tokens.

    resid_k > 0 runs the whole residual stream 2^-resid_k smaller (fp16 headroom): the embedding,
    every sublayer's output projection (wO, w3) and the decoder tokens are scaled by 2^-k and every
    RMSNorm eps by 4^-k. RMSNorm is scale-invariant, so the norm outputs, and the graph output, are
    the same numbers in fp32 (power-of-two scaling is exact).
    """

    def __init__(self, model, context_patches, horizon_patches, norm_mode="exact", norm_k=None, resid_k=0,
                 attn_k=0, key_center=None, attn_clamp=False):
        super().__init__()
        c = model.config
        assert c.patch_size == PATCH and c.head_dim == HEAD_DIM
        assert c.num_quantiles == N_Q and c.median_index == MEDIAN
        self.d_model = c.d_model
        norm_k = norm_k or {}
        key_center = key_center or {}
        self.L, self.H = int(context_patches), int(horizon_patches)
        self.embed = ResidualMLP(model.representation.patch_proj)
        self.blocks = nn.ModuleList(
            [Block(b, norm_mode, norm_k.get(i, {}), attn_k, key_center.get(i), attn_clamp) for i, b in
             enumerate(model.transformer.layers)])
        on = model.transformer.out_norm
        self.out_norm = RMSNorm(on.scale, on.eps, norm_mode, norm_k.get("out", {}).get("out", 0))
        rows = [step * N_Q + q for step in range(PATCH) for q in HEAD_ORDER]
        self.head = ResidualMLP(model.head.projection, out_rows=rows)
        self.mono = _linear(monotone_matrix(list(c.quantile_knots), c.quantile_width_factor).T.contiguous())
        self.resid_k = int(resid_k)
        if self.resid_k:
            s = 2.0 ** -self.resid_k
            with torch.no_grad():
                for lin in (self.embed.output, self.embed.residual):
                    lin.weight.mul_(s)
                    if lin.bias is not None:
                        lin.bias.mul_(s)
                for blk in self.blocks:
                    for a in blk.attns:
                        a.o.weight.mul_(s)
                    blk.mlp.w3.weight.mul_(s)
            for m in self.modules():
                if isinstance(m, RMSNorm):
                    m.eps = m.eps * s * s

    def forward(self, patches, dec, bias, cos, sin):
        r = patches.shape[0]
        if self.resid_k:
            dec = dec * (2.0 ** -self.resid_k)
        x = torch.cat([self.embed(patches), dec], dim=1)
        for blk in self.blocks:
            x = blk(x, bias, cos, sin)
        y = self.out_norm(x)[:, self.L:, :]
        raw = self.head(y).view(r, self.H, PATCH, N_Q)
        sp = F.relu(raw) + torch.log(1.0 + torch.exp(-torch.abs(raw)))
        return self.mono(torch.cat([sp, raw], dim=-1))


def rope_table(model, p_min, p_max):
    """cos/sin [P, 64] for positions p_min..p_max, built with the official SinCosRoPE."""
    pos = torch.arange(p_min, p_max + 1, dtype=torch.long)[None, :]
    dummy = torch.zeros(1, dtype=torch.float32)
    cos, sin = model.transformer.rope(dummy, pos)
    return cos[0].numpy().astype(np.float32), sin[0].numpy().astype(np.float32)


def sample_inputs(rows, context_patches, horizon_patches, d_model=D_MODEL):
    s = context_patches + horizon_patches
    return (
        torch.zeros(rows, context_patches, 2 * PATCH),
        torch.zeros(rows, horizon_patches, d_model),
        torch.zeros(1, 1, s, s),
        torch.ones(1, 1, s, HEAD_DIM),
        torch.zeros(1, 1, s, HEAD_DIM),
    )


INPUT_NAMES = ("patches", "dec", "bias", "cos", "sin")


def count_params(module):
    return sum(math.prod(p.shape) for p in module.parameters())
