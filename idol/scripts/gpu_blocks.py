"""GPU-shaped re-authoring of the Idol building blocks for LiteRT CompiledModel (ML Drift).

Every rewrite here is exact in real arithmetic against the idol-sdk modules it is built from:
- attention as rank-4 einsum products ([1, H, L, D]), split over the queries so the score tensor of one split
  stays a few hundred MB (the full joint score matrix of the denoiser is ~4.8 GB at fp16);
- LayerNorm reduced in a per-row down-scaled domain (SafeLayerNorm v2 with eps / S^2), so the fp16 sum of
  squares cannot overflow;
- per-head RMS normalisation of q / k divided by the row max first (the same fp16 reason);
- the denoiser's interleaved rotary pairs de-interleaved by permuting the q / k rows of the projection
  (q . k is invariant under a permutation shared by q and k); the softmax scale 1/8 folded into the q gain;
- the RePo angle map (a 3-to-32 frequency matrix per head) folded into the RePo output projection;
- DINOv3 LayerScale folded into o_proj / down_proj; its 1/8 scale folded into q_proj.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_layer_norm(x, eps, weight=None, bias=None):
    """LayerNorm over the last dim, exact: y = (x/S - mean) * rsqrt(var(x/S) + eps/S^2), S = max(1, max|x|/8)."""
    s = (x.abs().amax(-1, keepdim=True) * 0.125).clamp(min=1.0)
    xs = x / s
    d = xs - xs.mean(-1, keepdim=True)
    var = (d * d).mean(-1, keepdim=True)
    y = d * torch.rsqrt(var + eps / (s * s))
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def safe_rms_unit(x):
    """x / rms(x) over the last dim, computed on x / max|x| so the squares stay <= 1. idol's MultiHeadRMSNorm is
    F.normalize(x) * gamma * sqrt(D) = x / rms(x) * gamma. A mean (not a sum) of squares keeps the converter from
    fusing the pattern into L2_NORMALIZATION."""
    m = x.abs().amax(-1, keepdim=True).clamp(min=1e-6)
    xs = x / m
    return xs * torch.rsqrt((xs * xs).mean(-1, keepdim=True).clamp(min=1e-12))


KEY_CHUNK = 1024           # the A.V product sums over this many keys at a time ...
KEY_CHUNK_MIN_LEN = 4096   # ... when there are at least this many keys (DINOv3 and the context refiner: 4,101)


def attention(q, k, v, chunk=None):
    """softmax(q k^T) v for q [1, H, Lq, D] (already scaled), k / v [1, H, Lk, D] -> [1, Lq, H, D] (token-major).

    Rank-4 einsum products, the queries split into blocks of ``chunk``. Each block's output is transposed to
    token-major before the concatenation: a concatenation of the [1, H, n, D] products themselves is rewritten by
    the converter into one concatenation of the softmax blocks and a single product, which rebuilds the whole
    [1, H, Lq, Lk] score tensor (seen on litert-torch 0.10.0).

    Long key sets (>= KEY_CHUNK_MIN_LEN): v is centred by its mean over the keys (added back after the product;
    exact, each softmax row sums to 1) and the A.V product runs over key chunks of KEY_CHUNK, partial results added.
    For diffuse attention one fp16 product over 8192 keys loses 18 % (ML Drift on Metal, Gaussian decoder block 0);
    centred chunks of 1024 lose 0.6 % at the same speed."""
    lq, lk = q.shape[2], k.shape[2]
    long_keys = lk >= KEY_CHUNK_MIN_LEN
    if long_keys:
        v_mean = v.mean(2, keepdim=True)
        v = v - v_mean

    def weighted_sum(p):
        if not long_keys:
            return torch.einsum('bhqk,bhkd->bhqd', p, v)
        o = None
        for k0 in range(0, lk, KEY_CHUNK):
            t = torch.einsum('bhqk,bhkd->bhqd', p[..., k0:k0 + KEY_CHUNK], v[:, :, k0:k0 + KEY_CHUNK])
            o = t if o is None else o + t
        return o + v_mean

    if not chunk or chunk >= lq:
        s = torch.einsum('bhqd,bhkd->bhqk', q, k)
        return weighted_sum(torch.softmax(s, dim=-1)).permute(0, 2, 1, 3)
    outs = []
    for s0 in range(0, lq, chunk):
        s = torch.einsum('bhqd,bhkd->bhqk', q[:, :, s0:s0 + chunk], k)
        outs.append(weighted_sum(torch.softmax(s, dim=-1)).permute(0, 2, 1, 3))
    return torch.cat(outs, dim=1)


def _linear(weight, bias):
    lin = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    lin.weight.data = weight.detach().float().clone()
    if bias is not None:
        lin.bias.data = bias.detach().float().clone()
    return lin


def _gelu_tanh(x):
    return F.gelu(x, approximate='tanh')


# -- Denoiser ------------------------------------------------------------------------------------------------

def deinterleave_perm(head_dim):
    return list(range(0, head_dim, 2)) + list(range(1, head_dim, 2))


class RePoGPU(nn.Module):
    """RePo3DRotaryEmbedding -> (cos, sin) [1, L, H, 32] with the frequency matrix folded into one Linear."""

    def __init__(self, repo, num_heads=16, head_dim=64):
        super().__init__()
        self.norm_w = nn.Parameter(repo.norm.weight.detach().float().clone(), requires_grad=False)
        self.norm_b = nn.Parameter(repo.norm.bias.detach().float().clone(), requires_grad=False)
        self.eps = repo.norm.eps
        self.gate = _linear(repo.gate_map.weight, None)
        self.content = _linear(repo.content_map.weight, None)
        freqs = [repo.freqs_0.detach().float(), repo.freqs_1.detach().float(), repo.freqs_2.detach().float()]
        half = head_dim // 2
        m = torch.zeros(3, half)
        start = 0
        for axis, f in enumerate(freqs):
            m[axis, start:start + f.numel()] = f * math.pi
            start += f.numel()
        assert start == half
        wf = repo.final_map.weight.detach().float()            # [H*3, hidden], rows h*3 + axis
        w = torch.cat([m.t() @ wf[h * 3:(h + 1) * 3] for h in range(num_heads)], dim=0)   # [H*32, hidden]
        self.ang = _linear(w, None)
        self.num_heads, self.half = num_heads, half

    def forward(self, x):
        b, l, _ = x.shape
        h = safe_layer_norm(x, self.eps, self.norm_w, self.norm_b)
        a = self.ang(F.silu(self.gate(h)) * self.content(h)).reshape(b, l, self.num_heads, self.half)
        return torch.cos(a), torch.sin(a)


class DenoiserBlockGPU(nn.Module):
    """RePo layer + UnifiedTransformerBlock of the denoiser (joint blocks and both refiners).

    forward(x [1, L, 1024], mod [1, 6144] or None) -> [1, L, 1024]."""

    def __init__(self, block, repo, chunk=None, num_heads=16):
        super().__init__()
        c = block.attn.qkv.in_features
        d = c // num_heads
        self.c, self.h, self.d, self.chunk = c, num_heads, d, chunk
        self.repo = RePoGPU(repo, num_heads, d)
        self.modulation = block.modulation
        perm = deinterleave_perm(d)
        w = block.attn.qkv.weight.detach().float()
        bqkv = block.attn.qkv.bias.detach().float()

        def rows(i, permute):
            wi, bi = w[i * c:(i + 1) * c].reshape(num_heads, d, c), bqkv[i * c:(i + 1) * c].reshape(num_heads, d)
            if permute:
                wi, bi = wi[:, perm], bi[:, perm]
            return wi.reshape(c, c), bi.reshape(c)

        self.q = _linear(*rows(0, True))
        self.k = _linear(*rows(1, True))
        self.v = _linear(*rows(2, False))
        gq = block.attn.q_norm.gamma.detach().float()[:, perm]
        gk = block.attn.k_norm.gamma.detach().float()[:, perm]
        # F.normalize(x) * gamma * sqrt(d) = x / rms(x) * gamma; the softmax scale 1/sqrt(d) folded into the q gain
        self.gq = nn.Parameter((gq / math.sqrt(d)).reshape(1, 1, num_heads, d), requires_grad=False)
        self.gk = nn.Parameter(gk.reshape(1, 1, num_heads, d), requires_grad=False)
        self.out = _linear(block.attn.out.weight, block.attn.out.bias)
        self.fc1 = _linear(block.mlp.mlp[0].weight, block.mlp.mlp[0].bias)
        self.fc2 = _linear(block.mlp.mlp[2].weight, block.mlp.mlp[2].bias)
        self.eps1, self.eps2 = block.norm1.eps, block.norm2.eps
        if self.modulation:
            self.shift_table = nn.Parameter(block.shift_table.detach().float().clone(), requires_grad=False)
            self.n1w = self.n1b = self.n2w = self.n2b = None
        else:
            self.n1w = nn.Parameter(block.norm1.weight.detach().float().clone(), requires_grad=False)
            self.n1b = nn.Parameter(block.norm1.bias.detach().float().clone(), requires_grad=False)
            self.n2w = nn.Parameter(block.norm2.weight.detach().float().clone(), requires_grad=False)
            self.n2b = nn.Parameter(block.norm2.bias.detach().float().clone(), requires_grad=False)

    def attn(self, h, cos, sin):
        b, l, _ = h.shape
        half = self.d // 2

        def rope(t):
            t0, t1 = t[..., :half], t[..., half:]
            return torch.cat([t0 * cos - t1 * sin, t0 * sin + t1 * cos], dim=-1)

        q = safe_rms_unit(rope(self.q(h).reshape(b, l, self.h, self.d))) * self.gq
        k = safe_rms_unit(rope(self.k(h).reshape(b, l, self.h, self.d))) * self.gk
        v = self.v(h).reshape(b, l, self.h, self.d)
        o = attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), self.chunk)
        return self.out(o.reshape(b, l, self.c))

    def forward(self, x, mod=None):
        cos, sin = self.repo(x)
        if not self.modulation:
            x = x + self.attn(safe_layer_norm(x, self.eps1, self.n1w, self.n1b), cos, sin)
            return x + self.fc2(_gelu_tanh(self.fc1(safe_layer_norm(x, self.eps2, self.n2w, self.n2b))))
        m = mod + self.shift_table
        c = self.c
        shift_msa, scale_msa, gate_msa = m[:, 0:c], m[:, c:2 * c], m[:, 2 * c:3 * c]
        shift_mlp, scale_mlp, gate_mlp = m[:, 3 * c:4 * c], m[:, 4 * c:5 * c], m[:, 5 * c:6 * c]
        h = safe_layer_norm(x, self.eps1) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + self.attn(h, cos, sin) * gate_msa.unsqueeze(1)
        h = safe_layer_norm(x, self.eps2) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        return x + self.fc2(_gelu_tanh(self.fc1(h))) * gate_mlp.unsqueeze(1)


# -- Decoders ------------------------------------------------------------------------------------------------

class _MHA(nn.Module):
    """idol MultiHeadAttention (self or cross) with q / k RMS-normalised per head."""

    def __init__(self, mha, chunk=None):
        super().__init__()
        self.h = mha.num_heads
        self.cross = mha.cross
        self.chunk = chunk
        if self.cross:
            c = mha.to_q.out_features
            self.q = _linear(mha.to_q.weight, mha.to_q.bias)
            wkv, bkv = mha.to_kv.weight.detach().float(), mha.to_kv.bias.detach().float()
            self.k = _linear(wkv[:c], bkv[:c])
            self.v = _linear(wkv[c:], bkv[c:])
        else:
            c = mha.to_qkv.out_features // 3
            w, bb = mha.to_qkv.weight.detach().float(), mha.to_qkv.bias.detach().float()
            self.q = _linear(w[:c], bb[:c])
            self.k = _linear(w[c:2 * c], bb[c:2 * c])
            self.v = _linear(w[2 * c:], bb[2 * c:])
        self.c, self.d = c, c // self.h
        gq = mha.q_rms_norm.gamma.detach().float()
        gk = mha.k_rms_norm.gamma.detach().float()
        self.gq = nn.Parameter((gq / math.sqrt(self.d)).reshape(1, 1, self.h, self.d), requires_grad=False)
        self.gk = nn.Parameter(gk.reshape(1, 1, self.h, self.d), requires_grad=False)
        self.out = _linear(mha.to_out.weight, mha.to_out.bias)

    def forward(self, x, context=None):
        b, l, _ = x.shape
        src = context if self.cross else x
        lk = src.shape[1]
        q = safe_rms_unit(self.q(x).reshape(b, l, self.h, self.d)) * self.gq
        k = safe_rms_unit(self.k(src).reshape(b, lk, self.h, self.d)) * self.gk
        v = self.v(src).reshape(b, lk, self.h, self.d)
        o = attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), self.chunk)
        return self.out(o.reshape(b, l, self.c))


def _ln_params(ln):
    if ln.weight is None:
        return None, None
    return (nn.Parameter(ln.weight.detach().float().clone(), requires_grad=False),
            nn.Parameter(ln.bias.detach().float().clone(), requires_grad=False))


class GaussianBlockGPU(nn.Module):
    """TransformerCrossBlock: forward(x [1, L, 1024], context [1, Lc, 16]) -> [1, L, 1024]."""

    def __init__(self, block, chunk=None):
        super().__init__()
        self.self_attn = _MHA(block.self_attn, chunk)
        self.cross_attn = _MHA(block.cross_attn, chunk)
        self.fc1 = _linear(block.mlp.mlp[0].weight, block.mlp.mlp[0].bias)
        self.fc2 = _linear(block.mlp.mlp[2].weight, block.mlp.mlp[2].bias)
        self.e1, self.e2, self.e3 = block.norm1.eps, block.norm2.eps, block.norm3.eps
        self.n2w, self.n2b = _ln_params(block.norm2)

    def forward(self, x, context):
        x = x + self.self_attn(safe_layer_norm(x, self.e1))
        x = x + self.cross_attn(safe_layer_norm(x, self.e2, self.n2w, self.n2b), context)
        return x + self.fc2(_gelu_tanh(self.fc1(safe_layer_norm(x, self.e3))))


class OctreeBlockGPU(nn.Module):
    """ModulatedTransformerCrossOnlyBlock: forward(x [1, L, 1024], mod [1, 6144], context [1, Lc, 16])."""

    def __init__(self, block, chunk=None):
        super().__init__()
        self.cross_attn = _MHA(block.cross_attn, chunk)
        self.fc1 = _linear(block.mlp.mlp[0].weight, block.mlp.mlp[0].bias)
        self.fc2 = _linear(block.mlp.mlp[2].weight, block.mlp.mlp[2].bias)
        self.e1, self.e2 = block.norm1.eps, block.norm2.eps
        self.c = self.fc1.in_features

    def forward(self, x, mod, context):
        c = self.c
        shift_msa, scale_msa, gate_msa = mod[:, 0:c], mod[:, c:2 * c], mod[:, 2 * c:3 * c]
        shift_mlp, scale_mlp, gate_mlp = mod[:, 3 * c:4 * c], mod[:, 4 * c:5 * c], mod[:, 5 * c:6 * c]
        h = safe_layer_norm(x, self.e1) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + self.cross_attn(h, context) * gate_msa.unsqueeze(1)
        h = safe_layer_norm(x, self.e2) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        return x + self.fc2(_gelu_tanh(self.fc1(h))) * gate_mlp.unsqueeze(1)


# -- DINOv3 --------------------------------------------------------------------------------------------------

class DinoBlockGPU(nn.Module):
    """transformers DINOv3ViTLayer (gated MLP): forward(x [1, L, 1280], cos, sin [1, 1, L, 64]) -> [1, L, 1280].
    cos / sin rows of the 5 prefix tokens are 1 / 0 (rope applies to the patch tokens only)."""

    def __init__(self, layer, num_heads=20, chunk=None):
        super().__init__()
        att = layer.attention
        c = att.q_proj.in_features
        d = c // num_heads
        self.c, self.h, self.d, self.chunk = c, num_heads, d, chunk
        sc = att.scaling
        self.q = _linear(att.q_proj.weight.detach().float() * sc, att.q_proj.bias.detach().float() * sc)
        self.k = _linear(att.k_proj.weight, att.k_proj.bias)
        self.v = _linear(att.v_proj.weight, att.v_proj.bias)
        l1 = layer.layer_scale1.lambda1.detach().float()
        l2 = layer.layer_scale2.lambda1.detach().float()
        self.o = _linear(att.o_proj.weight.detach().float() * l1[:, None], att.o_proj.bias.detach().float() * l1)
        self.gate = _linear(layer.mlp.gate_proj.weight, layer.mlp.gate_proj.bias)
        self.up = _linear(layer.mlp.up_proj.weight, layer.mlp.up_proj.bias)
        self.down = _linear(layer.mlp.down_proj.weight.detach().float() * l2[:, None],
                            layer.mlp.down_proj.bias.detach().float() * l2)
        self.n1w, self.n1b = _ln_params(layer.norm1)
        self.n2w, self.n2b = _ln_params(layer.norm2)
        self.e1, self.e2 = layer.norm1.eps, layer.norm2.eps

    def forward(self, x, cos, sin):
        b, l, _ = x.shape
        half = self.d // 2

        def rope(t):    # t [1, H, L, D], rotate_half form
            return t * cos + torch.cat([-t[..., half:], t[..., :half]], dim=-1) * sin

        h = safe_layer_norm(x, self.e1, self.n1w, self.n1b)
        q = rope(self.q(h).reshape(b, l, self.h, self.d).permute(0, 2, 1, 3))
        k = rope(self.k(h).reshape(b, l, self.h, self.d).permute(0, 2, 1, 3))
        v = self.v(h).reshape(b, l, self.h, self.d).permute(0, 2, 1, 3)
        o = attention(q, k, v, self.chunk)
        x = x + self.o(o.reshape(b, l, self.c))
        h = safe_layer_norm(x, self.e2, self.n2w, self.n2b)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))
