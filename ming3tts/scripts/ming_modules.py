"""Export modules for Ming-3.0-tts-0.5B (pure torch, weights from the HF safetensors).

Re-authored from the upstream inference code (GitHub inclusionAI/Ming-3.0-tts @028c1eace193,
MIT, Copyright (c) Ant Group) and transformers' Qwen2 (Apache-2.0). Every rewrite is exact in
real arithmetic:
  - RMSNorm in the max-normalized form (m = max(1, max|x|)), fp16 sum-of-squares guard;
  - RoPE cos/sin are inputs (Qwen2, NeoX half-split); x_transformers' interleaved RoPE in the
    DiT / aggregator is turned into the half-split form by permuting the to_q/to_k rows of
    each head (q.k is invariant to a shared permutation), with constant tables for its fixed
    3 / 2 positions;
  - attention = einsum on rank-4 tensors, GQA by grouping the 7 query heads of a KV head
    into the query axis (no repeat_kv / BROADCAST_TO), additive mask as a runtime input;
  - KV cache = packed [1, 2*layers, P, 64] graph input (post-RoPE keys), new columns out;
  - x4 linear upsample (align_corners=False) as fixed 2-tap blends of shifted slices.
Module, class and attribute names are part of the exported bytes: do not rename.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

import common as C

HID = 896
NQ, NKV, HD = 14, 2, 64
GROUPS = NQ // NKV
LAYERS = 24
DIT_HID, DIT_HEADS = 1024, 16
LATENT = 64
P_LM = 511          # past slots of the LM cache (prompt + steps <= MAX_CACHE_LEN = 512)
P_DEC = 64          # past slots per decoder layer (sliding window 64)
MASK_NEG = -30000.0


def load_weights(prefixes):
    out = {}
    with safe_open(str(C.HF_DIR / "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if any(k.startswith(p) for p in prefixes):
                out[k] = f.get_tensor(k).float()
    return out


class SafeRMSNorm(nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight = nn.Parameter(weight.clone(), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        m = torch.clamp(torch.amax(torch.abs(x), dim=-1, keepdim=True), min=1.0)
        xs = x / m
        var = torch.mean(xs * xs, dim=-1, keepdim=True)
        return xs * torch.rsqrt(var + self.eps / (m * m)) * self.weight


def lin(w, b=None):
    m = nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
    with torch.no_grad():
        m.weight.copy_(w)
        if b is not None:
            m.bias.copy_(b)
    m.requires_grad_(False)
    return m


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def gqa_attention(q, k, v, mask, groups):
    """q [1, nkv*g, Q, d] (scaled), k/v [1, nkv, K, d], mask [1, 1, Q, K] additive."""
    _, hq, nq, d = q.shape
    nkv = k.shape[1]
    qg = q.reshape(1, nkv, groups * nq, d)
    s = torch.einsum("bhqd,bhkd->bhqk", qg, k)
    if nq > 1:
        mask = torch.cat([mask] * groups, dim=2)
    p = torch.softmax(s + mask, dim=-1)
    o = torch.einsum("bhqk,bhkd->bhqd", p, v)
    return o.reshape(1, hq, nq, d)


class QwenLayer(nn.Module):
    """rs = residual-stream scale (power of 2): the stream runs at rs*x, the o_proj / down_proj
    inputs are multiplied by rs and every RMSNorm eps by rs^2 - exact (the norms are scale
    invariant, so q/k/v, the cache and the final-norm output do not change). rs = 1/4 keeps
    Qwen2.5's massive activations (residual 74,395, down_proj 66,738 on the fixtures) inside
    fp16. rs = 1.0 emits no extra ops (the graph form exported first)."""

    def __init__(self, w, pre, eps=1e-6, rs=1.0):
        super().__init__()
        a = pre + "self_attn."
        self.rs = rs
        eps = eps * rs * rs
        self.in_norm = SafeRMSNorm(w[pre + "input_layernorm.weight"], eps)
        self.q = lin(w[a + "q_proj.weight"], w[a + "q_proj.bias"])
        self.k = lin(w[a + "k_proj.weight"], w[a + "k_proj.bias"])
        self.v = lin(w[a + "v_proj.weight"], w[a + "v_proj.bias"])
        self.o = lin(w[a + "o_proj.weight"])
        self.post_norm = SafeRMSNorm(w[pre + "post_attention_layernorm.weight"], eps)
        self.gate = lin(w[pre + "mlp.gate_proj.weight"])
        self.up = lin(w[pre + "mlp.up_proj.weight"])
        self.down = lin(w[pre + "mlp.down_proj.weight"])

    def forward(self, x, cos, sin, mask, pk, pv):
        nq = x.shape[1]
        h = self.in_norm(x)
        q = self.q(h).reshape(1, nq, NQ, HD).transpose(1, 2)
        k = self.k(h).reshape(1, nq, NKV, HD).transpose(1, 2)
        v = self.v(h).reshape(1, nq, NKV, HD).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        ka = torch.cat([pk, k], dim=2)
        va = torch.cat([pv, v], dim=2)
        o = gqa_attention(q * 0.125, ka, va, mask, GROUPS)
        o = o.transpose(1, 2).reshape(1, nq, HID)
        if self.rs != 1.0:
            o = o * self.rs
        x = x + self.o(o)
        h = self.post_norm(x)
        a = F.silu(self.gate(h)) * self.up(h)
        if self.rs != 1.0:
            a = a * self.rs
        x = x + self.down(a)
        return x, k, v, ka, va


class QwenStack(nn.Module):
    """24 Qwen2 layers + final norm over Q new positions with a packed past cache."""

    def __init__(self, w, pre, rs=1.0):
        super().__init__()
        self.rs = rs
        self.layers = nn.ModuleList([QwenLayer(w, f"{pre}layers.{i}.", rs=rs) for i in range(LAYERS)])
        self.norm = SafeRMSNorm(w[pre + "norm.weight"], 1e-6 * rs * rs)

    def forward(self, x, cos, sin, mask, pk, pv, keep_last=0):
        if self.rs != 1.0:
            x = x * self.rs
        ks, vs = [], []
        for i, layer in enumerate(self.layers):
            x, k, v, ka, va = layer(x, cos, sin, mask, pk[:, 2 * i:2 * i + 2], pv[:, 2 * i:2 * i + 2])
            if keep_last:
                ks.append(ka[:, :, -keep_last:])
                vs.append(va[:, :, -keep_last:])
            else:
                ks.append(k)
                vs.append(v)
        return self.norm(x), torch.cat(ks, dim=1), torch.cat(vs, dim=1)


class LMChunk(nn.Module):
    """LM over Q new input embeddings: (z, k_new, v_new); z = final-norm hidden."""

    def __init__(self, w, lm=None):
        super().__init__()
        self.lm = lm if lm is not None else QwenStack(w, "model.model.")

    def forward(self, x, cos, sin, mask, pk, pv):
        return self.lm(x, cos, sin, mask, pk, pv)


class LMStep(nn.Module):
    """One AR step: (z, stop_logits, k_new, v_new)."""

    def __init__(self, w, lm=None):
        super().__init__()
        self.lm = lm if lm is not None else QwenStack(w, "model.model.")
        self.stop = lin(w["stop_head.weight"], w["stop_head.bias"])

    def forward(self, x, cos, sin, mask, pk, pv):
        z, k, v = self.lm(x, cos, sin, mask, pk, pv)
        return z, self.stop(z), k, v


class LMStepZ(nn.Module):
    """One AR step without the stop head: (z, k_new, v_new). z stays a sole-leaf output (on the
    Mac GPU an output that another node also consumed read back as zeros); the 2x896 stop
    head runs on the host."""

    def __init__(self, w, lm=None):
        super().__init__()
        self.lm = lm if lm is not None else QwenStack(w, "model.model.")

    def forward(self, x, cos, sin, mask, pk, pv):
        return self.lm(x, cos, sin, mask, pk, pv)


def interleave_perm(dim):
    return list(range(0, dim, 2)) + list(range(1, dim, 2))


def permute_heads(w, b, heads, hd):
    perm = torch.tensor(interleave_perm(hd))
    idx = torch.cat([h * hd + perm for h in range(heads)])
    return w[idx].contiguous(), b[idx].contiguous()


def xt_rope_tables(inv_freq, n):
    """x_transformers RotaryEmbedding at positions 0..n-1, in the de-interleaved half-split order."""
    t = torch.arange(n).type_as(inv_freq)
    freqs = torch.einsum("i,j->ij", t, inv_freq) / 1.0
    emb = torch.cat([freqs, freqs], dim=-1)            # [n, hd]: f0..f31, f0..f31
    return emb.cos()[None, None], emb.sin()[None, None]


class DiTBlock(nn.Module):
    def __init__(self, w, pre, cos, sin):
        super().__init__()
        a = pre + "attn."
        self.norm1 = SafeRMSNorm(w[pre + "norm1.weight"], 1e-6)
        qw, qb = permute_heads(w[a + "to_q.weight"], w[a + "to_q.bias"], DIT_HEADS, HD)
        kw, kb = permute_heads(w[a + "to_k.weight"], w[a + "to_k.bias"], DIT_HEADS, HD)
        self.q = lin(qw, qb)
        self.k = lin(kw, kb)
        self.v = lin(w[a + "to_v.weight"], w[a + "to_v.bias"])
        self.o = lin(w[a + "to_out.0.weight"], w[a + "to_out.0.bias"])
        self.norm2 = SafeRMSNorm(w[pre + "norm2.weight"], 1e-6)
        self.ff1 = lin(w[pre + "mlp.ff.0.0.weight"], w[pre + "mlp.ff.0.0.bias"])
        self.ff2 = lin(w[pre + "mlp.ff.2.weight"], w[pre + "mlp.ff.2.bias"])
        self.register_buffer("cos", cos.clone(), persistent=False)
        self.register_buffer("sin", sin.clone(), persistent=False)

    def forward(self, x):
        b, n, _ = x.shape
        h = self.norm1(x)
        q = self.q(h).reshape(b, n, DIT_HEADS, HD).transpose(1, 2)
        k = self.k(h).reshape(b, n, DIT_HEADS, HD).transpose(1, 2)
        v = self.v(h).reshape(b, n, DIT_HEADS, HD).transpose(1, 2)
        q = q * self.cos + rotate_half(q) * self.sin
        k = k * self.cos + rotate_half(k) * self.sin
        s = torch.einsum("bhqd,bhkd->bhqk", q * 0.125, k)
        p = torch.softmax(s, dim=-1)
        o = torch.einsum("bhqk,bhkd->bhqd", p, v).transpose(1, 2).reshape(b, n, DIT_HID)
        x = x + self.o(o)
        x = x + self.ff2(F.gelu(self.ff1(self.norm2(x)), approximate="tanh"))
        return x


def epss_times(steps=10, sway=-1.0):
    """CFM.get_epss_timesteps + sway, float32, as torch computes them."""
    table = {5: [0, 2, 4, 8, 16, 32], 6: [0, 2, 4, 6, 8, 16, 32], 7: [0, 2, 4, 6, 8, 16, 24, 32],
             10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32]}
    t = (1 / 32) * torch.tensor(table[steps], dtype=torch.float32)
    t = t + sway * (torch.cos(torch.pi / 2 * t) - 1 + t)
    return t


class TimeEmbed(nn.Module):
    """fm/dit.py TimestepEmbedder (host-side table generator, not exported)."""

    def __init__(self, w):
        super().__init__()
        p = "flowloss.cfm.model.t_embedder.time_mlp."
        self.l1 = lin(w[p + "0.weight"], w[p + "0.bias"])
        self.l2 = lin(w[p + "2.weight"], w[p + "2.bias"])

    def forward(self, t):
        half = 128
        e = math.log(10000) / (half - 1)
        e = torch.exp(torch.arange(half).float() * -e)
        e = 1000 * t.unsqueeze(1) * e.unsqueeze(0)
        e = torch.cat([e.sin(), e.cos()], dim=-1).to(t.dtype)
        return self.l2(F.silu(self.l1(e)))


class DiTCore(nn.Module):
    """Shared DiT weights: embedders, 8 blocks, final layer."""

    def __init__(self, w):
        super().__init__()
        p = "flowloss.cfm.model."
        cos, sin = xt_rope_tables(w[p + "rotary_embed.inv_freq"], 3)
        self.x_emb = lin(w[p + "x_embedder.weight"], w[p + "x_embedder.bias"])
        self.c_emb = lin(w[p + "c_embedder.cond_embedder.weight"], w[p + "c_embedder.cond_embedder.bias"])
        self.blocks = nn.ModuleList([DiTBlock(w, f"{p}blocks.{i}.", cos, sin) for i in range(8)])
        self.final_norm = SafeRMSNorm(w[p + "final_layer.norm_final.weight"], 1e-6)
        self.final = lin(w[p + "final_layer.linear.weight"], w[p + "final_layer.linear.bias"])

    def velocity(self, tok0, xh2, y):
        """tok0 [2,1,1024] (t + c), xh2 [2,1,1024] history embedding, y [1,1,64] -> cfg velocity."""
        xn = self.x_emb(y)
        seq = torch.cat([tok0, xh2, torch.cat([xn, xn], dim=0)], dim=1)
        for blk in self.blocks:
            seq = blk(seq)
        out = self.final(self.final_norm(seq[:, 2:3]))
        pred, null = out[0:1], out[1:2]
        return pred + (pred - null) * 2.0


class Flow(nn.Module):
    """CFM.sample at temperature 0: EPSS + sway Euler steps, CFG 2.0 as batch 2.

    temb = TimestepEmbedder(t_k) table [1, steps, 1024] (host constant); dt baked.
    """

    def __init__(self, w, steps=10, core=None):
        super().__init__()
        self.core = core if core is not None else DiTCore(w)
        t = epss_times(steps)
        self.dts = [float(np.float32((t[i + 1] - t[i]).item())) for i in range(steps)]
        self.steps = steps

    def forward(self, z, hist, noise, temb):
        core = self.core
        c = core.c_emb(z)
        c0 = core.c_emb.bias.reshape(1, 1, DIT_HID)
        xh = core.x_emb(hist)
        xh2 = torch.cat([xh, xh], dim=0)
        y = noise
        for i in range(self.steps):
            tk = temb[:, i:i + 1]
            tok0 = torch.cat([tk + c, tk + c0], dim=0)
            y = y + core.velocity(tok0, xh2, y) * self.dts[i]
        return y


class Velocity(nn.Module):
    """One CFG DiT evaluation (host Euler loop option): (t_emb [1,1,1024], z, hist, y) -> v."""

    def __init__(self, w, core=None):
        super().__init__()
        self.core = core if core is not None else DiTCore(w)

    def forward(self, tk, z, hist, y):
        core = self.core
        c = core.c_emb(z)
        c0 = core.c_emb.bias.reshape(1, 1, DIT_HID)
        xh = core.x_emb(hist)
        tok0 = torch.cat([tk + c, tk + c0], dim=0)
        return core.velocity(tok0, torch.cat([xh, xh], dim=0), y)


class Aggregator(nn.Module):
    """fm/dit.py Aggregator on one latent patch: [cls, x] -> token 0 -> 896."""

    def __init__(self, w):
        super().__init__()
        p = "linear_proj_audio."
        cos, sin = xt_rope_tables(w[p + "rotary_embed.inv_freq"], 2)
        self.x_emb = lin(w[p + "x_embedder.weight"], w[p + "x_embedder.bias"])
        self.cls = nn.Parameter(w[p + "word_embedder.weight"].reshape(1, 1, DIT_HID).clone(), requires_grad=False)
        self.blocks = nn.ModuleList([DiTBlock(w, f"{p}blocks.{i}.", cos, sin) for i in range(8)])
        self.final_norm = SafeRMSNorm(w[p + "final_layer.norm_final.weight"], 1e-6)
        self.final = lin(w[p + "final_layer.linear.weight"], w[p + "final_layer.linear.bias"])

    def forward(self, lat):
        x = torch.cat([self.cls, self.x_emb(lat)], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.final(self.final_norm(x[:, 0:1]))


class Frame(nn.Module):
    """Fused frame: LM step -> stop head -> 10-step flow -> aggregator, one output row.

    out = [latent 64 | next input embedding 896 | stop logits 2 | z 896 | k_new 48*64 | v_new 48*64]
    """

    def __init__(self, w, steps=10, lm=None, core=None, agg=None):
        super().__init__()
        self.step = LMStep(w, lm=lm)
        self.flow = Flow(w, steps=steps, core=core)
        self.agg = agg if agg is not None else Aggregator(w)

    def forward(self, x, hist, noise, temb, cos, sin, mask, pk, pv):
        z, stop, k, v = self.step(x, cos, sin, mask, pk, pv)
        lat = self.flow(z, hist, noise, temb)
        e = self.agg(lat)
        return torch.cat([lat.reshape(1, LATENT), e.reshape(1, HID), stop.reshape(1, 2),
                          z.reshape(1, HID), k.reshape(1, 2 * LAYERS * HD),
                          v.reshape(1, 2 * LAYERS * HD)], dim=1)


FRAME_SLICES = {"latent": (0, 64), "e_next": (64, 960), "stop": (960, 962), "z": (962, 1858),
                "k": (1858, 4930), "v": (4930, 8002)}


class DecChunk(nn.Module):
    """VAE decoder over C latents: fc1 -> x4 linear upsample -> Qwen2 x24 (sliding window via
    the runtime mask, 64 past positions per layer) -> ISTFT-head linear output.

    lat [1, C+2, 64] = [previous latent | C latents | next latent] (edges replicate);
    cos/sin [1, 1, 4C, 64] at the absolute positions; mask [1, 1, 4C, 64+4C];
    pk/pv [1, 48, 64, 64] -> (head [1, 4C, 3530], pk', pv') with pk' = last 64 positions.
    """

    def __init__(self, w, chunk=8):
        super().__init__()
        p = "audio.decoder."
        self.chunk = chunk
        self.fc1 = lin(w[p + "fc1.weight"], w[p + "fc1.bias"])
        self.dec = QwenStack(w, p + "decoder.")
        self.head = lin(w[p + "head.out.weight"], w[p + "head.out.bias"])

    def forward(self, lat, cos, sin, mask, pk, pv):
        c = self.chunk
        h = self.fc1(lat)
        a, b, d = h[:, 0:c], h[:, 1:c + 1], h[:, 2:c + 2]
        u = torch.stack([a * 0.375 + b * 0.625, a * 0.125 + b * 0.875,
                         b * 0.875 + d * 0.125, b * 0.625 + d * 0.375], dim=2).reshape(1, 4 * c, HID)
        z, k, v = self.dec(u, cos, sin, mask, pk, pv, keep_last=P_DEC)
        return self.head(z), k, v
