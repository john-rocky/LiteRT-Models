"""Node-reduced rewrites of the Ming-3.0-tts DiT (flow / velocity) and aggregator (conversion venv).

Built from the ming_modules modules (same weights, same fixed 3 / 2 token positions), which stay
untouched. Every change is exact in real arithmetic and only power-of-2 constants (and exact 0 / +-1
tables) are folded into weights, so the wfp16 file keeps the checkpoint values (bf16 -> fp16 exact, as with ming_modules):
  - 2-D token-major stream [n * b, 1024] (rows (token, CFG batch); n = 3 / b = 2 for the flow, n = 2 /
    b = 1 for the aggregator): every FULLY_CONNECTED takes a rank-2 input, no flatten RESHAPE per block;
  - stream pre-scaled by RS = 1/16 (x_emb, c_emb, cls, temb, o-proj and ff2 weights x 2^-4), and
    RMSNorm(x) * w = xg * rsqrt(sum(xg^2) + 1024 e^2 eps) * (32 w), xg = g * xs, e = RS * g, with the
    sum + eps as one FULLY_CONNECTED (weight = ones [1, 1024], bias = the eps): no runtime amax, no DIV
    (the Qualcomm HTP's fp16 divide returns NaN for divisors in [2^15, 65504]; this form has no divide at all),
    no SUM + RESHAPE. g = 1 except the first-block norms whose inputs are the small embedding rows (agg
    cls RMS 0.0245, flow history row 0.0896): there g is sized from the norm input statistics of the test sentences so that
    sum(xg^2) stays >= 6.7x below the fp16 max and the squares of ordinary rows above the fp16 subnormals;
  - q and k from one FC (q rows x 1/8 = the attention scale), v from its own FC;
  - RoPE as q * cos + rot(q) * sin on [n, b, 32, 64] with rot = rotate_half as a 64 x 64 FULLY_CONNECTED
    of 0 / +-1 (exact) and per-token [n, 1, 1, 64] cos / sin tables (ming_modules' values; bias-sized constants), one
    TRANSPOSE for q and k together, rank-4 BMM attention (batch dim kept: ML Drift rank-3 trap);
  - flow: c_emb(z) and the time table combined once for all steps; CFG folded into the final layer:
    v = pred + 2 (pred - null) = [3 W | -2 W] [h_cond ; h_uncond] + b, one FULLY_CONNECTED on the two
    final-norm rows reshaped to [1, 2048] (3 W and -2 W of a bf16 W are exact in fp16). An earlier draft's
    weighted SUM over the batch rows (MUL by a baked [2, 1] constant) read back wrong on the Mac GPU at both
    precisions and is gone. The Euler dt rides on the final-norm weight
    (32 w dt, an fp32 MUL constant) and the bias (b dt, folded by the converter into the FC bias, fp32): a MUL
    by dt right after the FC is folded by the converter into the FC weights, which FLOAT_CASTING then rounds
    to fp16 (W dt is not exact; an earlier export put the latent 100x further from the reference).
Module, class and attribute names are part of the exported bytes: do not rename.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ming_modules as M

HID = M.DIT_HID          # 1024
HEADS = M.DIT_HEADS      # 16
HD = M.HD                # 64
RS = 1.0 / 16            # stream scale (power of 2)
FLOW_G = {(0, 1): 4.0, (0, 2): 2.0}   # (block, norm) -> g, from the norm input statistics of all test sentences
AGG_G = {(0, 1): 16.0}


def lin_from(m, scale=1.0):
    """Copy of nn.Linear m with a power-of-2 scale on weight and bias."""
    w = m.weight.detach().clone()
    b = m.bias.detach().clone() if m.bias is not None else None
    return M.lin(w * scale, None if b is None else b * scale)


class R4Norm(nn.Module):
    """RMSNorm of x = xs / RS over C = 1024: x * rsqrt(mean(x^2) + eps) * w
    == xg * rsqrt(sum(xg^2) + 1024 (RS g)^2 eps) * (32 w), xg = g * xs (g = 1: no extra op)."""

    def __init__(self, norm, g=1.0):
        super().__init__()
        self.g = float(g)
        eps = float(norm.eps * 1024.0 * (RS * g) ** 2)
        self.ss = M.lin(torch.ones(1, HID), torch.tensor([eps], dtype=torch.float32))
        self.weight = nn.Parameter(norm.weight.detach().clone() * 32.0, requires_grad=False)

    def forward(self, xs, scale=1.0):
        if self.g != 1.0:
            xs = xs * self.g
        w = self.weight if scale == 1.0 else self.weight * scale
        return xs * torch.rsqrt(self.ss(xs * xs)) * w


def rotate_half_matrix():
    """P with P @ [a | b] = [-b | a] per 64-dim head (rotate_half of the half-split order)."""
    p = torch.zeros(HD, HD)
    h = HD // 2
    for j in range(h):
        p[j, j + h] = -1.0
        p[j + h, j] = 1.0
    return p


class R4Block(nn.Module):
    def __init__(self, blk, n, b, g1=1.0, g2=1.0):
        super().__init__()
        self.n, self.b = n, b
        self.norm1 = R4Norm(blk.norm1, g1)
        wq, bq = blk.q.weight.detach() * 0.125, blk.q.bias.detach() * 0.125
        self.qk = M.lin(torch.cat([wq, blk.k.weight.detach()]), torch.cat([bq, blk.k.bias.detach()]))
        self.rot = M.lin(rotate_half_matrix())
        self.v = lin_from(blk.v)
        self.o = lin_from(blk.o, RS)
        self.norm2 = R4Norm(blk.norm2, g2)
        self.ff1 = lin_from(blk.ff1)
        self.ff2 = lin_from(blk.ff2, RS)
        self.register_buffer("cos", blk.cos[0, 0, :n][:, None, None].clone(), persistent=False)   # [n, 1, 1, 64]
        self.register_buffer("sin", blk.sin[0, 0, :n][:, None, None].clone(), persistent=False)

    def forward(self, x):
        n, b = self.n, self.b
        h = self.norm1(x)
        qk = self.qk(h).reshape(n, b, 2 * HEADS, HD)
        qk = (qk * self.cos + self.rot(qk) * self.sin).permute(1, 2, 0, 3)  # [b, 32, n, 64]
        q, k = qk[:, :HEADS], qk[:, HEADS:]
        v = self.v(h).reshape(n, b, HEADS, HD).permute(1, 2, 0, 3)        # [b, 16, n, 64]
        p = torch.softmax(torch.einsum("bhqd,bhkd->bhqk", q, k), dim=-1)
        o = torch.einsum("bhqk,bhkd->bhqd", p, v).permute(2, 0, 1, 3).reshape(n * b, HID)
        x = x + self.o(o)
        x = x + self.ff2(F.gelu(self.ff1(self.norm2(x)), approximate="tanh"))
        return x


class R4Core(nn.Module):
    """ming_modules.DiTCore re-laid for the 2-D token-major, RS-scaled stream (rows: t+c x2, history x2, y x2)."""

    def __init__(self, core):
        super().__init__()
        self.x_emb = lin_from(core.x_emb, RS)
        self.c_emb = lin_from(core.c_emb, RS)
        self.register_buffer("c0", (core.c_emb.bias.detach() * RS).reshape(1, HID).clone(), persistent=False)
        self.blocks = nn.ModuleList([R4Block(blk, 3, 2, FLOW_G.get((i, 1), 1.0), FLOW_G.get((i, 2), 1.0))
                                     for i, blk in enumerate(core.blocks)])
        self.final_norm = R4Norm(core.final_norm)
        w, b = core.final.weight.detach(), core.final.bias.detach()
        self.cfg = M.lin(torch.cat([w * 3.0, w * -2.0], dim=1))            # [64, 2048], no bias
        self.register_buffer("cfg_b", b.reshape(1, M.LATENT).clone(), persistent=False)

    def run(self, seq, dt=1.0):
        """seq [6, 1024] -> dt * CFG velocity [1, 64]."""
        for blk in self.blocks:
            seq = blk(seq)
        h = self.final_norm(seq[4:6], dt)
        return self.cfg(h.reshape(1, 2 * HID)) + self.cfg_b * dt


class R4Flow(nn.Module):
    """ming_modules.Flow (EPSS + sway Euler, CFG 2.0 as batch 2): (z, hist, noise, temb [1, S, 1024]) -> latent [1, 1, 64]."""

    def __init__(self, core, steps):
        super().__init__()
        self.core = core
        t = M.epss_times(steps)
        self.dts = [float(np.float32((t[i + 1] - t[i]).item())) for i in range(steps)]
        self.steps = steps

    def forward(self, z, hist, noise, temb):
        core, s = self.core, self.steps
        # [1, 2, C] built rank 3 by a CONCAT (a reshaped [2, C] is folded back to rank 2 by the converter and
        # its broadcast against [S, 1, C] maps the pair onto the BHWC batch axis): [S, 1, C] + [1, 2, C]
        c = core.c_emb(z.reshape(1, M.HID)).reshape(1, 1, HID)
        cc = torch.cat([c, core.c0.reshape(1, 1, HID)], dim=1)                     # [1, 2, 1024]
        tc = (temb.reshape(s, 1, HID) * RS + cc).reshape(2 * s, HID)               # rows (step, batch)
        xh = core.x_emb(hist.reshape(1, M.LATENT))
        y = noise.reshape(1, M.LATENT)
        for i in range(s):
            xn = core.x_emb(y)
            seq = torch.cat([tc[2 * i:2 * i + 2], xh, xh, xn, xn], dim=0)          # [6, 1024]
            y = y + core.run(seq, self.dts[i])
        return y.reshape(1, 1, M.LATENT)


class R4Velocity(nn.Module):
    """ming_modules.Velocity (one CFG DiT evaluation, host Euler option): (t_emb [1,1,1024], z, hist, y) -> v [1,1,64]."""

    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, tk, z, hist, y):
        core = self.core
        tok0 = tk.reshape(1, HID) * RS + torch.cat([core.c_emb(z.reshape(1, M.HID)), core.c0], dim=0)
        xh = core.x_emb(hist.reshape(1, M.LATENT))
        xn = core.x_emb(y.reshape(1, M.LATENT))
        seq = torch.cat([tok0, xh, xh, xn, xn], dim=0)                             # [6, 1024]
        return core.run(seq).reshape(1, 1, M.LATENT)


class R4Aggregator(nn.Module):
    """ming_modules.Aggregator on one latent patch: [cls, x_emb(lat)] -> token 0 -> 896."""

    def __init__(self, agg):
        super().__init__()
        self.x_emb = lin_from(agg.x_emb, RS)
        self.register_buffer("cls", (agg.cls.detach() * RS).reshape(1, HID).clone(), persistent=False)
        self.blocks = nn.ModuleList([R4Block(blk, 2, 1, AGG_G.get((i, 1), 1.0), AGG_G.get((i, 2), 1.0))
                                     for i, blk in enumerate(agg.blocks)])
        self.final_norm = R4Norm(agg.final_norm)
        self.final = lin_from(agg.final)

    def forward(self, lat):
        seq = torch.cat([self.cls, self.x_emb(lat.reshape(1, M.LATENT))], dim=0)  # [2, 1024]
        for blk in self.blocks:
            seq = blk(seq)
        return self.final(self.final_norm(seq[0:1])).reshape(1, 1, M.HID)


def forward_epsdiv(self, x):
    """One-point SafeRMSNorm change (eps / m / m instead of eps / (m * m)) for exports in the ming_modules form."""
    m = torch.clamp(torch.amax(torch.abs(x), dim=-1, keepdim=True), min=1.0)
    xs = x / m
    var = torch.mean(xs * xs, dim=-1, keepdim=True)
    return xs * torch.rsqrt(var + (self.eps / m) / m) * self.weight


def r1_core(w=None):
    w = w if w is not None else M.load_weights(["flowloss."])
    return M.DiTCore(w).eval(), w


def r1_agg(w=None):
    w = w if w is not None else M.load_weights(["linear_proj_audio."])
    return M.Aggregator(w).eval(), w
