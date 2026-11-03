#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1 (a re-authoring of nvidia/NVIDIA-NemotronLabs-RE-USE's forward)
# The selective-scan math follows mamba_ssm (Apache-2.0, state-spaces/mamba @ a14b1dff), see mamba_ref.py.
"""r3 GPU form of RE-USE: reuse_gpu.GPUReUSE (r1, unchanged) with the scan form chosen per axis
(scan_forms_r3: F0 = r1's Hillis-Steele, F3a / F3b = Hillis-Steele variants, F2 = sequential, F1q<Q> = blocks of Q)
and, with npu=True, a DIV-free fp16 range scheme for the HTP (see R3MambaBlock / div_free_*).

  R3ReUSE(model, time_form="F2", freq_form="F1q8", npu=False)    spec [1, 2, T, F] -> out [1, 3, T, F] (as r1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import reuse_gpu as G
import scan_forms_r3 as SF

EVAL_M_CHUNK = None   # torch evaluation only: run a scan over slices of M (rows are independent); None in export
STATS = None          # torch evaluation only: a dict that the npu form fills with fp16-range maxima (None in export)


def _stat(key, t, fn="amax"):
    if STATS is None:
        return
    v = float(t.abs().max()) if fn == "amax" else float(t.min())
    old = STATS.get(key)
    STATS[key] = v if old is None else (max(old, v) if fn == "amax" else min(old, v))


def _scan(fn, kw, delta, du, Bm, Cm, A, reverse, A4, AT, AP=None):
    M = delta.shape[2]
    if EVAL_M_CHUNK is None or M <= EVAL_M_CHUNK:
        return fn(delta, du, Bm, Cm, A, reverse, A4=A4, AT=AT, AP=AP, **kw)
    outs = []
    for a in range(0, M, EVAL_M_CHUNK):
        b = min(M, a + EVAL_M_CHUNK)
        outs.append(fn(delta[:, :, a:b], du[:, :, a:b], Bm[:, :, a:b], Cm[:, :, a:b], A, reverse, A4=A4, AT=AT, AP=AP,
                       **kw))
    return torch.cat(outs, dim=2)


class R3Mamba(G.GPUMamba):
    """reuse_gpu.GPUMamba with the scan form `form` (pre() is r1's with the scan call swapped)."""

    def __init__(self, m, reverse, s_y=1.0, form="F0"):
        super().__init__(m, reverse, s_y)
        self.form = form
        self.scan_fn, self.scan_kw = SF.get(form)
        D, N = self.A.shape
        self.register_buffer("A4", self.A.reshape(1, 1, D, N).clone(), persistent=False)
        self.register_buffer("AT", self.A.t().reshape(1, N, 1, D).contiguous(), persistent=False)
        self.register_buffer("AP", self.A.t().reshape(1, 1, N, D).contiguous(), persistent=False)

    def pre(self, x):
        m = self.m
        xi = G.lin(x, self.Wx)
        z = G.lin(x, self.Wz)
        xc = G.dwconv_L(xi, m.conv1d.weight, m.conv1d.bias, self.reverse)
        u = xc * torch.sigmoid(xc)
        dt = G.lin(G.lin(u, self.Wdt_low), m.dt_proj.weight, m.dt_proj.bias)
        delta = G.softplus_safe(dt)
        Bm = G.lin(u, self.WB)
        Cm = G.lin(u, self.WC)
        y = _scan(self.scan_fn, self.scan_kw, delta, delta * u, Bm, Cm, self.A, self.reverse, self.A4, self.AT, self.AP)
        y = y + u * self.Dsk
        return y, z * torch.sigmoid(z)


# ------------------------------------------------------------------ DIV-free fp16 range (npu=True)

NPU_S_Y = 64.0          # static scale of the Mamba output path (folded into C, D, the residual and output_proj's bias)
NPU_S_MIN = 2.0 ** -4   # per-row / per-channel norm scale floor: rows with amax < 1/4 are scaled UP (at most x16)
NPU_VAR_FLOOR = 2.0 ** -14   # rsqrt guard = the smallest normal fp16 number (r1's 1e-6 is an fp16 subnormal)


def _recip_rsqrt(S):
    """1 / S without DIV: rsqrt(S)^2. S in [1/16, ~1e3] here, so rsqrt(S) in [0.03, 4] and 1/S in [1e-3, 16]: normal
    fp16 numbers, and no divisor ever reaches the HTP's 2^14 band."""
    r = torch.rsqrt(S)
    return r * r


def div_free_layer_norm(x, ln, eps_scale):
    """LayerNorm over the last axis of [1, L, M, C] without DIV. x carries the static factor eps_scale (a Python
    float); a per-row S = max(amax / 4, 1/16) with 1/S = rsqrt(S)^2 brings every row to |x / S| <= 4 before squaring
    (large rows down, small rows up). eps follows every factor squared, so the result equals LayerNorm of the
    unscaled input in real arithmetic (the var floor only binds for a row whose std is < 0.8 % of its amax)."""
    amax = x.abs().amax(dim=-1, keepdim=True)
    inv = _recip_rsqrt(torch.clamp(amax * G.NORM_SHIFT, min=NPU_S_MIN))
    xs = x * inv
    mu = xs.mean(dim=-1, keepdim=True)
    d = xs - mu
    var = (d * d).mean(dim=-1, keepdim=True)
    v = var + (ln.eps * eps_scale * eps_scale) * (inv * inv)
    if STATS is not None:
        _stat("ln_in", x)
        _stat("ln_var_min", v, "min")
        STATS["ln_floor_rows"] = STATS.get("ln_floor_rows", 0) + int((v < NPU_VAR_FLOOR).sum())
    y = d * torch.rsqrt(torch.clamp(v, min=NPU_VAR_FLOOR))
    return y * ln.weight + ln.bias


def div_free_instance_norm(x, norm):
    """InstanceNorm2d(affine) on [1, C, H, W] without DIV (per-channel S = max(amax / 4, 1/16), 1/S = rsqrt(S)^2,
    eps follows the factor squared; spatial means chained over single axes as r1)."""
    amax = x.abs().amax(dim=3, keepdim=True).amax(dim=2, keepdim=True)
    inv = _recip_rsqrt(torch.clamp(amax * G.NORM_SHIFT, min=NPU_S_MIN))
    xs = x * inv
    mu = xs.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
    d = xs - mu
    var = (d * d).mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
    v = var + norm.eps * (inv * inv)
    if STATS is not None:
        _stat("in_in", x)
        _stat("in_var_min", v, "min")
        STATS["in_floor_ch"] = STATS.get("in_floor_ch", 0) + int((v < NPU_VAR_FLOOR).sum())
    y = d * torch.rsqrt(torch.clamp(v, min=NPU_VAR_FLOOR))
    return y * norm.weight.view(1, -1, 1, 1) + norm.bias.view(1, -1, 1, 1)


class R3MambaBlock(G.GPUMambaBlock):
    """reuse_gpu.GPUMambaBlock with R3Mamba branches. npu=False: r1's row scale (r = 1 / max(1, amax), a DIV).
    npu=True: no per-row DIV: the Mamba output path is divided by the static NPU_S_Y (folded into C and D, a power
    of two), the residual x and output_proj's bias by the same constant, and the LayerNorm runs DIV-free with
    eps * (1 / NPU_S_Y)^2 -- exact in real arithmetic (LayerNorm is invariant to a common factor when eps follows)."""

    def __init__(self, mb, form="F0", npu=False):
        nn.Module.__init__(self)
        self.npu = npu
        self.row_scale = not npu
        self.S_Y = NPU_S_Y if npu else 16.0
        self.fw = R3Mamba(mb.forward_blocks, reverse=False, s_y=self.S_Y, form=form)
        self.bw = R3Mamba(mb.backward_blocks, reverse=True, s_y=self.S_Y, form=form)
        Wo = mb.output_proj.weight.detach()
        d = Wo.shape[1] // 2
        self.register_buffer("Wo_f", Wo[:, :d].clone(), persistent=False)
        self.register_buffer("Wo_b", Wo[:, d:].clone(), persistent=False)
        self.register_buffer("bo", mb.output_proj.bias.detach().clone(), persistent=False)
        self.register_buffer("bo_s", (mb.output_proj.bias.detach() / self.S_Y).clone(), persistent=False)
        self.norm = mb.norm

    def forward(self, x):
        if not self.npu:
            return super().forward(x)
        yf, gf = self.fw.pre(x)
        yb, gb = self.bw.pre(x)
        rs = 1.0 / self.S_Y
        xs = x * rs
        f = self.fw.out(yf * gf) + xs
        b = self.bw.out(yb * gb) + xs
        if STATS is not None:
            for k, t in (("y_scan", yf), ("y_scan", yb), ("y_gated", yf * gf), ("y_gated", yb * gb),
                         ("branch_out", f), ("branch_out", b)):
                _stat(k, t)
        o = G.lin(f, self.Wo_f) + G.lin(b, self.Wo_b) + self.bo_s
        return div_free_layer_norm(o, self.norm, eps_scale=rs)


# ------------------------------------------------------------------ encoder / decoders with a chosen norm

def _dense_block(db, x, inorm):
    skip = x
    for i in range(db.depth):
        conv, norm, act = db.dense_block[i][0], db.dense_block[i][1], db.dense_block[i][2]
        y = G.conv2d(skip, conv.weight, conv.bias, padding=conv.padding, dilation=conv.dilation)
        y = G.prelu(inorm(y, norm), act.weight)
        skip = torch.cat([y, skip], dim=1)
    return y


def _up_branch(dec, x, inorm):
    y = _dense_block(dec.dense_block, x, inorm)
    u1 = dec.up_conv1
    y = G.prelu(inorm(G.sp_conv(u1[0], y), u1[1]), u1[2].weight)
    u2 = dec.up_conv2
    y = y.permute(0, 1, 3, 2)
    y = G.prelu(inorm(G.sp_conv(u2[0], y), u2[1]), u2[2].weight)
    return y.permute(0, 1, 3, 2)


class R3ReUSE(G.GPUReUSE):
    """spec [1, 2, T, F] -> out [1, 3, T, F]; scan forms per axis; npu=True = DIV-free norms and Mamba scaling."""

    def __init__(self, model, time_form="F0", freq_form="F0", npu=False):
        nn.Module.__init__(self)
        self.model = model
        self.npu = npu
        self.time_form, self.freq_form = time_form, freq_form
        self.blocks = nn.ModuleList()
        for blk in model.TSMamba:
            self.blocks.append(nn.ModuleList([R3MambaBlock(blk.time_mamba, time_form, npu),
                                              R3MambaBlock(blk.freq_mamba, freq_form, npu)]))
        self.inorm = div_free_instance_norm if npu else G.safe_instance_norm

    def encoder(self, x):
        if not self.npu:
            return super().encoder(x)
        de = self.model.dense_encoder
        c1 = de.dense_conv_1
        y = G.conv2d(x, c1[0].weight, c1[0].bias)
        y = G.prelu(self.inorm(y, c1[1]), c1[2].weight)
        y = _dense_block(de.dense_block, y, self.inorm)
        c2 = de.dense_conv_2
        y = G.conv2d(y, c2[0].weight, c2[0].bias, stride=c2[0].stride)
        return G.prelu(self.inorm(y, c2[1]), c2[2].weight)

    def forward(self, spec):
        if not self.npu:
            return super().forward(spec)
        _, _, T, Fq = spec.shape
        x = F.pad(spec, (0, 2, 0, 2))
        y = self.encoder(x)
        s = y.permute(0, 2, 3, 1)
        for tb, fb in self.blocks:
            s = tb(s) + s
            s = s.permute(0, 2, 1, 3)
            s = fb(s) + s
            s = s.permute(0, 2, 1, 3)
        y = s.permute(0, 3, 1, 2)
        md = self.model.mask_decoder
        mag = G.conv2d(_up_branch(md, y, self.inorm), md.final_conv.weight, md.final_conv.bias)
        pd = self.model.phase_decoder
        yp = _up_branch(pd, y, self.inorm)
        xr = G.conv2d(yp, pd.phase_conv_r.weight, pd.phase_conv_r.bias)
        xi = G.conv2d(yp, pd.phase_conv_i.weight, pd.phase_conv_i.bias)
        out = torch.cat([mag, xr, xi], dim=1)
        return out[:, :, :T, :Fq]
