#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1 (a re-authoring of nvidia/NVIDIA-NemotronLabs-RE-USE's forward)
# The selective-scan math follows mamba_ssm (Apache-2.0, state-spaces/mamba @ a14b1dff), see mamba_ref.py.
"""GPU form of RE-USE for LiteRT CompiledModel (ML Drift): a fixed-shape re-authoring of SEMamba.forward that
reads the reference module's own weights. Every rewrite is exact in real arithmetic:

  stream     the 30 TFMamba blocks run on a channels-last 4D stream [1, L, M, 64] (L = scan axis, M = the other
             position axis): time blocks on [1, T', F', 64], frequency blocks on [1, F', T', 64] (one 4D permute
             between them, as the original's permute/view does). No 3D [B*M, L, C] tensor, no flip.
  Mamba      in_proj / x_proj / dt_proj / out_proj = row-wise linears; the depthwise causal Conv1d (k = 4) = a sum
             of 4 shifted copies along L (zero pad on the left); the backward Mamba on the un-flipped sequence =
             the anti-causal conv (pad on the right, taps reversed) + a suffix scan: flip(f(flip(x))) == f'(x).
  scan       Hillis-Steele over L in log space on the state [1, L*M, 256, 16] (row l*M + m): level k combines row
             r with row r - 2^k*M (prefix) or r + 2^k*M (suffix), carrying la = sum(delta*A) and
             h <- exp(la)*h_shifted + h; the shifted copies are zero-PADded slices (log 1 = 0, h 0 = 0), so no
             constant ones / zeros tensors, no SELECT, no GATHER.
  softplus   relu(x) + log1p(exp(-|x|)) (no SELECT).
  norms      InstanceNorm2d and LayerNorm in the down-scaled domain (SafeLayerNorm v2 idea): per channel (IN) or per
             row (LN) S = max(amax * 2^-2, 1), x / S, mean, d = x/S - mean, var of d, y = d * rsqrt(var + eps/S^2);
             spatial means are chained single-axis means (T and F are not powers of two).
  PReLU      relu(x) - w * relu(-x).
  SPConvT    the 5D view -> [b, r, c, H*W] -> permute -> [b, c, H, W*r] (CMGAN recipe 3).
  phase      atan2 is not computed: the graph returns the magnitude and (x_r, x_i); the host forms
             cos = x_r / hypot, sin = x_i / hypot (CMGAN recipe 1).
  fp16 range the Mamba output path is divided by 16 (folded into C and D) and by a per-row factor r = 1/max(1, amax)
             before the gate; LayerNorm absorbs it (eps * rs^2), see GPUMambaBlock.
Input  spec [1, 2, T, F]: channel 0 = log1p(|STFT|), channel 1 = angle(STFT), time-major (host: center=True, reflect).
Output out  [1, 3, T, F]: channel 0 = compressed magnitude (before the host's zero-frame rule), 1 = x_r, 2 = x_i.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NORM_SHIFT = 0.25      # S = max(amax * NORM_SHIFT, 1): |x / S| <= 4 before squaring
# rewrite switches (the stacking record in r1 turns them on one at a time; the export uses the defaults)
SAFE_IN = True         # InstanceNorm2d in the down-scaled domain (else F.instance_norm)
SAFE_LN = True         # LayerNorm in the down-scaled domain (else F.layer_norm)
SAFE_SOFTPLUS = True   # relu(x) + log1p(exp(-|x|)) (else F.softplus)
VAR_FLOOR = 1e-6       # rsqrt guard in the scaled domain (only binds when std < 1e-3 * S)


# ------------------------------------------------------------------ building blocks

def prelu(x, w):
    """PReLU with a per-channel weight on dim 1 of [1, C, H, W]."""
    return F.relu(x) - w.view(1, -1, 1, 1) * F.relu(-x)


def safe_instance_norm(x, norm):
    """InstanceNorm2d(affine) on [1, C, H, W], in the per-channel down-scaled domain."""
    if not SAFE_IN:
        return F.instance_norm(x, weight=norm.weight, bias=norm.bias, eps=norm.eps)
    amax = x.abs().amax(dim=3, keepdim=True).amax(dim=2, keepdim=True)
    S = torch.clamp(amax * NORM_SHIFT, min=1.0)
    xs = x / S
    mu = xs.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
    d = xs - mu
    var = (d * d).mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
    y = d * torch.rsqrt(torch.clamp(var + norm.eps / (S * S), min=VAR_FLOOR))
    return y * norm.weight.view(1, -1, 1, 1) + norm.bias.view(1, -1, 1, 1)


def safe_layer_norm(x, ln, eps_scale=None):
    """LayerNorm over the last axis of [1, L, M, C], in the per-row down-scaled domain. eps_scale: the factor the
    caller multiplied the input by (scalar or per-row tensor); eps follows as eps * eps_scale^2 so the result equals
    LayerNorm of the unscaled input."""
    if not SAFE_LN:
        eps_t = ln.eps if eps_scale is None else ln.eps * eps_scale * eps_scale
        mu = x.mean(dim=-1, keepdim=True)
        d = x - mu
        return d * torch.rsqrt((d * d).mean(dim=-1, keepdim=True) + eps_t) * ln.weight + ln.bias
    amax = x.abs().amax(dim=-1, keepdim=True)
    S = torch.clamp(amax * NORM_SHIFT, min=1.0)
    xs = x / S
    mu = xs.mean(dim=-1, keepdim=True)
    d = xs - mu
    var = (d * d).mean(dim=-1, keepdim=True)
    eps = ln.eps if eps_scale is None else ln.eps * eps_scale * eps_scale
    y = d * torch.rsqrt(torch.clamp(var + eps / (S * S), min=VAR_FLOOR))
    return y * ln.weight + ln.bias


def softplus_safe(x):
    if not SAFE_SOFTPLUS:
        return F.softplus(x)
    return F.relu(x) + torch.log1p(torch.exp(-torch.abs(x)))


def lin(x, w, b=None):
    """Row-wise linear over the last axis of a 4D tensor (weights [out, in])."""
    return F.linear(x, w, b)


def dwconv_L(x, w, b, reverse):
    """Depthwise Conv1d (k taps) along axis 1 of [1, L, M, C]. Causal: y[l] = b + sum_k w[k] x[l-(K-1)+k].
    reverse (the backward Mamba on the un-flipped sequence): y[l] = b + sum_j w[K-1-j] x[l+j]."""
    K = w.shape[-1]
    L = x.shape[1]
    wk = w.reshape(w.shape[0], K)                          # [C, K]
    if not reverse:
        xp = F.pad(x, (0, 0, 0, 0, K - 1, 0))
        y = xp[:, 0:L] * wk[:, 0]
        for k in range(1, K):
            y = y + xp[:, k:k + L] * wk[:, k]
    else:
        xp = F.pad(x, (0, 0, 0, 0, 0, K - 1))
        y = xp[:, 0:L] * wk[:, K - 1]
        for j in range(1, K):
            y = y + xp[:, j:j + L] * wk[:, K - 1 - j]
    return y + b


EVAL_M_CHUNK = None   # torch evaluation only: run the scan over slices of M (rows are independent); None in export
EVAL_CONV_TILE = None  # torch evaluation only: conv2d over output-row tiles below this im2col size (bytes); None in export


def conv2d(x, w, b=None, stride=1, padding=0, dilation=1):
    """F.conv2d; in torch evaluation (EVAL_CONV_TILE) over output-row tiles (same sums, bounded im2col)."""
    if EVAL_CONV_TILE is None:
        return F.conv2d(x, w, b, stride=stride, padding=padding, dilation=dilation)
    st = (stride, stride) if isinstance(stride, int) else tuple(stride)
    pd = (padding, padding) if isinstance(padding, int) else tuple(padding)
    dl = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    _, C, Hh, Ww = x.shape
    kh, kw = w.shape[2], w.shape[3]
    Ho = (Hh + 2 * pd[0] - dl[0] * (kh - 1) - 1) // st[0] + 1
    Wo = (Ww + 2 * pd[1] - dl[1] * (kw - 1) - 1) // st[1] + 1
    cols = C * kh * kw * Ho * Wo * 4
    if cols <= EVAL_CONV_TILE or Ho < 2:
        return F.conv2d(x, w, b, stride=st, padding=pd, dilation=dl)
    xp = F.pad(x, (pd[1], pd[1], pd[0], pd[0]))
    rows = max(1, int(Ho * EVAL_CONV_TILE // cols))
    outs = []
    for a in range(0, Ho, rows):
        e = min(Ho, a + rows)
        i0, i1 = a * st[0], (e - 1) * st[0] + dl[0] * (kh - 1) + 1
        outs.append(F.conv2d(xp[:, :, i0:i1], w, b, stride=st, padding=0, dilation=dl))
    return torch.cat(outs, 2)


def scan_hs(delta, du, Bm, Cm, A, reverse):
    """Selective scan along axis 1 of [1, L, M, *] (Hillis-Steele, log space, zero-padded shifts).
    delta, du = delta*u: [1, L, M, D]; Bm, Cm: [1, L, M, N]; A: [D, N] (negative). Returns sum_n h*C: [1, L, M, D]."""
    M_all = delta.shape[2]
    if EVAL_M_CHUNK is not None and M_all > EVAL_M_CHUNK:
        outs = []
        for a in range(0, M_all, EVAL_M_CHUNK):
            b = min(M_all, a + EVAL_M_CHUNK)
            outs.append(_scan_hs(delta[:, :, a:b], du[:, :, a:b], Bm[:, :, a:b], Cm[:, :, a:b], A, reverse))
        return torch.cat(outs, dim=2)
    return _scan_hs(delta, du, Bm, Cm, A, reverse)


def _scan_hs(delta, du, Bm, Cm, A, reverse):
    _, L, M, D = delta.shape
    N = A.shape[1]
    P = L * M
    la = delta.reshape(1, P, D, 1) * A.reshape(1, 1, D, N)           # log of the per-step decay
    h = du.reshape(1, P, D, 1) * Bm.reshape(1, P, 1, N)
    s = 1
    while s < L:
        k = s * M
        last = 2 * s >= L
        if not reverse:
            h_sh = F.pad(h[:, :P - k], (0, 0, 0, 0, k, 0))
            la_sh = None if last else F.pad(la[:, :P - k], (0, 0, 0, 0, k, 0))
        else:
            h_sh = F.pad(h[:, k:], (0, 0, 0, 0, 0, k))
            la_sh = None if last else F.pad(la[:, k:], (0, 0, 0, 0, 0, k))
        h = torch.exp(la) * h_sh + h
        if not last:
            la = la + la_sh
        s *= 2
    y = (h * Cm.reshape(1, P, 1, N)).sum(dim=-1, keepdim=True)        # [1, P, D, 1]
    return y.reshape(1, L, M, D)


class GPUMamba(nn.Module):
    """mamba_ssm Mamba (slow path) on [1, L, M, d_model] along axis 1; reverse = the backward branch of
    MambaBlock (flip -> Mamba -> flip) computed without flips. pre() returns the scan output y / s_y (s_y, a power
    of two, folded into x_proj's C rows and D) and the gate silu(z); out() applies out_proj."""

    def __init__(self, m, reverse, s_y=1.0):
        super().__init__()
        self.m = m
        self.reverse = reverse
        di, dr, ds = m.d_inner, m.dt_rank, m.d_state
        W = m.in_proj.weight.detach()
        self.register_buffer("Wx", W[:di].clone(), persistent=False)
        self.register_buffer("Wz", W[di:].clone(), persistent=False)
        Xp = m.x_proj.weight.detach()
        self.register_buffer("Wdt_low", Xp[:dr].clone(), persistent=False)
        self.register_buffer("WB", Xp[dr:dr + ds].clone(), persistent=False)
        self.register_buffer("WC", (Xp[dr + ds:dr + 2 * ds] / s_y).clone(), persistent=False)
        self.register_buffer("A", (-torch.exp(m.A_log.detach().float())).clone(), persistent=False)
        self.register_buffer("Dsk", (m.D.detach().float() / s_y).clone(), persistent=False)

    def pre(self, x):
        m = self.m
        xi = lin(x, self.Wx)
        z = lin(x, self.Wz)
        xc = dwconv_L(xi, m.conv1d.weight, m.conv1d.bias, self.reverse)
        u = xc * torch.sigmoid(xc)
        dt = lin(lin(u, self.Wdt_low), m.dt_proj.weight, m.dt_proj.bias)
        delta = softplus_safe(dt)
        Bm = lin(u, self.WB)
        Cm = lin(u, self.WC)
        y = scan_hs(delta, delta * u, Bm, Cm, self.A, self.reverse)
        y = y + u * self.Dsk
        return y, z * torch.sigmoid(z)

    def out(self, yg):
        return lin(yg, self.m.out_proj.weight)


class GPUMambaBlock(nn.Module):
    """MambaBlock: LN(output_proj(cat[fw(x) + x, bw(x) + x])) on [1, L, M, 64].
    fp16 range (exact up to rounding): the scan outputs come out divided by S_Y = 16 (folded into C and D); with
    row_scale, each row (position) of both branches is further multiplied by r = 1 / max(1, amax_c |y_fw|, |y_bw|),
    so |y * r| <= 1 before the gate. Everything that meets in the LayerNorm input (the Mamba outputs, the residual x,
    output_proj's bias) carries the same row factor rs = r / S_Y; LayerNorm is invariant to it when its eps is
    taken as eps * rs^2 (done here)."""

    def __init__(self, mb, row_scale=True):
        super().__init__()
        self.row_scale = row_scale
        self.S_Y = 16.0 if row_scale else 1.0
        self.fw = GPUMamba(mb.forward_blocks, reverse=False, s_y=self.S_Y)
        self.bw = GPUMamba(mb.backward_blocks, reverse=True, s_y=self.S_Y)
        Wo = mb.output_proj.weight.detach()
        d = Wo.shape[1] // 2
        self.register_buffer("Wo_f", Wo[:, :d].clone(), persistent=False)
        self.register_buffer("Wo_b", Wo[:, d:].clone(), persistent=False)
        self.register_buffer("bo", mb.output_proj.bias.detach().clone(), persistent=False)
        self.norm = mb.norm

    def forward(self, x):
        yf, gf = self.fw.pre(x)
        yb, gb = self.bw.pre(x)
        if self.row_scale:
            mrow = torch.maximum(yf.abs().amax(dim=-1, keepdim=True), yb.abs().amax(dim=-1, keepdim=True))
            r = 1.0 / torch.clamp(mrow, min=1.0)                       # [1, L, M, 1]
            yf = yf * r
            yb = yb * r
            rs = r * (1.0 / self.S_Y)
        else:
            rs = 1.0 / self.S_Y
        xs = x if self.S_Y == 1.0 and not self.row_scale else x * rs
        f = self.fw.out(yf * gf) + xs
        b = self.bw.out(yb * gb) + xs
        if not self.row_scale and self.S_Y == 1.0:
            o = lin(f, self.Wo_f) + lin(b, self.Wo_b, self.bo)
            return safe_layer_norm(o, self.norm)
        o = lin(f, self.Wo_f) + lin(b, self.Wo_b) + self.bo * rs
        return safe_layer_norm(o, self.norm, eps_scale=rs)


def dense_block(db, x):
    skip = x
    for i in range(db.depth):
        conv, norm, act = db.dense_block[i][0], db.dense_block[i][1], db.dense_block[i][2]
        y = conv2d(skip, conv.weight, conv.bias, padding=conv.padding, dilation=conv.dilation)
        y = prelu(safe_instance_norm(y, norm), act.weight)
        skip = torch.cat([y, skip], dim=1)
    return y


def sp_conv(sp, x):
    """SPConvTranspose2d without the 5D view (CMGAN recipe 3)."""
    x = F.pad(x, (1, 1, 0, 0))
    out = conv2d(x, sp.conv.weight, sp.conv.bias)
    b, rc, H, W = out.shape
    c = rc // sp.r
    out = out.reshape(b, sp.r, c, H * W).permute(0, 2, 3, 1)
    return out.reshape(b, c, H, W * sp.r)


def up_branch(dec, x):
    y = dense_block(dec.dense_block, x)
    u1 = dec.up_conv1
    y = prelu(safe_instance_norm(sp_conv(u1[0], y), u1[1]), u1[2].weight)
    u2 = dec.up_conv2
    y = y.permute(0, 1, 3, 2)
    y = prelu(safe_instance_norm(sp_conv(u2[0], y), u2[1]), u2[2].weight)
    return y.permute(0, 1, 3, 2)


class GPUReUSE(nn.Module):
    """spec [1, 2, T, F] -> out [1, 3, T, F] (see the module docstring)."""

    def __init__(self, model, row_scale=True):
        super().__init__()
        self.model = model
        self.blocks = nn.ModuleList()
        for blk in model.TSMamba:
            self.blocks.append(nn.ModuleList([GPUMambaBlock(blk.time_mamba, row_scale),
                                              GPUMambaBlock(blk.freq_mamba, row_scale)]))

    def encoder(self, x):
        de = self.model.dense_encoder
        c1 = de.dense_conv_1
        y = conv2d(x, c1[0].weight, c1[0].bias)
        y = prelu(safe_instance_norm(y, c1[1]), c1[2].weight)
        y = dense_block(de.dense_block, y)
        c2 = de.dense_conv_2
        y = conv2d(y, c2[0].weight, c2[0].bias, stride=c2[0].stride)
        return prelu(safe_instance_norm(y, c2[1]), c2[2].weight)

    def forward(self, spec):
        _, _, T, Fq = spec.shape
        x = F.pad(spec, (0, 2, 0, 2))                         # the generator's zero rows / columns
        y = self.encoder(x)                                   # [1, 64, T', F']
        s = y.permute(0, 2, 3, 1)                             # [1, T', F', 64]
        for tb, fb in self.blocks:
            s = tb(s) + s                                     # time Mamba (scan over T')
            s = s.permute(0, 2, 1, 3)                         # [1, F', T', 64]
            s = fb(s) + s                                     # frequency Mamba (scan over F')
            s = s.permute(0, 2, 1, 3)
        y = s.permute(0, 3, 1, 2)                             # [1, 64, T', F']
        md = self.model.mask_decoder
        mag = conv2d(up_branch(md, y), md.final_conv.weight, md.final_conv.bias)
        pd = self.model.phase_decoder
        yp = up_branch(pd, y)
        xr = conv2d(yp, pd.phase_conv_r.weight, pd.phase_conv_r.bias)
        xi = conv2d(yp, pd.phase_conv_i.weight, pd.phase_conv_i.bias)
        out = torch.cat([mag, xr, xi], dim=1)                 # [1, 3, 4T', 2F']
        return out[:, :, :T, :Fq]
