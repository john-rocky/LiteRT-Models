#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 (the selective-scan recurrence of mamba_ssm, state-spaces/mamba @ a14b1dff)
"""r3 scan forms for the RE-USE GPU form: the same selective scan as reuse_gpu.scan_hs (r1, Hillis-Steele) written so
one forward reads and writes fewer state-sized tensors. All forms are exact in real arithmetic (the same recurrence,
re-associated); every value that enters exp() is <= 0 (a decay), never a cumulative sum divided back out.

Common API: f(delta, du, Bm, Cm, A, reverse, **kw) -> y
  delta, du = delta * u: [1, L, M, D]; Bm, Cm: [1, L, M, N]; A: [D, N] (negative); y = sum_n h * C: [1, L, M, D]
  (no D * u skip, no gate). Forward: h[l] = exp(delta[l] A) h[l-1] + du[l] B[l]. Reverse (the backward Mamba on the
  un-flipped sequence): h[l] = exp(delta[l] A) h[l+1] + du[l] B[l].

Forms:
  F0  scan_f0       r1's Hillis-Steele in log space on the state [1, L*M, D, N] (reuse_gpu._scan_hs, unchanged).
  F3a scan_hs_mul   Hillis-Steele in product form: the decays e = exp(delta A) are formed once and multiplied per
                    level (zero-padded shifts: a row whose window passes the sequence start only ever multiplies a
                    zero h), instead of a log sum and an exp per level.
  F3b scan_hs_lam   Hillis-Steele carrying the window sum of delta (lam = sum delta, [1, P, D, 1], 1/N of the state)
                    and rebuilding the decay exp(lam A) at each level, instead of the state-sized log sum.
  F2  scan_seq      the recurrence unrolled step by step along L on the state [1, N, M, D] (one step touches 1/L of
                    the state); delta / du slices are the stream's own [1, 1, M, D] rows, B / C are laid out once as
                    [1, N, L*M, 1]; y[l] = sum over N (axis 1) = a [1, 1, M, D] row of the stream; one tree concat.
  F1  scan_block    L split into nb = ceil(L / Q) blocks of Q (zero padded at the end: delta = du = 0 is the identity
                    step), all blocks in parallel on the state [1, N, nb*M, D]: pass 1 = the local recurrence from a
                    zero state -> each block's end state E; the block decay G = exp((sum of delta over the block) A);
                    the carry across blocks step by step (nb steps on [1, N, M, D]); pass 2 = the local recurrence
                    again from the carried-in state, emitting y. Inputs are laid out step-major once (one transpose
                    each), y is put back block-major with one transpose.
Every form keeps tensors at rank <= 4 and uses only SLICE / PAD / CONCATENATION / TRANSPOSE / RESHAPE / MUL / ADD / EXP
/ SUM: no GATHER(_ND), SELECT, REVERSE, CUMSUM, SPLIT, DIV.
"""
import math
import os

import torch
import torch.nn.functional as F

CAT_FANIN = int(os.environ.get("R3_CAT_FANIN", "8"))   # concat fan-in limit (a tree of CONCATENATION ops)


def cat_tree(xs, dim, fanin=None):
    """torch.cat as a tree of concats with at most `fanin` inputs each (same result, bounded op arity)."""
    fanin = fanin or CAT_FANIN
    while len(xs) > fanin:
        xs = [xs[i] if i + 1 == len(xs) else torch.cat(xs[i:i + fanin], dim=dim) for i in range(0, len(xs), fanin)]
    return xs[0] if len(xs) == 1 else torch.cat(xs, dim=dim)


XNN_FANIN = 5   # the most inputs of a CONCATENATION the LiteRT CPU accelerator (XNNPACK) takes (r3 probe 02:3x)


def cat_tree_x(xs):
    """torch.cat(xs, dim=1) for 4D pieces [1, a_i, V, C] as CONCATENATIONs of <= XNN_FANIN inputs that the converter
    cannot merge back into one (it folds a same-axis concat of concats: a plain tree exported as ONE concat): between
    levels each partial result is reshaped [1, K, V, C] -> [1, 1, K*V, C] (dims 1 and 2 merged, C kept, linear order
    kept) and the next level concatenates along dim 2; the level after splits back and concatenates along dim 1."""
    V, C = xs[0].shape[2], xs[0].shape[3]
    axis = 1
    while len(xs) > XNN_FANIN:
        groups = [xs[i:i + XNN_FANIN] for i in range(0, len(xs), XNN_FANIN)]
        xs = [g[0] if len(g) == 1 else torch.cat(g, dim=axis) for g in groups]
        if axis == 1:
            xs = [x.reshape(1, 1, x.shape[1] * V, C) for x in xs]
            axis = 2
        else:
            xs = [x.reshape(1, x.shape[2] // V, V, C) for x in xs]
            axis = 1
    out = xs[0] if len(xs) == 1 else torch.cat(xs, dim=axis)
    return out if axis == 1 else out.reshape(1, out.shape[2] // V, V, C)


def _shift(t, k, reverse, P):
    """Rows r <- r - k (prefix) or r + k (suffix) along axis 1 of [1, P, ...], zero filled."""
    if not reverse:
        return F.pad(t[:, :P - k], (0, 0, 0, 0, k, 0))
    return F.pad(t[:, k:], (0, 0, 0, 0, 0, k))


# ------------------------------------------------------------------ F0 = r1

def scan_f0(delta, du, Bm, Cm, A, reverse, **kw):
    import reuse_gpu as G
    return G._scan_hs(delta, du, Bm, Cm, A, reverse)


# ------------------------------------------------------------------ F3: Hillis-Steele variants

def scan_hs_mul(delta, du, Bm, Cm, A, reverse, A4=None, **kw):
    """F3a: product-form Hillis-Steele (exp once)."""
    _, L, M, D = delta.shape
    N = A.shape[1]
    P = L * M
    A4 = A.reshape(1, 1, D, N) if A4 is None else A4
    e = torch.exp(delta.reshape(1, P, D, 1) * A4)
    h = du.reshape(1, P, D, 1) * Bm.reshape(1, P, 1, N)
    s = 1
    while s < L:
        k = s * M
        last = 2 * s >= L
        h_sh = _shift(h, k, reverse, P)
        e_sh = None if last else _shift(e, k, reverse, P)
        h = e * h_sh + h
        if not last:
            e = e * e_sh
        s *= 2
    y = (h * Cm.reshape(1, P, 1, N)).sum(dim=-1, keepdim=True)
    return y.reshape(1, L, M, D)


def scan_hs_lam(delta, du, Bm, Cm, A, reverse, A4=None, **kw):
    """F3b: Hillis-Steele carrying the window sum of delta ([1, P, D, 1]); decay = exp(lam A) per level."""
    _, L, M, D = delta.shape
    N = A.shape[1]
    P = L * M
    A4 = A.reshape(1, 1, D, N) if A4 is None else A4
    lam = delta.reshape(1, P, D, 1)
    h = du.reshape(1, P, D, 1) * Bm.reshape(1, P, 1, N)
    s = 1
    while s < L:
        k = s * M
        last = 2 * s >= L
        h_sh = _shift(h, k, reverse, P)
        lam_sh = None if last else _shift(lam, k, reverse, P)
        h = torch.exp(lam * A4) * h_sh + h
        if not last:
            lam = lam + lam_sh
        s *= 2
    y = (h * Cm.reshape(1, P, 1, N)).sum(dim=-1, keepdim=True)
    return y.reshape(1, L, M, D)


# ------------------------------------------------------------------ F2: sequential

def scan_seq(delta, du, Bm, Cm, A, reverse, AT=None, **kw):
    """F2: step-by-step recurrence along L on the state [1, N, M, D]."""
    _, L, M, D = delta.shape
    N = A.shape[1]
    AT = A.t().reshape(1, N, 1, D) if AT is None else AT            # [1, N, 1, D]
    BT = Bm.permute(0, 3, 1, 2).reshape(1, N, L * M, 1)              # [1, N, L*M, 1], column l*M + m
    CT = Cm.permute(0, 3, 1, 2).reshape(1, N, L * M, 1)
    ys = [None] * L
    h = None
    for l in (range(L) if not reverse else range(L - 1, -1, -1)):
        b = du[:, l:l + 1] * BT[:, :, l * M:(l + 1) * M]             # [1, 1, M, D] x [1, N, M, 1] -> [1, N, M, D]
        h = b if h is None else torch.exp(delta[:, l:l + 1] * AT) * h + b
        ys[l] = (h * CT[:, :, l * M:(l + 1) * M]).sum(dim=1, keepdim=True)   # [1, 1, M, D]
    return cat_tree(ys, dim=1)                                        # [1, L, M, D]


# ------------------------------------------------------------------ F1: blocks of Q + carry

def scan_block(delta, du, Bm, Cm, A, reverse, Q=8, AT=None, cache=False, **kw):
    """F1: two-pass block-sequential scan (see the module docstring). cache=True (F1c): pass 1 keeps every step's
    decay e_j and input b_j and pass 2 reuses them (h = e_j h + b_j, y_j = sum_N h C_j) instead of rebuilding them."""
    _, L, M, D = delta.shape
    N = A.shape[1]
    AT = A.t().reshape(1, N, 1, D) if AT is None else AT
    Q = min(Q, L)
    nb = math.ceil(L / Q)
    Lp = nb * Q
    if Lp > L:
        p = (0, 0, 0, 0, 0, Lp - L)
        delta, du, Bm, Cm = F.pad(delta, p), F.pad(du, p), F.pad(Bm, p), F.pad(Cm, p)
    W = nb * M

    def steps_d(x):          # [1, Lp, M, D] -> [1, Q, nb*M, D]: row j holds step j of every block
        return x.reshape(1, nb, Q, M * D).permute(0, 2, 1, 3).reshape(1, Q, W, D)

    def steps_n(x):          # [1, Lp, M, N] -> [1, N, Q*nb*M, 1]: columns j*W + b*M + m
        x = x.reshape(1, nb, Q, M * N).permute(0, 2, 1, 3)
        return x.reshape(1, 1, Q * W, N).permute(0, 3, 2, 1)

    dS, uS = steps_d(delta), steps_d(du)
    BS, CS = steps_n(Bm), steps_n(Cm)
    order = list(range(Q)) if not reverse else list(range(Q - 1, -1, -1))

    def step(h, j):
        b = uS[:, j:j + 1] * BS[:, :, j * W:(j + 1) * W]             # [1, N, W, D]
        return b if h is None else torch.exp(dS[:, j:j + 1] * AT) * h + b

    hin = None
    es, bs = {}, {}
    if nb > 1:
        h = None
        for j in order:                                             # pass 1: local end states
            if cache:
                bs[j] = uS[:, j:j + 1] * BS[:, :, j * W:(j + 1) * W]
                es[j] = torch.exp(dS[:, j:j + 1] * AT)
                h = bs[j] if h is None else es[j] * h + bs[j]
            else:
                h = step(h, j)
        E = h
        G = torch.exp(dS.sum(dim=1, keepdim=True) * AT)             # [1, N, W, D] block decays
        blk = lambda t, b: t[:, :, b * M:(b + 1) * M]
        hs = []
        if not reverse:                                             # hin[b] = G[b-1] hin[b-1] + E[b-1], hin[0] = 0
            c = blk(E, 0)
            hs.append(c)
            for b in range(1, nb - 1):
                c = blk(G, b) * c + blk(E, b)
                hs.append(c)
            hin = F.pad(cat_tree(hs, dim=2), (0, 0, M, 0))
        else:                                                       # hin[b] = G[b+1] hin[b+1] + E[b+1], hin[nb-1] = 0
            c = blk(E, nb - 1)
            hs.append(c)
            for b in range(nb - 2, 0, -1):
                c = blk(G, b) * c + blk(E, b)
                hs.append(c)
            hs.reverse()
            hin = F.pad(cat_tree(hs, dim=2), (0, 0, 0, M))
    ys = [None] * Q
    h = hin
    for j in order:                                                 # pass 2: emit y
        if j in bs:
            h = bs[j] if h is None else es[j] * h + bs[j]
        else:
            h = step(h, j)
        ys[j] = (h * CS[:, :, j * W:(j + 1) * W]).sum(dim=1, keepdim=True)   # [1, 1, W, D]
    y = cat_tree(ys, dim=1)                                         # [1, Q, W, D]
    y = y.reshape(1, Q, nb, M * D).permute(0, 2, 1, 3).reshape(1, Lp, M, D)
    return y[:, :L] if Lp > L else y


def scan_seq_x(delta, du, Bm, Cm, A, reverse, AT=None, **kw):
    """F2x: F2 with the y assembly as XNNPACK-sized concats (cat_tree_x)."""
    _, L, M, D = delta.shape
    N = A.shape[1]
    AT = A.t().reshape(1, N, 1, D) if AT is None else AT
    BT = Bm.permute(0, 3, 1, 2).reshape(1, N, L * M, 1)
    CT = Cm.permute(0, 3, 1, 2).reshape(1, N, L * M, 1)
    ys = [None] * L
    h = None
    for l in (range(L) if not reverse else range(L - 1, -1, -1)):
        b = du[:, l:l + 1] * BT[:, :, l * M:(l + 1) * M]
        h = b if h is None else torch.exp(delta[:, l:l + 1] * AT) * h + b
        ys[l] = (h * CT[:, :, l * M:(l + 1) * M]).sum(dim=1, keepdim=True)
    return cat_tree_x(ys)


def scan_block_p(delta, du, Bm, Cm, A, reverse, Q=8, AP=None, **kw):
    """F1p: F1 (two passes, blocks of Q, pass 1 keeps e_j and b_j) on a positions-major state [1, nb*M, N, D] (row
    b*M + m, N on axis 2): a block is a contiguous row range, so the carried-in states assemble with cat_tree_x
    (XNNPACK-sized concats), and y_j = sum over N (axis 2)."""
    _, L, M, D = delta.shape
    N = A.shape[1]
    AP = A.t().reshape(1, 1, N, D) if AP is None else AP          # [1, 1, N, D]
    Q = min(Q, L)
    nb = math.ceil(L / Q)
    Lp = nb * Q
    if Lp > L:
        p = (0, 0, 0, 0, 0, Lp - L)
        delta, du, Bm, Cm = F.pad(delta, p), F.pad(du, p), F.pad(Bm, p), F.pad(Cm, p)
    W = nb * M

    def steps_d(x):          # [1, Lp, M, D] -> [1, Q*W, 1, D]: rows j*W + b*M + m
        return x.reshape(1, nb, Q, M * D).permute(0, 2, 1, 3).reshape(1, Q * W, 1, D)

    def steps_n(x):          # [1, Lp, M, N] -> [1, Q*W, N, 1]
        return x.reshape(1, nb, Q, M * N).permute(0, 2, 1, 3).reshape(1, Q * W, N, 1)

    dS, uS = steps_d(delta), steps_d(du)
    BS, CS = steps_n(Bm), steps_n(Cm)
    order = list(range(Q)) if not reverse else list(range(Q - 1, -1, -1))
    rows = lambda t, j: t[:, j * W:(j + 1) * W]
    es, bs = {}, {}
    for j in order:
        bs[j] = rows(uS, j) * rows(BS, j)                          # [1, W, 1, D] x [1, W, N, 1] -> [1, W, N, D]
        es[j] = torch.exp(rows(dS, j) * AP)
    hin = None
    if nb > 1:
        h = None
        for j in order:                                            # pass 1: local end states
            h = bs[j] if h is None else es[j] * h + bs[j]
        E = h
        lam = dS.reshape(1, Q, W, D).sum(dim=1, keepdim=True).reshape(1, W, 1, D)
        G = torch.exp(lam * AP)                                    # [1, W, N, D]
        blk = lambda t, b: t[:, b * M:(b + 1) * M]                 # [1, M, N, D]
        hs = []
        if not reverse:
            c = blk(E, 0)
            hs.append(c)
            for b in range(1, nb - 1):
                c = blk(G, b) * c + blk(E, b)
                hs.append(c)
            hin = F.pad(cat_tree_x(hs), (0, 0, 0, 0, M, 0))
        else:
            c = blk(E, nb - 1)
            hs.append(c)
            for b in range(nb - 2, 0, -1):
                c = blk(G, b) * c + blk(E, b)
                hs.append(c)
            hs.reverse()
            hin = F.pad(cat_tree_x(hs), (0, 0, 0, 0, 0, M))
    ys = [None] * Q
    h = hin
    for j in order:                                                # pass 2: emit y
        h = bs[j] if h is None else es[j] * h + bs[j]
        ys[j] = (h * rows(CS, j)).sum(dim=2, keepdim=True).reshape(1, 1, W, D)
    y = cat_tree_x(ys)                                             # [1, Q, W, D]
    y = y.reshape(1, Q, nb, M * D).permute(0, 2, 1, 3).reshape(1, Lp, M, D)
    return y[:, :L] if Lp > L else y


FORMS = {
    "F0": scan_f0,
    "F3a": scan_hs_mul,
    "F3b": scan_hs_lam,
    "F2": scan_seq,
    "F2x": scan_seq_x,
    "F1": scan_block,
}


def get(name):
    """'F1q8' -> (scan_block, {'Q': 8}); 'F1cq8' -> (scan_block, {'Q': 8, 'cache': True}); 'F2' -> (scan_seq, {})."""
    if name.startswith("F1q"):
        return scan_block, dict(Q=int(name[3:]))
    if name.startswith("F1cq"):
        return scan_block, dict(Q=int(name[4:]), cache=True)
    if name.startswith("F1pq"):
        return scan_block_p, dict(Q=int(name[4:]))
    return FORMS[name], {}


# ------------------------------------------------------------------ analytic counts

def state_tensors(name, L, Q=None):
    """Number of state-sized tensors (in units of the full state [L*M, D, N]) one scan writes, by form, counting each
    op output once (no fusion assumed): the r1 table's measure. Small tensors ([.., D] or [.., N]) are not counted."""
    if name == "F0":
        lev = math.ceil(math.log2(L)) if L > 1 else 0
        # la, h; per level: h slice, h pad, la slice, la pad, exp, mul, add (+ la add except the last level)
        n = 2 + lev * 7 + (lev - 1)
        return n + 1                         # h * C
    if name == "F3a":
        lev = math.ceil(math.log2(L)) if L > 1 else 0
        # la, e, h; per level: h slice, h pad, mul, add (+ e slice, e pad, e mul except the last level)
        return 3 + lev * 4 + (lev - 1) * 3 + 1
    if name == "F3b":
        lev = math.ceil(math.log2(L)) if L > 1 else 0
        # h; per level: h slice, h pad, lam*A, exp, mul, add
        return 1 + lev * 6 + 1
    if name in ("F2", "F2x"):
        # per step (1/L of the state each): b, la, exp, mul, add, h*C  (first step: b, h*C)
        return (6 * (L - 1) + 2) / L
    if name.startswith("F1cq") or name.startswith("F1pq"):
        Q = Q or int(name[4:])
        nb = math.ceil(L / Q)
        fr = nb * Q / L
        if nb == 1:
            return fr * (6 * (Q - 1) + 2) / Q
        # pass 1: b, la, e (every step), mul, add (not the first); G: 2/Q; carry ~4/Q; pass 2: mul, add, h*C
        return fr * ((3 * Q + 2 * (Q - 1)) / Q + 2 / Q + 4 / Q + 3)
    if name.startswith("F1"):
        Q = Q or int(name[3:])
        nb = math.ceil(L / Q)
        Lp = nb * Q
        fr = Lp / L                           # padding overhead
        if nb == 1:
            return fr * (6 * (Q - 1) + 2) / Q
        p1 = (5 * (Q - 1) + 1) / Q            # pass 1: b, la, exp, mul, add (first step: b)
        g = 2 / Q                             # block decays: lam*A, exp
        carry = 2 * 2 / Q                     # carry slices / mul / add on [N, M, D] per block (~4 per block, 1/Lp each)
        p2 = 6                                # pass 2: b, la, exp, mul, add, h*C per step
        return fr * (p1 + g + carry + p2)
    raise KeyError(name)
