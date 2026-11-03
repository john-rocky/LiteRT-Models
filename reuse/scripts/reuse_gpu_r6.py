#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1 (a re-authoring of nvidia/NVIDIA-NemotronLabs-RE-USE's forward)
# The selective-scan math follows mamba_ssm (Apache-2.0, state-spaces/mamba @ a14b1dff), see mamba_ref.py.
"""r6 form of RE-USE: reuse_gpu_r3.R3ReUSE (r3, unchanged: scan forms, decoders, norms) with the dense encoder's
dilated 3x3 convs written so that every CONV_2D passes TFLite's mobile im2col check.

The check (LiteRT v2.2.0 tflite/kernels/conv.cc 262-274, run by TFLite on every builtin node before the GPU delegate
takes the graph): on Android / iOS a conv that needs im2col with batch*out_h*out_w*c_in*k_h*k_w*4 B >= 1 GiB fails
Prepare when its dilation is not 1 ("Dilated im2col buffer size overflowed."). The 2 s chunk's 4th encoder dense conv
(in [1, 256, 403, 323], dilation (8, 1)) needs 1.117 GiB; the same conv of the 1 s chunk needs 0.563 GiB and was
prepared on both phones (r4). IM2COL_BUDGET = that 1 s size sets a conv's tile count from its output rows
(n = ceil(im2col / budget)), so a graph whose convs all fit (every 1 s graph) comes out unchanged. A tilecrop tile also
computes its halo rows: its im2col may pass the budget by up to 2p rows (2 s: 0.582 GiB, 1.72x below the line).

  enc_fix = "tile"   (A) a conv over the budget runs over n = ceil(im2col / budget) equal output-row (time) tiles:
                     the conv's own zero padding made explicit (PAD), one SLICE per tile with the dilation halo,
                     VALID conv, CONCATENATION along time. Each output element is the same sum over the same inputs.
  enc_fix = "tilecrop" (A') as A, without the PAD of the whole input: each tile is the conv with its own (SAME)
                     padding on the tile's rows plus the padding halo cut from the unpadded input, its output cropped
                     to the tile's rows (the edge tiles compute p extra rows). litert-torch's layout pass has no NHWC
                     rewriter for constant_pad_nd; on graphs above ~2000 nodes its greedy partitioner (the min-cut one
                     only runs below |V|^2|E| < 2000^3) then keeps A's padded tensor in NCHW and adds 3 TRANSPOSEs of
                     it per tiled conv (r6 05:16, the 2 s F1pq8/F1pq8 export); slices of 4D NHWC tensors stay NHWC.
  enc_fix = "split"  (B) the dense conv over cat([y_{i-1}, ..., y_0, x]) is a sum of convs over channel groups (the
                     weights split along c_in, the bias once): the newest 64-channel pieces alone until the remaining
                     older pieces fit the budget as one group, which is the dense block's own earlier concat. The
                     concats that no conv reads any more are not built.
  enc_fix = None     r3 as is.

  R6ReUSE(model, time_form, freq_form, npu=False, enc_fix="tile")   spec [1, 2, T, F] -> out [1, 3, T, F] (as r1)
  R6Encoder(...)                                                     spec [1, 2, T, F] -> encoder out (micro graph)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import reuse_gpu as G
import reuse_gpu_r3 as G3

IM2COL_LINE = 1024 ** 3                    # kMaxIm2colBufferSizeMobile (conv.cc 74)
IM2COL_BUDGET = 1 * 203 * 323 * 256 * 9 * 4  # 604,283,904 B: the 1 s graph's node 137 (prepared on S26 and 11 Pro)


def im2col_bytes(c_in, h_out, w_out, kh, kw):
    """TFLite's im2col size for a float32 batch-1 conv (conv.cc 504-533)."""
    return c_in * h_out * w_out * kh * kw * 4


def _out_hw(h, w, kh, kw, padding, dilation):
    return h + 2 * padding[0] - dilation[0] * (kh - 1), w + 2 * padding[1] - dilation[1] * (kw - 1)


def conv_tiles(x, w, b, padding, dilation, budget=IM2COL_BUDGET):
    """(A) stride-1 conv2d of [1, C, H, W] over n equal output-row tiles (n = 1: the plain conv)."""
    _, C, H, W = x.shape
    kh, kw = w.shape[2], w.shape[3]
    Ho, Wo = _out_hw(H, W, kh, kw, padding, dilation)
    n = -(-im2col_bytes(C, Ho, Wo, kh, kw) // budget)
    if n <= 1:
        return F.conv2d(x, w, b, padding=padding, dilation=dilation)
    halo = dilation[0] * (kh - 1)
    xp = F.pad(x, (padding[1], padding[1], padding[0], padding[0]))
    q, r = divmod(Ho, n)
    outs, a = [], 0
    for i in range(n):
        e = a + q + (1 if i < r else 0)
        outs.append(F.conv2d(xp[:, :, a:e + halo], w, b, dilation=dilation))
        a = e
    return torch.cat(outs, dim=2)


def conv_tiles_crop(x, w, b, padding, dilation, budget=IM2COL_BUDGET):
    """(A') stride-1, size-preserving conv2d of [1, C, H, W] over n equal output-row tiles without a PAD of x: tile
    [a, e) = conv(x[a - p : e + p] clipped to [0, H), the conv's own padding)[rows a .. e)."""
    _, C, H, W = x.shape
    kh, kw = w.shape[2], w.shape[3]
    Ho, Wo = _out_hw(H, W, kh, kw, padding, dilation)
    assert Ho == H and dilation[0] * (kh - 1) == 2 * padding[0], "size-preserving convs only"
    n = -(-im2col_bytes(C, Ho, Wo, kh, kw) // budget)
    if n <= 1:
        return F.conv2d(x, w, b, padding=padding, dilation=dilation)
    p = padding[0]
    q, r = divmod(Ho, n)
    outs, a = [], 0
    for i in range(n):
        e = a + q + (1 if i < r else 0)
        s0, s1 = max(0, a - p), min(H, e + p)
        y = F.conv2d(x[:, :, s0:s1], w, b, padding=padding, dilation=dilation)
        outs.append(y[:, :, a - s0:e - s0])
        a = e
    return torch.cat(outs, dim=2)


def dense_block_r6(db, x, inorm, enc_fix, budget=IM2COL_BUDGET):
    """DenseBlock (depth 4, 3x3, dilation (2^i, 1)) with the chosen fix; enc_fix None = r3's _dense_block."""
    if enc_fix is None:
        return G3._dense_block(db, x, inorm)
    if enc_fix in ("tile", "tilecrop"):
        tiles = conv_tiles if enc_fix == "tile" else conv_tiles_crop
        skip = x
        for i in range(db.depth):
            conv, norm, act = db.dense_block[i][0], db.dense_block[i][1], db.dense_block[i][2]
            y = tiles(skip, conv.weight, conv.bias, conv.padding, conv.dilation, budget)
            y = G.prelu(inorm(y, norm), act.weight)
            skip = torch.cat([y, skip], dim=1)
        return y
    if enc_fix != "split":
        raise ValueError(enc_fix)
    ys = []
    S = {0: x}            # S[j] = the input of conv j = cat([y_{j-1}, ..., y_0, x]) (built only when read)

    def suffix(j):
        if j not in S:
            S[j] = torch.cat([ys[j - 1], suffix(j - 1)], dim=1)
        return S[j]

    C = x.shape[1]
    for i in range(db.depth):
        conv, norm, act = db.dense_block[i][0], db.dense_block[i][1], db.dense_block[i][2]
        kh, kw = conv.weight.shape[2], conv.weight.shape[3]
        Ho, Wo = _out_hw(x.shape[2], x.shape[3], kh, kw, conv.padding, conv.dilation)
        groups, lo, j = [], 0, i
        while j > 0 and im2col_bytes(C * (j + 1), Ho, Wo, kh, kw) > budget:
            groups.append((ys[j - 1], lo, lo + C))        # the newest remaining piece alone
            lo += C
            j -= 1
        groups.append((suffix(j), lo, lo + C * (j + 1)))   # the older pieces as the block's own concat
        acc = None
        for k, (t, a, e) in enumerate(groups):
            o = F.conv2d(t, conv.weight[:, a:e], conv.bias if k == 0 else None, padding=conv.padding,
                         dilation=conv.dilation)
            acc = o if acc is None else acc + o
        ys.append(G.prelu(inorm(acc, norm), act.weight))
    return ys[-1]


class R6ReUSE(G3.R3ReUSE):
    """r3's R3ReUSE with the encoder's dense block fixed (see the module docstring)."""

    def __init__(self, model, time_form="F1pq8", freq_form="F1pq8", npu=False, enc_fix="tile",
                 budget=IM2COL_BUDGET):
        super().__init__(model, time_form=time_form, freq_form=freq_form, npu=npu)
        self.enc_fix = enc_fix
        self.budget = budget

    def encoder(self, x):
        de = self.model.dense_encoder
        c1 = de.dense_conv_1
        y = G.conv2d(x, c1[0].weight, c1[0].bias)
        y = G.prelu(self.inorm(y, c1[1]), c1[2].weight)
        y = dense_block_r6(de.dense_block, y, self.inorm, self.enc_fix, self.budget)
        c2 = de.dense_conv_2
        y = G.conv2d(y, c2[0].weight, c2[0].bias, stride=c2[0].stride)
        return G.prelu(self.inorm(y, c2[1]), c2[2].weight)

    def forward(self, spec):
        # R3ReUSE.forward (npu=False) goes to GPUReUSE.forward, which calls self.encoder = the method above;
        # npu=True runs R3ReUSE's own forward, which also calls self.encoder.
        return super().forward(spec)


class R6Encoder(nn.Module):
    """Micro graph for the fix comparison: the generator's zero rows / columns + the dense encoder only."""

    def __init__(self, model, enc_fix, budget=IM2COL_BUDGET):
        super().__init__()
        self.model = model
        self.enc_fix = enc_fix
        self.budget = budget
        self.inorm = G.safe_instance_norm

    def forward(self, spec):
        return R6ReUSE.encoder(self, F.pad(spec, (0, 2, 0, 2)))
