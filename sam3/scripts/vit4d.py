"""Exact 4-D re-authoring of the SAM3 ViT-L/14 trunk for LiteRT GPU (ML Drift / Metal).

The stock ``sam3.model.vitdet`` graph lowers, but with 5-D/6-D/8-D tensors
(qkv head split, window partition, tiled abs-pos broadcast, interleaved
real-RoPE) which (a) ML Drift rejects (>4-D) and (b) already mis-lower on the
2026-08 litert-torch (raw export corr 0.607 vs PyTorch, see precheck log). Every
rewrite here is numerically exact (weight permutations and reshapes only):

  * abs-pos: the tiled 24x24 -> 72x72 embedding is a constant -> baked buffer.
  * qkv: slice the fused projection along channels, heads via a 4-D reshape +
    transpose (no 5-D).
  * RoPE: the pairwise (2p, 2p+1) interleave is baked into the Q/K rows of the
    qkv weight (same permutation on q and k => identical q.k), so the rotation
    is a half-split  [x1*cos - x2*sin | x1*sin + x2*cos]  on 4-D tensors.
    (Recipe: [[reference_interleaved_rope_bake]].)
  * attention: explicit matmul + softmax (no SDPA composite).
  * window partition: reshape/transpose/reshape/transpose, all <=4-D, and EXACT
    (the SAM2 order-swap trick is not valid here because RoPE is position
    dependent inside the window; the extra transpose restores the layout).

Call ``patch_vit_4d(vit)`` once on the loaded trunk. Parity is asserted by the
caller (precheck_sam3.py --vit4d).
"""
import math
import types

import torch
import torch.nn.functional as F

from sam3.model import vitdet


class SafeLayerNorm(torch.nn.Module):
    """fp16-robust LayerNorm (zoo recipe, deep-ViT row): the residual stream of this ViT-L
    reaches |x|~300 (blocks 16-31), so sum((x-mean)^2) overflows fp16 (65504) inside the
    GPU delegate even for an fp32 graph. Scale before the square; LN is scale invariant and
    eps*scale^2 keeps it exact."""

    def __init__(self, ln, scale=1.0 / 32):
        super().__init__()
        self.scale = scale
        self.eps = ln.eps * scale * scale
        self.register_buffer("w", ln.weight.detach().clone())
        self.register_buffer("b", ln.bias.detach().clone())

    def forward(self, x):
        xs = x * self.scale
        mu = xs.mean(-1, keepdim=True)
        d = xs - mu
        var = (d * d).mean(-1, keepdim=True)
        return d * torch.rsqrt(var + self.eps) * self.w + self.b


def _deinterleave_rows(w, num_heads, head_dim, b=None):
    """Permute output rows of a (C_out, ...) weight so that within each head the
    even channels come first, then the odd ones."""
    C = num_heads * head_dim
    idx = torch.arange(C).view(num_heads, head_dim // 2, 2).permute(0, 2, 1).reshape(C)
    return w[idx], (b[idx] if b is not None else None)


def window_partition_4d(x, ws):
    """[B,H,W,C] -> [B*nH*nW, ws, ws, C], exact, <=4-D ops only. H, W multiples of ws."""
    B, H, W, C = x.shape
    nH, nW = H // ws, W // ws
    x = x.reshape(B * nH, ws, W, C)            # row bands
    x = x.transpose(1, 2)                      # [B*nH, W, ws, C]
    x = x.reshape(B * nH * nW, ws, ws, C)      # [.., ws_w, ws_h, C]  (window transposed)
    return x.transpose(1, 2).contiguous(), (H, W)  # [.., ws_h, ws_w, C]  exact layout


def window_unpartition_4d(win, ws, pad_hw, hw):
    H, W = hw
    C = win.shape[-1]
    nH, nW = H // ws, W // ws
    B = win.shape[0] // (nH * nW)
    x = win.transpose(1, 2)                    # [B*nH*nW, ws_w, ws_h, C]
    x = x.reshape(B * nH, W, ws, C)
    x = x.transpose(1, 2)                      # [B*nH, ws, W, C]
    return x.reshape(B, H, W, C)


def _attn_forward_4d(self, x):
    B, H, W, C = x.shape
    L = H * W
    nH, hd = self.num_heads, C // self.num_heads
    qkv = self.qkv(x.reshape(B, L, C))                       # (B, L, 3C)
    q = qkv[:, :, :C].reshape(B, L, nH, hd).transpose(1, 2)  # (B, nH, L, hd)
    k = qkv[:, :, C:2 * C].reshape(B, L, nH, hd).transpose(1, 2)
    v = qkv[:, :, 2 * C:].reshape(B, L, nH, hd).transpose(1, 2)
    if self.use_rope:
        cos, sin = self.rope_cos, self.rope_sin               # (1, 1, L, hd/2)
        h = hd // 2
        q1, q2 = q[..., :h], q[..., h:]
        q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], -1)
        k1, k2 = k[..., :h], k[..., h:]
        k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], -1)
    q = q * (1.0 / math.sqrt(hd))
    kt = k.transpose(-2, -1)
    nch = getattr(self, "q_chunks", 1)
    if nch > 1 and L % nch == 0:
        # exact query chunking: bounds the (nH, Lc, L) score tensor for phone GPUs
        Lc = L // nch
        outs = []
        for i in range(nch):
            s = torch.matmul(q[:, :, i * Lc:(i + 1) * Lc], kt)
            outs.append(torch.matmul(torch.softmax(s, dim=-1), v))
        o = torch.cat(outs, 2)                               # (B, nH, L, hd)
    else:
        attn = torch.softmax(torch.matmul(q, kt), dim=-1)
        o = torch.matmul(attn, v)                            # (B, nH, L, hd)
    o = o.transpose(1, 2).reshape(B, H, W, C)
    return self.proj(o)


def _vit_forward_4d(self, x):
    x = self.patch_embed(x)                                  # (B, 72, 72, C)
    x = x + self.pos_baked
    x = self.ln_pre(x)
    for blk in self.blocks:
        x = blk(x)
    x = self.ln_post(x)
    return [x.permute(0, 3, 1, 2)]


@torch.no_grad()
def patch_vit_4d(vit, safe_ln=True, global_chunks=1):
    assert not vit.retain_cls_token and vit.pos_embed is not None
    hw = (vit.patch_embed.proj.weight.new_zeros(1),)  # noqa: F841 (doc only)
    size = int(vit.pos_embed.shape[1] ** 0.5) if not vit.pretrain_use_cls_token else \
        int((vit.pos_embed.shape[1] - 1) ** 0.5)
    img_hw = vit.blocks[-1].attn.input_size if vit.blocks[-1].window_size == 0 else None
    assert img_hw is not None, "expects the last block to be global (72x72)"
    pos = vitdet.get_abs_pos(vit.pos_embed, vit.pretrain_use_cls_token, tuple(img_hw),
                             vit.retain_cls_token, tiling=vit.tile_abs_pos)
    vit.register_buffer("pos_baked", pos.detach().clone(), persistent=False)
    del size
    if safe_ln:
        vit.ln_pre = SafeLayerNorm(vit.ln_pre)
        for blk in vit.blocks:
            blk.norm1 = SafeLayerNorm(blk.norm1)
            blk.norm2 = SafeLayerNorm(blk.norm2)
    for blk in vit.blocks:
        a = blk.attn
        assert a.use_rope and a.use_rope_real and not a.use_rel_pos and not a.use_ve_rope
        nH = a.num_heads
        C = a.qkv.in_features
        hd = C // nH
        w = a.qkv.weight.data
        b = a.qkv.bias.data if a.qkv.bias is not None else None
        wq, bq = _deinterleave_rows(w[:C], nH, hd, b[:C] if b is not None else None)
        wk, bk = _deinterleave_rows(w[C:2 * C], nH, hd, b[C:2 * C] if b is not None else None)
        a.qkv.weight.data = torch.cat([wq, wk, w[2 * C:]], 0)
        if b is not None:
            a.qkv.bias.data = torch.cat([bq, bk, b[2 * C:]], 0)
        # freqs_cis_real/imag: (L, hd/2) -> (1,1,L,hd/2)
        a.register_buffer("rope_cos", a.freqs_cis_real.detach().clone()[None, None],
                          persistent=False)
        a.register_buffer("rope_sin", a.freqs_cis_imag.detach().clone()[None, None],
                          persistent=False)
        a.q_chunks = global_chunks if blk.window_size == 0 else 1
        a.forward = types.MethodType(_attn_forward_4d, a)
    vitdet.window_partition = window_partition_4d
    vitdet.window_unpartition = window_unpartition_4d
    vit.forward = types.MethodType(_vit_forward_4d, vit)
    return vit
