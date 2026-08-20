"""GPU-clean re-authoring for the SAM 3.1 tracker sub-graphs (stage 2).

SimpleRoPEAttention (memory attention, 4 layers x {self, cross}): the stock path is
head-split + interleaved real-RoPE on 5-D/8-D tensors + an in-place `k[:, :, :n] = rope(k)`
(SELECT) + SDPA. Re-authored exactly with <=4-D tensors:
  * the (pair, 2) interleave is undone by a 4-D reshape/transpose on q and k THEMSELVES
    (same within-head permutation on both -> q.k unchanged), then half-split rotation;
  * key RoPE only on the first `num_k_rope` keys (memory tokens; object pointers excluded)
    via slicing + concat instead of the in-place write;
  * `rope_k_repeat`: the 72x72 table repeated once per memory slot (constant buffer);
  * explicit matmul-softmax attention with optional query chunking (the cross attention
    has 5184 queries x (N*5184 + P*16) keys; chunking bounds the score tensor).
"""
import math
import types

NEG = -1e4

import torch
import torch.nn.functional as F


def _deinterleave(x):
    """(B, H, L, hd) with interleaved (re, im) pairs -> [re... | im...] per head, 4-D only."""
    B, H, L, hd = x.shape
    x = x.reshape(B * H, L, hd // 2, 2).transpose(2, 3)      # (B*H, L, 2, hd/2)
    return x.reshape(B, H, L, hd)


def _rope_half(x, cos, sin):
    """x (B,H,L,hd) de-interleaved; cos/sin (1,1,L,hd/2)."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


def simple_rope_attention_forward_4d(self, q, k, v, num_k_exclude_rope=0):
    """Drop-in for decoder.SimpleRoPEAttention.forward. q (B,Lq,C), k/v (B,Lk,C)."""
    B, Lq, C = q.shape
    Lk = k.shape[1]
    H = self.num_heads
    hd = C // H
    q = q.reshape(B, Lq, H, hd).transpose(1, 2)
    k = k.reshape(B, Lk, H, hd).transpose(1, 2)
    v = v.reshape(B, Lk, H, hd).transpose(1, 2)
    cos, sin = self.rope_cos, self.rope_sin                   # (1,1,Lq,hd/2)
    q = _rope_half(_deinterleave(q), cos, sin)
    n_rope = Lk - num_k_exclude_rope
    if n_rope == Lq:
        ck, sk = cos, sin
    else:
        # repeated table is a CONSTANT buffer built at patch time (a runtime .repeat()
        # lowers to an 8-D BROADCAST_TO)
        assert self.rope_k_repeat and self.rope_cos_k.shape[2] == n_rope, (self.rope_cos_k.shape, n_rope)
        ck, sk = self.rope_cos_k, self.rope_sin_k
    k_rope = _rope_half(_deinterleave(k[:, :, :n_rope]), ck, sk)
    if num_k_exclude_rope > 0:
        # excluded keys (object pointers) get the SAME de-interleave so q.k stays consistent
        k = torch.cat([k_rope, _deinterleave(k[:, :, n_rope:])], 2)
    else:
        k = k_rope
    q = q * (1.0 / math.sqrt(hd))
    kt = k.transpose(-2, -1)
    keep = getattr(self, "key_keep", None)          # (1,1,1,Lk) float, 1 = valid key, or None

    def sm(s):
        if keep is None:
            return torch.softmax(s, -1)
        # masked softmax in the form the Metal delegate executes correctly (gpu_patches)
        m = (s * keep + (1.0 - keep) * NEG).max(dim=-1, keepdim=True).values
        e = torch.exp((s - m) * keep) * keep
        return e / e.sum(dim=-1, keepdim=True)
    nch = getattr(self, "q_chunks", 1)
    if nch > 1 and Lq % nch == 0:
        Lc = Lq // nch
        outs = []
        for i in range(nch):
            s = torch.matmul(q[:, :, i * Lc:(i + 1) * Lc], kt)
            outs.append(torch.matmul(sm(s), v))
        o = torch.cat(outs, 2)
    else:
        o = torch.matmul(sm(torch.matmul(q, kt)), v)
    return o.transpose(1, 2).reshape(B, Lq, C)


def patch_memattn(trk, q_chunks=1, n_slots=7):
    from sam3.model.decoder import SimpleRoPEAttention
    n = 0
    for m in trk.transformer.modules():
        if isinstance(m, SimpleRoPEAttention):
            fr, fi = m.freqs_cis_real.detach().clone(), m.freqs_cis_imag.detach().clone()  # (L, hd/2)
            m.register_buffer("rope_cos", fr[None, None], persistent=False)
            m.register_buffer("rope_sin", fi[None, None], persistent=False)
            if m.rope_k_repeat:
                m.register_buffer("rope_cos_k", fr[None, None].repeat(1, 1, n_slots, 1), persistent=False)
                m.register_buffer("rope_sin_k", fi[None, None].repeat(1, 1, n_slots, 1), persistent=False)
            m.q_chunks = q_chunks
            m.forward = types.MethodType(simple_rope_attention_forward_4d, m)
            n += 1
    return n
