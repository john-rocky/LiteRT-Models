"""r4 forms of the re-authored Timestral graph (timestral_graph.py is imported and left as r1 wrote it).

dec_only_last: the last block computes only what the head reads. The head takes out_norm of the decoder
tokens only, and in the last block (Timestral-209M schedule "time_variate": time attention first, then
variate attention, then the MLP) every sublayer after the time attention's key / value projections is
per-token. So the last block runs
  xn = norm(x) on all S tokens (keys / values need every token),
  q, gate, attention output and wO on the H decoder tokens only (keys / values from all S),
  variate attention and the MLP on the H decoder tokens only.
The decoder rows see the same operations on the same values as in the full block (RMSNorm is per token,
so norm(x)[:, L:] = norm(x[:, L:])), i.e. the same function; only the rows nobody reads are dropped.
"""

import torch
import torch.nn.functional as F

import timestral_graph as tg


def _time_attention_dec_only(attn, xn, xn_dec, bias_dec, cos, sin, cos_dec, sin_dec):
    """tg.TimeAttention.forward with the queries restricted to the decoder tokens."""
    r, s, _ = xn.shape
    h = xn_dec.shape[1]
    q = attn.q(xn_dec).view(r, h, attn.nh, attn.hd).permute(0, 2, 1, 3)
    k = attn.k(xn).view(r, s, attn.nh, attn.hd).permute(0, 2, 1, 3)
    v = attn.v(xn).view(r, s, attn.nh, attn.hd).permute(0, 2, 1, 3)
    q = q * cos_dec + tg._rotate_half(q) * sin_dec
    k = k * cos + tg._rotate_half(k) * sin
    w = torch.einsum("bhqd,bhkd->bhqk", q, k) + bias_dec
    if attn.attn_k:
        w = (w - w.amax(dim=-1, keepdim=True)) * (2.0 ** attn.attn_k)
    if attn.attn_clamp:
        w = torch.clamp(w, min=-3.0e4)
    p = torch.softmax(w, dim=-1)
    o = torch.einsum("bhqk,bhkd->bhqd", p, v).permute(0, 2, 1, 3).reshape(r, h, attn.d)
    return attn.o(o * torch.sigmoid(attn.g(xn_dec)))


class TimestralGraphR4(tg.TimestralGraph):
    """tg.TimestralGraph (same weights, same inputs / output) with the r4 options."""

    def __init__(self, model, context_patches, horizon_patches, dec_only_last=False, **kw):
        super().__init__(model, context_patches, horizon_patches, **kw)
        self.dec_only_last = bool(dec_only_last)
        if self.dec_only_last:
            last = self.blocks[-1]
            assert last.kinds and last.kinds[0], "the last block must start with its time attention"
            assert not any(last.kinds[1:]), "only one time attention in the last block"

    def _last_block(self, blk, x, bias, cos, sin):
        ll = self.L
        xd = x[:, ll:, :]
        for j, (is_time, norm, attn) in enumerate(zip(blk.kinds, blk.attn_norms, blk.attns)):
            if is_time:
                xn = norm(x)
                xd = xd + _time_attention_dec_only(attn, xn, xn[:, ll:, :], bias[:, :, ll:, :], cos, sin,
                                                   cos[:, :, ll:, :], sin[:, :, ll:, :])
            else:
                xd = xd + attn(norm(xd))
        return xd + blk.mlp(blk.mlp_norm(xd))

    def forward(self, patches, dec, bias, cos, sin):
        if not self.dec_only_last:
            return super().forward(patches, dec, bias, cos, sin)
        r = patches.shape[0]
        if self.resid_k:
            dec = dec * (2.0 ** -self.resid_k)
        x = torch.cat([self.embed(patches), dec], dim=1)
        for blk in self.blocks[:-1]:
            x = blk(x, bias, cos, sin)
        y = self.out_norm(self._last_block(self.blocks[-1], x, bias, cos, sin))
        raw = self.head(y).view(r, self.H, tg.PATCH, tg.N_Q)
        sp = F.relu(raw) + torch.log(1.0 + torch.exp(-torch.abs(raw)))
        return self.mono(torch.cat([sp, raw], dim=-1))


def count_last_block_macs(d_model=768, hidden=2048, s=96, h=32, variate=True):
    """Per-row multiply-accumulates of the last block, full vs decoder-only (projections + MLP + attention)."""
    proj = 5 * d_model * d_model                          # q, k, v, gate, o
    mlp = 2 * hidden * d_model + d_model * hidden         # gate, up, w3
    var = 3 * d_model * d_model if variate else 0          # v, gate, o of the variate attention
    att = lambda nq: 2 * nq * s * d_model                  # scores + weighted sum (12 heads x 64)
    full = s * (proj + mlp + var) + att(s)
    dec = s * 2 * d_model * d_model + h * (3 * d_model * d_model + mlp + var) + att(h)
    return full, dec
