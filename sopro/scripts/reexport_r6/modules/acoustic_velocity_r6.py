"""Exact velocity rewrite of the published acoustic graph for LiteRT GPU.

Only InputEmbedding's explicit speaker expand changes. Multiplication of an
all-one [1,1,T,1] matrix by [1,1,1,80] has a one-term reduction and replicates
the speaker exactly, preserving concatenate -> Linear accumulation order. All
learned parameters, source activations and public inputs remain unchanged.
"""
import torch
from torch import nn
from acoustic_graphs import AcousticVelocityGraph


class BroadcastFreeInputEmbedding(nn.Module):
    def __init__(self, original, frames):
        super().__init__()
        self.proj = original.proj
        self.pos = original.pos
        self.cond_mask_proj = original.cond_mask_proj
        self.register_buffer("replication_ones", torch.ones(1, 1, frames, 1))

    def forward(self, x_t, cond_mel, cond_mask, mu, spk):
        x_btc = x_t.transpose(1, 2).contiguous()
        cond_btc = cond_mel.transpose(1, 2).contiguous()
        cond_mask_btc = cond_mask.transpose(1, 2).contiguous().to(x_btc.dtype)
        mu_btc = mu.transpose(1, 2).contiguous()
        spk_btc = (self.replication_ones @ spk[:, None, None, :]).reshape(1, x_t.shape[-1], spk.shape[-1])
        h = self.proj(torch.cat([x_btc, cond_btc, mu_btc, spk_btc], dim=-1))
        h = h + self.cond_mask_proj(cond_mask_btc)
        return h + self.pos(h)


class AcousticVelocityGraphR6(AcousticVelocityGraph):
    def __init__(self, head, frames=2048):
        super().__init__(head, frames)
        self.input_embed = BroadcastFreeInputEmbedding(self.input_embed, frames)
