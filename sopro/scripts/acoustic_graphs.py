"""Exact static acoustic boundaries of sopro 2.2.0 (N=512, T=2048).

The DiT's *source* GELU uses approximate='tanh'; we preserve that exact
source function. No erf GELU is changed into an approximation by this port.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from sopro.nn.layers import apply_rotary, rotary_cos_sin, sinusoidal_time_embedding

N_MAX = 512
T_MAX = 2048


class AcousticConditionGraph(nn.Module):
    input_names = ('semantic_tokens', 'token_mask', 'frame_to_token')
    output_names = ('mu',)

    def __init__(self, head):
        super().__init__()
        self.semantic_token_emb = copy.deepcopy(head.semantic_token_emb)
        self.semantic_prelook = copy.deepcopy(head.semantic_prelook)
        self.semantic_upsampler = copy.deepcopy(head.semantic_upsampler)
        self.mu_proj = copy.deepcopy(head.mu_proj)

    def forward(self, semantic_tokens, token_mask, frame_to_token):
        latents = self.semantic_token_emb(semantic_tokens).transpose(1, 2) * token_mask
        # Conv1 looks right, so zeros beyond the real token count reproduce
        # F.pad(..., (0, lookahead)). Conv2 and the upsampler are causal.
        pre = self.semantic_prelook(latents)
        rep = torch.index_select(pre, 2, frame_to_token)
        up = self.semantic_upsampler
        h = F.silu(up.in_proj(rep))
        h = F.silu(up.mix(h))
        mu = self.mu_proj(rep + up.out_proj(h))
        signal = (latents.abs().amax(dim=(1, 2)) > 0).to(mu.dtype)
        return mu * signal[:, None, None]


class AcousticVelocityGraph(nn.Module):
    input_names = ('x', 't', 'mu', 'cond_vec', 'cond_mel', 'cond_mask', 'key_bias')
    output_names = ('velocity',)

    def __init__(self, head, frames=T_MAX):
        super().__init__()
        self.frames = frames
        self.time_embed_dim = head.time_embed_dim
        for name in ('time_mlp', 'spk_proj', 'input_embed', 'blocks', 'out_norm', 'out_proj'):
            setattr(self, name, copy.deepcopy(getattr(head, name)))
        cos, sin = rotary_cos_sin(torch.arange(frames), head.dim_head, torch.float32)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)

    def forward(self, x, t, mu, cond_vec, cond_mel, cond_mask, key_bias):
        cond = F.normalize(cond_vec, dim=-1)
        signal = (cond_vec.abs().amax(dim=-1) > 0).to(x.dtype)
        spk = self.spk_proj(cond) * signal[:, None]
        emb = sinusoidal_time_embedding(t.to(torch.float32), self.time_embed_dim)
        emb = self.time_mlp(emb.to(self.time_mlp[0].weight.dtype)).to(x.dtype)
        h = self.input_embed(x, cond_mel, cond_mask, mu, spk)
        # Positional convolutions look only left. Pad queries never influence
        # valid queries once pad keys are masked in every attention layer.
        for block in self.blocks:
            normed, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.attn_norm(h, emb)
            attn = block.attn
            b, n, _ = normed.shape
            q = attn.to_q(normed).view(b, n, attn.heads, attn.dim_head).transpose(1, 2)
            k = attn.to_k(normed).view(b, n, attn.heads, attn.dim_head).transpose(1, 2)
            v = attn.to_v(normed).view(b, n, attn.heads, attn.dim_head).transpose(1, 2)
            q = apply_rotary(q, self.cos, self.sin)
            k = apply_rotary(k, self.cos, self.sin)
            scores = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(attn.dim_head))
            y = torch.softmax(scores + key_bias, dim=-1) @ v
            y = y.transpose(1, 2).reshape(b, n, attn.heads * attn.dim_head)
            h = h + gate_msa[:, None, :] * attn.to_out(y)
            ff = block.ff_norm(h) * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
            h = h + gate_mlp[:, None, :] * block.ff(ff)
        return self.out_proj(self.out_norm(h, emb)).transpose(1, 2)
