"""Exact float-onehot replacement for both acoustic condition gather sites.

A token onehot replaces the unsupported EMBEDDING_LOOKUP; a frame onehot replaces
runtime GATHER_ND. Each row must contain exactly one 1 and all other entries 0.
The rank-four frame matmul preserves the source gather without rank-three BMM.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class AcousticConditionGraphR6(nn.Module):
    input_names = ('semantic_onehot', 'token_mask', 'frame_onehot')
    output_names = ('mu',)

    def __init__(self, head):
        super().__init__()
        weight = head.semantic_token_emb.weight.detach().transpose(0, 1).contiguous().clone()
        self.semantic_projection = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        self.semantic_projection.weight = nn.Parameter(weight)
        for name in ('semantic_prelook', 'semantic_upsampler', 'mu_proj'):
            setattr(self, name, copy.deepcopy(getattr(head, name)))

    def forward(self, semantic_onehot, token_mask, frame_onehot):
        latents = self.semantic_projection(semantic_onehot).transpose(1, 2) * token_mask
        pre = self.semantic_prelook(latents)
        # [1,1,C,N] @ [1,1,N,T] -> [1,C,T], with one selected source per frame.
        rep = (pre.unsqueeze(1) @ frame_onehot.transpose(0, 1).unsqueeze(0).unsqueeze(0)).squeeze(1)
        up = self.semantic_upsampler
        h = F.silu(up.in_proj(rep))
        h = F.silu(up.mix(h))
        mu = self.mu_proj(rep + up.out_proj(h))
        signal = (latents.abs().amax(dim=(1, 2)) > 0).to(mu.dtype)
        return mu * signal[:, None, None]
