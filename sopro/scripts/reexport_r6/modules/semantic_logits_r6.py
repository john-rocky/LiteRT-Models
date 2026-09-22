"""Exact semantic digit logits; FSQ argmax/indexing belongs to the host.

The two fixed interpolation gathers are onehot fully connected selections.
Selection and blending remain separate, preserving source float arithmetic.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class SemanticLogitsGraphR6(nn.Module):
    input_names = ('semantic_mel',)
    output_names = ('digit_logits',)

    def __init__(self, semantic):
        super().__init__()
        self.semantic = copy.deepcopy(semantic)
        self.semantic.frontend = nn.Identity()
        src = ((torch.arange(235, dtype=torch.float32) + 0.5) *
               (torch.tensor(500., dtype=torch.float32) /
                torch.tensor(235., dtype=torch.float32)) - 0.5).clamp(0., 499.)
        left = src.floor().long()
        right = (left + 1).clamp_max(499)
        left_onehot = F.one_hot(left, 500).to(torch.float32).contiguous().clone()
        right_onehot = F.one_hot(right, 500).to(torch.float32).contiguous().clone()
        self.register_buffer('interp_left_onehot', left_onehot)
        self.register_buffer('interp_right_onehot', right_onehot)
        self.register_buffer('interp_weight', (src - left.float()).view(1, -1, 1).contiguous().clone())

    def forward(self, semantic_mel):
        sem = self.semantic
        x = F.gelu(sem.conv1(semantic_mel))
        x = F.gelu(sem.conv2(x)).permute(0, 2, 1)
        x = (x + sem.pos_emb[:x.shape[1]].to(x.dtype).unsqueeze(0))[:, :500]
        for layer in sem.layers:
            x = layer(x)
        x = sem.final_norm(x)
        source = x.transpose(1, 2)
        left = F.linear(source, self.interp_left_onehot).transpose(1, 2)
        right = F.linear(source, self.interp_right_onehot).transpose(1, 2)
        x = left * (1.0 - self.interp_weight) + right * self.interp_weight
        return sem.digit_head(sem.pre_head_norm(x))
