"""Exact static reference semantic encoder (sopro 2.2.0, ten-second bucket).

Unlike the author's ReferenceGraph, crop 501 convolution positions to the
package's 500 positions before attention. Spectral processing stays on host.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


def tokens_from_logits(logits, levels, bases):
    digits = torch.stack([part.argmax(dim=-1).to(torch.int32)
                          for part in torch.split(logits, levels, dim=-1)], dim=-1)
    return (digits * bases.view(1, 1, -1)).sum(dim=-1, dtype=torch.int32)


def native_digit_logits(semantic, semantic_mel):
    """Installed encode() after its spectral frontend, with actual bucket sizes."""
    x = F.gelu(semantic.conv1(semantic_mel))
    x = F.gelu(semantic.conv2(x)).permute(0, 2, 1)
    x = (x + semantic.pos_emb[:x.shape[1]].to(x.dtype).unsqueeze(0))[:, :500]
    for layer in semantic.layers:
        x = layer(x)
    x = semantic.final_norm(x)
    x = semantic._interpolate(x, 235)
    return semantic.digit_head(semantic.pre_head_norm(x))


class SemanticGraph(nn.Module):
    """One int32 token output, [1,80,1002] → [1,235]."""
    def __init__(self, semantic):
        super().__init__()
        self.semantic = copy.deepcopy(semantic)
        self.semantic.frontend = nn.Identity()
        # Compute this exactly as SemanticEncoder._interpolate in fp32. Baking
        # the fixed indexes/weights changes no interpolation arithmetic.
        src = ((torch.arange(235, dtype=torch.float32) + 0.5) *
               (torch.tensor(500., dtype=torch.float32) /
                torch.tensor(235., dtype=torch.float32)) - 0.5).clamp(0., 499.)
        left = src.floor().long()
        self.register_buffer('interp_left', left)
        self.register_buffer('interp_right', (left + 1).clamp_max(499))
        self.register_buffer('interp_weight', (src - left.float()).view(1, -1, 1))
        self.register_buffer('int32_bases', semantic._bases.to(torch.int32).clone())

    def digit_logits(self, semantic_mel):
        sem = self.semantic
        x = F.gelu(sem.conv1(semantic_mel))
        x = F.gelu(sem.conv2(x)).permute(0, 2, 1)
        x = (x + sem.pos_emb[:x.shape[1]].to(x.dtype).unsqueeze(0))[:, :500]
        for layer in sem.layers:
            x = layer(x)
        x = sem.final_norm(x)
        x = (x.index_select(1, self.interp_left) * (1.0 - self.interp_weight) +
             x.index_select(1, self.interp_right) * self.interp_weight)
        return sem.digit_head(sem.pre_head_norm(x))

    def forward(self, semantic_mel):
        logits = self.digit_logits(semantic_mel)
        return tokens_from_logits(logits, self.semantic.levels, self.int32_bases)


class SemanticDiagnosticGraph(SemanticGraph):
    """Same computation with digit logits exposed solely for numeric evidence."""
    def forward(self, semantic_mel):
        logits = self.digit_logits(semantic_mel)
        return tokens_from_logits(logits, self.semantic.levels, self.int32_bases), logits
