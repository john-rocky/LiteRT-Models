"""Exact style-prefix attention with an explicit leading singleton batch axis.

The published torch source already uses rank-four operands. LiteRT-Torch lowers
these to rank-three BATCH_MATMUL, so export additionally applies the lossless
leading-singleton flatbuffer normalization in flatbuffer_rewrites.py.
"""
import copy
import torch
from torch import nn
from ar_graphs import rms


class StylePrefixGraphR6(nn.Module):
    input_names = ('style_embeddings',)
    output_names = ('prefix_vectors',)

    def __init__(self, model):
        super().__init__()
        self.style = copy.deepcopy(model.style_prefix)

    def forward(self, style_embeddings):
        s = self.style
        kv = rms(style_embeddings, s.kv_norm)
        queries = s.queries.unsqueeze(0)
        q = s.q_proj(queries).reshape(1, 8, 8, 64).transpose(1, 2)
        k = s.k_proj(kv).reshape(1, 160, 8, 64).transpose(1, 2)
        v = s.v_proj(kv).reshape(1, 160, 8, 64).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / 8.0
        y = torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2).reshape(1, 8, 512)
        return rms(queries + s.out_proj(y), s.out_norm)
