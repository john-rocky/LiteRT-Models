"""Exact static AR re-authoring; all tensors have rank at most four.

The fp32 formulas match sopro.nn.ar. For mathematical padding proofs they
also preserve fp64, unlike the package's intentional float() casts.
"""
import copy
import math
import torch
from torch import nn
from sopro.nn.layers import rotary_cos_sin


def rms(x, norm):
    return x * torch.rsqrt((x*x).mean(dim=-1, keepdim=True) + norm.eps) * norm.weight


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x*cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1)*sin


class StylePrefixGraph(nn.Module):
    input_names = ('style_embeddings',)
    output_names = ('prefix_vectors',)

    def __init__(self, model):
        super().__init__()
        self.style = copy.deepcopy(model.style_prefix)

    def forward(self, style_embeddings):
        s = self.style
        kv = rms(style_embeddings, s.kv_norm)
        queries = s.queries.unsqueeze(0)
        q = s.q_proj(queries).view(1, 8, 8, 64).transpose(1, 2)
        k = s.k_proj(kv).view(1, 160, 8, 64).transpose(1, 2)
        v = s.v_proj(kv).view(1, 160, 8, 64).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / 8.0
        y = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(1, 8, 512)
        return rms(queries + s.out_proj(y), s.out_norm)


class ARBase(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ar = copy.deepcopy(model.ar_prior)

    def project(self, layer, x, cos, sin):
        a = layer.attn
        h = rms(x, layer.attn_norm)
        b, t, _ = h.shape
        q = a.q_proj(h).reshape(b, t, 8, 64).transpose(1, 2)
        k = a.k_proj(h).reshape(b, t, 8, 64).transpose(1, 2)
        v = a.v_proj(h).reshape(b, t, 8, 64).transpose(1, 2)
        return rotate(rms(q, a.q_norm), cos, sin), rotate(rms(k, a.k_norm), cos, sin), v

    def finish_layer(self, layer, x, q, k, v, bias):
        scores = (q @ k.transpose(-2, -1)) / 8.0 + bias
        y = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(x.shape)
        x = x + layer.attn_scale(layer.attn.out_proj(y))
        return x + layer.ffn_scale(layer.ffn(rms(x, layer.ffn_norm)))

    def logits(self, x):
        return self.ar.token_head(rms(x, self.ar.out_norm))[:, 0]


class ARPrefillGraph(ARBase):
    input_names = ('prefix_embeddings', 'attention_bias', 'last_index')
    output_names = ('logits', 'k', 'v')

    def __init__(self, model):
        super().__init__(model)
        cos, sin = rotary_cos_sin(torch.arange(256), 64, torch.float32)
        self.register_buffer('cos', cos[None, None].contiguous())
        self.register_buffer('sin', sin[None, None].contiguous())

    def forward(self, prefix_embeddings, attention_bias, last_index):
        x, keys, values = prefix_embeddings, [], []
        n = x.shape[1]
        cos, sin = self.cos[:, :, :n], self.sin[:, :, :n]
        for layer in self.ar.temporal.layers:
            q, k, v = self.project(layer, x, cos, sin)
            x = self.finish_layer(layer, x, q, k, v, attention_bias)
            keys.append(k)
            values.append(v)
        row = torch.index_select(x, 1, last_index.to(torch.int64))
        return self.logits(row), torch.cat(keys, dim=1), torch.cat(values, dim=1)


class ARStepGraph(ARBase):
    input_names = ('token_embedding', 'cos', 'sin', 'attention_bias', 'pk', 'pv')
    output_names = ('logits', 'new_k', 'new_v')

    def forward(self, token_embedding, cos, sin, attention_bias, pk, pv):
        # Bias is zero on 0..p (INCLUDING p) and -10000 beyond p. Its
        # right edge identifies p without an additional integer contract input.
        valid = 1.0 + attention_bias / 10000.0
        successor = torch.cat((valid[..., 1:], torch.zeros_like(valid[..., :1])), dim=-1)
        current = (valid - successor).transpose(-1, -2)
        keep = 1.0-current
        x, keys, values = token_embedding, [], []
        for i, layer in enumerate(self.ar.temporal.layers):
            q, k, v = self.project(layer, x, cos, sin)
            all_k = pk[:, i*8:(i+1)*8] * keep + k * current
            all_v = pv[:, i*8:(i+1)*8] * keep + v * current
            x = self.finish_layer(layer, x, q, all_k, all_v, attention_bias)
            keys.append(k)
            values.append(v)
        return self.logits(x), torch.cat(keys, dim=1), torch.cat(values, dim=1)
