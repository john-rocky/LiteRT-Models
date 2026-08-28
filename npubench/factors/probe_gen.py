"""Synthetic transformer-encoder probes: one variable = sequence length.

Graph style matches the zoo's NPU-winning ViT conversions: rank-3 (H,T,64)
attention batch-matmuls, torch LayerNorm (lowers to MEAN/SQUARED_DIFFERENCE),
sigmoid-GELU, fp32 weights (no DEQUANTIZE).
"""
import sys
import torch
import torch.nn as nn


D, H, L = 384, 6, 4
HD = D // H


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.ln2 = nn.LayerNorm(D)
        self.fc1 = nn.Linear(D, 4 * D)
        self.fc2 = nn.Linear(4 * D, D)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, T, 3, H, HD).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0].reshape(H, T, HD), qkv[1].reshape(H, T, HD), qkv[2].reshape(H, T, HD)
        att = torch.matmul(q, k.transpose(-1, -2)) * (HD ** -0.5)
        att = att.softmax(-1)
        o = torch.matmul(att, v)                      # (H, T, HD)
        o = o.reshape(1, H, T, HD).permute(0, 2, 1, 3).reshape(B, T, D)
        x = x + self.proj(o)
        h = self.ln2(x)
        h = self.fc1(h)
        h = h * torch.sigmoid(1.702 * h)              # sigmoid-GELU
        return x + self.fc2(h)


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(L))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def build(t, out):
    torch.manual_seed(0)
    m = Probe().eval()
    x = torch.randn(1, t, D)
    import litert_torch
    litert_torch.convert(m, (x,)).export(out)
    print('wrote', out)


if __name__ == '__main__':
    for t in [int(a) for a in sys.argv[1:]]:
        build(t, f'probe_t{t}.tflite')
