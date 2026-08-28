"""Style probes: isolate GELU flavor and LayerNorm implementation at fixed T.

Base = probe_gen's winner-style block (native LN -> MEAN/SQDIF, sigmoid-GELU).
Variants change exactly one thing:
  tanh   - tanh-GELU (dinov2's TANH ops) instead of sigmoid-GELU
  manln  - manual LayerNorm via SUM/MUL chains (dinov2's SUM+MUL pattern)
  dino   - both + T=1025 (full dinov2 style reproduction)
"""
import math
import sys
import torch
import torch.nn as nn

D, H, L = 384, 6, 4
HD = D // H
C = math.sqrt(2.0 / math.pi)


def gelu_sig(x):
    return x * torch.sigmoid(1.702 * x)


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(C * (x + 0.044715 * x * x * x)))


def gelu_tanh_via_sigmoid(x):
    # identical formula structure to gelu_tanh, tanh(y) rewritten as 2*sigmoid(2y)-1
    y = C * (x + 0.044715 * x * x * x)
    return 0.5 * x * (2.0 * torch.sigmoid(2.0 * y))


class ManualLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Parameter(torch.ones(D))
        self.b = nn.Parameter(torch.zeros(D))

    def forward(self, x):
        mu = x.sum(-1, keepdim=True) * (1.0 / D)
        xc = x - mu
        var = (xc * xc).sum(-1, keepdim=True) * (1.0 / D)
        return xc * torch.rsqrt(var + 1e-6) * self.g + self.b


class Block(nn.Module):
    def __init__(self, gelu, ln_cls):
        super().__init__()
        self.gelu = gelu
        self.ln1 = ln_cls()
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.ln2 = ln_cls()
        self.fc1 = nn.Linear(D, 4 * D)
        self.fc2 = nn.Linear(4 * D, D)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, T, 3, H, HD).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[i].reshape(H, T, HD) for i in range(3))
        att = (torch.matmul(q, k.transpose(-1, -2)) * (HD ** -0.5)).softmax(-1)
        o = torch.matmul(att, v).reshape(1, H, T, HD).permute(0, 2, 1, 3).reshape(B, T, D)
        x = x + self.proj(o)
        h = self.gelu(self.fc1(self.ln2(x)))
        return x + self.fc2(h)


class Probe(nn.Module):
    def __init__(self, gelu, ln_cls):
        super().__init__()
        self.blocks = nn.ModuleList(Block(gelu, ln_cls) for _ in range(L))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def build(tag, t, gelu, ln_cls):
    torch.manual_seed(0)
    m = Probe(gelu, ln_cls).eval()
    x = torch.randn(1, t, D)
    import litert_torch
    litert_torch.convert(m, (x,)).export(f'probe_{tag}.tflite')
    print('wrote', f'probe_{tag}.tflite')


if __name__ == '__main__':
    std_ln = lambda: nn.LayerNorm(D)
    for tag in sys.argv[1:]:
        if tag == 'tanh1024':
            build(tag, 1024, gelu_tanh, std_ln)
        elif tag == 'manln1024':
            build(tag, 1024, gelu_sig, ManualLN)
        elif tag == 'dino1025':
            build(tag, 1025, gelu_tanh, ManualLN)
        elif tag == 'tanhsig1024':
            build(tag, 1024, gelu_tanh_via_sigmoid, std_ln)