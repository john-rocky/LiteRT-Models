"""Matched-compute probes: GELU flavor on the LiteRT Hexagon NPU accelerator.

Three variants of the same 4-block transformer (d=384, h=6, T=1024, fp32,
identical seed/weights). One variable each:

  base     - sigmoid-GELU  x*sigmoid(1.702x)
  tanh     - hand-written tanh-GELU  0.5x(1+tanh(0.79788(x+0.044715x^3)))
  tanhsig  - the SAME polynomial chain with tanh(y) rewritten as 2*sigmoid(2y)-1
             (exact identity; isolates the TANH op from the chain around it)

Measured on Galaxy S26 (SM8850, Hexagon v81), LiteRT CompiledModel 2.2.0,
on-device JIT, one accelerator per process, N=50 medians, thermal status NONE:

  base     9.36 ms
  tanh    53.81 ms   (5.75x)
  tanhsig 22.11 ms   (2.36x -- so the chain costs ~2.4x and TANH-in-chain ~2.4x more)

All three compile fully onto the NPU in a single partition. The same swaps do
not move the Adreno GPU. Real-model confirmation (one variable, official
weights, builtin GELU op instead of the tanh chain):
  DINOv2-S      85.9 -> 41.9 ms   (features corr 0.999992)
  TIPSv2-B14    326.9 -> 142.0 ms (outputs corr 0.99998+)

Env: litert-torch 0.9.3, torch 2.x. Usage: python gelu_probe_repro.py base tanh tanhsig
"""
import math
import sys

import torch
import torch.nn as nn

D, H, L, T = 384, 6, 4, 1024
HD = D // H
C = math.sqrt(2.0 / math.pi)


def gelu_sig(x):
    return x * torch.sigmoid(1.702 * x)


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(C * (x + 0.044715 * x * x * x)))


def gelu_tanh_via_sigmoid(x):
    y = C * (x + 0.044715 * x * x * x)
    return 0.5 * x * (2.0 * torch.sigmoid(2.0 * y))


class Block(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.fc1, self.fc2 = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)

    def forward(self, x):
        B, N, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, N, 3, H, HD).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[i].reshape(H, N, HD) for i in range(3))
        att = (torch.matmul(q, k.transpose(-1, -2)) * (HD ** -0.5)).softmax(-1)
        o = torch.matmul(att, v).reshape(1, H, N, HD).permute(0, 2, 1, 3).reshape(B, N, D)
        x = x + self.proj(o)
        return x + self.fc2(self.act(self.fc1(self.ln2(x))))


ACTS = {'base': gelu_sig, 'tanh': gelu_tanh, 'tanhsig': gelu_tanh_via_sigmoid}

if __name__ == '__main__':
    import litert_torch
    for tag in sys.argv[1:]:
        torch.manual_seed(0)
        m = nn.Sequential(*[Block(ACTS[tag]) for _ in range(L)]).eval()
        litert_torch.convert(m, (torch.randn(1, T, D),)).export(f'probe_{tag}.tflite')
        print('wrote', f'probe_{tag}.tflite')
