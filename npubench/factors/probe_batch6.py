"""Sixth probe batch — (a) fusion-breaker op family, (b) op-granularity curve.

(a) Fusion breakers. Phase-3 established that the tanh-GELU polynomial chain
costs 5.75x with TANH but only 2.36x with LOGISTIC substituted into the same
chain — i.e. TANH breaks a QNN elementwise-chain fusion that LOGISTIC keeps.
These arms slot one op X into the identical chain
    act(y) = 0.5 * y * (1 + X(c * (y + 0.044715 * y^3)))
inside the phase-3 probe (D=384, H=6, L=4, T=1024, native LN, builtin softmax,
fp32). One arm = one X. c is dropped to 1e-4 on exp/pow arms so activations
stay bounded (constant value changes nothing structurally). rsqrt/sqrt need a
positive domain, so X(z) = op(z*z + 1) there — two extra elementwise ops,
noted in the row.

  fx_tanh  - TANH            (in-batch re-anchor of the 53.8 ms arm)
  fx_sig   - LOGISTIC        (clean sigma(z)-in-slot arm)
  fx_exp   - EXP             (c=1e-4)
  fx_abs   - ABS
  fx_max   - MAXIMUM(z, 0.01) (max-vs-0 lowers to RELU and gets fused away —
             that build is kept as fx_relu, the "no op in the slot" control)
  fx_relu  - RELU(z), fused into the preceding MUL by the exporter
  fx_rsqrt - RSQRT(z*z+1)
  fx_sqrt  - SQRT(z*z+1)
  fx_pow   - POW(z, 3)       (c=1e-4; opscan the file — may lower to MULs)
  fx_erf   - torch.erf       (no ERF builtin in TFLite; see what it lowers to)

(b) Granularity: same-cost serial 1x1 Conv+ReLU chains, N = op-pair count.
Per conv: [1,32,32,32] @ 32x32 kernel-1 = 2.1 MFLOP. Time-vs-N slope = per-op
overhead; zipformer (3085 ops, 12.6 GF) is the real-model anchor.

  gr10 gr30 gr100 gr300 gr1000 gr3000
  grmono - one Conv2d(1792,1792,1) on 32x32 = 6.58 GF in a single fat op,
           the equal-FLOPs monolithic reference for gr3000 (6.29 GF).

  python probe_batch6.py fx_tanh gr100 ...
"""
import sys
import torch
import torch.nn as nn

D, H, L, T = 384, 6, 4, 1024
HD = D // H


class Block(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.fc1 = nn.Linear(D, 4 * D)
        self.fc2 = nn.Linear(4 * D, D)

    def forward(self, x):
        B, Tn, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, Tn, 3, H, HD).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[i].reshape(H, Tn, HD) for i in range(3))
        att = (torch.matmul(q, k.transpose(-1, -2)) * (HD ** -0.5)).softmax(-1)
        o = torch.matmul(att, v).reshape(1, H, Tn, HD).permute(0, 2, 1, 3).reshape(B, Tn, D)
        x = x + self.proj(o)
        return x + self.fc2(self.act(self.fc1(self.ln2(x))))


class FxProbe(nn.Module):
    def __init__(self, f, c):
        super().__init__()
        self.register_buffer('thr', torch.full((), 0.01))

        def act(y):
            z = c * (y + 0.044715 * y * y * y)
            return 0.5 * y * (1.0 + f(self, z))

        self.blocks = nn.ModuleList(Block(act) for _ in range(L))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


C0 = 0.7978845608
FX = {
    'fx_tanh':  (lambda s, z: torch.tanh(z), C0),
    'fx_sig':   (lambda s, z: torch.sigmoid(z), C0),
    'fx_exp':   (lambda s, z: torch.exp(z), 1e-4),
    'fx_abs':   (lambda s, z: torch.abs(z), C0),
    'fx_max':   (lambda s, z: torch.maximum(z, s.thr), C0),
    'fx_relu':  (lambda s, z: torch.relu(z), C0),
    'fx_rsqrt': (lambda s, z: torch.rsqrt(z * z + 1.0), C0),
    'fx_sqrt':  (lambda s, z: torch.sqrt(z * z + 1.0), C0),
    'fx_pow':   (lambda s, z: torch.pow(z, 3.0), 1e-4),
    'fx_erf':   (lambda s, z: torch.erf(z), C0),
}

GR_CH, GR_RES = 32, 32


class GrChain(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(GR_CH, GR_CH, 1) for _ in range(n))

    def forward(self, x):
        for c in self.convs:
            x = torch.relu(c(x))
        return x


class GrMono(nn.Module):
    CH = 1792

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(self.CH, self.CH, 1)

    def forward(self, x):
        return self.conv(x)


def build(tag):
    torch.manual_seed(0)
    import litert_torch
    if tag in FX:
        f, c = FX[tag]
        m = FxProbe(f, c).eval()
        x = torch.randn(1, T, D)
    elif tag == 'grmono':
        m = GrMono().eval()
        x = torch.randn(1, GrMono.CH, GR_RES, GR_RES)
    elif tag.startswith('gr'):
        m = GrChain(int(tag[2:])).eval()
        x = torch.randn(1, GR_CH, GR_RES, GR_RES)
    else:
        raise SystemExit(f'unknown tag {tag}')
    litert_torch.convert(m, (x,)).export(f'probe_{tag}.tflite')
    print('wrote', f'probe_{tag}.tflite', flush=True)


if __name__ == '__main__':
    for tag in sys.argv[1:]:
        build(tag)
