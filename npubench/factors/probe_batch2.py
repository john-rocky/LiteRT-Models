"""Second probe batch: more mathematically-equivalent rewrites, one variable each.

  rank4_1024   - attention in rank-4 [1,H,T,d] layout (vit4d style, separate q/k/v
                 Linears). No rank-5 tensor -> also compiles on ML Drift (GPU data!).
  t1536        - sequence 1536 = 12*128 (the whisper pad-target candidate)
  mlptanh1024  - plain tanh activation in the MLP (no polynomial chain)
  mlptsig1024  - same, tanh(x) rewritten as 2*sigmoid(2x)-1 (exact identity)
  smaxman1024  - softmax decomposed to max/sub/exp/sum/div (vs builtin SOFTMAX)
  safeln1024   - the zoo's SafeLayerNorm (pre-square scale s=64) vs native LN
"""
import sys
import torch
import torch.nn as nn

D, H, L = 384, 6, 4
HD = D // H
LN_S = 64.0


def gelu_sig(x):
    return x * torch.sigmoid(1.702 * x)


class SafeLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Parameter(torch.ones(D))
        self.b = nn.Parameter(torch.zeros(D))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        d = x - mu
        var = (d * (1.0 / LN_S)).pow(2).mean(-1, keepdim=True) * (LN_S * LN_S)
        return d * torch.rsqrt(var + 1e-6) * self.g + self.b


def softmax_manual(x):
    m = x.max(-1, keepdim=True).values
    e = torch.exp((x - m) * 1.0)
    s = e.sum(-1, keepdim=True)
    return e * torch.reciprocal(s + 1e-12)


class Block(nn.Module):
    def __init__(self, ln_cls=None, act=gelu_sig, smax=None, rank4=False):
        super().__init__()
        ln_cls = ln_cls or (lambda: nn.LayerNorm(D))
        self.act, self.rank4 = act, rank4
        self.smax = smax or (lambda a: a.softmax(-1))
        self.ln1, self.ln2 = ln_cls(), ln_cls()
        if rank4:
            self.q = nn.Linear(D, D)
            self.k = nn.Linear(D, D)
            self.v = nn.Linear(D, D)
        else:
            self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.fc1 = nn.Linear(D, 4 * D)
        self.fc2 = nn.Linear(4 * D, D)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.ln1(x)
        if self.rank4:
            def sp(t):
                return t.view(B, T, H, HD).permute(0, 2, 1, 3)   # [1,H,T,d]
            q, k, v = sp(self.q(h)), sp(self.k(h)), sp(self.v(h))
            att = self.smax(torch.matmul(q, k.transpose(-1, -2)) * (HD ** -0.5))
            o = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(B, T, D)
        else:
            qkv = self.qkv(h).reshape(B, T, 3, H, HD).permute(2, 0, 3, 1, 4)
            q, k, v = (qkv[i].reshape(H, T, HD) for i in range(3))
            att = self.smax(torch.matmul(q, k.transpose(-1, -2)) * (HD ** -0.5))
            o = torch.matmul(att, v).reshape(1, H, T, HD).permute(0, 2, 1, 3).reshape(B, T, D)
        x = x + self.proj(o)
        h = self.act(self.fc1(self.ln2(x)))
        return x + self.fc2(h)


class Probe(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.blocks = nn.ModuleList(Block(**kw) for _ in range(L))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def build(tag, t, **kw):
    torch.manual_seed(0)
    m = Probe(**kw).eval()
    import litert_torch
    litert_torch.convert(m, (torch.randn(1, t, D),)).export(f'probe_{tag}.tflite')
    print('wrote', f'probe_{tag}.tflite')


VARIANTS = {
    'rank4_1024': dict(t=1024, rank4=True),
    't1536': dict(t=1536),
    'mlptanh1024': dict(t=1024, act=torch.tanh),
    'mlptsig1024': dict(t=1024, act=lambda x: 2.0 * torch.sigmoid(2.0 * x) - 1.0),
    'smaxman1024': dict(t=1024, smax=softmax_manual),
    'safeln1024': dict(t=1024, ln_cls=SafeLN),
}

if __name__ == '__main__':
    for tag in sys.argv[1:]:
        kw = dict(VARIANTS[tag])
        t = kw.pop('t')
        build(tag, t, **kw)
