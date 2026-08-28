"""Fourth probe batch — LLM prefill: decoder-style block, one variable per arm.

Block = RMSNorm + GQA attention (16 q-heads / 4 kv-heads, hd=64, rank-4 layout,
baked RoPE tables, additive causal mask) + SwiGLU MLP (d_ff=2816). d=1024, L=2,
fp32 weights, same seed for every arm.

Sweep arms (only T varies):
  T in {1151,1152,1153, 1279,1280,1281, 1407,1408,1409, 1500,
        1535,1536,1537, 1663,1664,1665}
  Hypothesis under test: prefill T === 0 (mod 128) is fast on Hexagon, +/-1 and
  1500-style numbers are slow (generalizes the encoder T-shape curve to a
  decoder-style graph).

RMSNorm-flavor arms (T=1536 fixed, norm impl is the one variable;
naive == the sweep's t1536 arm):
  rmssafe1536 - SafeRMS scale-before-square, constant s=64   [rtmpose recipe]
  rmsmax1536  - max-norm SafeRMS, runtime s=amax|x|          [qwen3-emb recipe]
Both are exact-math identities of naive RMSNorm (fp accumulation differs only).
No builtin RMS_NORM exists in the LiteRT flatbuffer schema (checked 2.1.6 and
2.3.0.dev20260823: 210 builtins, none) — decomposition choice is the only lever.

  python probe_batch4.py pf1536 pfrmssafe1536 ...   # build named arms
  python probe_batch4.py verify                     # host equivalence check
"""
import sys
import torch
import torch.nn as nn

D, DFF, NH, NKV, HD, L = 1024, 2816, 16, 4, 64, 2
G = NH // NKV


class RMSNaive(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Parameter(torch.ones(D))

    def forward(self, x):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + 1e-6) * self.g


class RMSSafe(nn.Module):
    """Scale-before-square with constant s (Mali-fp16 overflow guard)."""
    S = 64.0

    def __init__(self):
        super().__init__()
        self.g = nn.Parameter(torch.ones(D))

    def forward(self, x):
        d = x * (1.0 / self.S)
        v = d.pow(2).mean(-1, keepdim=True) * (self.S * self.S)
        return x * torch.rsqrt(v + 1e-6) * self.g


class RMSMax(nn.Module):
    """Scale-before-square with runtime s = amax|x| (max-norm SafeRMS)."""

    def __init__(self):
        super().__init__()
        self.g = nn.Parameter(torch.ones(D))

    def forward(self, x):
        m = torch.clamp(x.abs().amax(-1, keepdim=True), min=1e-6)
        d = x / m
        v = d.pow(2).mean(-1, keepdim=True) * (m * m)
        return x * torch.rsqrt(v + 1e-6) * self.g


class Attn(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.q = nn.Linear(D, NH * HD)
        self.k = nn.Linear(D, NKV * HD)
        self.v = nn.Linear(D, NKV * HD)
        self.o = nn.Linear(NH * HD, D)
        pos = torch.arange(t, dtype=torch.float32)
        inv = 1.0 / (10000.0 ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
        ang = pos[:, None] * inv[None, :]                        # [T, HD/2]
        self.register_buffer('cos', torch.cat([ang.cos(), ang.cos()], -1)[None, None])
        self.register_buffer('sin', torch.cat([ang.sin(), ang.sin()], -1)[None, None])
        mask = torch.zeros(1, 1, t, t)
        mask.masked_fill_(torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1), -1e4)
        self.register_buffer('mask', mask)

    def rot(self, x):                                            # [1, h, T, HD]
        x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
        return x * self.cos + torch.cat([-x2, x1], -1) * self.sin

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, NH, HD).permute(0, 2, 1, 3)
        k = self.k(x).view(B, T, NKV, HD).permute(0, 2, 1, 3)
        v = self.v(x).view(B, T, NKV, HD).permute(0, 2, 1, 3)
        q, k = self.rot(q), self.rot(k)
        # kv-head repeat as rank-4 CONCATENATION (repeat_interleave/.repeat both
        # export high-rank BROADCAST_TO, which ML Drift refuses; head order is
        # probe-irrelevant)
        k = torch.cat([k] * G, dim=1)
        v = torch.cat([v] * G, dim=1)
        att = torch.matmul(q, k.transpose(-1, -2)) * (HD ** -0.5) + self.mask
        att = att.softmax(-1)
        o = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(B, T, NH * HD)
        return self.o(o)


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(D, DFF)
        self.up = nn.Linear(D, DFF)
        self.down = nn.Linear(DFF, D)

    def forward(self, x):
        h = self.gate(x)
        return self.down(h * torch.sigmoid(h) * self.up(x))      # SiLU gate


class Block(nn.Module):
    def __init__(self, t, rms_cls):
        super().__init__()
        self.n1, self.n2 = rms_cls(), rms_cls()
        self.attn, self.mlp = Attn(t), Mlp()

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class Probe(nn.Module):
    def __init__(self, t, rms_cls=RMSNaive):
        super().__init__()
        self.blocks = nn.ModuleList(Block(t, rms_cls) for _ in range(L))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


SWEEP_T = [1151, 1152, 1153, 1279, 1280, 1281, 1407, 1408, 1409, 1500,
           1535, 1536, 1537, 1663, 1664, 1665]
RMS = {'rmssafe1536': RMSSafe, 'rmsmax1536': RMSMax}


def build(tag):
    torch.manual_seed(0)
    if tag.startswith('pfrms'):
        t, rms_cls = 1536, RMS[tag[2:]]
    else:
        t, rms_cls = int(tag[2:]), RMSNaive
    m = Probe(t, rms_cls).eval()
    import litert_torch
    litert_torch.convert(m, (torch.randn(1, t, D),)).export(f'probe_{tag}.tflite')
    print('wrote', f'probe_{tag}.tflite', flush=True)


def verify():
    """Host equivalence: same weights, three norm flavors, one input."""
    torch.manual_seed(0)
    ref = Probe(256, RMSNaive).eval()
    x = torch.randn(1, 256, D) * 3.0
    with torch.no_grad():
        y0 = ref(x)
        for name, cls in [('safe', RMSSafe), ('max', RMSMax)]:
            torch.manual_seed(0)
            m = Probe(256, cls).eval()
            with torch.no_grad():
                y = m(x)
            d = (y - y0).abs().max().item()
            c = torch.corrcoef(torch.stack([y.flatten(), y0.flatten()]))[0, 1].item()
            print(f'rms_{name} vs naive: max_abs_diff={d:.3e} corr={c:.9f}')


if __name__ == '__main__':
    if sys.argv[1:] == ['verify']:
        verify()
    else:
        for tag in sys.argv[1:]:
            build(tag)
