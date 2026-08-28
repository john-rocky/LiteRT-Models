"""Third probe batch — conv-family equivalent rewrites.

  ps    - conv stack + PixelShuffle x2 (lowers to DEPTH_TO_SPACE)
  zs    - identical weights, PixelShuffle -> zero-stuff + Conv2d (EDSR recipe, exact)
  rb    - conv stack with two bilinear x2 upsamples, align_corners=True (RESIZE_BILINEAR)
  rbmm  - identical weights, upsample as two constant-matrix matmuls (tipsv2 recipe, exact)
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CH, RES = 48, 160


class ZeroStuffUp(nn.Module):
    """PixelShuffle(2) as zero-stuff + conv, weights fixed to the shuffle permutation."""
    def __init__(self, cin, res):
        super().__init__()
        r = 2
        cout = cin // (r * r)
        w = torch.zeros(cin, cout, r, r)
        for c in range(cout):
            for p in range(r):
                for q in range(r):
                    w[c * r * r + p * r + q, c, p, q] = 1.0
        wf = w.flip(2).flip(3).permute(1, 0, 2, 3).contiguous()
        self.register_buffer('w', wf)
        m = np.zeros((res * r, res * r), np.float32)
        m[::r, ::r] = 1.0
        self.register_buffer('mask', torch.from_numpy(m)[None, None])
        self.r, self.k, self.res = r, r, res

    def forward(self, x):
        xn = F.interpolate(x, size=(self.res * self.r, self.res * self.r), mode='nearest') * self.mask
        y = F.conv2d(xn, self.w, padding=self.k - 1)
        return y[:, :, 0:self.res * self.r, 0:self.res * self.r]


def up_matrix(n):
    m = np.zeros((2 * n, n), np.float32)
    for i in range(2 * n):
        s = i * (n - 1) / (2 * n - 1)
        i0 = int(np.floor(s))
        i1 = min(i0 + 1, n - 1)
        w = s - i0
        m[i, i0] += 1.0 - w
        m[i, i1] += w
    return torch.from_numpy(m)


class SRProbe(nn.Module):
    def __init__(self, zerostuff=False):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(CH, CH, 3, padding=1) for _ in range(8))
        if zerostuff:
            self.up1 = ZeroStuffUp(CH, RES)
            self.up2 = ZeroStuffUp(CH, RES * 2)
        else:
            self.up1 = nn.PixelShuffle(2)
            self.up2 = nn.PixelShuffle(2)
        self.mid = nn.Conv2d(CH // 4, CH, 3, padding=1)
        self.out = nn.Conv2d(CH // 4, 3, 3, padding=1)

    def forward(self, x):
        for c in self.convs:
            x = torch.relu(c(x))
        x = self.mid(self.up1(x))
        return self.out(self.up2(x))


class UpProbe(nn.Module):
    def __init__(self, matmul=False):
        super().__init__()
        self.matmul = matmul
        self.c1 = nn.ModuleList(nn.Conv2d(64, 64, 3, padding=1) for _ in range(3))
        self.c2 = nn.ModuleList(nn.Conv2d(64, 64, 3, padding=1) for _ in range(3))
        self.c3 = nn.Conv2d(64, 3, 3, padding=1)
        self.register_buffer('m1', up_matrix(96))
        self.register_buffer('m2', up_matrix(192))

    def up(self, x, m):
        if self.matmul:
            return torch.matmul(torch.matmul(m, x), m.t())
        n = x.shape[-1] * 2
        return F.interpolate(x, size=(n, n), mode='bilinear', align_corners=True)

    def forward(self, x):
        for c in self.c1:
            x = torch.relu(c(x))
        x = self.up(x, self.m1)
        for c in self.c2:
            x = torch.relu(c(x))
        x = self.up(x, self.m2)
        return self.c3(x)


def build(tag):
    torch.manual_seed(0)
    import litert_torch
    if tag in ('ps', 'zs'):
        m = SRProbe(zerostuff=(tag == 'zs')).eval()
        x = torch.randn(1, CH, RES, RES)
    else:
        m = UpProbe(matmul=(tag == 'rbmm')).eval()
        x = torch.randn(1, 64, 96, 96)
    litert_torch.convert(m, (x,)).export(f'probe_{tag}.tflite')
    print('wrote', f'probe_{tag}.tflite')


if __name__ == '__main__':
    for tag in sys.argv[1:]:
        build(tag)
