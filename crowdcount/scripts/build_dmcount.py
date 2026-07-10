"""Converts DM-Count (UCF-QNRF) to a fully GPU-compatible LiteRT model.

Expects the official repo cloned next to this script (or set DMCOUNT_SRC):
    git clone https://github.com/cvlab-stonybrook/DM-Count.git dmcount-src
The MIT-licensed pretrained weights ship in the repo via git-lfs
(pretrained_models/model_qnrf.pth).

The only graph change is exact: `F.upsample_bilinear` (align_corners=True,
banned as RESIZE_BILINEAR on the ML Drift GPU delegate) is a linear operator,
so it is re-authored as two constant-matrix multiplies with the constant on
the RHS (lowers to FULLY_CONNECTED, which the delegate accepts). Output is
bit-identical to PyTorch up to fp32 rounding.
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC = os.environ.get("DMCOUNT_SRC", os.path.join(os.path.dirname(__file__), "dmcount-src"))
sys.path.insert(0, SRC)

SIZE = 512  # fixed input; the density map comes out at SIZE/8


def bilinear_matrix(n_in, n_out):
    """Returns the [n_out, n_in] align_corners=True bilinear resize matrix."""
    matrix = np.zeros((n_out, n_in), np.float32)
    scale = (n_in - 1) / (n_out - 1)
    for i in range(n_out):
        pos = i * scale
        lo = int(np.floor(pos))
        hi = min(lo + 1, n_in - 1)
        frac = np.float32(pos - lo)
        matrix[i, lo] += 1 - frac
        matrix[i, hi] += frac
    return matrix


def exact_upsample(x, size=None, scale_factor=None):
    """Exact align_corners=True 2x bilinear upsample as constant-RHS matmuls."""
    h, w = int(x.shape[2]), int(x.shape[3])
    a_h_t = torch.from_numpy(bilinear_matrix(h, 2 * h)).t().contiguous()  # [h, 2h]
    a_w_t = torch.from_numpy(bilinear_matrix(w, 2 * w)).t().contiguous()  # [w, 2w]
    y = x.transpose(2, 3) @ a_h_t   # [1,C,w,2h]
    y = y.transpose(2, 3) @ a_w_t   # [1,C,2h,w] @ [w,2w] -> [1,C,2h,2w]
    return y


F.upsample_bilinear = exact_upsample

from models import VGG, make_layers, cfg  # noqa: E402  (needs SRC on sys.path)


class DensityHead(nn.Module):
    """Wraps DM-Count to emit only the raw density map (count = map sum)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        density, _ = self.net(x)   # [1,1,SIZE/8,SIZE/8]
        return density


def main():
    net = VGG(make_layers(cfg["E"]))
    weights = torch.load(
        os.path.join(SRC, "pretrained_models", "model_qnrf.pth"), map_location="cpu")
    net.load_state_dict(weights, strict=True)
    model = DensityHead(net).eval()

    dummy = torch.rand(1, 3, SIZE, SIZE)
    with torch.no_grad():
        out = model(dummy)
    print("out:", tuple(out.shape), "count", float(out.sum()))
    np.save("ref_in.npy", dummy.numpy())
    np.save("ref_out.npy", out.numpy())

    import litert_torch
    litert_torch.convert(model, (dummy,)).export("dmcount.tflite")
    print("saved %.1f MB" % (os.path.getsize("dmcount.tflite") / 1e6))


if __name__ == "__main__":
    main()
