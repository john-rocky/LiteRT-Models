"""Z-Image VAE (AutoencoderKL flux-dev) decoder op-compat probe.

AutoencoderKL's mid-block attention applies group_norm to a 3D [B, HW, C]->[B, C, HW]
tensor, so the toolkit's 4D-only ManualGroupNorm is replaced here with a 3D/4D-aware
version. The decoder upsamples with interpolate+conv (no ConvTranspose), so the
conv-transpose auto-patch is skipped.
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/Users/majimadaisuke/Downloads/depthanything-android")
from litert_gpu_toolkit.patches import patch_gelu, patch_swish, patch_interpolate
from check_ops import check_ops


class ManualGroupNormND(nn.Module):
    """GPU-friendly GroupNorm accepting 3D [B,C,L] or 4D [B,C,H,W] input."""

    def __init__(self, gn):
        super().__init__()
        self.num_groups = gn.num_groups
        self.num_channels = gn.num_channels
        self.eps = gn.eps
        self.weight = nn.Parameter(gn.weight.clone()) if gn.weight is not None else None
        self.bias = nn.Parameter(gn.bias.clone()) if gn.bias is not None else None

    def forward(self, x):
        shape = x.shape
        if x.dim() == 3:
            b, c, l = shape
            x4 = x.reshape(b, c, l, 1)
        else:
            x4 = x
        b, c = x4.shape[0], x4.shape[1]
        g = self.num_groups
        xg = x4.reshape(b * g, c // g, x4.shape[2], x4.shape[3])
        mean = xg.mean(dim=(1, 2, 3), keepdim=True)
        var = ((xg - mean) * (xg - mean)).mean(dim=(1, 2, 3), keepdim=True)
        xn = (xg - mean) * torch.rsqrt(var + self.eps)
        xn = xn.reshape(b, c, x4.shape[2], x4.shape[3])
        if self.weight is not None:
            xn = xn * self.weight.reshape(1, c, 1, 1) + self.bias.reshape(1, c, 1, 1)
        return xn.reshape(shape)


def patch_groupnorm_nd(model):
    count = 0
    for _, module in list(model.named_modules()):
        for cn, child in list(module.named_children()):
            if isinstance(child, nn.GroupNorm):
                setattr(module, cn, ManualGroupNormND(child))
                count += 1
    print(f"[patch] {count} GroupNorm -> ManualGroupNormND")


def main(latent_hw=32, out="zvae_dec_probe.tflite"):
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained("Tongyi-MAI/Z-Image-Turbo", subfolder="vae").eval()

    class DecWrap(nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, z):
            return self.vae.decode(z, return_dict=False)[0]

    model = DecWrap(vae).eval()
    z = torch.randn(1, 16, latent_hw, latent_hw)

    patch_gelu(model)
    patch_swish(model)
    patch_interpolate()
    patch_groupnorm_nd(model)

    with torch.no_grad():
        ref = model(z)
    print(f"[forward] latent {tuple(z.shape)} -> image {tuple(ref.shape)}"
          f" range [{ref.min():.3f},{ref.max():.3f}]")

    import litert_torch
    print("[convert] ...")
    result = litert_torch.convert(model, (z,))
    result.export(out)
    import os
    print(f"[convert] saved {out} ({os.path.getsize(out)/1e6:.1f} MB)")
    check_ops(out)


if __name__ == "__main__":
    main()
