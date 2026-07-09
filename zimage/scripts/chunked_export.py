"""Split the S3-DiT into <1GB int8 chunks that run sequentially (peak RAM = one
chunk, fits Pixel 8a; each loads+compiles fully-GPU in ~16s per device test).

Chunks (unified hidden [1,288,3840] passes between them):
  prep : x_embed + noise_refiner + cap_embed + context_refiner + cat -> unified
  m0..m5 : 5 main layers each (30 total) -> unified   (== ExportDiT, device-proven)
  final : GpuFinal -> patches [1,288,64]
Verifies the sequential composition == the monolithic DeployDiT output.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, "/Users/majimadaisuke/Downloads/depthanything-android")
sys.path.insert(0, ".")
from build_zimage import ExportDiT, DIM
from full_dit import GpuBlock, GpuFinal
from check_ops import check_ops

N = 288
CHUNK = 5  # main layers per chunk (device sweet spot: 5L=16s; 10L blows up)


class Prep(nn.Module):
    """embedders + refiners + cat -> unified hidden (with pad-token masks)."""
    def __init__(self, rm):
        super().__init__()
        self.x_embed = rm.all_x_embedder["2-1"]
        self.cap_embed = rm.cap_embedder
        self.noise_refiner = nn.ModuleList([GpuBlock(b) for b in rm.noise_refiner])
        self.context_refiner = nn.ModuleList([GpuBlock(b) for b in rm.context_refiner])
        self.register_buffer("x_pad_token", rm.x_pad_token.detach().clone())
        self.register_buffer("cap_pad_token", rm.cap_pad_token.detach().clone())

    def forward(self, img, cap, adaln, xc, xs, cc, cs, xpad, cpad):
        x = self.x_embed(img)
        x = x * (1 - xpad) + self.x_pad_token * xpad
        for b in self.noise_refiner:
            x = b(x, xc, xs, adaln)
        c = self.cap_embed(cap)
        c = c * (1 - cpad) + self.cap_pad_token * cpad
        for b in self.context_refiner:
            c = b(c, cc, cs)
        return torch.cat([x, c], dim=1)


def q_export(model, ins, name):
    import litert_torch
    from litert_torch.generative.quantize import quant_recipes as qrs
    from litert_torch.generative.quantize.quant_attrs import Dtype, Granularity
    cfg = qrs.full_dynamic_recipe(weight_dtype=Dtype.INT8, granularity=Granularity.CHANNELWISE)
    out = f"{name}.tflite"
    litert_torch.convert(model, ins, quant_config=cfg).export(out)
    print(f"[{name}] {os.path.getsize(out)/1e6:.0f} MB")
    return out


def main():
    from diffusers import ZImageTransformer2DModel
    from deploy_dit import DeployDiT
    rm = ZImageTransformer2DModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", subfolder="transformer", torch_dtype=torch.float32).eval()

    # reference: full DeployDiT (fp32) on shared inputs
    full = DeployDiT(rm).eval()
    torch.manual_seed(3)
    img = torch.randn(1, 256, 64); cap = torch.randn(1, 32, 2560); adaln = torch.randn(1, 256)
    xc = torch.randn(1, 1, 256, 64); xs = torch.randn(1, 1, 256, 64)
    cc = torch.randn(1, 1, 32, 64); cs = torch.randn(1, 1, 32, 64)
    xpad = torch.zeros(1, 256, 1); cpad = torch.zeros(1, 32, 1)
    uc = torch.cat([xc, cc], dim=2); us = torch.cat([xs, cs], dim=2)
    with torch.no_grad():
        ref = full(img, cap, adaln, xc, xs, cc, cs, xpad, cpad).numpy()

    # build + torch-run chunks sequentially, verify == ref
    prep = Prep(rm).eval()
    mains = [ExportDiT(list(rm.layers[i:i + CHUNK])).eval() for i in range(0, 30, CHUNK)]
    final = GpuFinal(rm.all_final_layer["2-1"]).eval()
    with torch.no_grad():
        u = prep(img, cap, adaln, xc, xs, cc, cs, xpad, cpad)
        for m in mains:
            u = m(u, uc, us, adaln)
        out = final(u, adaln).numpy()
    corr = np.corrcoef(ref.flatten(), out.flatten())[0, 1]
    print(f"[compose] chunks-sequential vs monolithic DeployDiT: corr {corr:.6f} "
          f"(prep + {len(mains)}x{CHUNK}main + final)")

    if "--export" in sys.argv:
        h = torch.randn(1, N, DIM)
        q_export(prep, (img, cap, adaln, xc, xs, cc, cs, xpad, cpad), "zc_prep")
        for i, m in enumerate(mains):
            q_export(m, (h, uc, us, adaln), f"zc_main{i}")
        q_export(final, (h, adaln), "zc_final")


if __name__ == "__main__":
    main()
