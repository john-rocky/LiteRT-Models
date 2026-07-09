"""Final prep split that clears the Mali C19 wall (a MUL right after the
embed FC mis-assigns the token dim N to the BATCH axis -> "Unable to parse bc
coord for BATCH axis" compile fail). The pad-token substitution is moved to the
host, so every graph is either a bare FC (embed) or a refiner stack -- both
proven to compile on the Pixel Mali OpenCL delegate.

Graphs:
  z_embx : x_embed            img[1,256,64]   -> x_raw[1,256,3840]
  z_embc : cap_embed          cap[1,32,2560]  -> c_raw[1,32,3840]
  z_refx : noise_refiner      x[1,256,3840]   -> x_ref[1,256,3840]   (modulated)
  z_refc : context_refiner    c[1,32,3840]    -> c_ref[1,32,3840]    (no adaLN)
Host between embed and refiner:  x = x_raw*(1-xpad) + x_pad_token*xpad
Host after both refiners:        u = cat([x_ref, c_ref], dim=1) -> [1,288,3840]

Verifies the full host+GPU composition == the monolithic DeployDiT reference,
and writes the pad-token vectors as .bin for the on-device host mask.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, "/Users/majimadaisuke/Downloads/depthanything-android")
sys.path.insert(0, ".")
from build_zimage import ExportDiT, DIM
from full_dit import GpuBlock, GpuFinal

CHUNK = 5


class EmbX(nn.Module):
    def __init__(self, rm):
        super().__init__()
        self.x_embed = rm.all_x_embedder["2-1"]

    def forward(self, img):
        return self.x_embed(img)


class EmbC(nn.Module):
    def __init__(self, rm):
        super().__init__()
        self.cap_embed = rm.cap_embedder

    def forward(self, cap):
        return self.cap_embed(cap)


class RefX(nn.Module):
    def __init__(self, rm):
        super().__init__()
        self.blocks = nn.ModuleList([GpuBlock(b) for b in rm.noise_refiner])

    def forward(self, x, xc, xs, adaln):
        for b in self.blocks:
            x = b(x, xc, xs, adaln)
        return x


class RefC(nn.Module):
    def __init__(self, rm):
        super().__init__()
        self.blocks = nn.ModuleList([GpuBlock(b) for b in rm.context_refiner])

    def forward(self, c, cc, cs):
        for b in self.blocks:
            c = b(c, cc, cs)
        return c


def q_export(model, ins, name):
    import litert_torch
    from litert_torch.generative.quantize import quant_recipes as qrs
    from litert_torch.generative.quantize.quant_attrs import Dtype, Granularity
    cfg = qrs.full_dynamic_recipe(weight_dtype=Dtype.INT8, granularity=Granularity.CHANNELWISE)
    out = f"{name}.tflite"
    litert_torch.convert(model, ins, quant_config=cfg).export(out)
    print(f"[{name}] {os.path.getsize(out)/1e6:.1f} MB")


def save(t, name):
    t.detach().cpu().numpy().astype("<f4").tofile(f"chunk_bins/{name}.bin")


def main():
    from diffusers import ZImageTransformer2DModel
    from deploy_dit import DeployDiT
    rm = ZImageTransformer2DModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", subfolder="transformer", torch_dtype=torch.float32).eval()

    full = DeployDiT(rm).eval()
    torch.manual_seed(3)
    img = torch.randn(1, 256, 64); cap = torch.randn(1, 32, 2560); adaln = torch.randn(1, 256)
    xc = torch.randn(1, 1, 256, 64); xs = torch.randn(1, 1, 256, 64)
    cc = torch.randn(1, 1, 32, 64); cs = torch.randn(1, 1, 32, 64)
    xpad = torch.zeros(1, 256, 1); cpad = torch.zeros(1, 32, 1)
    uc = torch.cat([xc, cc], dim=2); us = torch.cat([xs, cs], dim=2)
    with torch.no_grad():
        ref = full(img, cap, adaln, xc, xs, cc, cs, xpad, cpad).numpy()

    embx, embc = EmbX(rm).eval(), EmbC(rm).eval()
    refx, refc = RefX(rm).eval(), RefC(rm).eval()
    mains = [ExportDiT(list(rm.layers[i:i + CHUNK])).eval() for i in range(0, 30, CHUNK)]
    final = GpuFinal(rm.all_final_layer["2-1"]).eval()
    xpt = rm.x_pad_token.detach()      # [1,1,3840]
    cpt = rm.cap_pad_token.detach()
    with torch.no_grad():
        x = embx(img); x = x * (1 - xpad) + xpt * xpad; x = refx(x, xc, xs, adaln)
        c = embc(cap); c = c * (1 - cpad) + cpt * cpad; c = refc(c, cc, cs)
        u = torch.cat([x, c], dim=1)
        for m in mains:
            u = m(u, uc, us, adaln)
        out = final(u, adaln).numpy()
    corr = np.corrcoef(ref.flatten(), out.flatten())[0, 1]
    print(f"[compose] emb+hostmask+refiner(x,c) + host-cat + 6x5main + final "
          f"vs monolithic: corr {corr:.6f}")

    if "--export" in sys.argv:
        save(xpt, "xpt"); save(cpt, "cpt")
        q_export(embx, (img,), "z_embx")
        q_export(embc, (cap,), "z_embc")
        q_export(refx, (x, xc, xs, adaln), "z_refx")
        q_export(refc, (c, cc, cs), "z_refc")


if __name__ == "__main__":
    main()
