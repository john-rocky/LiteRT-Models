"""Z-Image VAE (AutoencoderKL) decoder → LiteRT INTEGER int8 deploy graph.

Fixed 256px (latent [1,16,32,32] -> image [1,3,256,256]). 3D/4D-aware ManualGroupNorm
(mid-attn group_norm on 3D). Runs ONCE per image, so int8 error does not compound.
Verifies decode corr vs the fp32 diffusers VAE on a real latent.
"""
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/Users/majimadaisuke/Downloads/depthanything-android")
from litert_gpu_toolkit.patches import patch_gelu, patch_swish, patch_interpolate
from probe_vae import ManualGroupNormND, patch_groupnorm_nd
from ai_edge_litert.interpreter import Interpreter

LAT = 32  # 256px


def main():
    from diffusers import AutoencoderKL
    import litert_torch
    from litert_torch.generative.quantize import quant_recipes as qrs
    from litert_torch.generative.quantize.quant_attrs import Dtype, Granularity

    vae = AutoencoderKL.from_pretrained("Tongyi-MAI/Z-Image-Turbo", subfolder="vae").eval()

    class Dec(nn.Module):
        def __init__(s):
            super().__init__(); s.vae = vae

        def forward(s, z):
            return s.vae.decode(z, return_dict=False)[0]

    m = Dec().eval()
    z = torch.randn(1, 16, LAT, LAT)
    patch_gelu(m); patch_swish(m); patch_interpolate(); patch_groupnorm_nd(m)
    with torch.no_grad():
        ref = m(z).numpy()
    print(f"[forward] latent {tuple(z.shape)} -> {ref.shape}")

    cfg = qrs.full_dynamic_recipe(weight_dtype=Dtype.INT8, granularity=Granularity.CHANNELWISE)
    out = "zvae_int8_256.tflite"
    litert_torch.convert(m, (z,), quant_config=cfg).export(out)
    import os
    print(f"[int8] {out}  {os.path.getsize(out)/1e6:.0f} MB")

    it = Interpreter(model_path=out); it.allocate_tensors()
    d = it.get_input_details()[0]; it.set_tensor(d["index"], z.numpy().astype("<f4")); it.invoke()
    q = it.get_tensor(it.get_output_details()[0]["index"])
    corr = np.corrcoef(ref.flatten(), q.flatten())[0, 1]
    print(f"[decode] int8 vs fp32: corr {corr:.5f}, mean|diff| {np.abs(ref-q).mean():.4e}")


if __name__ == "__main__":
    main()
