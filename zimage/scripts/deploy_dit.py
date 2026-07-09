"""Deploy DiT graph: the full GPU-clean transformer over host-prepped fixed-shape
inputs (patchify / pos-ids / RoPE cos-sin / adaln / unpatchify stay on host).

Convert to tflite + int4-quantize to measure the REAL full-model on-device size
(vs the earlier extrapolation) and confirm op-compat at full depth.

Graph inputs (fixed for a chosen resolution + cap length):
    img_tokens [1, n_img, 64]       patchified latent (patch_dim = in_ch*2*2)
    cap_feats  [1, n_cap, 2560]     text-encoder output (padded)
    adaln      [1, 256]             host-computed t_embedder output
    x_cos/x_sin   [1,1,n_img,64]    RoPE for image tokens (noise_refiner)
    cap_cos/cap_sin [1,1,n_cap,64]  RoPE for caption tokens (context_refiner)
Output: unified patches [1, n_img+n_cap, 64]  (host unpatchifies the image slice)
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/Users/majimadaisuke/Downloads/depthanything-android")
from check_ops import check_ops
from full_dit import GpuBlock, GpuFinal


class DeployDiT(nn.Module):
    def __init__(self, rm, n_layers=None):
        super().__init__()
        self.x_embed = rm.all_x_embedder["2-1"]
        self.cap_embed = rm.cap_embedder
        self.noise_refiner = nn.ModuleList([GpuBlock(b) for b in rm.noise_refiner])
        self.context_refiner = nn.ModuleList([GpuBlock(b) for b in rm.context_refiner])
        layers = rm.layers if n_layers is None else rm.layers[:n_layers]
        self.layers = nn.ModuleList([GpuBlock(b) for b in layers])
        self.final = GpuFinal(rm.all_final_layer["2-1"])
        # Learned pad tokens the real _prepare_sequence substitutes at padded
        # positions (SELECT is GPU-banned -> arithmetic mask: y = y*(1-m)+pad*m).
        self.register_buffer("x_pad_token", rm.x_pad_token.detach().clone())      # [1, dim]
        self.register_buffer("cap_pad_token", rm.cap_pad_token.detach().clone())

    def forward(self, img_tokens, cap_feats, adaln, x_cos, x_sin, cap_cos, cap_sin,
                x_pad_mask, cap_pad_mask):
        x = self.x_embed(img_tokens)
        x = x * (1.0 - x_pad_mask) + self.x_pad_token * x_pad_mask
        for layer in self.noise_refiner:
            x = layer(x, x_cos, x_sin, adaln)
        cap = self.cap_embed(cap_feats)
        cap = cap * (1.0 - cap_pad_mask) + self.cap_pad_token * cap_pad_mask
        for layer in self.context_refiner:
            cap = layer(cap, cap_cos, cap_sin)
        unified = torch.cat([x, cap], dim=1)
        u_cos = torch.cat([x_cos, cap_cos], dim=2)
        u_sin = torch.cat([x_sin, cap_sin], dim=2)
        for layer in self.layers:
            unified = layer(unified, u_cos, u_sin, adaln)
        return self.final(unified, adaln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None, help="subset of main layers (default all 30)")
    ap.add_argument("--latent", type=int, default=16, help="latent H=W (256px img -> 32; use 16 for a fast full-depth size probe)")
    ap.add_argument("--cap", type=int, default=32)
    ap.add_argument("--quant", action="store_true")
    args = ap.parse_args()

    from diffusers import ZImageTransformer2DModel
    print("[load] real transformer (fp32) ...")
    rm = ZImageTransformer2DModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", subfolder="transformer", torch_dtype=torch.float32).eval()
    model = DeployDiT(rm, args.layers).eval()
    n_main = len(model.layers)

    n_img = (args.latent // 2) ** 2
    n_cap = args.cap
    img = torch.randn(1, n_img, 64)
    cap = torch.randn(1, n_cap, rm.config.cap_feat_dim)
    adaln = torch.randn(1, 256)
    xcos = torch.randn(1, 1, n_img, 64); xsin = torch.randn(1, 1, n_img, 64)
    ccos = torch.randn(1, 1, n_cap, 64); csin = torch.randn(1, 1, n_cap, 64)
    ins = (img, cap, adaln, xcos, xsin, ccos, csin)

    with torch.no_grad():
        out = model(*ins)
    print(f"[forward] {n_main} main layers, seq {n_img}+{n_cap} -> out {tuple(out.shape)}")

    import litert_torch
    out_path = f"zdeploy_L{n_main}.tflite"
    print("[convert] ...")
    litert_torch.convert(model, ins).export(out_path)
    mb = os.path.getsize(out_path) / 1e6
    print(f"[convert] {out_path}  {mb:.0f} MB fp32 ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")
    check_ops(out_path)

    if args.quant:
        from ai_edge_quantizer import quantizer, recipe
        qt = quantizer.Quantizer(out_path)
        qt.load_quantization_recipe(recipe.weight_only_wi4_afp32())
        q = f"zdeploy_L{n_main}_wi4.tflite"
        qt.quantize().export_model(q)
        qmb = os.path.getsize(q) / 1e6
        print(f"[int4] {q}  {qmb:.0f} MB  ({qmb/mb*100:.0f}% of fp32)")
        check_ops(q)


if __name__ == "__main__":
    main()
