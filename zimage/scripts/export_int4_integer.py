"""LiteRT INTEGER-compute int4 export of the Z-Image DiT (the working path).

Team finding (~/Downloads/meeting/runtime-pitch-apple-gpu-parity.md): weight-only
int4 + FLOAT compute hangs/fails in the Apple GPU delegate; the path that RUNS on
Apple GPU is INTEGER-compute dynamic-range int4 (int4×int8). In litert_torch that is
`full_dynamic_recipe(weight_dtype=INT4, granularity=BLOCKWISE_*)`, applied at convert
time. This script produces that graph and CPU-verifies int4 quality vs fp32 (the
Apple-GPU run itself needs the WebGPU delegate / device — not selectable from the
ai_edge_litert Python CompiledModel API, which forces ML Drift Metal).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/Users/majimadaisuke/Downloads/depthanything-android")
from deploy_dit import DeployDiT
from ai_edge_litert.interpreter import Interpreter

LATENT, N_CAP = 32, 32
N_IMG = (LATENT // 2) ** 2


def run_cpu(path, feeds):
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    ins = sorted(it.get_input_details(), key=lambda d: d["index"])
    for d, a in zip(ins, feeds):
        it.set_tensor(d["index"], a)
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None, help="main layers (default all 30)")
    ap.add_argument("--block", type=int, default=128, choices=[32, 64, 128, 256])
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8])
    args = ap.parse_args()

    from diffusers import ZImageTransformer2DModel
    import litert_torch
    from litert_torch.generative.quantize import quant_recipes as qrs
    from litert_torch.generative.quantize.quant_attrs import Dtype, Granularity

    print("[load] real transformer (fp32) ...")
    rm = ZImageTransformer2DModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", subfolder="transformer", torch_dtype=torch.float32).eval()
    model = DeployDiT(rm, args.layers).eval()
    n_main = len(model.layers)

    torch.manual_seed(7)
    img = torch.randn(1, N_IMG, 64)
    cap = torch.randn(1, N_CAP, rm.config.cap_feat_dim)
    adaln = torch.randn(1, 256)
    xcos = torch.randn(1, 1, N_IMG, 64); xsin = torch.randn(1, 1, N_IMG, 64)
    ccos = torch.randn(1, 1, N_CAP, 64); csin = torch.randn(1, 1, N_CAP, 64)
    x_pad = torch.zeros(1, N_IMG, 1); cap_pad = torch.zeros(1, N_CAP, 1)
    ins = (img, cap, adaln, xcos, xsin, ccos, csin, x_pad, cap_pad)

    with torch.no_grad():
        ref = model(*ins).numpy()
    print(f"[forward] {n_main} main layers, out {ref.shape}")

    if args.bits == 8:
        cfg = qrs.full_dynamic_recipe(weight_dtype=Dtype.INT8, granularity=Granularity.CHANNELWISE)
        out = f"zdit_int8integer_L{n_main}.tflite"
        print("[convert] INTEGER-compute int8 dynamic, CHANNELWISE ...")
    else:
        gran = {32: Granularity.BLOCKWISE_32, 64: Granularity.BLOCKWISE_64,
                128: Granularity.BLOCKWISE_128, 256: Granularity.BLOCKWISE_256}[args.block]
        cfg = qrs.full_dynamic_recipe(weight_dtype=Dtype.INT4, granularity=gran)
        out = f"zdit_int4integer_L{n_main}_blk{args.block}.tflite"
        print(f"[convert] INTEGER-compute int4 dynamic, BLOCKWISE_{args.block} ...")
    litert_torch.convert(model, ins, quant_config=cfg).export(out)
    mb = os.path.getsize(out) / 1e6
    print(f"[int4] {out}  {mb:.0f} MB  ({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")

    # int4 quality gate: run on CPU (XNNPack), corr vs fp32 torch reference
    feeds = [a.numpy().astype("<f4") for a in ins]
    q = run_cpu(out, feeds)
    corr = np.corrcoef(ref.flatten(), q.flatten())[0, 1]
    mad = np.abs(ref - q).mean()
    print(f"[quality] int4(CPU) vs fp32(torch): corr {corr:.6f}, mean|diff| {mad:.4e}, "
          f"range ref[{ref.min():.3f},{ref.max():.3f}] q[{q.min():.3f},{q.max():.3f}]")


if __name__ == "__main__":
    main()
