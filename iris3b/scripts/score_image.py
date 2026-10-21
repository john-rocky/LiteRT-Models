"""Image parity scorer for the iris-3b lane (one definition for every round).

Compares a candidate image with a reference image, both in the sampler's
output range [-1, 1] (the clamped `clean` of the last step). Inputs may be
`.pt` files holding a tensor or a dict with an "image" key ([1, 3, H, W] or
[3, H, W]), `.npy` arrays in the same layout, or 8-bit `.png` files (mapped
from [0, 255] back to [-1, 1]).

Metrics, on (x + 1) / 2 in [0, 1]:
  psnr_db  10 * log10(1 / MSE)
  ssim     skimage.metrics.structural_similarity, channel_axis=-1,
           data_range=1.0, default 7x7 uniform window
  corr     Pearson over every value
  max_abs  max |candidate - reference|

    .venv/bin/python scripts/score_image.py REF CAND [--json OUT]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_image(path: str) -> np.ndarray:
    """Returns an [H, W, 3] float64 array in [0, 1]."""
    p = Path(path)
    if p.suffix == ".png":
        from PIL import Image

        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0
        return arr
    if p.suffix == ".npy":
        arr = np.load(p).astype(np.float64)
    else:
        import torch

        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, dict):
            obj = obj["image"]
        arr = obj.detach().float().numpy().astype(np.float64)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, -1.0, 1.0)
    return (arr + 1.0) / 2.0


def score(ref: np.ndarray, cand: np.ndarray) -> dict:
    from skimage.metrics import structural_similarity

    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch: ref {ref.shape} vs cand {cand.shape}")
    diff = cand - ref
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse == 0.0 else 10.0 * np.log10(1.0 / mse)
    ssim = float(structural_similarity(ref, cand, channel_axis=-1, data_range=1.0))
    corr = float(np.corrcoef(ref.ravel(), cand.ravel())[0, 1])
    return {
        "psnr_db": round(psnr, 3),
        "ssim": round(ssim, 5),
        "corr": round(corr, 6),
        "max_abs": round(float(np.max(np.abs(diff))), 6),
        "shape": list(ref.shape),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--json", default=None, help="append the result as one JSON line to this file")
    args = ap.parse_args()
    result = {"ref": args.ref, "cand": args.cand, **score(load_image(args.ref), load_image(args.cand))}
    line = json.dumps(result)
    print(line)
    if args.json:
        with open(args.json, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    sys.exit(main())
