#!/usr/bin/env python3
"""
Convert Ultralytics yolo26n-pose to a GPU-friendly TFLite model.

Path: litert-torch (NOT onnx2tf — the YOLO26 backbone trips an onnx2tf
channel-tracking bug at model.2/m.0/Add). litert-torch preserves NCHW and
goes through the same pipeline used for mobilesam / rmbg / dsine in this repo.

Output:  yolo-pose/app/src/main/assets/yolo26n_pose.tflite
Format:  NCHW [1, 3, 384, 384] -> [1, 56, 3024]
         56 = 4 bbox + 1 person conf + 17 keypoints * 3 (x, y, visibility)
         3024 = 48^2 + 24^2 + 12^2 multi-scale anchor cells for 384x384
         Coordinates are in input image pixel space (0..384).

Usage:
    pip install ultralytics litert-torch
    python convert_yolo26n_pose.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = REPO_ROOT / "yolo-pose"
ASSETS_DIR = MODULE_DIR / "app" / "src" / "main" / "assets"

# Make litert_gpu_toolkit importable from repo root
sys.path.insert(0, str(REPO_ROOT))

IMG_SIZE = 384
SOURCE = "yolo26n-pose.pt"
TARGET = "yolo26n_pose.tflite"


class Yolo26PoseRawHead(nn.Module):
    """Wrap an Ultralytics YOLO26 pose model so forward returns the raw
    one-to-many head tensor [1, 56, N] without the end-to-end TopK/Gather."""

    def __init__(self, ultralytics_model: nn.Module) -> None:
        super().__init__()
        self.model = ultralytics_model
        head = self.model.model[-1]
        head.end2end = False  # bypass NMS-free TopK / Gather
        head.export = True    # use the export-mode forward path
        head.format = "tflite"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Install ultralytics first: pip install ultralytics")

    try:
        from litert_gpu_toolkit import convert_for_gpu
    except ImportError:
        sys.exit(
            "litert_gpu_toolkit not importable from repo root.\n"
            f"Expected at: {REPO_ROOT / 'litert_gpu_toolkit'}"
        )

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Loading {SOURCE}")
    yolo = YOLO(SOURCE)
    base = yolo.model
    base.eval()

    wrapper = Yolo26PoseRawHead(base).eval()

    # Sanity check forward shape before patching
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        sample = wrapper(dummy)
    print(f"      forward output: {tuple(sample.shape)}  "
          f"(min={float(sample.min()):.2f}, max={float(sample.max()):.2f})")
    if sample.shape != (1, 56, 3024):
        print(f"      WARNING: unexpected pose output shape {tuple(sample.shape)}")

    print(f"[2/3] Converting via litert_gpu_toolkit (litert-torch)")
    out_path = ASSETS_DIR / TARGET
    convert_for_gpu(
        wrapper,
        dummy_input=dummy,
        output_path=str(out_path),
        check=True,
        verbose=True,
    )

    if not out_path.exists():
        sys.exit(f"Conversion finished but {out_path} is missing")

    size_mb = out_path.stat().st_size / 1_000_000
    print(f"[3/3] Done. {out_path.name} ({size_mb:.1f} MB) -> {out_path}")


if __name__ == "__main__":
    main()
