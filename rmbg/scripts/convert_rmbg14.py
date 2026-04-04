#!/usr/bin/env python3
"""
Convert RMBG-1.4 (ISNet) to TFLite for Android CompiledModel GPU.

Pure CNN model — no attention, no deformable conv, no GPU-incompatible ops.

Requirements:
    pip install litert-torch safetensors huggingface_hub torch

Usage:
    python convert_rmbg14.py --output_dir output/
"""

import argparse
import os
import sys
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_model():
    """Load RMBG-1.4 from HuggingFace, bypassing transformers compatibility issues."""
    from huggingface_hub import snapshot_download

    repo_path = snapshot_download("briaai/RMBG-1.4")
    print(f"Model downloaded to: {repo_path}")

    # Minimal config (ISNet only uses in_ch/out_ch)
    class RMBGConfig:
        model_type = "SegformerForSemanticSegmentation"
        def __init__(self, in_ch=3, out_ch=1):
            self.in_ch = in_ch
            self.out_ch = out_ch

    # Load model code with patched imports
    with open(os.path.join(repo_path, "briarmbg.py")) as f:
        src = f.read()

    src = src.replace("from .MyConfig import RMBGConfig", "")
    src = src.replace("from transformers import PreTrainedModel", "")
    src = src.replace("class BriaRMBG(PreTrainedModel):", "class BriaRMBG(nn.Module):")
    src = src.replace("super().__init__(config)", "super().__init__()")

    ns = {"nn": nn, "F": F, "torch": torch, "RMBGConfig": RMBGConfig,
          "__name__": "__main__", "__builtins__": __builtins__}
    exec(compile(src, "briarmbg.py", "exec"), ns)

    model = ns["BriaRMBG"]()
    from safetensors.torch import load_file
    model.load_state_dict(load_file(os.path.join(repo_path, "model.safetensors")))
    model.eval()

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    return model


class ISNetWrapper(nn.Module):
    """Wrapper that returns only the primary mask output."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0][0]  # First mask: [1, 1, 1024, 1024]


def convert(output_dir):
    base_model = load_model()
    model = ISNetWrapper(base_model)
    model.eval()

    # Patch F.interpolate to avoid align_corners GPU issue
    _orig = F.interpolate
    F.interpolate = lambda inp, size=None, scale_factor=None, mode='nearest', align_corners=None, **kw: \
        _orig(inp, size=size, scale_factor=scale_factor, mode=mode,
              align_corners=False if mode in ('bilinear', 'bicubic') else align_corners, **kw)

    dummy = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        out = model(dummy)
        print(f"Output: {out.shape} [{out.min():.3f}, {out.max():.3f}]")

    print("Converting with litert-torch...")
    import litert_torch
    result = litert_torch.convert(model, (dummy,))

    out_path = os.path.join(output_dir, "rmbg14.tflite")
    result.export(out_path)
    print(f"Saved: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    print(f"  Input:  [1, 3, 1024, 1024] NCHW float32")
    print(f"  Output: [1, 1, 1024, 1024] sigmoid mask (0-1)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Convert RMBG-1.4 to TFLite")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    path = convert(args.output_dir)

    print(f"\nDone! Copy to Android assets/:")
    print(f"  cp {path} rmbg/app/src/main/assets/")


if __name__ == "__main__":
    main()
