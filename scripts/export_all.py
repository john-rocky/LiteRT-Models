#!/usr/bin/env python3
"""
Export Depth Anything V2 Small to ONNX + TFLite formats for Android.

Produces:
  - depth_anything_v2.onnx               (ONNX FP32, NCHW, single file)
  - depth_anything_v2_fp32.tflite        (TFLite FP32, NHWC via ai-edge-torch)
  - depth_anything_v2_fp32_clamped.tflite (TFLite FP32 with FP16-safe clamp, NHWC)
  - depth_anything_v2_int8.tflite        (TFLite dynamic-range quantized)
"""

import os
import shutil
import numpy as np
import torch
import torch.nn as nn

INPUT_H, INPUT_W = 518, 518
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "app", "src", "main", "assets")


# ---- Wrappers ----

class PlainWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).predicted_depth


class ClampedWrapper(nn.Module):
    """Inserts clamp inside ViT attention to prevent FP16 overflow."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._install_hooks()

    def _install_hooks(self):
        for name, mod in self.model.named_modules():
            cls_name = mod.__class__.__name__
            if cls_name in ("DPTViTAttention", "ViTSdpaSelfAttention",
                            "ViTSelfAttention", "Attention"):
                orig_forward = mod.forward

                def _make_hook(orig):
                    def hooked(*a, **kw):
                        out = orig(*a, **kw)
                        if isinstance(out, tuple):
                            return tuple(
                                torch.clamp(t, -60000.0, 60000.0)
                                if isinstance(t, torch.Tensor) and t.is_floating_point()
                                else t
                                for t in out
                            )
                        if isinstance(out, torch.Tensor) and out.is_floating_point():
                            return torch.clamp(out, -60000.0, 60000.0)
                        return out
                    return hooked

                mod.forward = _make_hook(orig_forward)

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).predicted_depth


# ---- Export: ONNX ----

def export_onnx(wrapper, path, label="ONNX"):
    print(f"\nExporting {label} -> {os.path.basename(path)}")
    dummy = torch.randn(1, 3, INPUT_H, INPUT_W)

    # Export to a temp dir first, then consolidate to single file
    tmp_path = path + ".tmp"
    torch.onnx.export(
        wrapper, dummy, tmp_path,
        opset_version=18,
        input_names=["pixel_values"],
        output_names=["depth"],
    )

    # Consolidate external data into single file
    import onnx
    if os.path.exists(tmp_path + ".data"):
        model = onnx.load(tmp_path, load_external_data=True)
        onnx.save_model(model, path,
                        save_as_external_data=False)
        os.remove(tmp_path)
        os.remove(tmp_path + ".data")
    else:
        os.rename(tmp_path, path)

    print(f"  OK  {os.path.getsize(path)/1e6:.1f} MB")


# ---- Export: TFLite via ai-edge-torch ----

def export_tflite(wrapper, path, label="TFLite"):
    print(f"\nExporting {label} -> {os.path.basename(path)}")
    import litert_torch

    dummy = (torch.randn(1, 3, INPUT_H, INPUT_W),)
    edge_model = litert_torch.convert(wrapper, dummy)
    edge_model.export(path)
    print(f"  OK  {os.path.getsize(path)/1e6:.1f} MB")


def export_tflite_int8(wrapper, path):
    """INT8 quantized TFLite via PT2E + XNNPACK quantizer."""
    print(f"\nExporting TFLite INT8 -> {os.path.basename(path)}")
    try:
        import litert_torch
        from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )

        dummy = (torch.randn(1, 3, INPUT_H, INPUT_W),)
        exported = torch.export.export(wrapper, dummy, strict=False)

        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True)
        )
        prepared = prepare_pt2e(exported, quantizer)

        # Calibrate with random data
        for _ in range(10):
            prepared(torch.randn(1, 3, INPUT_H, INPUT_W))

        quantized = convert_pt2e(prepared)
        edge_model = litert_torch.convert(quantized, dummy)
        edge_model.export(path)
        print(f"  OK  {os.path.getsize(path)/1e6:.1f} MB")
    except Exception as e:
        print(f"  SKIP ({e})")


# ---- Main ----

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading Depth-Anything-V2-Small from HuggingFace...")
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    model.eval()

    plain = PlainWrapper(model)
    clamped = ClampedWrapper(model)

    # 1. ONNX (single file, for ONNX Runtime mode)
    onnx_path = os.path.join(OUTPUT_DIR, "depth_anything_v2.onnx")
    export_onnx(plain, onnx_path, "ONNX FP32")

    # 2. TFLite FP32 (for XNNPACK FP16 CPU mode)
    fp32_path = os.path.join(OUTPUT_DIR, "depth_anything_v2_fp32.tflite")
    export_tflite(plain, fp32_path, "TFLite FP32")

    # 3. TFLite FP32 clamped (for GPU FP16 delegate)
    clamped_path = os.path.join(OUTPUT_DIR, "depth_anything_v2_fp32_clamped.tflite")
    export_tflite(clamped, clamped_path, "TFLite FP32 clamped")

    # 4. TFLite INT8 (PT2E quantization)
    int8_path = os.path.join(OUTPUT_DIR, "depth_anything_v2_int8.tflite")
    export_tflite_int8(PlainWrapper(model), int8_path)

    # Clean up any leftover clamped ONNX temp
    for f in os.listdir(OUTPUT_DIR):
        if "clamped" in f and f.endswith(".onnx"):
            os.remove(os.path.join(OUTPUT_DIR, f))

    print("\n=== Done ===")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        full = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(full):
            print(f"  {f:50s} {os.path.getsize(full)/1e6:8.1f} MB")


if __name__ == "__main__":
    main()
