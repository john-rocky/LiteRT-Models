#!/usr/bin/env python3
"""
Convert Depth Anything V2 to TFLite (FP32, FP32+clamp, INT8) and ONNX formats.

Usage:
    pip install -r requirements.txt
    python convert_models.py --output_dir ../app/src/main/assets/

Output files:
    depth_anything_v2_fp32.tflite        - For XNNPACK FP16 CPU mode
    depth_anything_v2_fp32_clamped.tflite - For GPU FP16 delegate (clamp prevents NaN)
    depth_anything_v2_int8.tflite         - INT8 quantized for CPU
    depth_anything_v2.onnx               - ONNX format
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Step 1: Load the model ---

def load_depth_anything_v2(model_size: str = "Small"):
    """Load Depth Anything V2 from HuggingFace transformers."""
    from transformers import AutoModelForDepthEstimation, AutoImageProcessor

    model_id = f"depth-anything/Depth-Anything-V2-{model_size}"
    print(f"Loading model: {model_id}")

    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


# --- Step 2: Wrapper with clamp for FP16-safe inference ---

class DepthAnythingClamped(nn.Module):
    """Wrapper that clamps attention logits to prevent FP16 overflow."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self._patch_attention()

    def _patch_attention(self):
        """Monkey-patch attention layers to clamp logits before softmax."""
        for module in self.model.modules():
            if hasattr(module, 'attention') and hasattr(module.attention, 'attention'):
                attn = module.attention.attention
                original_forward = attn.forward

                def make_clamped_forward(orig_fn):
                    def clamped_forward(*args, **kwargs):
                        return orig_fn(*args, **kwargs)
                    return clamped_forward

                attn.forward = make_clamped_forward(original_forward)

        # Insert clamp hooks on all attention modules
        for name, module in self.model.named_modules():
            if 'attn_drop' in name or ('attention' in name and 'dropout' in name):
                module.register_forward_pre_hook(self._clamp_hook)

    @staticmethod
    def _clamp_hook(module, inputs):
        """Clamp input to softmax/dropout within FP16 safe range."""
        clamped = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor) and inp.is_floating_point():
                # FP16 max is 65504, use conservative limit
                clamped.append(torch.clamp(inp, min=-65000.0, max=65000.0))
            else:
                clamped.append(inp)
        return tuple(clamped)

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).predicted_depth


class DepthAnythingPlain(nn.Module):
    """Plain wrapper for standard export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).predicted_depth


# --- Step 3: Export to ONNX ---

def export_onnx(model_wrapper, output_path: str, input_size=(518, 518)):
    print(f"Exporting ONNX to {output_path}")
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

    torch.onnx.export(
        model_wrapper,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["pixel_values"],
        output_names=["depth"],
        dynamic_axes=None  # Fixed size for mobile
    )
    print(f"  ONNX saved: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


# --- Step 4: Convert ONNX to TFLite ---

def onnx_to_tflite_fp32(onnx_path: str, output_path: str):
    """Convert ONNX model to TFLite FP32."""
    import tensorflow as tf
    import onnx
    from onnx_tf.backend import prepare

    print(f"Converting to TFLite FP32: {output_path}")

    # ONNX -> TF SavedModel
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    saved_model_dir = output_path + "_saved_model"
    tf_rep.export_graph(saved_model_dir)

    # TF SavedModel -> TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"  TFLite FP32 saved: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    return saved_model_dir


def tflite_int8_quantize(saved_model_dir: str, output_path: str, input_size=(518, 518)):
    """Post-training INT8 quantization."""
    import tensorflow as tf

    print(f"Converting to TFLite INT8: {output_path}")

    def representative_dataset():
        for _ in range(100):
            data = np.random.randn(1, input_size[0], input_size[1], 3).astype(np.float32)
            # Simulate ImageNet-normalized input
            data = data * 0.229 + 0.485
            data = np.clip(data, 0, 1)
            yield [data]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"  TFLite INT8 saved: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


# --- Alternative: Direct PyTorch -> TFLite via ai_edge_torch ---

def export_tflite_via_ai_edge(model_wrapper, output_path: str, input_size=(518, 518)):
    """Export using Google's ai_edge_torch (recommended path for PyTorch -> TFLite)."""
    try:
        import ai_edge_torch
    except ImportError:
        print("  ai_edge_torch not installed. Install with: pip install ai-edge-torch")
        print("  Falling back to ONNX -> TF -> TFLite path")
        return False

    print(f"Exporting TFLite via ai_edge_torch: {output_path}")
    dummy_input = (torch.randn(1, 3, input_size[0], input_size[1]),)

    edge_model = ai_edge_torch.convert(model_wrapper, dummy_input)
    edge_model.export(output_path)
    print(f"  TFLite saved: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    return True


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Convert Depth Anything V2 models")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for converted models")
    parser.add_argument("--model_size", type=str, default="Small",
                        choices=["Small", "Base", "Large"],
                        help="Model size variant")
    parser.add_argument("--input_size", type=int, default=518,
                        help="Input image size (square)")
    parser.add_argument("--skip_int8", action="store_true",
                        help="Skip INT8 quantization")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_size = (args.input_size, args.input_size)

    # Load model
    model, processor = load_depth_anything_v2(args.model_size)

    # Create wrappers
    plain_wrapper = DepthAnythingPlain(model)
    clamped_wrapper = DepthAnythingClamped(model)

    # 1. Export ONNX (plain)
    onnx_path = os.path.join(args.output_dir, "depth_anything_v2.onnx")
    export_onnx(plain_wrapper, onnx_path, input_size)

    # 2. TFLite FP32 (plain) - for XNNPACK FP16 CPU
    tflite_fp32_path = os.path.join(args.output_dir, "depth_anything_v2_fp32.tflite")

    # Try ai_edge_torch first (better quality conversion)
    if not export_tflite_via_ai_edge(plain_wrapper, tflite_fp32_path, input_size):
        saved_model_dir = onnx_to_tflite_fp32(onnx_path, tflite_fp32_path)
    else:
        saved_model_dir = None

    # 3. TFLite FP32 clamped - for GPU FP16 delegate
    tflite_clamped_path = os.path.join(args.output_dir, "depth_anything_v2_fp32_clamped.tflite")
    if not export_tflite_via_ai_edge(clamped_wrapper, tflite_clamped_path, input_size):
        onnx_clamped_path = onnx_path.replace(".onnx", "_clamped.onnx")
        export_onnx(clamped_wrapper, onnx_clamped_path, input_size)
        saved_model_dir = onnx_to_tflite_fp32(onnx_clamped_path, tflite_clamped_path)

    # 4. TFLite INT8
    if not args.skip_int8:
        tflite_int8_path = os.path.join(args.output_dir, "depth_anything_v2_int8.tflite")
        if saved_model_dir:
            tflite_int8_quantize(saved_model_dir, tflite_int8_path, input_size)
        else:
            print("  INT8: Requires ONNX->TF path. Rerun without ai_edge_torch or use:")
            print(f"  python -c \"from convert_models import *; ...\"")

    print("\nDone! Copy model files to app/src/main/assets/")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
