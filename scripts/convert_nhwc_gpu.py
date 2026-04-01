#!/usr/bin/env python3
"""
Convert Depth Anything V2 Small to GPU-compatible NHWC TFLite models.

Produces 4 models:
  - depth_anything_v2_nhwc.tflite           (FP32, GPU FP32, 94MB, 575ms)
  - depth_anything_v2_nhwc_fp16w.tflite     (FP16 weights, GPU FP32, 47MB, 576ms)
  - depth_anything_v2_nhwc_clamped.tflite   (FP32 clamped, GPU FP16, 99MB, 209ms)
  - depth_anything_v2_nhwc_clamped_fp16w.tflite (FP16w clamped, GPU FP16, 50MB, 208ms)

Requirements:
  pip install torch torchvision transformers onnx onnxsim onnx2tf tf_keras psutil

  onnx2tf requires patches (applied automatically by this script):
  1. download_test_image_data() numpy pickle fix
  2. get_weights_constant_or_variable() ConvTranspose axis fix

Tested with: torch 2.9.1, transformers 5.4.0, onnx2tf 1.28.8, tensorflow 2.21.0

Usage:
  python convert_nhwc_gpu.py --output_dir ../app/src/main/assets/
"""

import argparse
import os
import sys
import types
import shutil
import numpy as np
import torch
import torch.nn as nn


def patch_onnx2tf():
    """Patch onnx2tf bugs before import."""
    try:
        import onnx2tf.utils.common_functions as cf
        src = os.path.abspath(cf.__file__)

        with open(src, "r") as f:
            code = f.read()

        patched = False

        # Patch 1: download_test_image_data pickle error
        if "def download_test_image_data" in code and "return np.random" not in code:
            code = code.replace(
                "def download_test_image_data() -> np.ndarray:",
                "def download_test_image_data() -> np.ndarray:\n"
                "    return np.random.rand(20, 230, 310, 3).astype(np.float32)"
            )
            patched = True

        # Patch 2: ConvTranspose weight transpose axis mismatch
        old = "        convertion_table = [i for i in range(2, kernel_size + 2)] + [1, 0]\n        values = values.transpose(convertion_table)"
        new = (
            "        convertion_table = [i for i in range(2, kernel_size + 2)] + [1, 0]\n"
            "        if len(convertion_table) != values.ndim:\n"
            "            convertion_table = [i for i in range(2, values.ndim)] + [1, 0]\n"
            "        values = values.transpose(convertion_table)"
        )
        if old in code and "values.ndim" not in code:
            code = code.replace(old, new)
            patched = True

        if patched:
            with open(src, "w") as f:
                f.write(code)
            print("[patch] onnx2tf patched successfully")
    except ImportError:
        print("[patch] onnx2tf not installed, skipping patch")


# --- Step 1: Load and patch PyTorch model ---

def load_and_patch_model():
    """Load DepthAnything V2 Small and apply GPU-compatibility patches."""
    from transformers import AutoModelForDepthEstimation

    print("[1/5] Loading Depth-Anything-V2-Small from HuggingFace...")
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    model.eval()

    # Patch 1: Replace GELU(exact) with GELU(tanh) to avoid Erf op
    gelu_count = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.GELU):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], nn.GELU(approximate="tanh"))
            gelu_count += 1
    print(f"  Replaced {gelu_count} GELU -> GELU(tanh)")

    # Patch 2: CLS token concat in 4D (GPU delegate requires 4D CONCATENATION)
    emb = model.backbone.embeddings

    def patched_forward(self, pixel_values, **kwargs):
        B = pixel_values.shape[0]
        patch_emb = self.patch_embeddings(pixel_values)
        cls = self.cls_token.expand(B, -1, -1)
        # 4D concat: [B,1,1,D] + [B,1,N,D] -> squeeze -> [B,1+N,D]
        combined = torch.cat(
            [cls.unsqueeze(1), patch_emb.unsqueeze(1)], dim=2
        ).squeeze(1)
        return combined + self.position_embeddings

    emb.forward = types.MethodType(patched_forward, emb)
    print("  Patched CLS token concat to 4D")

    return model


# --- Step 2: Export to ONNX ---

def export_onnx(model, output_path, input_h=518, input_w=518):
    """Export patched model to ONNX."""
    print(f"\n[2/5] Exporting ONNX -> {os.path.basename(output_path)}")

    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(pixel_values=x).predicted_depth

    wrapper = Wrapper(model)
    dummy = torch.randn(1, 3, input_h, input_w)

    with torch.no_grad():
        out = wrapper(dummy)
        print(f"  Test: input={dummy.shape} -> output={out.shape}")

    torch.onnx.export(
        wrapper, dummy, output_path + ".tmp",
        opset_version=18,
        input_names=["pixel_values"],
        output_names=["depth"],
    )

    # Consolidate external data to single file
    import onnx
    m = onnx.load(output_path + ".tmp", load_external_data=True)
    onnx.save_model(m, output_path, save_as_external_data=False)

    # Cleanup
    for ext in [".tmp", ".tmp.data"]:
        p = output_path + ext
        if os.path.exists(p):
            os.remove(p)

    print(f"  Size: {os.path.getsize(output_path) / 1e6:.1f} MB")
    return output_path


# --- Step 3: Fix ONNX graph ---

def fix_onnx(input_path, output_path):
    """Fix ConvTranspose kernel_shape, replace Erf->Tanh, simplify."""
    print(f"\n[3/5] Fixing ONNX graph...")
    import onnx
    from onnx import helper
    import onnxsim

    m = onnx.load(input_path)

    # Fix ConvTranspose missing kernel_shape
    ct_fixed = 0
    for node in m.graph.node:
        if node.op_type == "ConvTranspose":
            if not any(a.name == "kernel_shape" for a in node.attribute):
                for init in m.graph.initializer:
                    if init.name in node.input and len(init.dims) == 4:
                        ks = list(init.dims[2:])
                        node.attribute.append(
                            helper.make_attribute("kernel_shape", ks)
                        )
                        ct_fixed += 1
    print(f"  Fixed {ct_fixed} ConvTranspose kernel_shape")

    # Replace Erf -> Tanh
    erf_count = 0
    for i, node in enumerate(list(m.graph.node)):
        if node.op_type == "Erf":
            m.graph.node.remove(node)
            m.graph.node.insert(
                i,
                helper.make_node(
                    "Tanh", node.input, node.output, name=f"tanh_{i}"
                ),
            )
            erf_count += 1
    print(f"  Replaced {erf_count} Erf -> Tanh")

    # Simplify
    m2, _ = onnxsim.simplify(m)
    onnx.save_model(m2, output_path, save_as_external_data=False)
    print(f"  Simplified: {os.path.getsize(output_path) / 1e6:.1f} MB")

    # Verify
    remaining_erf = sum(1 for n in m2.graph.node if n.op_type == "Erf")
    assert remaining_erf == 0, f"Still {remaining_erf} Erf ops remaining!"

    return output_path


# --- Step 4: Insert clamps for FP16 safety ---

def insert_clamps(input_path, output_path, clamp_range=60000.0):
    """Insert Clip ops before Softmax/LayerNorm and after MatMul."""
    print(f"\n[4/5] Inserting FP16 clamps...")
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(input_path)
    clip_count = 0

    nodes = list(model.graph.node)

    # Collect insertion points (don't modify while iterating)
    clip_before = []  # (node_index, input_index)
    clip_after = []  # (node_index,)

    for i, node in enumerate(nodes):
        if node.op_type in {"Softmax", "LayerNormalization"}:
            clip_before.append(i)
        if node.op_type == "MatMul":
            clip_after.append(i)

    # Insert clips (process in reverse to preserve indices)
    all_insertions = []

    for i in clip_before:
        node = model.graph.node[i]
        inp = node.input[0]
        clipped = f"{inp}_c{clip_count}"
        min_t = numpy_helper.from_array(
            np.float32(-clamp_range), f"mn_{clip_count}"
        )
        max_t = numpy_helper.from_array(
            np.float32(clamp_range), f"mx_{clip_count}"
        )
        clip_node = helper.make_node(
            "Clip", [inp, min_t.name, max_t.name], [clipped],
            name=f"clip_{clip_count}",
        )
        model.graph.initializer.extend([min_t, max_t])
        node.input[0] = clipped
        all_insertions.append((i, clip_node))
        clip_count += 1

    for i in clip_after:
        node = model.graph.node[i]
        out = node.output[0]
        clipped_out = f"{out}_c{clip_count}"
        min_t = numpy_helper.from_array(
            np.float32(-clamp_range), f"mn_{clip_count}"
        )
        max_t = numpy_helper.from_array(
            np.float32(clamp_range), f"mx_{clip_count}"
        )
        clip_node = helper.make_node(
            "Clip", [out, min_t.name, max_t.name], [clipped_out],
            name=f"clip_{clip_count}",
        )
        model.graph.initializer.extend([min_t, max_t])
        # Redirect consumers
        for n2 in model.graph.node:
            if n2 is not node:
                for k in range(len(n2.input)):
                    if n2.input[k] == out:
                        n2.input[k] = clipped_out
        for go in model.graph.output:
            if go.name == out:
                go.name = clipped_out
        all_insertions.append((i + 1, clip_node))
        clip_count += 1

    # Insert in reverse order to preserve indices
    for idx, clip_node in sorted(all_insertions, key=lambda x: x[0], reverse=True):
        model.graph.node.insert(idx, clip_node)

    print(f"  Inserted {clip_count} Clip nodes "
          f"(Softmax:{len(clip_before) - sum(1 for i in clip_before if nodes[i].op_type == 'LayerNormalization')}, "
          f"LayerNorm:{sum(1 for i in clip_before if nodes[i].op_type == 'LayerNormalization')}, "
          f"MatMul:{len(clip_after)})")

    onnx.save(model, output_path, save_as_external_data=False)
    print(f"  Size: {os.path.getsize(output_path) / 1e6:.1f} MB")
    return output_path


# --- Step 5: Convert to TFLite via onnx2tf ---

def convert_tflite(onnx_path, output_dir, name_prefix):
    """Convert ONNX to TFLite NHWC via onnx2tf."""
    print(f"\n[5/5] Converting to TFLite via onnx2tf -> {name_prefix}*")
    import subprocess

    tmp_dir = output_dir + f"/_tmp_{name_prefix}"
    r = subprocess.run(
        [
            sys.executable, "-m", "onnx2tf",
            "-i", onnx_path,
            "-o", tmp_dir,
            "-osd", "-coion",
        ],
        capture_output=True, text=True, timeout=600,
    )

    if "complete" not in r.stdout.lower():
        print(f"  FAILED: {r.stderr[-300:]}")
        return {}

    # Find and rename output files
    results = {}
    onnx_basename = os.path.splitext(os.path.basename(onnx_path))[0]
    for f in os.listdir(tmp_dir):
        if f.endswith(".tflite"):
            if "float16" in f:
                dst = f"{name_prefix}_fp16w.tflite"
            elif "float32" in f:
                dst = f"{name_prefix}.tflite"
            else:
                continue
            src = os.path.join(tmp_dir, f)
            dst_path = os.path.join(output_dir, dst)
            shutil.copy2(src, dst_path)
            results[dst] = os.path.getsize(dst_path)
            print(f"  {dst}: {os.path.getsize(dst_path) / 1e6:.0f} MB")

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return results


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Convert DepthAnything V2 for Android GPU")
    parser.add_argument("--output_dir", default="./output",
                        help="Output directory for model files")
    parser.add_argument("--input_size", type=int, default=518,
                        help="Input image size (must be multiple of 14)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tmp = os.path.join(args.output_dir, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    # Patch onnx2tf first
    patch_onnx2tf()

    # Step 1: Load model
    model = load_and_patch_model()

    # Step 2: Export ONNX
    onnx_path = os.path.join(tmp, "model.onnx")
    export_onnx(model, onnx_path, args.input_size, args.input_size)

    # Step 3: Fix ONNX
    fixed_path = os.path.join(tmp, "model_fixed.onnx")
    fix_onnx(onnx_path, fixed_path)

    # Step 4a: Convert plain (no clamp) -> FP32 GPU + FP16w GPU
    convert_tflite(fixed_path, args.output_dir, "depth_anything_v2_nhwc")

    # Step 4b: Insert clamps -> FP16 GPU
    clamped_path = os.path.join(tmp, "model_clamped.onnx")
    insert_clamps(fixed_path, clamped_path)
    convert_tflite(clamped_path, args.output_dir, "depth_anything_v2_nhwc_clamped")

    # Cleanup
    shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    print("Done! Output files:")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".tflite"):
            path = os.path.join(args.output_dir, f)
            print(f"  {f:50s} {os.path.getsize(path) / 1e6:6.0f} MB")
    print("=" * 60)
    print("\nCopy to app/src/main/assets/ and build with Android Studio.")


if __name__ == "__main__":
    main()
