#!/usr/bin/env python3
"""
Convert Depth Anything V2 Small with onnx2tf 1.29.x improvements.

This script tests multiple onnx2tf conversion strategies to find
the best quality NHWC TFLite model for ML Drift GPU inference.

Variants produced:
  1. v129         — onnx2tf 1.29.24 default (upgraded from 1.28.8)
  2. v129_cotof   — with -cotof per-op accuracy checking
  3. v129_ebu     — with -ebu (BatchMatMul unfold)
  4. v129_dsft    — with -dsft (disable FlexTranspose suppression)

Each variant also produces a clamped version for FP16 GPU mode.

Usage:
  pip install --upgrade onnx2tf
  python convert_v129.py --output_dir ../app/src/main/assets/
"""

import argparse
import os
import sys
import types
import shutil
import subprocess
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


def load_and_patch_model():
    """Load DepthAnything V2 Small and apply GPU-compatibility patches."""
    from transformers import AutoModelForDepthEstimation

    print("[1/4] Loading Depth-Anything-V2-Small from HuggingFace...")
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
    patch_size = emb.patch_embeddings.patch_size[0]

    def patched_forward(self, pixel_values, **kwargs):
        B = pixel_values.shape[0]
        patch_emb = self.patch_embeddings(pixel_values)
        num_patches = patch_emb.shape[1]
        cls = self.cls_token.expand(B, -1, -1)
        combined = torch.cat(
            [cls.unsqueeze(1), patch_emb.unsqueeze(1)], dim=2
        ).squeeze(1)
        pos_emb = self.position_embeddings
        if pos_emb.shape[1] != num_patches + 1:
            cls_pos = pos_emb[:, :1, :]
            patch_pos = pos_emb[:, 1:, :]
            orig_size = int(patch_pos.shape[1] ** 0.5)
            h_patches = pixel_values.shape[2] // patch_size
            w_patches = pixel_values.shape[3] // patch_size
            patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
            patch_pos = nn.functional.interpolate(
                patch_pos, size=(h_patches, w_patches), mode="bicubic", align_corners=False
            )
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, pos_emb.shape[2])
            pos_emb = torch.cat([cls_pos, patch_pos], dim=1)
        return combined + pos_emb

    emb.forward = types.MethodType(patched_forward, emb)
    print("  Patched CLS token concat to 4D + position embedding interpolation")

    return model


def export_onnx(model, output_path, input_h=518, input_w=518):
    """Export patched model to ONNX."""
    print(f"\n[2/4] Exporting ONNX -> {os.path.basename(output_path)}")
    import onnx

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

    m = onnx.load(output_path + ".tmp", load_external_data=True)
    onnx.save_model(m, output_path, save_as_external_data=False)

    for ext in [".tmp", ".tmp.data"]:
        p = output_path + ext
        if os.path.exists(p):
            os.remove(p)

    print(f"  Size: {os.path.getsize(output_path) / 1e6:.1f} MB")
    return output_path


def fix_onnx(input_path, output_path):
    """Fix ConvTranspose kernel_shape, replace Erf->Tanh, simplify."""
    print(f"\n[3/4] Fixing ONNX graph...")
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

    remaining_erf = sum(1 for n in m2.graph.node if n.op_type == "Erf")
    assert remaining_erf == 0, f"Still {remaining_erf} Erf ops remaining!"

    return output_path


def insert_clamps(input_path, output_path, clamp_range=60000.0):
    """Insert Clip ops before Softmax/LayerNorm and after MatMul."""
    print(f"  Inserting FP16 clamps...")
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(input_path)
    clip_count = 0

    nodes = list(model.graph.node)
    clip_before = []
    clip_after = []

    for i, node in enumerate(nodes):
        if node.op_type in {"Softmax", "LayerNormalization"}:
            clip_before.append(i)
        if node.op_type == "MatMul":
            clip_after.append(i)

    all_insertions = []

    for i in clip_before:
        node = model.graph.node[i]
        inp = node.input[0]
        clipped = f"{inp}_c{clip_count}"
        min_t = numpy_helper.from_array(np.float32(-clamp_range), f"mn_{clip_count}")
        max_t = numpy_helper.from_array(np.float32(clamp_range), f"mx_{clip_count}")
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
        min_t = numpy_helper.from_array(np.float32(-clamp_range), f"mn_{clip_count}")
        max_t = numpy_helper.from_array(np.float32(clamp_range), f"mx_{clip_count}")
        clip_node = helper.make_node(
            "Clip", [out, min_t.name, max_t.name], [clipped_out],
            name=f"clip_{clip_count}",
        )
        model.graph.initializer.extend([min_t, max_t])
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

    for idx, clip_node in sorted(all_insertions, key=lambda x: x[0], reverse=True):
        model.graph.node.insert(idx, clip_node)

    print(f"  Inserted {clip_count} Clip nodes")
    onnx.save(model, output_path, save_as_external_data=False)
    return output_path


def convert_tflite(onnx_path, output_dir, name_prefix, extra_flags=None):
    """Convert ONNX to TFLite NHWC via onnx2tf."""
    print(f"\n  Converting: {name_prefix}")
    tmp_dir = os.path.join(output_dir, f"_tmp_{name_prefix}")

    cmd = [
        sys.executable, "-m", "onnx2tf",
        "-i", onnx_path,
        "-o", tmp_dir,
        "-osd", "-coion",
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    print(f"  Command: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)

    if r.returncode != 0 and "complete" not in r.stdout.lower():
        print(f"  FAILED (rc={r.returncode})")
        stderr_tail = r.stderr[-500:] if r.stderr else ""
        stdout_tail = r.stdout[-500:] if r.stdout else ""
        print(f"  stderr: {stderr_tail}")
        print(f"  stdout: {stdout_tail}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {}

    # Find and rename output files
    results = {}
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
            print(f"  -> {dst}: {os.path.getsize(dst_path) / 1e6:.0f} MB")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert DepthAnything V2 with onnx2tf 1.29.x improvements"
    )
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--input_height", type=int, default=518)
    parser.add_argument("--input_width", type=int, default=518)
    parser.add_argument("--variant", type=str, default="all",
                        choices=["all", "v129", "cotof", "ebu", "dsft", "cotof_agje"],
                        help="Which variant to convert (default: all)")
    args = parser.parse_args()

    assert args.input_height % 14 == 0
    assert args.input_width % 14 == 0

    os.makedirs(args.output_dir, exist_ok=True)
    tmp = os.path.join(args.output_dir, "_tmp_v129")
    os.makedirs(tmp, exist_ok=True)

    # Check onnx2tf version
    import onnx2tf
    print(f"onnx2tf version: {onnx2tf.__version__}")
    if onnx2tf.__version__ < "1.29":
        print("WARNING: onnx2tf < 1.29.0, upgrade with: pip install --upgrade onnx2tf")

    patch_onnx2tf()

    # Step 1: Load model
    model = load_and_patch_model()

    # Step 2: Export ONNX
    onnx_path = os.path.join(tmp, "model.onnx")
    export_onnx(model, onnx_path, args.input_height, args.input_width)

    # Step 3: Fix ONNX
    fixed_path = os.path.join(tmp, "model_fixed.onnx")
    fix_onnx(onnx_path, fixed_path)

    # Step 3b: Clamped ONNX for FP16 variants
    clamped_path = os.path.join(tmp, "model_clamped.onnx")
    insert_clamps(fixed_path, clamped_path)

    # Define conversion variants
    variants = {
        "v129": {
            "flags": [],
            "desc": "onnx2tf 1.29 default",
        },
        "cotof": {
            "flags": ["-cotof", "-cotoa", "0.1"],
            "desc": "onnx2tf 1.29 + per-op accuracy check (atol=0.1)",
        },
        "ebu": {
            "flags": ["-ebu"],
            "desc": "onnx2tf 1.29 + BatchMatMul unfold",
        },
        "dsft": {
            "flags": ["-dsft"],
            "desc": "onnx2tf 1.29 + disable FlexTranspose suppression",
        },
        "cotof_agje": {
            "flags": ["-cotof", "-cotoa", "0.01", "-agje"],
            "desc": "onnx2tf 1.29 + accuracy check + auto JSON error fix",
        },
    }

    if args.variant != "all":
        variants = {args.variant: variants[args.variant]}

    # Step 4: Convert each variant
    print("\n" + "=" * 60)
    print("[4/4] Converting variants...")
    print("=" * 60)

    all_results = {}
    for name, cfg in variants.items():
        print(f"\n{'='*40}")
        print(f"Variant: {name} — {cfg['desc']}")
        print(f"{'='*40}")

        # No-clamp version (for FP32 GPU mode)
        prefix = f"depth_anything_v2_{name}"
        results = convert_tflite(fixed_path, args.output_dir, prefix, cfg["flags"])
        all_results.update(results)

        # Clamped version (for FP16 GPU mode)
        clamped_prefix = f"depth_anything_v2_{name}_clamped"
        results = convert_tflite(clamped_path, args.output_dir, clamped_prefix, cfg["flags"])
        all_results.update(results)

    # Cleanup
    shutil.rmtree(tmp, ignore_errors=True)

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    for name, size in sorted(all_results.items()):
        print(f"  {name:55s} {size / 1e6:6.0f} MB")
    print("=" * 60)
    print("\nCopy models to app/src/main/assets/ and benchmark on device.")


if __name__ == "__main__":
    main()
