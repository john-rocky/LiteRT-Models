#!/usr/bin/env python3
"""
Compare quality of TFLite models against PyTorch/ONNX ground truth.

Computes correlation coefficient and MSE between model outputs.

Usage:
  python compare_quality.py --model_dir ../app/src/main/assets/
"""

import argparse
import os
import sys
import glob
import numpy as np


def get_pytorch_reference(input_h=518, input_w=518):
    """Get PyTorch model output as ground truth."""
    import torch
    import torch.nn as nn
    import types
    from transformers import AutoModelForDepthEstimation

    print("Loading PyTorch model (ground truth)...")
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf"
    )
    model.eval()

    # Apply same patches as conversion
    for name, mod in model.named_modules():
        if isinstance(mod, nn.GELU):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], nn.GELU(approximate="tanh"))

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

    # Use deterministic input
    np.random.seed(42)
    dummy_np = np.random.rand(1, 3, input_h, input_w).astype(np.float32)
    dummy = torch.from_numpy(dummy_np)

    with torch.no_grad():
        output = model(pixel_values=dummy).predicted_depth
        ref = output.numpy().flatten()

    print(f"  PyTorch output shape: {output.shape}, range: [{ref.min():.4f}, {ref.max():.4f}]")
    return dummy_np, ref


def eval_onnx(model_path, input_np):
    """Evaluate ONNX model."""
    import onnxruntime as ort
    print(f"  Evaluating ONNX: {os.path.basename(model_path)}")
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    expected_shape = sess.get_inputs()[0].shape  # e.g. [1, 3, 392, 518]
    if list(input_np.shape) != list(expected_shape):
        print(f"    Skipping: input shape {input_np.shape} != expected {expected_shape}")
        return None
    result = sess.run(None, {input_name: input_np})[0]
    return result.flatten()


def eval_tflite(model_path, input_np):
    """Evaluate TFLite model using TF Interpreter."""
    try:
        import tensorflow as tf
        print(f"  Evaluating TFLite: {os.path.basename(model_path)}")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        inp_shape = input_details[0]["shape"]
        # Determine if NHWC or NCHW
        if inp_shape[1] == 3:  # NCHW
            feed = input_np.copy()
        else:  # NHWC
            feed = np.transpose(input_np, (0, 2, 3, 1))

        # Resize if shape mismatch
        if list(feed.shape) != list(inp_shape):
            interpreter.resize_tensor_input(input_details[0]["index"], list(feed.shape))
            interpreter.allocate_tensors()

        interpreter.set_tensor(input_details[0]["index"], feed.astype(np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        return output.flatten()
    except ImportError:
        print("  WARNING: tensorflow not available, skipping TFLite eval")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def compute_metrics(ref, pred):
    """Compute correlation and MSE."""
    # Handle size mismatch by interpolating
    if ref.shape != pred.shape:
        from scipy.ndimage import zoom
        ratio = ref.shape[0] / pred.shape[0]
        pred = zoom(pred, ratio, order=1)
    corr = np.corrcoef(ref, pred)[0, 1]
    mse = np.mean((ref - pred) ** 2)
    return corr, mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="../app/src/main/assets/")
    parser.add_argument("--input_height", type=int, default=518)
    parser.add_argument("--input_width", type=int, default=518)
    args = parser.parse_args()

    input_np, ref = get_pytorch_reference(args.input_height, args.input_width)

    print(f"\nScanning models in {args.model_dir}...")
    results = []

    # ONNX models
    for f in sorted(glob.glob(os.path.join(args.model_dir, "*.onnx"))):
        pred = eval_onnx(f, input_np)
        if pred is not None:
            corr, mse = compute_metrics(ref, pred)
            results.append((os.path.basename(f), corr, mse))

    # TFLite models
    for f in sorted(glob.glob(os.path.join(args.model_dir, "*.tflite"))):
        pred = eval_tflite(f, input_np)
        if pred is not None:
            corr, mse = compute_metrics(ref, pred)
            results.append((os.path.basename(f), corr, mse))

    # Print results
    print("\n" + "=" * 70)
    print(f"{'Model':50s} {'Corr':>8s} {'MSE':>10s}")
    print("-" * 70)
    for name, corr, mse in sorted(results, key=lambda x: -x[1]):
        print(f"{name:50s} {corr:8.6f} {mse:10.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
