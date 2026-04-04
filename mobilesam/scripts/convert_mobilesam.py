#!/usr/bin/env python3
"""
Convert MobileSAM to TFLite (encoder) + ONNX (decoder) for Android.

Encoder: litert-torch (preserves ViT attention accuracy, NCHW layout)
Decoder: ONNX export via SamOnnxModel (ONNX Runtime on Android)

Requirements:
    pip install mobile_sam litert-torch onnxruntime torch

Usage:
    python convert_mobilesam.py --checkpoint mobile_sam.pt --output_dir output/
"""

import argparse
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidGELU(nn.Module):
    """GELU approximation using sigmoid: x * sigmoid(1.702 * x).
    Required because Erf is not a native TFLite op.
    Max error vs real GELU: ~0.01 (negligible for inference).
    """
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


def patch_gelu(module):
    """Replace all nn.GELU modules with SigmoidGELU."""
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, SigmoidGELU())
        else:
            patch_gelu(child)


def convert_encoder(model, output_dir, image_size=1024):
    """Convert MobileSAM encoder to TFLite via litert-torch."""
    import litert_torch

    encoder = model.image_encoder
    encoder.eval()
    patch_gelu(encoder)

    # Patch F.gelu globally for any functional calls
    original_gelu = F.gelu
    F.gelu = lambda input, approximate='none': input * torch.sigmoid(1.702 * input)

    dummy = torch.randn(1, 3, image_size, image_size)

    print(f"Converting encoder ({sum(p.numel() for p in encoder.parameters()):,} params)...")
    result = litert_torch.convert(encoder, (dummy,))

    out_path = os.path.join(output_dir, "mobilesam_encoder.tflite")
    data = result.export_flatbuffer()
    with open(out_path, "wb") as f:
        f.write(data)

    F.gelu = original_gelu

    print(f"Encoder saved: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    print(f"  Input:  [1, 3, {image_size}, {image_size}] NCHW float32")
    print(f"  Output: [1, 256, 64, 64] NCHW float32")
    return out_path


def convert_decoder(model, output_dir):
    """Convert MobileSAM decoder to ONNX via SamOnnxModel."""
    from mobile_sam.utils.onnx import SamOnnxModel

    patch_gelu(model)

    onnx_model = SamOnnxModel(model, return_single_mask=True)

    # Fixed 2-point input (foreground + background)
    dummy_inputs = {
        "image_embeddings": torch.randn(1, 256, 64, 64, dtype=torch.float),
        "point_coords": torch.randint(0, 1024, (1, 2, 2), dtype=torch.float),
        "point_labels": torch.randint(0, 4, (1, 2), dtype=torch.float),
        "mask_input": torch.randn(1, 1, 256, 256, dtype=torch.float),
        "has_mask_input": torch.tensor([1.0]),
        "orig_im_size": torch.tensor([1024.0, 1024.0]),
    }

    out_path = os.path.join(output_dir, "mobilesam_decoder.onnx")

    print(f"Exporting decoder ({sum(p.numel() for p in model.mask_decoder.parameters()):,} params)...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with torch.no_grad():
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                out_path,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=["masks", "iou_predictions", "low_res_masks"],
                dynamo=False,
            )

    print(f"Decoder saved: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    print(f"  Inputs:  image_embeddings [1,256,64,64] + point_coords [1,2,2] + ...")
    print(f"  Outputs: masks [1,1,1024,1024] + iou [1,1] + low_res [1,1,256,256]")
    return out_path


def verify(encoder_path, decoder_path, checkpoint):
    """Verify converted models against PyTorch reference."""
    import numpy as np

    print("\nVerifying encoder accuracy...")
    from mobile_sam import sam_model_registry
    model = sam_model_registry["vit_t"](checkpoint=checkpoint)
    model.eval()

    np.random.seed(42)
    img = np.random.randint(0, 255, (1024, 1024, 3)).astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img_nchw = ((img - mean) / std).transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    # PyTorch reference
    with torch.no_grad():
        pt_out = model.image_encoder(torch.from_numpy(img_nchw)).numpy()

    # TFLite
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=encoder_path)
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], img_nchw)
    interp.invoke()
    tf_out = interp.get_tensor(interp.get_output_details()[0]["index"])

    from scipy.stats import pearsonr
    corr, _ = pearsonr(tf_out.flatten(), pt_out.flatten())
    print(f"  Correlation: {corr:.6f} (target: >0.98)")

    # Decoder smoke test
    import onnxruntime as ort
    sess = ort.InferenceSession(decoder_path)
    masks, iou, _ = sess.run(None, {
        "image_embeddings": tf_out.astype(np.float32),
        "point_coords": np.array([[[512.0, 512.0], [0.0, 0.0]]], dtype=np.float32),
        "point_labels": np.array([[1.0, -1.0]], dtype=np.float32),
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.array([0.0], dtype=np.float32),
        "orig_im_size": np.array([1024.0, 1024.0], dtype=np.float32),
    })
    print(f"  Decoder IoU: {iou[0, 0]:.4f}, mask positive: {(masks > 0).mean() * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Convert MobileSAM for Android")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to mobile_sam.pt")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--verify", action="store_true", help="Run accuracy verification")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from mobile_sam import sam_model_registry
    model = sam_model_registry["vit_t"](checkpoint=args.checkpoint)
    model.eval()

    enc_path = convert_encoder(model, args.output_dir)
    dec_path = convert_decoder(model, args.output_dir)

    if args.verify:
        verify(enc_path, dec_path, args.checkpoint)

    print(f"\nDone! Copy to Android assets/:")
    print(f"  cp {enc_path} mobilesam/app/src/main/assets/")
    print(f"  cp {dec_path} mobilesam/app/src/main/assets/")


if __name__ == "__main__":
    main()
