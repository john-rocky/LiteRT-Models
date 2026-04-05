#!/usr/bin/env python3
"""
Convert DSINE surface normal estimator to TFLite for Android.

Converts the encoder + decoder (initial prediction) only — skips the ConvGRU
iterative refinement which uses ops unsupported by TFLite (boolean masking,
axis-angle rotation, 7D tensors).

The initial prediction is still high quality — refinement adds ~2-3° improvement.

Requirements:
    pip install litert-torch geffnet torch numpy

Usage:
    python convert_dsine.py --output_dir output/
    python convert_dsine.py --output_dir output/ --verify
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add DSINE repo to path
DSINE_REPO = os.path.expanduser("~/.cache/torch/hub/baegwangbin_DSINE_main")
if not os.path.exists(DSINE_REPO):
    print("DSINE repo not found. Downloading...")
    torch.hub.load("baegwangbin/DSINE", "DSINE", trust_repo=True)
sys.path.insert(0, DSINE_REPO)


# DSINE model config (from exp001_cvpr2024/dsine.txt)
class DSINEConfig:
    NNET_encoder_B = 5
    NNET_decoder_NF = 2048
    NNET_decoder_BN = False
    NNET_decoder_down = 8
    NNET_learned_upsampling = True
    NNET_output_dim = 3
    NNET_feature_dim = 64
    NNET_hidden_dim = 64
    NRN_prop_ps = 5
    NRN_num_iter_train = 5
    NRN_num_iter_test = 5
    NRN_ray_relu = True


def load_dsine():
    """Load DSINE v02 model with pretrained weights."""
    from models.dsine.v02 import DSINE_v02

    args = DSINEConfig()
    model = DSINE_v02(args)

    ckpt_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/dsine.pt")
    if not os.path.exists(ckpt_path):
        url = "https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt"
        state_dict = torch.hub.load_state_dict_from_url(url, file_name="dsine.pt", map_location="cpu")
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(state_dict["model"], strict=True)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"DSINE loaded: {params:,} params")
    return model


def get_intrins_from_fov(fov_deg, H, W):
    """Compute camera intrinsics matrix from field of view."""
    fov_rad = math.radians(fov_deg)
    fx = W / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx
    cx = W / 2.0
    cy = H / 2.0
    intrins = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    return intrins


class DSINEInitialWrapper(nn.Module):
    """Wrapper that runs DSINE encoder + decoder initial prediction only.

    Skips iterative refinement (ConvGRU) for TFLite compatibility.
    Uses bilinear upsampling instead of learned convex upsampling
    (which requires F.unfold with 7D tensors).

    Input:  [1, 3, H, W] normalized image (ImageNet stats)
    Output: [1, 3, H, W] surface normal map (unit vectors, [-1, 1])
    """

    def __init__(self, model, H, W, fov=60.0):
        super().__init__()
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.down = model.downsample_ratio

        # Pre-compute UV ray encodings as constant buffers
        intrins = get_intrins_from_fov(fov, H, W).unsqueeze(0)
        # Add 0.5 offset (as done in DSINE forward)
        intrins[:, 0, 2] += 0.5
        intrins[:, 1, 2] += 0.5

        self.register_buffer("uv_32", self._compute_uv(intrins, H // 32, W // 32, H, W))
        self.register_buffer("uv_16", self._compute_uv(intrins, H // 16, W // 16, H, W))
        self.register_buffer("uv_8", self._compute_uv(intrins, H // 8, W // 8, H, W))

    def _compute_uv(self, intrins, h, w, orig_H, orig_W):
        """Compute UV ray direction encoding at given resolution."""
        fu = intrins[:, 0, 0].unsqueeze(-1).unsqueeze(-1) * (w / orig_W)
        cu = intrins[:, 0, 2].unsqueeze(-1).unsqueeze(-1) * (w / orig_W)
        fv = intrins[:, 1, 1].unsqueeze(-1).unsqueeze(-1) * (h / orig_H)
        cv = intrins[:, 1, 2].unsqueeze(-1).unsqueeze(-1) * (h / orig_H)

        # Pixel coordinate grid
        pixel_y, pixel_x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32) + 0.5,
            torch.arange(w, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        pixel_x = pixel_x.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
        pixel_y = pixel_y.unsqueeze(0).unsqueeze(0)

        u = (pixel_x - cu) / fu
        v = (pixel_y - cv) / fv
        uv = torch.cat([u, v], dim=1)  # (1, 2, h, w)
        return uv

    def forward(self, img):
        # Encoder
        features = self.encoder(img)

        # Decoder — initial prediction at 1/8 resolution
        pred_norm, _, _ = self.decoder(features, (self.uv_32, self.uv_16, self.uv_8))

        # Bilinear upsample to full resolution
        pred_norm = F.interpolate(
            pred_norm,
            scale_factor=self.down,
            mode="bilinear",
            align_corners=False,
        )
        pred_norm = F.normalize(pred_norm, dim=1)
        return pred_norm


def patch_swish(module):
    """Replace Swish/SiLU with x * sigmoid(x) for TFLite compatibility."""
    for name, child in module.named_children():
        if isinstance(child, (nn.SiLU,)):
            setattr(module, name, SigmoidSwish())
        elif hasattr(child, '__class__') and child.__class__.__name__ in ('Swish', 'SwishMe'):
            setattr(module, name, SigmoidSwish())
        else:
            patch_swish(child)


class SigmoidSwish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def bake_weight_standardization(module):
    """Replace Conv2d_WS with regular Conv2d by pre-computing standardized weights.
    Conv2d_WS does weight standardization at runtime, which causes TFLite issues.
    """
    from models.conv_encoder_decoder.submodules import Conv2d_WS

    for name, child in list(module.named_children()):
        if isinstance(child, Conv2d_WS):
            # Pre-compute standardized weight
            with torch.no_grad():
                weight = child.weight.clone()
                weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                weight = weight - weight_mean
                std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
                weight = weight / std.expand_as(weight)

            # Create regular Conv2d with baked weights
            conv = nn.Conv2d(
                child.in_channels, child.out_channels, child.kernel_size,
                stride=child.stride, padding=child.padding,
                dilation=child.dilation, groups=child.groups,
                bias=child.bias is not None,
            )
            conv.weight = nn.Parameter(weight)
            if child.bias is not None:
                conv.bias = nn.Parameter(child.bias.clone())
            setattr(module, name, conv)
        else:
            bake_weight_standardization(child)


def patch_normalize(module):
    """Replace F.normalize calls in the decoder with TFLite-safe version."""
    pass  # We handle this by monkey-patching F.normalize globally


class BakedGroupNorm(nn.Module):
    """GPU-friendly GroupNorm replacement using only 4D ops.
    TFLite GPU delegate doesn't support nn.GroupNorm or 5D tensors,
    so we reshape to (B*G, C//G, H, W) — stays 4D throughout.
    """
    def __init__(self, gn: nn.GroupNorm):
        super().__init__()
        self.num_groups = gn.num_groups
        self.num_channels = gn.num_channels
        self.eps = gn.eps
        self.weight = nn.Parameter(gn.weight.clone()) if gn.weight is not None else None
        self.bias = nn.Parameter(gn.bias.clone()) if gn.bias is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
        G = self.num_groups
        # Stay 4D: (B*G, C//G, H, W)
        x = x.reshape(B * G, C // G, H, W)
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = ((x - mean) * (x - mean)).mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.reshape(B, C, H, W)
        if self.weight is not None:
            x = x * self.weight.reshape(1, C, 1, 1)
        if self.bias is not None:
            x = x + self.bias.reshape(1, C, 1, 1)
        return x


def replace_groupnorm(module):
    """Replace all nn.GroupNorm with GPU-friendly BakedGroupNorm."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, BakedGroupNorm(child))
        else:
            replace_groupnorm(child)


def convert(model, output_dir, H=480, W=640):
    """Convert DSINE to TFLite via litert-torch."""
    import litert_torch

    wrapper = DSINEInitialWrapper(model, H=H, W=W, fov=60.0)
    wrapper.eval()

    # Patch any Swish activations in EfficientNet
    patch_swish(wrapper)

    # Bake weight standardization into regular Conv2d
    bake_weight_standardization(wrapper)

    # Replace GroupNorm with GPU-friendly implementation
    replace_groupnorm(wrapper)

    # Patch F.interpolate to avoid align_corners GPU issue
    _orig_interp = F.interpolate
    def safe_interpolate(inp, size=None, scale_factor=None, mode='nearest', align_corners=None, **kw):
        if mode in ('bilinear', 'bicubic') and align_corners is True:
            align_corners = False
        return _orig_interp(inp, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, **kw)
    F.interpolate = safe_interpolate

    # Patch F.normalize to avoid TFLite div broadcast issue
    _orig_normalize = F.normalize
    def safe_normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
        norm = torch.sqrt(torch.sum(input * input, dim=dim, keepdim=True).clamp(min=eps * eps))
        return input / norm
    F.normalize = safe_normalize

    dummy = torch.randn(1, 3, H, W)

    print(f"Testing forward pass ({H}x{W})...")
    with torch.no_grad():
        out = wrapper(dummy)
        print(f"  Output: {out.shape}, range: [{out.min():.3f}, {out.max():.3f}]")
        print(f"  Norm check (should be ~1.0): {out.norm(dim=1).mean():.4f}")

    print("Converting with litert-torch...")
    result = litert_torch.convert(wrapper, (dummy,))

    out_path = os.path.join(output_dir, "dsine.tflite")
    result.export(out_path)

    F.interpolate = _orig_interp
    F.normalize = _orig_normalize

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Saved: {out_path} ({size_mb:.1f} MB)")
    print(f"  Input:  [1, 3, {H}, {W}] NCHW float32 (ImageNet normalized)")
    print(f"  Output: [1, 3, {H}, {W}] surface normals (unit vectors)")
    return out_path


def verify(tflite_path, model, H=480, W=640):
    """Verify TFLite output against PyTorch reference."""
    print("\nVerifying accuracy...")

    wrapper = DSINEInitialWrapper(model, H=H, W=W, fov=60.0)
    wrapper.eval()

    np.random.seed(42)
    img = np.random.randn(1, 3, H, W).astype(np.float32)

    # PyTorch reference
    with torch.no_grad():
        pt_out = wrapper(torch.from_numpy(img)).numpy()

    # TFLite
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], img)
    interp.invoke()
    tf_out = interp.get_tensor(interp.get_output_details()[0]["index"])

    from scipy.stats import pearsonr
    corr, _ = pearsonr(tf_out.flatten(), pt_out.flatten())

    # Angular error between normal vectors
    dot = np.sum(pt_out * tf_out, axis=1).clip(-1, 1)
    angle_error = np.degrees(np.arccos(dot))
    mean_angle = angle_error.mean()

    print(f"  Correlation: {corr:.6f} (target: >0.98)")
    print(f"  Mean angular error: {mean_angle:.2f}° (target: <1°)")


def main():
    parser = argparse.ArgumentParser(description="Convert DSINE to TFLite")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_dsine()
    path = convert(model, args.output_dir, H=args.height, W=args.width)

    if args.verify:
        verify(path, model, H=args.height, W=args.width)

    print(f"\nDone! Copy to Android assets/:")
    print(f"  cp {path} dsine/app/src/main/assets/")


if __name__ == "__main__":
    main()
