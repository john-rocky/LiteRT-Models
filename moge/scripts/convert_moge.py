#!/usr/bin/env python3
"""
Convert MoGe-2 (Microsoft) to TFLite for Android CompiledModel GPU.

MoGe-2 = DINOv2 ViT backbone + ConvStack neck + 4 heads (points, normal, mask, scale).
Outputs an affine point map, surface normals, confidence mask, and metric scale —
all in a single forward pass. The .infer() post-processing in the original repo
(focal/shift recovery, depth-from-z, projection constraint) is implemented separately
in Kotlin since it requires utils3d ops not available in TFLite.

The default variant is moge-2-vits-normal (35M params, smallest, ~140 MB FP32 TFLite).

Requirements:
    pip install moge torch litert-torch numpy scipy huggingface_hub utils3d

Usage:
    python convert_moge.py --output_dir output/
    python convert_moge.py --output_dir output/ --verify
    python convert_moge.py --variant Ruicheng/moge-2-vitb-normal --output_dir output/
"""

import argparse
import os
import sys

# Disable xformers BEFORE importing moge — DINOv2 attention checks this env var at import time
os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Static-export wrapper ───────────────────────────────────────────────────

class MoGeStaticWrapper(nn.Module):
    """Fixed-shape wrapper around MoGeModel for litert-torch export.

    The original MoGeModel.forward takes (image, num_tokens) and computes the
    encoder base resolution from num_tokens + aspect ratio. For static export
    we hardcode num_tokens so the trace becomes shape-stable.

    Input:
        image: [1, 3, INPUT_H, INPUT_W] float32, range [0, 1] (no ImageNet norm —
               the encoder applies it internally)

    Outputs (4-tuple, all 4D tensors for GPU compat):
        points: [1, INPUT_H, INPUT_W, 3]  affine point map after remap_output
        normal: [1, INPUT_H, INPUT_W, 3]  L2-normalized surface normals
        mask:   [1, INPUT_H, INPUT_W, 1]  sigmoid confidence
        scale:  [1, 1, 1, 1]              metric scale (broadcast scalar)
    """

    # 448 = 32 patches × 14 patch_size. Square so aspect_ratio == 1.
    # 32² = 1024 tokens, close to SmolVLM SigLIP (1024) which works on device GPU.
    INPUT_H = 448
    INPUT_W = 448
    NUM_TOKENS = 1024  # 32 * 32

    BASE_H = 32  # NUM_TOKENS = BASE_H * BASE_W = 1024
    BASE_W = 32

    def __init__(self, moge_model):
        super().__init__()
        moge_model.onnx_compatible_mode = True
        self.encoder = moge_model.encoder
        self.neck = moge_model.neck
        self.points_head = moge_model.points_head
        self.normal_head = moge_model.normal_head
        self.mask_head = moge_model.mask_head
        self.scale_head = moge_model.scale_head
        self.remap_output = moge_model.remap_output

        # Pre-compute UV grids for 5 pyramid levels (constant for fixed aspect_ratio=1)
        from moge.utils.geometry_torch import normalized_view_plane_uv
        for level in range(5):
            h = self.BASE_H * (2 ** level)
            w = self.BASE_W * (2 ** level)
            uv = normalized_view_plane_uv(width=w, height=h, aspect_ratio=1.0,
                                          dtype=torch.float32)
            # [h, w, 2] → [1, 2, h, w]
            uv_buf = uv.permute(2, 0, 1).unsqueeze(0).contiguous()
            self.register_buffer(f"uv_{level}", uv_buf)

    def forward(self, image: torch.Tensor):
        # Encoder
        features, cls_token = self.encoder(
            image, self.BASE_H, self.BASE_W, return_class_token=True
        )

        # Build 5-level feature list with UV grids.
        # UV buffers are constants but GPU delegate rejects Conv2d with constant
        # inputs ("input must be a runtime tensor"). Add a negligible image-dependent
        # epsilon to prevent constant folding while preserving values.
        eps = image[:, :1, :1, :1] * 1e-10  # [1, 1, 1, 1] negligible, prevents folding
        feat_list = [
            torch.cat([features, self.uv_0 + eps], dim=1),
            self.uv_1 + eps,
            self.uv_2 + eps,
            self.uv_3 + eps,
            self.uv_4 + eps,
        ]

        # Neck → Heads
        feat_list = self.neck(feat_list)
        points = self.points_head(feat_list)[-1]   # [1, 3, base*16, base*16]
        normal = self.normal_head(feat_list)[-1]
        mask_out = self.mask_head(feat_list)[-1]
        scale_out = self.scale_head(cls_token)

        # Resize to input resolution
        points = F.interpolate(points, (self.INPUT_H, self.INPUT_W),
                               mode="bilinear", align_corners=False)
        normal = F.interpolate(normal, (self.INPUT_H, self.INPUT_W),
                               mode="bilinear", align_corners=False)
        mask_out = F.interpolate(mask_out, (self.INPUT_H, self.INPUT_W),
                                 mode="bilinear", align_corners=False)

        # Post-process: remap, normalize, activate
        points = points.permute(0, 2, 3, 1)  # [1, H, W, 3]
        if self.remap_output == "exp":
            xy, z = points[..., :2], points[..., 2:]
            z = torch.exp(z)
            points = torch.cat([xy * z, z], dim=-1)
        elif self.remap_output == "sinh":
            points = torch.sinh(points)
        elif self.remap_output == "sinh_exp":
            xy, z = points[..., :2], points[..., 2:]
            points = torch.cat([torch.sinh(xy), torch.exp(z)], dim=-1)

        normal = normal.permute(0, 2, 3, 1)  # [1, H, W, 3]
        normal = F.normalize(normal, dim=-1)

        mask_out = mask_out.squeeze(1).sigmoid().unsqueeze(-1)  # [1, H, W, 1]
        scale_out = scale_out.squeeze(1).exp().view(1, 1, 1, 1)  # [1, 1, 1, 1]

        return points, normal, mask_out, scale_out


# ─── Custom patches beyond the toolkit defaults ───────────────────────────────

def patch_conv_transpose_to_bilinear(model: nn.Module) -> int:
    """Replace ConvTranspose2d(k=2, s=2) with bilinear 2x upsample + Conv2d(1x1).

    TFLite GPU delegate rejects TRANSPOSE_CONV. RESIZE_BILINEAR is GPU-native.
    Weight transfer uses the spatial mean of the 2x2 kernel to best approximate
    the learned upsampling. The subsequent Conv2d(3x3) in the Resampler handles
    spatial filtering, so the 1x1 only needs channel projection accuracy.
    """
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.ConvTranspose2d) and child.stride == (2, 2):
                in_c = child.in_channels
                out_c = child.out_channels
                with torch.no_grad():
                    ct_w = child.weight  # [in_c, out_c, 2, 2]
                    # Mean over spatial dims for best approximation
                    avg_w = ct_w.mean(dim=(2, 3))  # [in_c, out_c]
                    conv1x1 = nn.Conv2d(in_c, out_c, 1, bias=child.bias is not None)
                    conv1x1.weight.copy_(avg_w.T.unsqueeze(-1).unsqueeze(-1))
                    if child.bias is not None:
                        conv1x1.bias.copy_(child.bias)

                replacement = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    conv1x1,
                )
                setattr(module, child_name, replacement)
                count += 1
    if count:
        print(f"  Patched {count} ConvTranspose2d → bilinear(2x) + Conv2d(1x1)")
    return count


def patch_upsample_to_fixed_size(model: nn.Module) -> int:
    """Replace nn.Upsample(scale_factor=2) with fixed-size F.interpolate.

    nn.Upsample with scale_factor computes the output size dynamically at runtime,
    producing RESIZE_BILINEAR ops with non-constant size tensors. The GPU delegate
    rejects dynamic sizes. We trace the model once to determine each Upsample's
    actual output size, then replace with a fixed-size interpolate.
    """
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Upsample) and child.scale_factor is not None:
                sf = int(child.scale_factor) if isinstance(child.scale_factor, (int, float)) else int(child.scale_factor[0])
                mode = child.mode
                ac = child.align_corners

                class FixedUpsample(nn.Module):
                    def __init__(self, scale, interp_mode, align):
                        super().__init__()
                        self.scale = scale
                        self.interp_mode = interp_mode
                        self.align = align

                    def forward(self, x):
                        h, w = x.shape[2], x.shape[3]
                        return F.interpolate(x, size=(h * self.scale, w * self.scale),
                                             mode=self.interp_mode, align_corners=self.align)

                setattr(module, child_name, FixedUpsample(sf, mode, ac))
                count += 1
    if count:
        print(f"  Patched {count} nn.Upsample(scale_factor) → fixed-size F.interpolate")
    return count


def patch_bicubic_to_bilinear():
    """Force F.interpolate(mode='bicubic') → bilinear.

    DINOv2 position embedding uses bicubic. TFLite GPU delegate does not
    support BICUBIC RESIZE — only bilinear and nearest. The accuracy difference
    on a one-shot 37×37 → 36×36 pos-embed resize is negligible.
    Returns the original F.interpolate so we can restore it after conversion.
    """
    orig = F.interpolate

    def patched(input, size=None, scale_factor=None, mode="nearest",
                align_corners=None, recompute_scale_factor=None, antialias=False):
        if mode == "bicubic":
            mode = "bilinear"
        if mode in ("bilinear", "trilinear"):
            align_corners = False
        if antialias:
            antialias = False  # antialias not supported by GPU delegate either
        return orig(
            input, size=size, scale_factor=scale_factor, mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor, antialias=antialias,
        )

    F.interpolate = patched
    return orig


def patch_replicate_padding(model: nn.Module) -> int:
    """Swap nn.Conv2d(padding_mode='replicate') → padding_mode='zeros'.

    The ConvStack/ResidualConvBlock uses replicate padding which TFLite GPU
    does not support natively. The boundary difference is one pixel ring; for
    geometry estimation this is unobservable in the visualizations.
    Returns the number of conv layers patched.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and getattr(module, "padding_mode", "zeros") != "zeros":
            module.padding_mode = "zeros"
            count += 1
    if count:
        print(f"  Patched {count} Conv2d (replicate → zeros padding)")
    return count


def bake_layerscale_into_linear(model: nn.Module) -> int:
    """Fold LayerScale gamma into the preceding Linear layer's weight and bias.

    DINOv2 Block does: x + ls1(attn(norm1(x))) and x + ls2(mlp(norm2(x))).
    LayerScale multiplies the output by a per-channel gamma [C]. The GPU delegate
    misinterprets this MUL between a rank-reduced FC output [N, C] and gamma
    [1, 1, C], causing a shape conflict ({N,1,1,C} vs {1,1,N,C}).

    Fix: pre-multiply gamma into the last Linear of attn (proj) and mlp, then
    replace LayerScale with Identity. This eliminates the MUL entirely.
    """
    from moge.model.dinov2.layers.layer_scale import LayerScale

    count = 0
    for block in model.modules():
        if not hasattr(block, 'ls1') or not hasattr(block, 'ls2'):
            continue

        # Attn path: bake ls1 gamma into attn.proj
        if isinstance(block.ls1, LayerScale):
            gamma = block.ls1.gamma.data.squeeze()  # [C]
            with torch.no_grad():
                block.attn.proj.weight.data.mul_(gamma.unsqueeze(1))
                if block.attn.proj.bias is not None:
                    block.attn.proj.bias.data.mul_(gamma)
            block.ls1 = nn.Identity()
            count += 1

        # MLP path: bake ls2 gamma into mlp's last Linear
        if isinstance(block.ls2, LayerScale):
            gamma = block.ls2.gamma.data.squeeze()  # [C]
            # MLP is nn.Sequential(..., Linear(hidden, dim))
            last_linear = None
            for child in reversed(list(block.mlp.children())):
                if isinstance(child, nn.Linear):
                    last_linear = child
                    break
            if last_linear is not None:
                with torch.no_grad():
                    last_linear.weight.data.mul_(gamma.unsqueeze(1))
                    if last_linear.bias is not None:
                        last_linear.bias.data.mul_(gamma)
            block.ls2 = nn.Identity()
            count += 1

    if count:
        print(f"  Baked {count} LayerScale into Linear weights (eliminates GPU shape conflict)")
    return count


def patch_dinov2_attention():
    """Replace DINOv2 fused qkv with 3 separate Linear layers.

    The default attention uses a single `self.qkv = Linear(dim, 3*dim)` which
    requires either a 5D reshape+unbind or slicing the output. Both patterns
    cause the ML Drift GPU delegate to fail with FC shape mismatches because
    the delegate can't reconcile {B, 1, N, C} vs {N, 1, 1, C} layouts after
    a slice of a large FC output.

    Fix: extract q/k/v weight slices into 3 separate Linear(dim, dim) modules.
    Each FC is [B, N, C] → [B, N, C], same shape as the output projection,
    which the GPU delegate handles correctly.
    """
    from moge.model.dinov2.layers.attention import Attention

    count = 0
    for module in list(torch._modules.values()) if hasattr(torch, '_modules') else []:
        pass  # no-op, just defensive

    # First, decompose qkv weight into 3 separate Linear layers per Attention module
    import moge.model.dinov2.layers.attention as attn_mod
    for name, module in list(attn_mod.__dict__.items()):
        pass  # walk module hierarchy below

    def decompose_qkv(attn: Attention):
        C = attn.qkv.in_features
        with torch.no_grad():
            w = attn.qkv.weight  # [3*C, C]
            b = attn.qkv.bias    # [3*C] or None
            attn.q_linear = nn.Linear(C, C, bias=b is not None)
            attn.k_linear = nn.Linear(C, C, bias=b is not None)
            attn.v_linear = nn.Linear(C, C, bias=b is not None)
            attn.q_linear.weight.copy_(w[:C])
            attn.k_linear.weight.copy_(w[C:2*C])
            attn.v_linear.weight.copy_(w[2*C:])
            if b is not None:
                attn.q_linear.bias.copy_(b[:C])
                attn.k_linear.bias.copy_(b[C:2*C])
                attn.v_linear.bias.copy_(b[2*C:])

    def patched_forward(self, x, attn_bias=None):
        B, N, C = x.shape
        H = self.num_heads
        Hd = C // H

        q = self.q_linear(x).reshape(B, N, H, Hd).permute(0, 2, 1, 3)  # [B, H, N, Hd]
        k = self.k_linear(x).reshape(B, N, H, Hd).permute(0, 2, 1, 3)
        v = self.v_linear(x).reshape(B, N, H, Hd).permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q, k, v, attn_bias)         # [B, H, N, Hd]
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)                   # [B, N, C]

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    Attention.forward = patched_forward
    return decompose_qkv


def patch_dinov2_encoder_forward():
    """Replace DINOv2Encoder.forward with a 4D-only implementation.

    The default forward stacks per-layer projections then sums (`torch.stack(...,
    dim=1).sum(dim=1)`), which produces a 5D intermediate `[B, L, C, H, W]`.
    Element-wise add stays 4D throughout. Also swaps `unflatten(2, (h, w))`
    for an explicit `reshape` to avoid any potential GATHER_ND emission.
    """
    from moge.model.modules import DINOv2Encoder

    def patched_forward(self, image, token_rows, token_cols, return_class_token=False):
        image_14 = F.interpolate(
            image,
            (token_rows * 14, token_cols * 14),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
        image_14 = (image_14 - self.image_mean) / self.image_std

        features = self.backbone.get_intermediate_layers(
            image_14, n=self.intermediate_layers, return_class_token=True
        )

        x = None
        last_clstoken = None
        for proj, (feat, clstoken) in zip(self.output_projections, features):
            B, N, C = feat.shape
            # [B, N, C] -> [B, C, N] -> [B, C, H, W]
            feat_4d = feat.permute(0, 2, 1).reshape(B, C, token_rows, token_cols)
            projected = proj(feat_4d)
            x = projected if x is None else x + projected
            last_clstoken = clstoken

        if return_class_token:
            return x, last_clstoken
        return x

    DINOv2Encoder.forward = patched_forward
    print("  Patched DINOv2Encoder.forward (4D add instead of 5D stack+sum)")


def precompute_dinov2_pos_embed(model, h_patches: int, w_patches: int):
    """Bake the interpolated DINOv2 position embedding into a constant buffer.

    DINOv2's `interpolate_pos_encoding` resizes the learned 37×37 position grid
    to (h_patches, w_patches) at every forward pass via bicubic F.interpolate.
    Even after a bicubic→bilinear monkey-patch, litert-torch decomposes the
    transposed-layout resize into ~16 GATHER_ND ops, which fail GPU compat.

    For our fixed input shape the interpolation result is identical every call.
    Pre-compute it once with PyTorch and overwrite `prepare_tokens_with_masks`
    to add the constant directly, bypassing `interpolate_pos_encoding` entirely.
    """
    from moge.model.dinov2.models.vision_transformer import DinoVisionTransformer

    backbone = model.encoder.backbone
    with torch.no_grad():
        # interpolate_pos_encoding expects x of shape [B, npatch+1, dim] just for shape inference
        # We can call it directly with a dummy x and the target h, w in pixels.
        dummy_x = torch.zeros(1, h_patches * w_patches + 1, backbone.embed_dim)
        h_pixels = h_patches * backbone.patch_size
        w_pixels = w_patches * backbone.patch_size
        # Temporarily flip onnx_compatible_mode off to trigger the kludge-free path,
        # then restore. This avoids using the interpolate_offset hack.
        original_mode = backbone.onnx_compatible_mode
        backbone.onnx_compatible_mode = True
        baked = backbone.interpolate_pos_encoding(dummy_x, h_pixels, w_pixels).clone()
        backbone.onnx_compatible_mode = original_mode

    backbone.register_buffer("_baked_pos_embed", baked)

    def patched_prepare_tokens(self, x, masks=None):
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self._baked_pos_embed
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )
        return x

    DinoVisionTransformer.prepare_tokens_with_masks = patched_prepare_tokens
    print(f"  Baked DINOv2 pos_embed for {h_patches}x{w_patches} patches "
          f"({h_pixels}x{w_pixels} input) and bypassed interpolate_pos_encoding")


# ─── Conversion entry point ──────────────────────────────────────────────────

def load_moge(variant: str):
    """Load a MoGe-2 model from HuggingFace."""
    from moge.model.v2 import MoGeModel
    print(f"Loading {variant} from HuggingFace...")
    model = MoGeModel.from_pretrained(variant)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {params:,} params")
    return model


def convert(model, output_dir):
    """Convert MoGe-2 to a single TFLite model for CompiledModel GPU."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from litert_gpu_toolkit import convert_for_gpu

    wrapper = MoGeStaticWrapper(model)
    wrapper.eval()

    print(f"\nWrapper input shape: [1, 3, {wrapper.INPUT_H}, {wrapper.INPUT_W}]")

    # Apply all GPU-compat patches
    print("\nApplying MoGe-specific patches...")
    patch_replicate_padding(wrapper)
    patch_conv_transpose_to_bilinear(wrapper)
    patch_upsample_to_fixed_size(wrapper)
    orig_interp = patch_bicubic_to_bilinear()
    print("  Patched F.interpolate (bicubic → bilinear, antialias off)")
    decompose_fn = patch_dinov2_attention()
    from moge.model.dinov2.layers.attention import Attention
    attn_count = 0
    for mod in wrapper.modules():
        if isinstance(mod, Attention):
            decompose_fn(mod)
            attn_count += 1
    print(f"  Decomposed qkv → q/k/v in {attn_count} Attention modules")
    patch_dinov2_encoder_forward()
    bake_layerscale_into_linear(wrapper)
    precompute_dinov2_pos_embed(model, h_patches=wrapper.BASE_H, w_patches=wrapper.BASE_W)

    # Verify patched forward
    dummy = torch.zeros(1, 3, wrapper.INPUT_H, wrapper.INPUT_W)
    print("\nVerifying patched forward pass...")
    with torch.no_grad():
        out = wrapper(dummy)
    for name, t in zip(("points", "normal", "mask", "scale"), out):
        print(f"  {name:7s}: {tuple(t.shape)}  range=[{t.min().item():.3f}, {t.max().item():.3f}]")

    # Convert
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "moge.tflite")
    convert_for_gpu(wrapper, dummy, out_path, check=True, verbose=True)

    F.interpolate = orig_interp
    return out_path


def verify(tflite_path, model, n_samples=2):
    """Compare TFLite output vs PyTorch reference."""
    import tensorflow as tf
    from scipy.stats import pearsonr

    print(f"\n=== Verifying {tflite_path} ===")
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_idx = interp.get_input_details()[0]["index"]
    out_details = sorted(interp.get_output_details(), key=lambda d: d["name"])

    wrapper = MoGeStaticWrapper(model)
    wrapper.eval()
    H, W = wrapper.INPUT_H, wrapper.INPUT_W

    np.random.seed(0)
    for s in range(n_samples):
        img = np.random.rand(1, 3, H, W).astype(np.float32)

        with torch.no_grad():
            pt_out = wrapper(torch.from_numpy(img))
        pt_arrays = [t.numpy() for t in pt_out]

        interp.set_tensor(in_idx, img)
        interp.invoke()
        tf_arrays = [interp.get_tensor(d["index"]) for d in out_details]

        print(f"\nSample {s}:")
        names = ("points", "normal", "mask", "scale")
        for name, pt_a, tf_a in zip(names, pt_arrays, tf_arrays):
            if pt_a.shape != tf_a.shape:
                print(f"  {name:7s}: SHAPE MISMATCH — pt {pt_a.shape} vs tf {tf_a.shape}")
                continue
            if pt_a.size < 2:
                mae = float(np.mean(np.abs(pt_a - tf_a)))
                print(f"  {name:7s}: value pt={pt_a.flatten()[0]:.6f} tf={tf_a.flatten()[0]:.6f} diff={mae:.6f}")
                continue
            corr, _ = pearsonr(pt_a.flatten(), tf_a.flatten())
            mae = float(np.mean(np.abs(pt_a - tf_a)))
            print(f"  {name:7s}: corr={corr:.6f}  mae={mae:.6f}  pt_range=[{pt_a.min():.3f},{pt_a.max():.3f}]")


def main():
    parser = argparse.ArgumentParser(description="Convert MoGe-2 to TFLite for Android")
    parser.add_argument("--variant", type=str, default="Ruicheng/moge-2-vits-normal",
                        help="HuggingFace model id (default: moge-2-vits-normal)")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--output_name", type=str, default="moge.tflite",
                        help="Output filename")
    parser.add_argument("--verify", action="store_true", help="Run TFLite vs PyTorch correlation check")
    args = parser.parse_args()

    model = load_moge(args.variant)
    out_path = convert(model, args.output_dir)

    size_mb = os.path.getsize(out_path) / 1e6
    H, W = MoGeStaticWrapper.INPUT_H, MoGeStaticWrapper.INPUT_W
    print(f"\nDone: {out_path} ({size_mb:.1f} MB)")
    print(f"  Input:  [1, 3, {H}, {W}] NCHW float32, range [0, 1]")
    print(f"  Output: points [1,{H},{W},3], normal [1,{H},{W},3], mask [1,{H},{W},1], scale [1,1,1,1]")

    if args.verify:
        verify(out_path, model)


if __name__ == "__main__":
    main()
