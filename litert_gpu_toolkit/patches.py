"""
Auto-patching pipeline for GPU-incompatible PyTorch patterns.

Each patch function returns the number of modifications made.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

log = logging.getLogger("litert_gpu_toolkit")

# ─── GELU → Sigmoid approximation ────────────────────────────────────────────

class SigmoidGELU(nn.Module):
    """GELU approximation: x * sigmoid(1.702 * x). Max error ~0.01."""
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


_original_gelu = F.gelu


def patch_gelu(model: nn.Module) -> int:
    """Replace all nn.GELU modules with SigmoidGELU.
    Also monkey-patches F.gelu globally.
    Returns number of modules replaced.
    """
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.GELU):
                setattr(module, child_name, SigmoidGELU())
                count += 1

    F.gelu = lambda x, approximate='none': x * torch.sigmoid(1.702 * x)

    if count:
        log.info(f"Patched {count} nn.GELU → SigmoidGELU")
    return count


def restore_gelu():
    """Restore original F.gelu."""
    F.gelu = _original_gelu


# ─── DeformableConv2d → Conv2d ────────────────────────────────────────────────

def patch_deformable_conv(model: nn.Module) -> int:
    """Replace DeformableConv2d with its internal regular_conv.
    Expects the module to have a `regular_conv` child (common pattern).
    Returns number of modules replaced.
    """
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if type(child).__name__ == 'DeformableConv2d':
                if not hasattr(child, 'regular_conv'):
                    log.warning(f"DeformableConv2d at {name}.{child_name} has no regular_conv — skipped")
                    continue
                conv = child.regular_conv
                ks = conv.weight.shape[2]
                stride = child.stride[0] if hasattr(child, 'stride') else 1
                new_conv = nn.Conv2d(
                    conv.in_channels, conv.out_channels,
                    kernel_size=ks, stride=stride, padding=ks // 2,
                    bias=conv.bias is not None,
                )
                new_conv.weight.data.copy_(conv.weight.data)
                if conv.bias is not None:
                    new_conv.bias.data.copy_(conv.bias.data)
                setattr(module, child_name, new_conv)
                count += 1

    if count:
        log.info(f"Patched {count} DeformableConv2d → Conv2d")
    return count


# ─── F.interpolate align_corners fix ─────────────────────────────────────────

_original_interpolate = F.interpolate


def patch_interpolate() -> None:
    """Force align_corners=False for bilinear/bicubic interpolation.
    GPU rejects half_pixel_centers=True + align_corners=True.
    """
    def safe_interpolate(input, size=None, scale_factor=None, mode='nearest',
                         align_corners=None, recompute_scale_factor=None, antialias=False):
        if mode in ('bilinear', 'bicubic', 'trilinear'):
            align_corners = False
        return _original_interpolate(
            input, size=size, scale_factor=scale_factor, mode=mode,
            align_corners=align_corners, recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )

    F.interpolate = safe_interpolate
    log.info("Patched F.interpolate → align_corners=False for bilinear/bicubic")


def restore_interpolate():
    """Restore original F.interpolate."""
    F.interpolate = _original_interpolate


# ─── WindowAttention relative position bias pre-compute ──────────────────────

def patch_window_attention(model: nn.Module) -> int:
    """Pre-compute relative_position_bias to eliminate GATHER_ND.
    Detects modules with relative_position_bias_table + relative_position_index.
    Returns number of modules patched.
    """
    count = 0
    wa_class = None

    for name, module in model.named_modules():
        if (hasattr(module, 'relative_position_bias_table') and
            hasattr(module, 'relative_position_index') and
            hasattr(module, 'num_heads') and
            hasattr(module, 'window_size')):

            table = module.relative_position_bias_table
            index = module.relative_position_index.view(-1)
            ws = module.window_size[0] if isinstance(module.window_size, (list, tuple)) else module.window_size
            nH = module.num_heads

            bias = table[index].view(ws * ws, ws * ws, nH).permute(2, 0, 1).contiguous()
            module.register_buffer('_precomputed_pos_bias', bias.unsqueeze(0))

            if wa_class is None:
                wa_class = type(module)
            count += 1

    if wa_class is not None and count > 0:
        _orig_forward = wa_class.forward

        def patched_wa_forward(self, x, mask=None):
            B_, N, C = x.shape
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn + self._precomputed_pos_bias
            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        wa_class.forward = patched_wa_forward
        log.info(f"Patched {count} WindowAttention modules (pre-computed position bias)")

    return count


# ─── PatchMerging → pixel_unshuffle ──────────────────────────────────────────

def patch_patch_merging(model: nn.Module) -> int:
    """Replace PatchMerging stride-2 slicing with pixel_unshuffle + weight permutation.
    Eliminates GATHER_ND from spatial downsampling.
    Returns number of modules patched.
    """
    count = 0
    pm_class = None

    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if type(child).__name__ == 'PatchMerging' and hasattr(child, 'reduction'):
                C = child.reduction.in_features // 4

                # Build permutation: pixel_unshuffle order → PatchMerging cat order
                perm = torch.zeros(4 * C, dtype=torch.long)
                for c in range(C):
                    perm[c * 4 + 0] = c            # even_h, even_w → x0
                    perm[c * 4 + 1] = 2 * C + c    # even_h, odd_w → x2
                    perm[c * 4 + 2] = C + c         # odd_h, even_w → x1
                    perm[c * 4 + 3] = 3 * C + c     # odd_h, odd_w → x3

                child.norm.weight.data = child.norm.weight.data[perm].clone()
                child.norm.bias.data = child.norm.bias.data[perm].clone()
                child.reduction.weight.data = child.reduction.weight.data[:, perm].clone()

                if pm_class is None:
                    pm_class = type(child)
                count += 1

    if pm_class is not None and count > 0:
        def patched_pm_forward(self, x, H, W):
            B, L, C = x.shape
            x = x.view(B, H, W, C)
            if (H % 2 == 1) or (W % 2 == 1):
                x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
                _, H, W, _ = x.shape
            x_nchw = x.permute(0, 3, 1, 2)
            x_ps = F.pixel_unshuffle(x_nchw, 2)
            x = x_ps.permute(0, 2, 3, 1).reshape(B, -1, 4 * C)
            return self.reduction(self.norm(x))

        pm_class.forward = patched_pm_forward
        log.info(f"Patched {count} PatchMerging → pixel_unshuffle + weight permutation")

    return count


# ─── einops.rearrange replacement ─────────────────────────────────────────────

def patch_einops() -> None:
    """Replace einops.rearrange with manual reshape/permute implementations.
    Covers the 3 patterns used in BiRefNet/ISNet decoders.
    """
    try:
        import einops
    except ImportError:
        return

    def manual_rearrange(x, pattern, **kwargs):
        hg = kwargs.get('hg', 2)
        wg = kwargs.get('wg', 2)
        if pattern == 'b c (hg h) (wg w) -> (b hg wg) c h w':
            B, C, H, W = x.shape
            h, w = H // hg, W // wg
            return x.view(B, C, hg, h, wg, w).permute(0, 2, 4, 1, 3, 5).contiguous().view(B * hg * wg, C, h, w)
        elif pattern == 'b c (hg h) (wg w) -> b (c hg wg) h w':
            B, C, H, W = x.shape
            h, w = H // hg, W // wg
            return x.view(B, C, hg, h, wg, w).permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C * hg * wg, h, w)
        elif pattern == '(b hg wg) c h w -> b c (hg h) (wg w)':
            BHW, C, h, w = x.shape
            B = BHW // (hg * wg)
            return x.view(B, hg, wg, C, h, w).permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, hg * h, wg * w)
        else:
            raise ValueError(f"Unsupported einops pattern: {pattern}")

    einops.rearrange = manual_rearrange
    log.info("Patched einops.rearrange → manual reshape/permute")


# ─── Master apply ────────────────────────────────────────────────────────────

def apply_all_patches(model: nn.Module) -> dict:
    """Apply all available patches. Returns summary of changes made."""
    summary = {}
    summary['gelu'] = patch_gelu(model)
    summary['deformable_conv'] = patch_deformable_conv(model)
    summary['window_attention'] = patch_window_attention(model)
    summary['patch_merging'] = patch_patch_merging(model)
    patch_interpolate()
    patch_einops()
    summary['interpolate'] = True
    summary['einops'] = True
    return summary
