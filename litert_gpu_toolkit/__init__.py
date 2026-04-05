"""
litert_gpu_toolkit — Automatic GPU-compatible TFLite conversion for PyTorch models.

Wraps litert-torch with pre-conversion patches that eliminate common
GPU-incompatible patterns (GELU, Swish, GroupNorm, Conv2d_WS, F.normalize,
GATHER_ND, SELECT, align_corners, etc.).

Usage:
    from litert_gpu_toolkit import convert_for_gpu

    tflite_path = convert_for_gpu(
        model,
        dummy_input=torch.randn(1, 3, 1024, 1024),
        output_path="model.tflite",
    )
"""

from litert_gpu_toolkit.converter import convert_for_gpu
from litert_gpu_toolkit.patches import apply_all_patches
from litert_gpu_toolkit.checker import check_gpu_compatibility

__all__ = ["convert_for_gpu", "apply_all_patches", "check_gpu_compatibility"]
