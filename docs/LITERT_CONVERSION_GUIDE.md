# LiteRT Model Conversion Guide

Practical findings from converting various model architectures to TFLite for CompiledModel GPU inference on Android.

## Conversion Tools Comparison

| Tool | Best For | Avoid For | Layout |
|------|----------|-----------|--------|
| **litert-torch** | Vision Transformers, attention models | Models with dynamic control flow | NCHW (preserved) |
| **onnx2tf** | Pure CNN models (YOLO, ESRGAN) | ViT, attention layers (destroys accuracy) | NHWC (converted) |
| **SavedModel → TFLiteConverter** | Models already in TF/Keras | PyTorch-only models | NHWC |
| **Native Keras reimplementation** | Maximum accuracy control | Quick prototyping | NHWC |

## Vision Transformer (ViT) Models

### The Problem

onnx2tf converts NCHW → NHWC during conversion. For CNNs this works fine, but **attention mechanisms break** because onnx2tf incorrectly transposes batch/spatial dimensions in MatMul operations.

Measured accuracy loss with onnx2tf on TinyViT (MobileSAM encoder):
- **onnx2tf**: corr = 0.29 (unusable)
- **litert-torch**: corr = 0.99 (excellent)

### The Solution: litert-torch

```python
import litert_torch

model.eval()
dummy = torch.randn(1, 3, 1024, 1024)
result = litert_torch.convert(model, (dummy,))

# Save — returns TfLiteModel object
result.export("model.tflite")
```

**Key points**:
- Preserves NCHW layout (no transpose errors)
- Output is `TfLiteModel` object — use `.export("path")` to save
- Requires `torch.export` compatibility (no dynamic control flow)
- `F.interpolate` must use `align_corners=False` (GPU rejects `half_pixel_centers=True` + `align_corners=True`)

### GELU Handling

TFLite has no native `Erf` op. The standard GELU `x * 0.5 * (1 + erf(x/√2))` produces FlexErf ops.

**Solution**: Replace with sigmoid approximation before conversion:

```python
class SigmoidGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

# Replace all nn.GELU modules
for name, child in model.named_modules():
    if isinstance(child, nn.GELU):
        setattr(parent, name, SigmoidGELU())

# Also patch functional calls
F.gelu = lambda x, approximate='none': x * torch.sigmoid(1.702 * x)
```

Max error vs real GELU: ~0.01 (negligible).

### ONNX Graph Surgery (Alternative)

If you must use onnx2tf (e.g., for a mixed CNN+attention model), you can replace Erf nodes in the ONNX graph:

```python
# erf(z) ≈ 2 * sigmoid(2.407 * z) - 1
# Coefficient: 1.702 * √2 = 2.407
```

**Warning**: Even with correct Erf replacement, onnx2tf still breaks attention accuracy. This only eliminates FlexErf ops — the underlying NCHW→NHWC issue remains.

## CNN Models (YOLO, ESRGAN)

### onnx2tf Works Well

For pure CNN architectures, onnx2tf is the recommended path:

```bash
onnx2tf -i model.onnx -o output/ -osd
```

Key flags:
- `-osd`: Output SavedModel directory
- `-ois input:1,3,H,W`: Override input shape
- `-dsm`: Disable strict mode (skip accuracy correction if it errors)
- `-ebu`: Enable BatchMatMul unfold (for models with matmul ops)

### GPU-Incompatible Ops

Common ops that prevent CompiledModel GPU:
- `TOPK_V2`, `GATHER`, `GATHER_ND` — Reconvert with these ops removed
- `PACK`, `SPLIT` — Use SavedModel export path instead
- `CAST` (float↔int) — Keep everything as float
- `Erf` (FlexErf) — Replace with sigmoid approximation
- Dynamic `RESHAPE` with -1 dimensions — Use static shapes
- `RESIZE_BILINEAR` with `align_corners=True` — Use `align_corners=False`

### 4D Tensor Limit (Critical)

CompiledModel GPU (ML Drift) only supports **4D tensors (BHWC)**. Any intermediate tensor with 5+ dimensions causes compilation failure. This is the fundamental reason why **Swin Transformer, window attention, and similar architectures cannot run on CompiledModel GPU** — window partition inherently requires 6D reshapes.

Standard ViT (global attention) works because Q/K/V are always 4D: `(B, heads, tokens, dim)`.

## Model-Specific Notes

### MobileSAM

| Component | Format | Converter | Reason |
|-----------|--------|-----------|--------|
| Encoder (TinyViT) | TFLite | litert-torch | ViT attention |
| Decoder (MaskDecoder) | ONNX | torch.onnx.export | Boolean indexing + cross-attention incompatible with all TFLite converters |

Decoder limitations tried:
- onnx2tf: BatchMatMul shape mismatch in cross-attention
- litert-torch: `NonConcreteBooleanIndexError` in mask selection
- onnx_tf: Works but produces FlexErf ops (no GPU)

### RMBG-1.4 (ISNet)

Converter: litert-torch (pure CNN, 247 ops, all GPU-compatible).

Key points:
- ISNet is a U2-Net variant — only Conv2d, BN, ReLU, MaxPool, bilinear upsample, concat, sigmoid
- Model outputs 6 side masks — wrap with `model(x)[0][0]` to get primary mask
- Normalization: `(pixel/255 - 0.5)`, NOT ImageNet mean/std
- Output is sigmoid-activated (0-1), no additional sigmoid needed
- `F.interpolate` must use `align_corners=False` for GPU compatibility

### BiRefNet-lite (Swin Transformer) — NOT GPU-compatible

**Attempted and failed.** Swin Transformer's window attention creates 5D+ tensors (`[B, num_windows, 1, 49, 49]`) that CompiledModel GPU rejects (4D max). This is an architectural limitation, not a conversion issue. Patches attempted:
- Replaced GATHER_ND (relative position bias → pre-computed static tensors)
- Replaced SELECT/NOT_EQUAL (attention masks → pre-computed)
- Replaced DeformableConv2d → regular Conv2d
- Replaced GELU → sigmoid approximation
- Duplicated backbone for dual-resolution pass

All ops became TFLite-native but the 5D tensor constraint blocked GPU compilation. **Swin Transformer ≠ CompiledModel GPU.**

### YOLO11 / YOLO26

Converter: SavedModel → TFLiteConverter (eliminates PACK/SPLIT from Ultralytics export).

### Real-ESRGAN

Converter: onnx2tf (pure CNN, no issues).
