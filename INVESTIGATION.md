# DepthAnything V2 Android FP16 Investigation

## Summary

DepthAnything V2 Small (24.7M params, 518x518 input) on Pixel 10 Pro (Tensor G5, PowerVR GPU).

### Final Results

| Mode | Avg (ms) | Quality | Model Size |
|------|----------|---------|------------|
| GPU clamp+FP16w+FP16 | **208** | OK | 50 MB |
| GPU clamp+FP16 | 209 | OK | 99 MB |
| GPU FP16 (no clamp) | 207 | Broken | 94 MB |
| GPU FP32 | 575 | OK | 94 MB |
| GPU FP16w (FP32 compute) | 576 | OK | 47 MB |
| ONNX Runtime CPU | 1404 | OK | 94 MB |
| TFLite XNNPACK CPU | ~2700 | OK | 94 MB |

**Best: GPU clamp+FP16w+FP16 at 208ms (50MB model)**

### iOS Comparison (Apple official benchmark)

| Device | Time | Compute |
|--------|------|---------|
| iPhone 16 Pro | 26ms | Neural Engine |
| iPhone 15 Pro Max | 34ms | Neural Engine |
| **Pixel 10 Pro** | **208ms** | GPU delegate |

iOS uses dedicated Neural Engine (NPU). Android Pixel uses GPU via TFLite GPU delegate.

---

## Problem

Client reported:
1. TFLite GPU delegate + FP16 (`isPrecisionLossAllowed=true`) causes NaN/Inf → output is single color
2. FP32 works but ~2x slower than iOS equivalent device
3. Pixel 10 Pro (Tensor G5) GPU delegate support unclear

## Root Cause Analysis

### Why FP16 produces NaN

ViT attention mechanism: Q*K^T dot product can produce values exceeding FP16 range (±65,504).
LayerNormalization's internal RSQRT can also overflow. These cause NaN propagation through the network.

iOS CoreML handles this transparently with automatic mixed precision. TFLite GPU delegate has no per-op precision control — `isPrecisionLossAllowed` is all-or-nothing.

### Why GPU delegate appeared "broken" initially

Three issues compounded:

1. **NCHW layout**: `litert-torch` (formerly ai-edge-torch) exports models with NCHW internal layout. TFLite GPU delegate requires NHWC for effective GPU execution. With NCHW, most ops fall back to CPU → GPU delegate gives no speedup.

2. **3D CLS token concat**: ViT's CLS token concatenation uses 3D tensors `[1, 1, 384]`. GPU delegate's CONCATENATION op requires 4D `BxHxWxC` → delegate initialization fails.

3. **Erf op**: GELU activation uses the Erf function, which is not a TFLite built-in. Requires FlexDelegate (CPU-only), preventing full GPU execution.

## Solution

### Model Conversion Pipeline

```
PyTorch (NCHW) → ONNX → onnx2tf → TFLite (NHWC, GPU-compatible)
```

Key steps:

1. **Patch PyTorch model before export:**
   - Replace `nn.GELU()` → `nn.GELU(approximate="tanh")` (avoids Erf op)
   - Modify CLS token concatenation to use 4D tensors:
     ```python
     # Original: torch.cat([cls, patches], dim=1)  # 3D
     # Fixed:    torch.cat([cls.unsqueeze(1), patches.unsqueeze(1)], dim=2).squeeze(1)  # via 4D
     ```

2. **ONNX post-processing:**
   - Add missing `kernel_shape` attribute to ConvTranspose nodes
   - Replace remaining Erf → Tanh in ONNX graph
   - Simplify with onnxsim

3. **For FP16 GPU: Insert clamp ops at 112 locations:**
   - Before every Softmax (12 nodes) — prevents attention score overflow
   - Before every LayerNormalization (28 nodes) — prevents variance overflow
   - After every MatMul (72 nodes) — clamps intermediate values
   - Clamp range: ±60,000 (within FP16 safe range)

4. **Convert with onnx2tf:**
   ```bash
   onnx2tf -i model.onnx -o output/ -osd -coion
   ```
   - `-osd`: output saved model directory
   - `-coion`: built-in ops only (no FlexDelegate)
   - Produces both FP32 and FP16 weight variants

### Why onnx2tf, not litert-torch?

| | litert-torch | onnx2tf |
|---|---|---|
| Internal layout | NCHW (PyTorch convention) | **NHWC** (TF/TFLite convention) |
| Transpose ops | Adds boundary transposes only | **Full weight/op conversion** |
| Op count | 714 | ~650 (fewer transposes) |
| GPU delegate | Falls back to CPU | **Full GPU execution** |
| Result | 2400ms | **208ms** |

### onnx2tf patches required

onnx2tf 1.28.8 has bugs with our model:

1. `download_test_image_data()` — np.load pickle error. Fix: early return with dummy data
2. `get_weights_constant_or_variable()` — ConvTranspose weight transpose axis mismatch. Fix: fallback to `values.ndim`

## Benchmark Evolution

| Stage | Mode | Time | Change |
|-------|------|------|--------|
| Initial | ONNX CPU | 1404ms | baseline |
| Initial | TFLite XNNPACK CPU | 2700ms | 2x slower |
| Initial | TFLite GPU (NCHW) | 2400ms | GPU not working |
| Fix NHWC | TFLite GPU FP32 (NHWC) | 575ms | **4.7x faster** |
| Fix FP16 | TFLite GPU clamp+FP16 | **208ms** | **6.7x faster** |

## Files

### Android App
- `app/src/main/java/com/depthanything/ml/` — Inference engine (TFLite, ONNX)
- `app/src/main/java/com/depthanything/ui/` — Benchmark UI
- Model files go in `app/src/main/assets/` (see Releases)

### Conversion Scripts
- `scripts/export_all.py` — Initial ONNX + TFLite export via litert-torch
- `scripts/export_nhwc.py` — NHWC export attempt via litert-torch (insufficient)
- `scripts/convert_models.py` — Full conversion pipeline reference
- `scripts/convert_onnx_mixed_precision.py` — ONNX FP16 mixed precision tool

### Key Learnings

1. **ViT models on Android GPU require NHWC layout throughout** — not just I/O boundaries
2. **litert-torch is not suitable for GPU-targeted TFLite models** — use onnx2tf
3. **FP16 on GPU is 2.8x faster than FP32** — but requires clamping at 112+ locations
4. **Clamp before Softmax alone is insufficient** — LayerNorm and MatMul outputs also overflow
5. **GPU delegate on Pixel 10 Pro (PowerVR) works** — contrary to initial investigation suggesting otherwise
