# DepthAnything V2 Android GPU Investigation

## Summary

DepthAnything V2 Small (24.7M params, 518x518) on Pixel 8a (Tensor G3, Mali-G715 GPU).

### Final Results

| Mode | Speed | Quality (vs PyTorch) | Model Size |
|------|-------|---------------------|------------|
| **onnx2tf + ML Drift FP32** | **~8ms** | corr=0.846 | 98 MB |
| onnx2tf + ML Drift FP16 clamped | ~7ms | corr=0.846 + FP16 noise | 49-99 MB |
| onnx2tf + old GPU delegate FP32 | 575ms | corr=0.846 | 94 MB |
| onnx2tf + old GPU delegate FP16 clamped | 208ms | corr=0.846 + FP16 noise | 94 MB |
| ONNX Runtime CPU | 1283ms | **corr=1.000** | 96 MB |
| TFLite XNNPACK CPU | ~2700ms | corr=0.846 | 94 MB |

### iOS Comparison

| Device | Time | Compute |
|--------|------|---------|
| iPhone 16 Pro | 26ms | Neural Engine |
| **Pixel 8a** | **8ms** | ML Drift GPU |

---

## Conversion Pipelines Investigated

### What works

| # | Pipeline | GPU | Quality | Status |
|---|----------|-----|---------|--------|
| 1 | **onnx2tf** (PINTO0309) | ML Drift + old delegate | corr=0.846 | **Production ready** |
| 2 | Qualcomm AI Hub | Old delegate | corr≈0.85 | External dependency |

### What doesn't work and why

| # | Pipeline | Failure Reason |
|---|----------|---------------|
| 3 | litert-torch direct | Internal ops are NCHW → ML Drift requires NHWC throughout |
| 4 | litert-torch + NHWC I/O wrapper | Only wraps boundaries, internal Conv/MatMul still NCHW |
| 5 | litert-torch + FlatWrapper + NHWC permute | Same: internal computation NCHW, ML Drift rejects |
| 6 | ONNX direct in CompiledModel | CompiledModel doesn't support ONNX format |
| 7 | TF version of DepthAnything | Does not exist in HuggingFace transformers |
| 8 | onnx-tf (official) | Deprecated, broken with modern onnx versions |
| 9 | nobuco (PyTorch→Keras) | Keras 2 only, no ViT support, high effort |

### Key insight: ML Drift NHWC requirement

ML Drift (LiteRT 2.x CompiledModel GPU engine) requires **all internal tensor operations** in NHWC layout. Adding NHWC transposes at I/O boundaries is insufficient. Only onnx2tf performs full NCHW→NHWC conversion of weights and operations.

litert-torch traces PyTorch computation (which is NCHW) and preserves it. Even with I/O wrappers, internal Conv2D/MatMul operate in NCHW.

---

## Quality Analysis

### Ground truth comparison

| Runtime | vs PyTorch | MSE |
|---------|-----------|-----|
| **ONNX Runtime** | **corr=1.000000** | 0.000000 |
| onnx2tf TFLite (all variants) | corr=0.846 | 0.057 |
| litert-torch TFLite (CPU) | corr=1.000000 | 0.000000 |

### What causes the 0.846 correlation in onnx2tf?

| Factor | Impact |
|--------|--------|
| Erf→Tanh replacement | **None** (verified corr=1.0 in Python) |
| GELU exact→tanh approximation | **None** (verified corr=1.0 in Python) |
| FP16 clamp ops | **None** (same corr with/without clamps) |
| FP16 vs FP32 weights | **None** (same corr) |
| **onnx2tf graph transformation** | **Sole cause** (op decomposition, layout conversion artifacts) |

The 15% quality loss is a structural limitation of the ONNX→TF→TFLite conversion pipeline. No parameter tuning can fix it.

---

## ML Drift (LiteRT 2.x) Findings

### CompiledModel API

```kotlin
implementation("com.google.ai.edge.litert:litert:2.1.0")

val options = CompiledModel.Options(Accelerator.GPU)
// Optional: force FP32 precision
options.gpuOptions = CompiledModel.GpuOptions(
    precision = CompiledModel.GpuOptions.Precision.FP32
)
val model = CompiledModel.create(context.assets, "model.tflite", options, null)
```

### GpuOptions.Precision

| Setting | Behavior | Quality |
|---------|----------|---------|
| DEFAULT | FP16 compute (fast, overflow risk) | Needs clamps |
| **FP32** | FP32 compute (same speed on this device!) | Best |

FP32 and DEFAULT showed same ~8ms speed on Pixel 8a. FP32 is strictly better.

### Supported TFLite ops on ML Drift

| Op | Supported | Notes |
|----|-----------|-------|
| GELU | **Yes** | Tested, works |
| BATCH_MATMUL | **Yes** | Works in NHWC model |
| SOFTMAX | **Yes** | |
| CONV_2D | **Yes** | |
| GATHER_ND | Unknown | litert-torch model failed but may be layout issue |
| BROADCAST_TO | Unknown | Same |

### ML Drift vs old GPU delegate

| | ML Drift (v2.x) | Old delegate (v1.x) |
|---|---|---|
| Speed | **~8ms** | 575ms (FP32), 208ms (FP16) |
| FP32/FP16 control | GpuOptions.Precision | isPrecisionLossAllowed |
| NHWC requirement | **Strict** (internal ops too) | Partial (NCHW partially works) |
| API | CompiledModel | Interpreter + GpuDelegate |

---

## FP16 Overflow Solution

### Problem
ViT attention Q*K^T produces values exceeding FP16 range (±65,504) → NaN.

### Solution: Targeted clamping
Insert Clip(±60000) at 64 locations in ONNX graph before onnx2tf conversion:
- Before Softmax: 12 nodes
- Before LayerNormalization: 28 nodes  
- After Attention MatMul (Q*K^T, attn*V): 24 nodes
- FFN MatMul: NOT clamped (degrades quality)

### Clamping results

| Clamp strategy | Clamp count | Quality |
|---------------|-------------|---------|
| None | 0 | Broken (NaN) |
| Softmax only | 12 | Broken |
| Softmax + LayerNorm | 40 | Broken |
| **Softmax + LN + Attention MatMul** | **64** | **OK** |
| All MatMul | 112 | Slightly worse (FFN clamped unnecessarily) |

---

## Conversion Pipeline (onnx2tf)

```bash
python scripts/convert_nhwc_gpu.py --output_dir app/src/main/assets/ \
    --input_height 518 --input_width 518
```

### Steps
1. Load DepthAnything V2 from HuggingFace
2. Patch: GELU(tanh), 4D CLS concat, position embedding interpolation
3. Export to ONNX (opset 18)
4. Fix: ConvTranspose kernel_shape, Erf→Tanh
5. Simplify with onnxsim
6. (Optional) Insert FP16 clamps at 64 attention-related locations
7. Convert with onnx2tf (`-osd -coion`)

### onnx2tf patches required
1. `download_test_image_data()`: numpy pickle error → early return with dummy
2. `get_weights_constant_or_variable()`: ConvTranspose axis mismatch → ndim fallback

---

## Files

### Models (in app/src/main/assets/, excluded from git)
- `depth_anything_v2_nhwc.tflite` — onnx2tf NHWC, no clamp (GPU FP32)
- `depth_anything_v2_nhwc_clamped.tflite` — onnx2tf NHWC, 112 clamps (GPU FP16)
- `depth_anything_v2_nhwc_clamped_fp16w.tflite` — same, FP16 weights (smaller)
- `depth_anything_v2.onnx` — ONNX FP32 (for ONNX Runtime CPU)
- `depth_anything_v2_direct.tflite` — litert-torch NCHW (CPU only, corr=1.0)
- `depth_anything_v2_flat.tflite` — litert-torch NCHW flat (experimental)
- `depth_anything_v2_nhwc_direct.tflite` — litert-torch NHWC I/O (ML Drift incompatible)

### Scripts
- `scripts/convert_nhwc_gpu.py` — Main conversion script (onnx2tf pipeline)
- `scripts/export_nhwc.py` — litert-torch NHWC export (experimental)
- `scripts/convert_onnx_mixed_precision.py` — ONNX FP16 mixed precision

### Android App
- LiteRT 2.1.0 CompiledModel API for GPU (ML Drift)
- ONNX Runtime 1.24.3 for CPU reference
- Jetpack Compose UI with benchmark comparison
