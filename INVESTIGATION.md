# DepthAnything V2 Android GPU Investigation

## Final Results

DepthAnything V2 Small (24.8M params, 518x518) depth estimation on Android.

| Mode | Speed | Quality (corr vs PyTorch) | Model Size | Notes |
|------|-------|--------------------------|------------|-------|
| **Keras Native GPU FP32** | **9ms** | **corr=0.9995** | 99 MB | Best quality + speed |
| Keras Native GPU FP16w | ~9ms | corr≈0.999 | 50 MB | FP16 weights, same speed |
| onnx2tf GPU FP32 | 8ms | corr=0.846 | 94 MB | Quality limited by conversion |
| onnx2tf GPU FP16 clamped | 7ms | corr=0.846 | 50 MB | Needs FP16 clamps |
| ONNX Runtime CPU | 1283ms | corr=1.000 | 96 MB | Ground truth, slow |

Device: Pixel 8a (Tensor G3, Mali-G715). LiteRT 2.1.3 CompiledModel (ML Drift GPU).

### iOS Comparison

| Device | Time | Compute |
|--------|------|---------|
| iPhone 16 Pro | 26ms | Neural Engine |
| **Pixel 8a** | **9ms** | ML Drift GPU |

---

## Recommended Conversion: Native Keras Pipeline

### Why not onnx2tf?

onnx2tf converts PyTorch→ONNX→TF→TFLite. The NCHW→NHWC transformation introduces
structural quality loss (corr=0.846). This is unfixable — tested all onnx2tf flags
(v1.28.8 and v1.29.24: default, -cotof, -ebu, -dsft, -agje). All produce identical corr=0.845.

### Why not litert-torch?

litert-torch preserves PyTorch's NCHW layout internally. ML Drift GPU requires NHWC for
ALL internal ops. `to_channel_last_io()` only wraps I/O, internal Conv/MatMul remain NCHW.
Confirmed as of litert-torch v0.8.0 (Jan 2026). Google has no fix planned.

### Native Keras approach

Rewrite the model architecture natively in TF/Keras with NHWC layout throughout.
Load PyTorch weights with transposition. Export directly to TFLite.

```bash
cd scripts
pip install torch transformers tensorflow tf_keras
python convert_keras_native.py --output_dir ../app/src/main/assets/
```

Result: `depth_anything_v2_keras.tflite` (99 MB, corr=0.9995)

---

## Architecture Know-How: DepthAnything V2 → TFLite

### Model Structure

```
DepthAnythingForDepthEstimation
├── backbone: DINOv2 ViT-S/14
│   ├── embeddings: Conv2D(3→384, 14x14) + CLS token + position embeddings
│   ├── encoder: 12x TransformerBlock
│   │   ├── LayerNorm → MultiHeadAttention(6 heads, dim=64) → LayerScale
│   │   └── LayerNorm → MLP(384→1536→384, GELU) → LayerScale
│   └── layernorm: final LayerNorm
├── neck: DPT
│   ├── reassemble: 4x (Conv2D 1x1 + resize) from layers [2,5,8,11]
│   ├── convs: 4x Conv2D(→64, 3x3, no bias)
│   └── fusion: 4x FusionLayer (reversed, bottom-up)
└── head: Conv2D(64→32) → resize(518x518) → Conv2D(32→32) → ReLU → Conv2D(32→1) → ReLU
```

### Critical Gotchas (discovered through debugging)

#### 1. `apply_layernorm=True` on feature maps

The DINOv2 backbone applies its final LayerNorm to EACH feature map before passing
to the neck, not just the last hidden state. Without this, corr drops from 1.0 to ~0.01.

```python
# WRONG: raw hidden states
features[i] = x  # after block i

# CORRECT: apply LayerNorm to each
features[i] = final_norm(x)  # matches backbone.apply_layernorm=True
```

#### 2. Fusion stage reverses features internally

`DepthAnythingFeatureFusionStage.forward()` reverses the input list before processing:
```python
hidden_states = hidden_states[::-1]  # [148x148, 74x74, 37x37, 19x19] → [19x19, 37x37, 74x74, 148x148]
```

The neck does NOT reverse (older code had `features[::-1]` in neck, removed in current version).

#### 3. Fusion argument order: (fused_state, current_feature)

After the first layer, fusion passes the accumulated fused state as `hidden_state`
and the current backbone feature as `residual`:

```python
# First iteration (no previous fused state):
fused = layer(features[0], size=next_size)

# Subsequent iterations:
fused = layer(fused, features[i], size=next_size)
#              ↑ hidden_state  ↑ residual
```

#### 4. Fusion upsample uses target_size, not scale_factor=2

Each fusion layer upsamples to match the NEXT feature's spatial dimensions,
not a fixed 2x scale. Only the last layer uses scale_factor=2.

```python
size = hidden_states[idx + 1].shape[2:] if idx < len-1 else None
# size=(37,37) for first, (74,74) for second, (148,148) for third, None for last
```

#### 5. `head_in_index = -1` (uses LAST fusion output)

The head uses `hidden_states[-1]` (the largest spatial resolution, 296x296),
NOT `hidden_states[0]`. This is the LAST element of the fusion output list.

#### 6. Head activation order: conv → relu (not relu → conv)

```python
# WRONG (assumed from architecture printout):
out = relu(resize(conv1(x)))
out = relu(conv2(out))
out = conv3(out)

# CORRECT (from actual HuggingFace source):
out = conv1(x)
out = resize(out)
out = conv2(out)        # NO relu before conv2
out = relu(out)         # relu AFTER conv2
out = conv3(out)        # NO relu before conv3
out = relu(out) * max_depth  # relu AFTER conv3, then scale
```

### Weight Transposition Rules

| PyTorch Layer | PT Shape | TF Layer | TF Shape | Transpose |
|---------------|----------|----------|----------|-----------|
| Conv2d | [O, I, H, W] | Conv2D | [H, W, I, O] | (2, 3, 1, 0) |
| ConvTranspose2d | [I, O, H, W] | Conv2DTranspose | [H, W, O, I] | (2, 3, 1, 0) |
| Linear | [O, I] | Dense | [I, O] | .T |
| LayerNorm weight | [D] | LayerNormalization gamma | [D] | none |
| LayerNorm bias | [D] | LayerNormalization beta | [D] | none |
| LayerScale lambda | [D] | add_weight | [D] | none |
| CLS token | [1, 1, D] | add_weight | [1, 1, D] | none |
| Position embeddings | [1, N, D] | add_weight | [1, N, D] | none (+ interpolation if size differs) |

### TFLite GPU (ML Drift) Compatibility

#### Must-have requirements
- **All ops in NHWC** — ML Drift requires NHWC for all internal tensor operations
- **Static shapes** — no `tf.shape()`, use Python `.shape` attributes instead
- **Built-in ops only** — no FlexDelegate ops (blocks GPU entirely)

#### Supported ops (verified on ML Drift)
ADD, BATCH_MATMUL, CONCATENATION, CONV_2D, FULLY_CONNECTED, GELU, MEAN, MUL, NEG,
RELU, RESHAPE, RESIZE_BILINEAR, RSQRT, SOFTMAX, SQUARED_DIFFERENCE, STRIDED_SLICE,
TRANSPOSE, TRANSPOSE_CONV

#### Known issues
- `RESIZE_BILINEAR` with `align_corners=True` may NOT be supported. Use default
  `half_pixel_centers` mode (`tf.image.resize`, not `tf.compat.v1.image.resize`).
  Quality impact: corr drops from 0.999998 to 0.9995 (negligible).
- `BROADCAST_TO`, `SHAPE`, `PACK` — avoid by using static shapes
- Dynamic batch size — hardcode batch=1 for inference

#### TFLite export from Keras 3
- Use `tf.lite.TFLiteConverter.from_keras_model(model)` — captures all variables
- Do NOT use `from_concrete_functions` or `from_saved_model` — may lose variables
  and produce tiny (0.2 MB) broken models

---

## All Conversion Pipelines Investigated

| # | Pipeline | GPU | Quality | Status |
|---|----------|-----|---------|--------|
| **1** | **Native Keras** | **ML Drift 9ms** | **corr=0.9995** | **Production** |
| 2 | onnx2tf (PINTO0309) | ML Drift 8ms | corr=0.846 | Quality limited |
| 3 | Qualcomm AI Hub | Old delegate | corr≈0.85 | External dependency |
| 4 | litert-torch direct | ML Drift fails | corr=1.0 on CPU | NCHW blocks GPU |
| 5 | litert-torch + NHWC I/O | ML Drift fails | — | Internal ops still NCHW |
| 6 | ONNX Runtime GPU | N/A | — | No GPU EP for Android |
| 7 | ExecuTorch Vulkan | 200+ sec | — | Immature for ViT |
| 8 | onnx-tf (official) | — | — | Deprecated, broken |
| 9 | nobuco (PyTorch→Keras) | — | — | Keras 2 only, no ViT |

---

## onnx2tf Quality Investigation (exhaustive)

All onnx2tf variants produce identical quality. No flags improve it.

| onnx2tf Version | Flags | Correlation |
|-----------------|-------|-------------|
| 1.28.8 | default | 0.845 |
| 1.29.24 | default | 0.845 |
| 1.29.24 | -cotof -cotoa 0.1 | 0.845 |
| 1.29.24 | -cotof -cotoa 0.01 -agje | 0.845 |
| 1.29.24 | -ebu | 0.845 |
| 1.29.24 | -dsft | 0.845 |

Root cause: ONNX→TF graph transformation decomposes and restructures ops (Reshape/Transpose/MatMul
chains in multi-head attention) during NCHW→NHWC conversion. This is an inherent limitation.

---

## FP16 Overflow (for onnx2tf models)

ViT attention Q*K^T can exceed FP16 range (±65,504) → NaN.

Solution: Insert Clip(±60000) at 64 locations in ONNX graph:
- Before Softmax: 12 nodes
- Before LayerNormalization: 28 nodes
- After Attention MatMul: 24 nodes
- FFN MatMul: NOT clamped (degrades quality)

For Keras native models: use `GpuOptions.Precision.FP32` (same speed on Tensor G3).

---

## ML Drift (LiteRT 2.x CompiledModel)

```kotlin
// Gradle
implementation("com.google.ai.edge.litert:litert:2.1.3")

// Kotlin
val options = CompiledModel.Options(Accelerator.GPU)
options.gpuOptions = CompiledModel.GpuOptions(
    null, null, null,
    CompiledModel.GpuOptions.Precision.FP32,  // FP32 = same speed, better quality
    null, null, null, null, null, null, null, null, null, null, null
)
val model = CompiledModel.create(context.assets, "model.tflite", options, null)
```

FP32 and FP16 showed same ~8-9ms speed on Pixel 8a. FP32 is strictly better.

---

## Files

### Models (in app/src/main/assets/, git-ignored)
- `depth_anything_v2_keras.tflite` — **Keras native NHWC FP32** (recommended)
- `depth_anything_v2_keras_fp16w.tflite` — Keras native FP16 weights
- `depth_anything_v2.onnx` — ONNX FP32 (for ONNX Runtime CPU reference)
- `depth_anything_v2_nhwc_clamped_fp16w.tflite` — onnx2tf clamped (legacy)

### Scripts
- **`scripts/convert_keras_native.py`** — Native Keras conversion (recommended)
- `scripts/convert_nhwc_gpu.py` — onnx2tf conversion (legacy)
- `scripts/convert_v129.py` — onnx2tf 1.29 multi-variant testing
- `scripts/compare_quality.py` — Quality comparison vs PyTorch ground truth
- `scripts/export_all.py` — litert-torch export (experimental)

### Android App
- LiteRT 2.1.3 CompiledModel API for GPU (ML Drift)
- ONNX Runtime 1.24.3 for CPU reference (+ XNNPACK EP mode)
- Jetpack Compose UI with benchmark comparison
