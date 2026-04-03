# LiteRT-Models

Converted TFLite Model Zoo for Android with [LiteRT CompiledModel](https://ai.google.dev/edge/litert) GPU acceleration.

All models run on-device with **CompiledModel GPU (ML Drift)** — no CPU fallback, no MediaPipe, no Qualcomm AI Hub runtime.

Each model includes a standalone Android sample app (Kotlin) with real-time camera inference.

**If you like this repository, please give it a star.**

# Models

- [**Object Detection**](#object-detection)
  - [YOLO11n](#yolo11n)

- [**Monocular Depth Estimation**](#monocular-depth-estimation)
  - [Depth Anything V2](#depth-anything-v2)

# How to use

1. Download the `.tflite` model from the GitHub Release link below.
2. Place it in your app's `assets/` directory.
3. Use the `CompiledModel` API with `Accelerator.GPU` to load and run inference.

```kotlin
val options = CompiledModel.Options(Accelerator.GPU)
val model = CompiledModel.create(context.assets, "model.tflite", options, null)
val inputBuffers = model.createInputBuffers()

// Write input data
inputBuffers[0].writeFloat(floatArray)

// Run inference
val outputBuffers = model.run(inputBuffers)
val result = outputBuffers[0].readFloat()
```

**Dependency** (build.gradle.kts):
```kotlin
implementation("com.google.ai.edge.litert:litert:2.1.3")
```

# Object Detection

### YOLO11n

YOLO11: Ultralytics latest YOLO with improved backbone and neck architecture. Pure CNN — runs at **18+ FPS** on Pixel 8a GPU via CompiledModel.

Converted from SavedModel to eliminate GPU-incompatible ops (PACK/SPLIT).

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [yolo11n.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v1/yolo11n.tflite) | 10 MB | Float32 [1, 384, 384, 3] NHWC | Float32 [1, 84, 3024] | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) | [yolo/](yolo/) |

**Output format**: `[1, 84, N]` — 84 = 4 bbox (cx, cy, w, h normalized 0-1) + 80 COCO class scores (sigmoid). Requires NMS post-processing.

**Preprocessing**: RGB normalized to 0-1 (divide by 255). No ImageNet mean/std.

# Monocular Depth Estimation

### Depth Anything V2

Depth Anything V2: Monocular depth estimation with ViT-Small backbone. Native Keras conversion achieves **correlation 0.9995** with original PyTorch model. Runs at **3+ FPS** on Pixel 8a GPU.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [depth_anything_v2_fp16w.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v1/depth_anything_v2_fp16w.tflite) | 47 MB | Float32 [1, 392, 518, 3] NHWC | Float32 [1, 392, 518, 1] | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) | [Apache-2.0](https://github.com/DepthAnything/Depth-Anything-V2/blob/main/LICENSE) | [sample-app/](sample-app/) |

**Output format**: `[1, H, W, 1]` — relative inverse depth (higher = closer). Apply min-max normalization for visualization.

**Preprocessing**: RGB with ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

# GPU Compatibility Notes

CompiledModel GPU requires **all ops** to be GPU-compatible. Key constraints:

- All tensors must be 4D or less
- No dynamic dimensions (-1) in reshape
- Avoid: TOPK_V2, GATHER, GATHER_ND, CAST (float-int), GELU, PACK, SPLIT

**Proven conversion paths**:
1. **SavedModel → TFLiteConverter** — Eliminates PACK/SPLIT ops (used for YOLO11)
2. **Native Keras → from_keras_model()** — Full op control (used for Depth Anything V2)

# Conversion Scripts

[**scripts/**](scripts/) — Python scripts for model conversion.

| Script | Description |
|--------|-------------|
| `convert_keras_native.py` | Native Keras DepthAnything V2 → TFLite |
| `compare_quality.py` | Quality comparison vs PyTorch ground truth |

# License

MIT (sample apps). Model licenses follow their original projects.
