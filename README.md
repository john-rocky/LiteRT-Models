# LiteRT-Models

Converted TFLite Model Zoo for Android with [LiteRT CompiledModel](https://ai.google.dev/edge/litert) GPU acceleration.

All models run on-device with **CompiledModel GPU (ML Drift)** — no CPU fallback, no MediaPipe, no Qualcomm AI Hub runtime.

Each model includes a standalone Android sample app (Kotlin) with real-time camera inference.

**If you like this repository, please give it a star.**

# Models

- [**Object Detection**](#object-detection)
  - [YOLO11n](#yolo11n)
  - [YOLO26n](#yolo26n)

- [**Segmentation**](#segmentation)
  - [MobileSAM](#mobilesam)

- [**Background Removal**](#background-removal)
  - [RMBG-1.4 (ISNet)](#rmbg-14-isnet)

- [**Inpainting**](#inpainting)
  - [LaMa-Dilated](#lama-dilated)

- [**Super Resolution**](#super-resolution)
  - [Real-ESRGAN x4v3](#real-esrgan-x4v3)

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

### YOLO26n

YOLO26: Edge-first vision AI with NMS-free end-to-end detection. Up to 43% faster CPU inference vs YOLO11 with DFL removal and ProgLoss.

Original model outputs `[1, 300, 6]` (NMS-free with top-k), but top-k uses GPU-incompatible ops (TOPK_V2, GATHER). Reconverted with top-k removed — raw output matches YOLO11 format.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [yolo26n.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v1/yolo26n.tflite) | 9.3 MB | Float32 [1, 384, 384, 3] NHWC | Float32 [1, 84, 3024] | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) | [yolo/](yolo/) |

**Output format**: Same as YOLO11n — `[1, 84, N]` with NMS post-processing in app. Bbox coords are normalized 0-1.

**Preprocessing**: RGB normalized to 0-1 (divide by 255). No ImageNet mean/std.

# Segmentation

### MobileSAM

<img src="https://github.com/user-attachments/assets/e9d0cc88-4d49-4ba4-ae8a-86c39befa80e" width="300">

MobileSAM: Fast Segment Anything on mobile. Tap anywhere to segment — encoder runs once per image (GPU), decoder runs per tap (CPU). Based on TinyViT encoder (6.1M params) + SAM mask decoder (4.1M params).

Encoder converted via **litert-torch** (the only converter that preserves Vision Transformer attention accuracy). Decoder runs on ONNX Runtime due to TFLite conversion limitations with cross-attention.

| Model | Download Link | Size | Input | Output | API |
| ----- | ------------- | ---- | ----- | ------ | --- |
| Encoder | [mobilesam_encoder.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v1/mobilesam_encoder.tflite) | 28 MB | Float32 [1, 3, 1024, 1024] NCHW | Float32 [1, 256, 64, 64] NCHW | CompiledModel GPU |
| Decoder | [mobilesam_decoder.onnx](https://github.com/john-rocky/LiteRT-Models/releases/download/v1/mobilesam_decoder.onnx) | 16 MB | Embeddings + point coords | Mask [1, 1, 1024, 1024] + IoU | ONNX Runtime CPU |

**Preprocessing**: RGB with `mean=[123.675, 116.28, 103.53]`, `std=[58.395, 57.12, 57.375]`. NCHW planar layout.

**Decoder inputs**: `image_embeddings` [1,256,64,64] + `point_coords` [1,2,2] + `point_labels` [1,2] + `mask_input` [1,1,256,256] + `has_mask_input` [1] + `orig_im_size` [2]

**Sample app**: [mobilesam/](mobilesam/) — Image picker + tap-to-segment with mask overlay.

**Original project**: [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM) | [Apache-2.0](https://github.com/ChaoningZhang/MobileSAM/blob/master/LICENSE)

# Background Removal

### RMBG-1.4 (ISNet)

RMBG-1.4: High-quality background removal based on ISNet (U2-Net variant). Pure CNN architecture — 44M params, runs on CompiledModel GPU. Outputs alpha matte for clean foreground extraction.

Converted via **litert-torch** from [briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4). Output is sigmoid-activated (0-1 mask), no post-processing needed.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [rmbg14.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v1/rmbg14.tflite) | 176 MB | Float32 [1, 3, 1024, 1024] NCHW | Float32 [1, 1, 1024, 1024] | [briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) | [bria-rmbg-1.4](https://bria.ai/bria-huggingface-model-license-agreement/) | [rmbg/](rmbg/) |

**Preprocessing**: RGB normalized as `(pixel/255 - 0.5)`. NCHW planar layout.

**Output format**: Sigmoid mask (0-1). Apply as alpha channel to original image for transparent background.

# Inpainting

### LaMa-Dilated

LaMa-Dilated: Large Mask Inpainting with dilated convolutions. Draw a mask over unwanted objects and the model fills in the region naturally. Based on [LaMa](https://github.com/advimman/lama) with FFT blocks replaced by dilated convolutions for GPU compatibility.

Pre-converted TFLite from [Qualcomm AI Hub](https://aihub.qualcomm.com/models/lama_dilated). Pure CNN, 361 ops, all GPU-native.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [lama_dilated.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v1/lama_dilated.tflite) | 174 MB | Float32 [1, 512, 512, 3] + [1, 512, 512, 1] NHWC | Float32 [1, 512, 512, 3] NHWC | [advimman/lama](https://github.com/advimman/lama) | [Apache-2.0](https://github.com/advimman/lama/blob/main/LICENSE) | [lama/](lama/) |

**Preprocessing**: Image RGB normalized to 0-1 (divide by 255). Mask is single channel 0-1 (1 = area to inpaint).

**Sample app**: [lama/](lama/) — Image picker + finger drawing mask + inpainting with before/after toggle.

# Super Resolution

### Real-ESRGAN x4v3

<img src="https://github.com/user-attachments/assets/ba4466ad-ed23-4bbe-bffd-53b3b3cc3a4a" width="300">

Real-ESRGAN: Practical image restoration and upscaling. The General-x4v3 variant is a lightweight model (1.21M params) with excellent quality for 4x super resolution.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [real_esrgan_x4v3.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v1/real_esrgan_x4v3.tflite) | 4.7 MB | Float32 [1, 128, 128, 3] NHWC | Float32 [1, 512, 512, 3] NHWC | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | [BSD-3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) | [real-esrgan/](real-esrgan/) |

**Output format**: `[1, 512, 512, 3]` — 4x upscaled RGB image (0-1 range).

**Preprocessing**: RGB normalized to 0-1 (divide by 255). For images larger than 128x128, process as overlapping tiles and stitch.

# GPU Compatibility Notes

CompiledModel GPU requires **all ops** to be GPU-compatible. Key constraints:

- All tensors must be 4D or less
- No dynamic dimensions (-1) in reshape
- Avoid: TOPK_V2, GATHER, GATHER_ND, CAST (float-int), GELU, PACK, SPLIT

**Proven conversion paths**:
1. **SavedModel → TFLiteConverter** — Eliminates PACK/SPLIT ops (used for YOLO11)
2. **Native Keras → from_keras_model()** — Full op control for ViT models
3. **litert-torch** — Only viable converter for Vision Transformers (ViT, TinyViT). onnx2tf breaks attention layers (corr≈0.3). See [docs/](docs/) for details.

> **Note**: MobileSAM encoder uses NCHW layout (not NHWC) because litert-torch preserves PyTorch's native layout. The decoder runs on ONNX Runtime as no TFLite converter supports SAM's cross-attention + boolean indexing.

# License

MIT (sample apps). Model licenses follow their original projects.
