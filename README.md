# LiteRT-Models

<img src="https://github.com/user-attachments/assets/3c764bf0-32ac-4b65-814b-a73e1aba1cfc" width="100%">

Converted TFLite Model Zoo for Android with [LiteRT CompiledModel](https://ai.google.dev/edge/litert) GPU acceleration.

All models run on-device with **CompiledModel GPU (ML Drift)** — no CPU fallback, no MediaPipe, no Qualcomm AI Hub runtime.

Each model includes a standalone Android sample app (Kotlin) with real-time camera inference.

**If you like this repository, please give it a star.**

# Models

- [**Object Detection**](#object-detection)
  - [YOLO11n](#yolo11n)
  - [YOLO26n](#yolo26n)

- [**Multi-Object Tracking**](#multi-object-tracking)
  - [YOLO + DeepSORT (OSNet)](#yolo--deepsort-osnet)

- [**Pose Estimation**](#pose-estimation)
  - [YOLO26n-pose](#yolo26n-pose)
  - [RTMPose-s](#rtmpose-s)
  - [RTMW-m (whole-body, 133 keypoints)](#rtmw-m-whole-body-133-keypoints)
  - [RTMPose-Hand (21 keypoints)](#rtmpose-hand-21-keypoints)
  - [YOLO26n-pose](#yolo26n-pose)

- [**Semantic Segmentation**](#semantic-segmentation)
  - [LR-ASPP MobileNetV3](#lr-aspp-mobilenetv3)
- [**Segmentation**](#segmentation)
  - [MobileSAM](#mobilesam)
  - [EdgeTAM (SAM2)](#edgetam-sam2)
  - [EdgeTAM Video (SAM2 tracking)](#edgetam-video-sam2-tracking)

- [**Background Removal**](#background-removal)
  - [RMBG-1.4 (ISNet)](#rmbg-14-isnet)

- [**Inpainting**](#inpainting)
  - [LaMa-Dilated](#lama-dilated)

- [**Zero-Shot Classification**](#zero-shot-classification)
  - [CLIP ViT-B/32](#clip-vit-b32)
  - [MobileNetV3-Large (ImageNet)](#mobilenetv3-large-imagenet)
  - [Places365 ResNet18 (scene recognition)](#places365-resnet18-scene-recognition)

- [**Surface Normal Estimation**](#surface-normal-estimation)
  - [DSINE](#dsine)

- [**Speech Recognition**](#speech-recognition)
  - [Whisper-tiny](#whisper-tiny)

- [**Text-to-Speech**](#text-to-speech)
  - [Kokoro-82M](#kokoro-82m)
  - [Matcha-TTS](#matcha-tts)

- [**Vision-Language Model**](#vision-language-model)
  - [SmolVLM-256M](#smolvlm-256m)

- [**Voice Assistant**](#voice-assistant)
  - [Whisper + SmolLM2 + Kokoro pipeline](#whisper--smollm2--kokoro-pipeline)

- [**Audio Codec**](#audio-codec)
  - [DAC 16kHz](#dac-16khz)
  - [Mimi (Kyutai 2024)](#mimi-kyutai-2024)

- [**Audio Classification**](#audio-classification)
  - [wav2vec2 Keyword Spotting](#wav2vec2-keyword-spotting)

- [**Line Detection**](#line-detection)

  - [M-LSD-tiny](#m-lsd-tiny)

- [**OCR**](#ocr)
  - [PP-OCRv5](#pp-ocrv5)

- [**Super Resolution**](#super-resolution)
  - [Real-ESRGAN x4v3](#real-esrgan-x4v3)
- [**Image Restoration**](#image-restoration)
  - [NAFNet (deblur)](#nafnet-deblur)

- [**Monocular Geometry Estimation**](#monocular-geometry-estimation)
  - [MoGe-2 ViT-S](#moge-2-vit-s)
  - [Depth Anything 3 ViT-S (Small)](#depth-anything-3-vit-s-small)
  - [Metric3D v2 ViT-S](#metric3d-v2-vit-s)

- [**Text Generation (LLM)**](#text-generation-llm)
  - [Falcon3-3B-Instruct](#falcon3-3b-instruct)
  - [Llama-3.2-3B-Instruct](#llama-32-3b-instruct)
  - [Ministral-3-3B-Instruct-2512](#ministral-3-3b-instruct-2512)
  - [SmolLM3-3B](#smollm3-3b)

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

# Multi-Object Tracking

### YOLO + DeepSORT (OSNet)

<img src="https://github.com/user-attachments/assets/d4cacaa1-da30-49f4-8e63-9789ee5bd1a7" width="300">

| | |
|---|---|
| **Module** | [`yolo-tracking/`](yolo-tracking/) |
| **Detection** | YOLO11n — 384×384, 10 MB |
| **Re-ID** | OSNet x0.25 — 256×128 → 512-dim embedding, ~1.4 MB |
| **Tracker** | DeepSORT (Kalman + Hungarian + cascade matching) |
| **Pipeline** | YOLO detect → OSNet Re-ID per crop → DeepSORT track |

Real-time multi-object tracking with appearance-based re-identification. YOLO11n detects objects, OSNet x0.25 extracts 512-dim appearance embeddings from each detection crop, and DeepSORT maintains track identities using cosine similarity on embeddings gated by Mahalanobis distance from Kalman-predicted positions.

Both ML models run on **CompiledModel GPU**. The tracker logic (Kalman filter, Hungarian algorithm, cascade matching) runs in Kotlin on CPU.

| Model | Download Link | Size | Input | Output | Original Project | License |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- |
| YOLO11n | [yolo11n.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v1/yolo11n.tflite) | 10 MB | Float32 [1, 384, 384, 3] NHWC | Float32 [1, 84, 3024] | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | AGPL-3.0 |
| OSNet x0.25 | [osnet_x0_25.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v3/osnet_x0_25.tflite) | 867 KB | Float32 [1, 3, 256, 128] NCHW | Float32 [1, 512] | [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) | [MIT](https://github.com/KaiyangZhou/deep-person-reid/blob/master/LICENSE) |

**Preprocessing**: YOLO — RGB 0-1. OSNet — RGB with ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), NCHW layout.

**Conversion**: `yolo-tracking/scripts/convert_osnet.py` — uses `litert_gpu_toolkit.convert_for_gpu()`. OSNet is a pure CNN, no special GPU patches needed.

# Pose Estimation

### YOLO26n-pose

<img src=https://github.com/user-attachments/assets/55e864bc-5e26-4025-a814-a6fcd5683a4d width=300>

Real-time human pose estimation with Ultralytics YOLO26n-pose. 17 COCO keypoints + skeleton overlay, runs on CompiledModel GPU. Three input modes in the sample app: live camera, picked image, picked video — all share the same pose decoder.

Converted via **litert-torch** by wrapping the head with `end2end=False, export=True, format='tflite'`. This bypasses the default end-to-end NMS-free path (which compiles to GPU-incompatible `TOPK_V2`/`GATHER`) and exposes the legacy one-to-many head output. Ultralytics' default ONNX → onnx2tf path breaks on the YOLO26 backbone (`model.2/m.0/Add` channel mismatch), so the conversion goes PyTorch → litert-torch directly, the same path used for MobileSAM, RMBG, and DSINE in this repo.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [yolo26n_pose.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/yolo26n_pose.tflite) | 12 MB | Float32 [1, 3, 384, 384] NCHW | Float32 [1, 56, 3024] | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) | [yolo-pose/](yolo-pose/) |

**Output format**: `[1, 56, N]` — `4 bbox (cx, cy, w, h) + 1 person conf + 17 keypoints * 3 (x, y, vis)`. Bbox is YOLO **xywh center format** (the legacy one-to-many head emits xywh, not xyxy). Bbox and keypoint xy values are in input image pixel space (0..384); person confidence and per-keypoint visibility are sigmoid-activated (0..1). Requires NMS post-processing.

**Preprocessing**: RGB normalized to 0-1 (divide by 255), planar NCHW layout. No ImageNet mean/std.

**Sample app**: [yolo-pose/](yolo-pose/) — Camera / Image / Video mode toggle, skeleton overlay matching either FILL_CENTER (camera) or FIT_CENTER (image/video).

### RTMPose-s

RTMPose-s (mmpose, CSPNeXt + RTMCC/SimCC head): the SOTA real-time **top-down** 2D human pose model — 17 COCO keypoints for a centered person — running **fully on the GPU** (`256/256` LITERT_CL on a Pixel 8a, **~4 ms**, fp16 **11.1 MB**). Apache-2.0 (vs the YOLO pose model's AGPL), device-vs-PyTorch SimCC corr **0.999**, keypoints within **0.3 px**.

Converted via **litert-torch** with two numerically-exact, on-device-only re-authorings (both pass the desktop op-check yet were needed for a correct Mali result — *residency ≠ correctness*): (1) the RTMCC **`ScaleNorm` (RMS)** input reaches ≈|274| so its `Σx²`≈3.6M **overflows fp16** on Mali → `norm=∞` → all-zero head; fixed by scaling `x` down before squaring (same class as the NAFNet SafeLayerNorm). (2) The GAU attention **`act@act` BMM** → broadcast-multiply + reduce-sum (K=17 tokens).

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [rtmpose_s_fp16.tflite](https://huggingface.co/litert-community/RTMPose-s-LiteRT) | 11.1 MB | Float32 [1, 3, 256, 192] NCHW | simcc_x [1,17,384], simcc_y [1,17,512] | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) | [Apache-2.0](https://github.com/open-mmlab/mmpose/blob/main/LICENSE) | [rtmpose/](rtmpose/) |

**Output format**: two 1D SimCC distributions per keypoint; argmax over the bins (÷ split=2) → pixel x/y. **Preprocessing**: center-crop to 3:4, resize 192×256, ImageNet 0-255 normalize, NCHW. Top-down (one centered person).

**Sample app**: [rtmpose/](rtmpose/) — image picker + COCO skeleton overlay.

### RTMW-m (whole-body, 133 keypoints)

RTMW-m (mmpose, CSPNeXt + CSPNeXtPAFPN neck + RTMW/SimCC head): **whole-body** 2D pose — **133 COCO-WholeBody keypoints** (17 body + 6 feet + 68 face + 42 hands) for a centered person. The model ControlNet/animation pipelines use. Runs **fully on the GPU** (`531/531` LITERT_CL on a Pixel 8a, **~6 ms**, fp16 **66 MB**), device-vs-PyTorch SimCC corr **0.999**, keypoints within **0.2 px**.

Converted via **litert-torch** with the RTMPose-family re-authorings (SafeRMSNorm for the `ScaleNorm` fp16 overflow + GAU `act@act` BMM → broadcast-reduce) **plus** `nn.PixelShuffle` → depth-to-space `ZeroStuffConvT2d` (the RTMW head's PixelShuffle upsample lowers to a 6D tensor; the fixed depth-to-space `ConvTranspose2d` keeps it 4D — reused from NAFNet/Metric3D).

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [rtmw_fp16.tflite](https://huggingface.co/litert-community/RTMW-m-WholeBody-LiteRT) | 66 MB | Float32 [1, 3, 256, 192] NCHW | simcc_x [1,133,384], simcc_y [1,133,512] | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) | [Apache-2.0](https://github.com/open-mmlab/mmpose/blob/main/LICENSE) | [rtmw/](rtmw/) |

**Sample app**: [rtmw/](rtmw/) — image picker + whole-body skeleton (body/feet/face/hands color-coded).

### RTMPose-Hand (21 keypoints)

RTMPose-m hand (mmpose, CSPNeXt + RTMCC/SimCC head): **hand** pose — the **21 standard hand keypoints** (wrist + 4 joints × 5 fingers) for a centered hand. Runs **fully on the GPU** (`333/333` LITERT_CL on a Pixel 8a, **~4 ms**, fp16 **28 MB**), device-vs-PyTorch SimCC corr **0.999**. Same RTMPose-family re-authorings as the body model (no PixelShuffle — no neck).

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [rtmhand_fp16.tflite](https://huggingface.co/litert-community/RTMPose-Hand-LiteRT) | 28 MB | Float32 [1, 3, 256, 256] NCHW | simcc_x [1,21,512], simcc_y [1,21,512] | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) | [Apache-2.0](https://github.com/open-mmlab/mmpose/blob/main/LICENSE) | [rtmhand/](rtmhand/) |

**Sample app**: [rtmhand/](rtmhand/) — image picker + 21-keypoint hand skeleton (per-finger color).

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

### EdgeTAM (SAM2)

EdgeTAM (Meta, CVPR 2025): on-device Segment Anything 2. Tap an object to segment it. RepViT backbone + FPN neck encoder (runs once per image) and a SAM2 mask decoder (runs per tap) — **both on CompiledModel GPU**. Image-segment mode only (the novel 2D Spatial Perceiver lives in the video-memory path, skipped). 9.1M params.

Converted via **litert-torch** (MobileSAM-style split): image **encoder** (image → embeddings + FPN), a tiny **prompt encoder in Kotlin** (point → sparse embedding, bit-exact vs the model — the positional `coords @ Gaussian` trips a `batch_matmul` converter pass, so it stays off-graph), and the mask **decoder** (embeddings + FPN + sparse → masks). Both graphs use a single concatenated input/output so CompiledModel never maps same-sized tensors by order. GPU-compat patches: **SqueezeExcite global avg-pool `mean((2,3))` → two single-axis means** (a single multi-axis `SUM` over ~65k elements silently returns NaN on the Pixel 8a ML Drift delegate — see [GPU Compatibility Notes](#gpu-compatibility-notes)), **ConvTranspose2d → zero-stuff + Conv2d** (`TRANSPOSE_CONV` rejected on-device), and a 4D mask decoder. Exact erf-GELU is kept (it is GPU-correct; the sigmoid approximation hurt mask quality).

| Model | Download Link | Size | Input | Output | API |
| ----- | ------------- | ---- | ----- | ------ | --- |
| Encoder | [edgetam_encoder.tflite](https://huggingface.co/mlboydaisuke/EdgeTAM-LiteRT) | 10 MB | Float32 [1, 3, 1024, 1024] NCHW | Float32 [1, 4194304] (ie \| fpn0 \| fpn1) | CompiledModel GPU |
| Decoder | [edgetam_decoder.tflite](https://huggingface.co/mlboydaisuke/EdgeTAM-LiteRT) | 17 MB | Float32 [1, 4194816] (ie \| sparse \| fpn0 \| fpn1) | Masks [1, 3, 256, 256] | CompiledModel GPU |

**Preprocessing**: resize to 1024×1024, divide by 255, ImageNet mean/std `[0.485,0.456,0.406]`/`[0.229,0.224,0.225]`, NCHW planar.

**Fidelity**: split pipeline corr **1.0** vs the full PyTorch model; on-device circle self-test `mask_fg=11376` (PyTorch ≈ 11816). Pixel 8a GPU: encoder ~110–220 ms (first run includes shader compile), decoder ~60 ms/tap → interactive tap-to-segment.

**Sample app**: [edgetam/](edgetam/) — image picker + tap-to-segment with mask overlay. Conversion: [edgetam/scripts/convert_edgetam.py](edgetam/scripts/convert_edgetam.py).

**Original project**: [facebook/EdgeTAM](https://github.com/facebookresearch/EdgeTAM) ([yonigozlan/EdgeTAM-hf](https://huggingface.co/yonigozlan/EdgeTAM-hf)) | [Apache-2.0](https://github.com/facebookresearch/EdgeTAM/blob/main/LICENSE)

### EdgeTAM Video (SAM2 tracking)

EdgeTAM's full **video object tracking** (Segment Anything 2 memory mechanism) running on-device on CompiledModel GPU — tap an object on the first frame and it is segmented and **tracked across the following frames**, on the GPU. This is the SAM2 memory pipeline (not per-frame re-segmentation): each frame's mask conditions the next via a rolling memory bank.

Four stateless per-frame graphs run on the GPU; the rolling memory bank (7 spatial-memory frames + up to 16 object-pointer frames) is managed in Kotlin — the standard on-device SAM2 split. Verified frame-by-frame **IoU ~1.0** vs the HF PyTorch model, and **on-device GPU tracking verified** on a Pixel 8a (the mask follows a moving target within ~1 px/frame).

| Graph | Role | Size (FP16) |
| ----- | ---- | ----------- |
| [encode.tflite](https://huggingface.co/mlboydaisuke/EdgeTAM-LiteRT)   | frame → image features + FPN                          | 10 MB |
| [memcond.tflite](https://huggingface.co/mlboydaisuke/EdgeTAM-LiteRT) | memory attention over the fixed-7 bank (masked) → conditioned features | 26 MB |
| [decode.tflite](https://huggingface.co/mlboydaisuke/EdgeTAM-LiteRT)   | mask decoder → 3 masks + IoU + object pointers + score | 18 MB |
| [memorize.tflite](https://huggingface.co/mlboydaisuke/EdgeTAM-LiteRT) | memory encoder + 2D Spatial Perceiver → new memory | 5 MB |

**Conversion** (litert-torch): the memory attention's **RoPE** (5D `rotate_pairwise`) is rewritten as a baked even/odd projection permutation + `rotate_half` with constant cos/sin; the cross-attention temporal/spatial 5D regroup is replaced by a constant masked `cos_k`; the 2D Spatial Perceiver's **Swin-style window partition (6D)** is replaced by a **grouped one-hot `Conv2d` space-to-depth** (stays 4D — the trick that lets window attention run on CompiledModel GPU at all); and several on-device-only fixes (constant-input `MEAN`/`DIV`/`SELECT` are rejected by ML Drift: latents tainted to runtime, single-key softmax skipped, sine position-encodings baked). See [GPU Compatibility Notes](#gpu-compatibility-notes) and [edgetam-video/scripts/convert_edgetam_video.py](edgetam-video/scripts/convert_edgetam_video.py).

**Preprocessing**: per-frame resize to 1024×1024, divide by 255, ImageNet mean/std, NCHW planar.

**Sample app**: [edgetam-video/](edgetam-video/) — pick a video, tap an object on the first frame, watch the tracked mask overlay play back. Pixel 8a GPU ~0.45 s/frame (offline). Reference pipeline: [edgetam-video/scripts/deploy_ref_flat.py](edgetam-video/scripts/deploy_ref_flat.py).

**Original project**: [facebook/EdgeTAM](https://github.com/facebookresearch/EdgeTAM) ([yonigozlan/EdgeTAM-hf](https://huggingface.co/yonigozlan/EdgeTAM-hf)) | [Apache-2.0](https://github.com/facebookresearch/EdgeTAM/blob/main/LICENSE)

# Semantic Segmentation

### LR-ASPP MobileNetV3

Lite R-ASPP with a MobileNetV3-Large backbone (torchvision, COCO-VOC 21 classes): real-time **semantic** segmentation — labels every pixel as one of 21 classes (person, dog, car, chair, …). Pure CNN → runs **fully on the GPU** (`242/242` LITERT_CL on a Pixel 8a, **~5 ms** at 512×512, device-vs-PyTorch corr **0.99998** / argmax agreement 99.85%). At **6.7 MB** fp16 it is the smallest model in this repo.

Converted via **litert-torch** with a single re-authoring: the MobileNetV3 Squeeze-Excite blocks and the R-ASPP scale branch use `AdaptiveAvgPool2d(1)` (global pool), each replaced by `mean(3).mean(2)` (two single-axis means — a single multi-axis pool is mis-computed on Mali). Everything else is already GPU-clean (`Hardswish`/`Hardsigmoid` → native `HARD_SWISH`, `align_corners=False`). Per-pixel argmax runs on the CPU.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [lraspp_fp16.tflite](https://huggingface.co/mlboydaisuke/LRASPP-MobileNetV3-LiteRT) | 6.7 MB | Float32 [1, 3, 512, 512] NCHW | Logits [1, 512, 512, 21] NHWC | [torchvision](https://github.com/pytorch/vision) | [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) | [lraspp/](lraspp/) |

**Preprocessing**: divide by 255, ImageNet mean/std `[0.485,0.456,0.406]`/`[0.229,0.224,0.225]`, NCHW. Output is 21-class logits; argmax per pixel for the class map.

**Sample app**: [lraspp/](lraspp/) — image picker + VOC class colormap overlay.

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

# Zero-Shot Classification

### CLIP ViT-B/32

CLIP: Zero-shot image classification using OpenAI's CLIP ViT-B/32 image encoder with pre-computed text embeddings. Classify any image into 96 diverse labels without task-specific training.

Converted via **litert-torch** (ViT architecture). Text embeddings pre-computed with prompt template "a photo of a {label}".

| Model | Download Link | Size | Input | Output | API |
| ----- | ------------- | ---- | ----- | ------ | --- |
| Image Encoder | [clip_image_encoder.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/clip_image_encoder.tflite) | 352 MB | Float32 [1, 3, 224, 224] NCHW | Float32 [1, 512] | CompiledModel GPU |
| Text Embeddings | [text_embeddings.bin](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/text_embeddings.bin) | 192 KB | — | Float32 [96, 512] | Pre-computed |

**Preprocessing**: RGB with CLIP normalization (mean=[122.77, 116.75, 104.09], std=[68.50, 66.63, 70.32]). Center-crop to square, resize to 224x224. NCHW planar layout.

**Classification**: Cosine similarity between image embedding and text embeddings → softmax with temperature 100.

**Sample app**: [clip/](clip/) — Image picker + top-10 classification results with confidence bars.

**Original project**: [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | [MIT](https://github.com/mlfoundations/open_clip/blob/main/LICENSE)

### MobileNetV3-Large (ImageNet)

MobileNetV3-Large (torchvision, ImageNet-1k): a standard supervised image classifier (top-1/top-5 of 1000 classes) — the supervised counterpart to the CLIP zero-shot model above. Pure CNN → runs **fully on the GPU** (`233/233` LITERT_CL on a Pixel 8a, **~4 ms**, device-vs-PyTorch corr **0.99995**, top-1 match). At **11.2 MB** fp16 it is one of the smallest models here.

Converted via **litert-torch** with a single re-authoring: the 9 `AdaptiveAvgPool2d(1)` global pools (Squeeze-Excite blocks + the final classifier pool) → `mean(3).mean(2)`. `Hardswish`/`Hardsigmoid` lower to the native `HARD_SWISH` builtin. Softmax + top-k run on the CPU.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [mnv3_fp16.tflite](https://huggingface.co/mlboydaisuke/MobileNetV3-Large-ImageNet-LiteRT) | 11.2 MB | Float32 [1, 3, 224, 224] NCHW | Logits [1, 1000] | [torchvision](https://github.com/pytorch/vision) | [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) | [mobilenetv3/](mobilenetv3/) |

**Preprocessing**: center-crop, resize to 224×224, /255, ImageNet mean/std, NCHW. Output 1000-class logits; softmax + argmax for top-k.

**Sample app**: [mobilenetv3/](mobilenetv3/) — image picker + top-5 predictions.

### Places365 ResNet18 (scene recognition)

ResNet18 trained on [Places365](http://places2.csail.mit.edu/) (CSAILVision, MIT): **scene/place recognition** across **365 categories** (beach, kitchen, forest, office, restaurant, …) — a distinct task from object classification (it answers *what kind of place* a photo is). Pure CNN → runs **fully on the GPU** (`61/61` LITERT_CL on a Pixel 8a, **~2 ms**, fp16 **22.8 MB**, device-vs-PyTorch corr **1.0**, top-1 match).

Converted via **litert-torch** with two numerically-exact re-authorings: the global `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)`, and the ResNet stem `MaxPool2d(3,s2,p1)` → **zero-pad + valid max-pool** (PyTorch's max-pool pads with `-inf` → a `PADV2` the Mali delegate won't delegate; since the pool follows a ReLU, a 0-pad is exactly equivalent and emits a delegatable `PAD`).

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [places_fp16.tflite](https://huggingface.co/litert-community/Places365-ResNet18-LiteRT) | 22.8 MB | Float32 [1, 3, 224, 224] NCHW | Logits [1, 365] | [CSAILVision/places365](https://github.com/CSAILVision/places365) | [MIT](https://github.com/CSAILVision/places365/blob/master/LICENSE) | [places365/](places365/) |

**Preprocessing**: center-crop, resize to 224×224, /255, ImageNet mean/std, NCHW. Output 365-class scene logits; softmax + argmax for top-k.

**Sample app**: [places365/](places365/) — image picker + top-5 scene categories.

# Surface Normal Estimation

### DSINE

DSINE (CVPR 2024): Per-pixel surface normal estimation from a single image. Outputs unit normal vectors visualized as RGB color map. Uses EfficientNet-B5 encoder with a custom decoder incorporating camera ray direction encoding.

Converted via **litert-torch** with encoder + decoder initial prediction only (ConvGRU iterative refinement skipped for TFLite compatibility). Additional patches: GroupNorm → 4D manual ops, Conv2d_WS weights baked, F.normalize replaced.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [dsine.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/dsine.tflite) | 282 MB | Float32 [1, 3, 480, 640] NCHW | Float32 [1, 3, 480, 640] | [baegwangbin/DSINE](https://github.com/baegwangbin/DSINE) | [MIT](https://github.com/baegwangbin/DSINE/blob/main/LICENSE) | [dsine/](dsine/) |

**Preprocessing**: RGB with ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). NCHW planar layout.

**Output format**: `[1, 3, 480, 640]` — unit normal vectors (X, Y, Z) in [-1, 1]. Visualize as `RGB = (normal + 1) / 2 * 255`.

# Speech Recognition

### Whisper-tiny

Whisper: OpenAI's speech recognition model running on-device. First implementation with LiteRT GPU-accelerated encoder. Supports microphone recording and audio file input with 10 language options.

Encoder converted via **litert-torch** with SigmoidGELU patch. Decoder exported to ONNX with manual attention (SDPA disabled for ONNX compatibility). No KV-cache — acceptable for tiny's 4 decoder layers. Mel spectrogram computed in pure Kotlin (FFT + filterbank).

| Model | Download Link | Size | Input | Output | API |
| ----- | ------------- | ---- | ----- | ------ | --- |
| Encoder | [whisper_encoder.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/whisper_encoder.tflite) | 33 MB | Float32 [1, 80, 3000] | Float32 [1, 1500, 384] | CompiledModel GPU |
| Decoder | [whisper_decoder.onnx](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/whisper_decoder.onnx) | 199 MB | Tokens [1, seq] int64 + Audio [1, 1500, 384] | Logits [1, seq, 51865] | ONNX Runtime CPU |

**Preprocessing**: 16kHz mono audio → log-mel spectrogram (80 bins, 3000 frames). Mel computation in Kotlin with Cooley-Tukey FFT.

**Decoding**: Autoregressive greedy decoding. Prompt: `[SOT, language, transcribe, no_timestamps]`. Max 224 tokens.

**Sample app**: [whisper/](whisper/) — Microphone recording + audio file picker + language selector + transcription display.

**Original project**: [openai/whisper](https://github.com/openai/whisper) | [MIT](https://github.com/openai/whisper/blob/main/LICENSE)

# Text-to-Speech

### Kokoro-82M

Kokoro: 82M parameter neural TTS based on StyleTTS2. Bilingual English / Japanese with 5 bundled voices, 24 kHz mono output, single ONNX graph (no model splitting). Pixel 8a CPU achieves **RTF 0.60** (3.9 s of audio synthesized in 2.4 s) — comfortably realtime.

Runs on **ONNX Runtime with NNAPI EP fallback to XNNPACK CPU**. Phonemization is pure Kotlin/Java (no NDK). **English** is robust to free-form input: numbers / currency / symbols are normalized to words ("$42.99" → "forty two dollars and ninety nine cents"), in-dictionary words use the CMU Pronouncing Dictionary (126k entries, ARPABET → Misaki IPA, bit-identical to misaki), and **out-of-dictionary words (names, brands, new words) fall back to a neural G2P** ([DeepPhonemizer](https://github.com/as-ideas/DeepPhonemizer), MIT) instead of being dropped — so nothing goes silent. **Japanese** uses the kuromoji-ipadic morphological analyzer with katakana → IPA lookup (yōon and long vowel handling).

| Model | Download Link | Size | Input | Output | API |
| ----- | ------------- | ---- | ----- | ------ | --- |
| TTS | [model_fp16.onnx](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_fp16.onnx) | 163 MB | input_ids [1, seq] int64 + style [1, 256] + speed [1] | waveform [1, samples] @ 24 kHz | ONNX Runtime |
| Voices | [voices/*.bin](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main/voices) | 510 KB each | — | Style vectors [N, 1, 256] | — |
| English G2P (OOV) | [dp_g2p_litert.tflite](https://huggingface.co/mlboydaisuke/Kokoro-G2P-en-US-LiteRT/resolve/main/dp_g2p_litert.tflite) | 51 MB FP32 | text [1, 96] float (char ids, 0-padded) | logits [1, 96, 42] (ARPABET) | LiteRT CompiledModel (CPU) |

**Bundled voices**: af_heart, am_michael, bf_emma (English), jf_alpha, jm_kumo (Japanese). Add more from [the HF voices folder](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main/voices).

**Phonemizer assets**: `cmudict.txt` (3.3 MB plain text, generated by `scripts/build_cmudict.py`), `kokoro_vocab.json` (IPA → Kokoro token IDs), and `dp_g2p_litert.tflite` (neural OOV G2P on **LiteRT**, drop into `app/src/main/assets/`; download from [Hugging Face](https://huggingface.co/mlboydaisuke/Kokoro-G2P-en-US-LiteRT) or build with `scripts/convert_dp_g2p_litert.py`). The neural model is optional — without it the app degrades to CMU + normalization. It runs on the **LiteRT CompiledModel CPU** accelerator, using a static `[1, 96]` graph with an in-graph padding mask (the fixed-shape design that suits edge runtimes) and a fused-QKV attention layout that fits the CPU path.

**Sample app**: [kokoro/](kokoro/) — Free-form text input with auto-language detection, voice picker, preset phrase fallback, AudioTrack PCM_FLOAT playback.

**Original project**: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | [Apache-2.0](https://huggingface.co/hexgrad/Kokoro-82M)

### Matcha-TTS

Matcha-TTS (LJSpeech): conditional flow-matching acoustic model + **HiFi-GAN time-domain vocoder**. This is the **FFT-free** TTS lane — there is **no FFT/iSTFT anywhere** in the synthesis path (spectral vocoders like Kokoro/Vocos are blocked on the missing ML Drift FFT kernel). 22.05 kHz output. All three graphs convert **GPU-clean** (parity 1.0); on the Pixel 8a the text encoder + vocoder run on the GPU and the CFM decoder runs on the **CPU** (a Mali ML Drift transformer-fusion bug — see the matcha/ README), keeping the pipeline realtime (RTF ~0.8). The Euler ODE loop, duration/length-regulator and embedding run host-side.

| Model | Download Link | Size | Input | Output | API |
| ----- | ------------- | ---- | ----- | ------ | --- |
| Text encoder | [matcha_textenc_fp16.tflite](https://huggingface.co/litert-community/Matcha-TTS) | 15 MB | emb [1,256,192] + mask [1,1,256] | mu [1,80,256] + logw [1,1,256] | CompiledModel GPU |
| CFM decoder | [matcha_decoder_fp16.tflite](https://huggingface.co/litert-community/Matcha-TTS) | 23 MB | x,mu [1,80,512] + t_sin [1,160] + mask [1,1,512] | v [1,80,512] | CompiledModel CPU |
| HiFi-GAN vocoder | [matcha_vocoder_fp16.tflite](https://huggingface.co/litert-community/Matcha-TTS) | 29 MB | mel [1,80,512] | wav [1,1,131072] | CompiledModel GPU |
| English G2P | [dp_g2p_matcha_fp16.tflite](https://huggingface.co/litert-community/Matcha-TTS) | 26 MB | text [1,96] float (char ids) | logits [1,96,64] (IPA) | CompiledModel CPU |

**Fixed shapes** (`MAX_TEXT=256` phonemes, `MAX_MEL=512` frames ≈ 5.9 s); a **runtime float mask** makes padded positions a no-op (additive attention bias), so one compiled graph handles any length without recompiling.

**G2P (espeak-free)**: Matcha-LJSpeech is trained on espeak en-us IPA, but espeak is GPL. The clean replacement is a hybrid (same shape as kokoro's): a **275k-entry espeak-IPA dictionary** (from [OpenPhonemizer](https://github.com/NeuralVox/OpenPhonemizer), Clear BSD) as primary, with [DeepPhonemizer](https://github.com/as-ideas/DeepPhonemizer) (MIT, espeak-IPA checkpoint) on **LiteRT CompiledModel CPU** for out-of-dictionary words. Output IPA maps 1:1 onto the keithito 178-symbol set.

**Conversion** (litert-torch): `GroupNorm → 4D`, `Mish → SELECT-free softplus`, `ConvTranspose1d → ZeroStuffConvT1d` (no `TRANSPOSE_CONV`), diffusers `Attention → manual additive-masked` (mask is a runtime input — decoder adds the raw 0/1 mask = `AttnProcessor2_0`'s soft bias, text-enc adds `(mask-1)·1e4`), half-res mask via reshape-decimate (a step-2 slice → `GATHER_ND`), time embedding host-side (weight-free sin/cos) with `time_mlp` on GPU. Per-graph tflite-vs-torch corr 1.000000; end-to-end waveform corr ≥0.99. See [matcha/scripts/build_matcha.py](matcha/scripts/build_matcha.py) and [GPU Compatibility Notes](#gpu-compatibility-notes).

**Sample app**: [matcha/](matcha/) — type text, synthesize on the GPU, AudioTrack PCM_FLOAT playback.

**Original project**: [shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) | [MIT](https://github.com/shivammehta25/Matcha-TTS/blob/main/LICENSE)

# Vision-Language Model

### SmolVLM-256M

SmolVLM: On-device vision-language model that can describe images and answer questions about them. SigLIP vision encoder compresses images to just 64 visual tokens via pixel shuffle, feeding into a SmolLM2 language model for text generation with streaming output.

Vision encoder converted via **litert-torch** with SigLIP position embedding pre-computation (bypassing torch.bucketize). LM decoder exported to ONNX with causal mask patch. Repetition penalty prevents generation loops.

| Model | Download Link | Size | Input | Output | API |
| ----- | ------------- | ---- | ----- | ------ | --- |
| Vision Encoder | [smolvlm_vision.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/smolvlm_vision.tflite) | 357 MB | Float32 [1, 3, 512, 512] NCHW | Float32 [1, 64, 576] | CompiledModel GPU |
| LM Decoder | [smolvlm_decoder.onnx](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/smolvlm_decoder.onnx) | 515 MB | Float32 [1, seq, 576] | Float32 [1, seq, 49280] | ONNX Runtime CPU |
| Token Embeddings | [embed_tokens.bin](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/embed_tokens.bin) | 108 MB | — | Float32 [49280, 576] | — |

**Preprocessing**: Image normalized to [-1, 1] (pixel/127.5 - 1). Center-crop to square, resize to 512x512. NCHW layout.

**Generation**: Greedy decoding with repetition penalty 1.2x. Prompt format: `<|im_start|>User:<image>{prompt}<end_of_utterance>\nAssistant:`

**Sample app**: [smolvlm/](smolvlm/) — Image picker + text prompt + streaming response.

**Original project**: [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | [Apache-2.0](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/main/LICENSE)

# Voice Assistant

### Whisper + SmolLM2 + Kokoro pipeline

Full on-device conversational pipeline running entirely offline on a Pixel-class device. Hands-free with **Silero VAD** driving turn taking and barge-in:

```
mic ─► AudioRecord (VOICE_COMMUNICATION + AEC/NS/AGC)
     ─► Silero VAD v5 (ONNX CPU, 32 ms chunks) ─► SegmentTracker hysteresis
     ─► [SPEECH_END] Whisper-tiny STT (TFLite GPU + ONNX CPU decoder)
     ─► SmolLM2-135M chat (ONNX CPU, cancellable streaming)
     ─► EnglishPhonemizer (CMU dict + ARPABET → IPA)
     ─► Kokoro-82M TTS (ONNX, NNAPI EP)
     ─► AudioTrack streaming playback (hard-stoppable on barge-in)
```

**Hands-free turn taking.** Tap **Listen** once. The mic stays open, VAD watches every 32 ms chunk, and a 600 ms trailing silence after a speech segment automatically submits the captured audio to the pipeline. A 5-chunk preroll ring buffer is prepended to the capture so the first phoneme that triggered VAD is not clipped.

**Barge-in.** While the assistant is replying, mic + VAD keep running. A new SPEECH_START event during THINKING/SPEAKING flips the in-flight turn's cancellation flag, hard-stops the AudioTrack, aborts the LM generation between tokens, and starts capturing the new utterance as the next turn.

**Streaming TTS** drops time-to-first-audio from ~4.5 s to **~1.5 s** on Pixel 8a: the LM token-by-token callback detects sentence boundaries, each completed sentence is phonemized and synthesized while the LM keeps generating, and audio chunks are pushed to a player thread via a blocking queue. Each chunk plays via a one-shot MODE_STATIC AudioTrack, decoupling the LM/TTS producer from playback duration.

**Per-stage on Pixel 8a** (short replies):
- STT: ~700 ms
- LM: ~1000-1500 ms
- TTS total: ~1100 ms (sentence-chunked)
- End-to-end: ~5 s for a typical Q&A turn

| Component | Model | Size |
| --------- | ----- | ---- |
| VAD | [silero_vad.onnx](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/silero_vad.onnx) | 2.3 MB |
| STT encoder | [whisper_encoder.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/whisper_encoder.tflite) | 33 MB |
| STT decoder | [whisper_decoder.onnx](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/whisper_decoder.onnx) | 199 MB |
| LM decoder | [smolvlm_decoder.onnx](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/smolvlm_decoder.onnx) | 515 MB |
| LM embeddings | [embed_tokens.bin](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/embed_tokens.bin) | 108 MB |
| TTS | [model_fp16.onnx](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_fp16.onnx) | 163 MB |
| TTS voices | [voices/*.bin](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main/voices) | 510 KB each |

**Sample app**: [voiceassistant/](voiceassistant/) — Listen toggle, idle/listening/capturing/thinking/replying state indicator, transcript + streaming response display, hands-free turn taking, barge-in. English-only for the MVP.

# Audio Codec

### DAC 16kHz

[Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) (DAC, 16 kHz) — a neural audio codec on **CompiledModel GPU**. Compresses 1 s of audio to 12×50 = 600 int codes (~43:1) and reconstructs it; the convolutional encoder/decoder run on the GPU, the RVQ on CPU (~1 ms). The sample app round-trips a clip and plays original vs. reconstructed to A/B by ear.

**On-device (Pixel 8a, Tensor G3 — verified):** encoder **367/367** + decoder **398/398** nodes on the LiteRT GPU delegate (`LITERT_CL`, no CPU fallback), warm **RTF ≈ 0.82** (faster than real-time), reconstruction **corr 1.0** vs PyTorch DAC.

| Model | Download | Size | Input → Output | Original Project | License | Sample App |
| ----- | -------- | ---- | -------------- | ---------------- | ------- | ---------- |
| Encoder | [HF: mlboydaisuke/DAC-16kHz-LiteRT](https://huggingface.co/mlboydaisuke/DAC-16kHz-LiteRT) | 43 MB FP16 | audio [1,1,16000] → latent [1,1024,50] | [descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec) | MIT | [dac/](dac/) |
| Decoder | (same HF repo) | 105 MB FP16 | latent [1,1024,50] → audio | — | MIT | [dac/](dac/) |

**Pipeline**: audio → encoder.tflite (GPU) → RVQ encode (CPU) → codes[12,50] → RVQ decode (CPU) → decoder.tflite (GPU) → audio.

**GPU compatibility**: the decoder's `ConvTranspose1d` are rewritten to a GPU-clean **zero-stuff** form (`ZeroStuffConvT1d` — the real DAC's odd stride-5 transposed conv fails converter legalization and `TRANSPOSE_CONV` is rejected by Mali); the RVQ (`EMBEDDING_LOOKUP` + int64 indices, Mali-rejected) runs on CPU. The big tflites are pushed to the app's filesDir via `dac/scripts/install_to_device.sh`.

### Mimi (Kyutai 2024)

[Mimi](https://huggingface.co/kyutai/mimi) (Kyutai/Moshi streaming neural codec, 24 kHz, 12.5 Hz) — a **hybrid** on-device codec: the heavy SEANet convolutional halves run on **CompiledModel GPU**, the two 8-layer Transformers + split RVQ on CPU. A 2 s clip round-trips faster than real-time; the app A/Bs original vs. reconstructed.

**On-device (Pixel 8a, Tensor G3 — verified):** enc_conv **189/189** + deconly **220/220** nodes on the LiteRT GPU delegate (`LITERT_CL`); encoder/decoder Transformers on CPU (XNNPACK). encode ≈ 0.49 s · decode ≈ 0.18 s for 2 s → **RTF ≈ 0.35**; reconstruction at the codec's own quality floor (device-vs-input corr = the PyTorch Mimi reference).

| Model | Download | Size | Input → Output | Placement |
| ----- | -------- | ---- | -------------- | --------- |
| enc_conv | [HF: litert-community/Mimi](https://huggingface.co/litert-community/Mimi) | 24 MB FP16 | audio [1,1,L] → feat [1,512,Se] | CompiledModel GPU |
| enc_tx | (same) | 50 MB FP16 | feat [1,Se,512] → emb [1,512,Tc] | CompiledModel CPU |
| dec_tx | (same) | 48 MB FP16 | emb [1,512,Tc] → conv_in [1,512,seq] | CompiledModel CPU |
| deconly | (same) | 28 MB FP16 | conv_in [1,512,seq] → audio [1,1,L] | CompiledModel GPU |
| RVQ weights | `mimi_rvq.bin` | 69 MB | codes ↔ emb (32 codebooks) | CPU |

**Why hybrid (the C33 result)**: every op is GPU-clean and the convs are fp16-exact on Mali (decoder-only fed the exact transformer output = **48 dB**), but the decoder transformer's residual stream reaches **|x|=27** and the Mali fp16 compute loses precision there (full-GPU decode ~12 dB on real speech). It behaves **identically standalone and fused** on device, so this is fp16 **precision**, not a fusion collapse — **the Matcha "C33" transformer-fusion bug does NOT generalize** to Mimi's own transformer (it is diffusers-specific). Transformers → CPU (tiny, exact); convs → GPU. RVQ → CPU (Euclidean argmin + int64, Mali-rejected).

**Re-authoring** (litert-torch): GELU→tanh-GELU, RoPE→baked cos/sin + rotate_half, causal mask→baked additive bias, `MimiLayerScale`→bake into Linear, depthwise `ConvTranspose1d`→grouped `ZeroStuffConvT1d`, `MimiConv1d` causal pad→baked `F.pad`, `nn.ELU`→`relu(x)−relu(1−exp(min(x,0)))`, replicate-pad→SLICE+CONCAT. Per-graph tflite-vs-torch corr 1.0; full round-trip corr 1.0. See [mimi/scripts/](mimi/scripts/) and [GPU Compatibility Notes](#gpu-compatibility-notes).

**Sample app**: [mimi/](mimi/) — round-trips a clip, plays original vs. reconstructed (AudioTrack).

**Original project**: [kyutai/mimi](https://huggingface.co/kyutai/mimi) | [CC-BY-4.0](https://huggingface.co/kyutai/mimi)

# Audio Classification

### wav2vec2 Keyword Spotting

[wav2vec2](https://huggingface.co/facebook/wav2vec2-base) keyword spotting ([`superb/wav2vec2-base-superb-ks`](https://huggingface.co/superb/wav2vec2-base-superb-ks)) running **fully on CompiledModel GPU**. Classifies 1 s of 16 kHz audio into 12 Speech-Commands labels. **No FFT anywhere** — the raw waveform goes straight into a 1D-conv feature extractor (no mel step), so the whole model rides the GPU delegate. The sample classifies a bundled clip and records keywords from the mic.

**On-device (Pixel 8a, Tensor G3 — verified):** frontend **134/134** + head **893/893** nodes on the LiteRT GPU delegate (`LITERT_CL`), end-to-end ~19 ms for a 1 s clip (**RTF ≈ 0.02**); real-speech validation 10/10 keywords, device-vs-CPU logits corr 0.9995.

| Model | Download | Size | Input → Output | Placement |
| ----- | -------- | ---- | -------------- | --------- |
| frontend | [HF: litert-community/wav2vec2-keyword-spotting](https://huggingface.co/litert-community/wav2vec2-keyword-spotting) | 9 MB FP16 | audio [1,16000] → feat [1,49,768] | CompiledModel GPU |
| head | (same repo) | 181 MB FP16 | feat [1,49,768] → logits [1,12] | CompiledModel GPU |

**Why two graphs**: the model is op-clean but the full 1008-node graph exceeds the Mali shader-compile limit (fails fused); splitting at the conv-frontend / transformer-encoder boundary makes each half compile (134/134 + 893/893). Both run on the GPU.

**Re-authoring** (litert-torch): GELU→tanh-GELU, feature-extractor GroupNorm→GN4D, pos-conv `weight_norm` fold, `create_bidirectional_mask`→None, and the `use_weighted_layer_sum` head accumulated incrementally with **baked** softmax layer-weights (stack-13 + runtime `w[i]` gathers split the Mali partition). Residual peaks at `|x|≈3.2` so it is fp16-exact on GPU (no CPU fallback). Per-graph tflite-vs-torch corr 1.0. See [wav2vec2-kws/scripts/](wav2vec2-kws/scripts/).

**Sample app**: [wav2vec2-kws/](wav2vec2-kws/) — bundled-clip classify on launch + mic Record button.

**Original project**: [superb/wav2vec2-base-superb-ks](https://huggingface.co/superb/wav2vec2-base-superb-ks) | [Apache-2.0](https://huggingface.co/facebook/wav2vec2-base)

# Line Detection

### M-LSD-tiny

[M-LSD](https://github.com/navervision/mlsd) (NAVER, AAAI 2022): light-weight real-time **line segment detection** — straight line segments for building edges, document borders, wireframes, and room layout. The **tiny** variant (MobileNetV2 backbone, 0.62M params) runs **fully on the GPU** (`99/99` LITERT_CL on a Pixel 8a, **~2 ms**, device-vs-PyTorch corr **0.997**). At **1.4 MB** fp16 it is the **smallest model in this zoo**.

Converted via **litert-torch** with a single re-authoring: the decoder's `F.interpolate(bilinear, align_corners=True)` → `align_corners=False` (the delegate bans `align_corners=True`). MobileNetV2 has no max-pool (strided convs → no `PADV2`) and the upsample is `RESIZE_BILINEAR` (not a transposed conv) → fully GPU-clean. The output is a "TP map" (center heatmap + displacement); the decode (sigmoid + NMS + displacement → endpoints) runs in the app.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [mlsd_fp16.tflite](https://huggingface.co/litert-community/M-LSD-tiny-LiteRT) | 1.4 MB | Float32 [1, 4, 512, 512] NCHW (RGB + ones) | tpMap [1, 9, 256, 256] | [navervision/mlsd](https://github.com/navervision/mlsd) | [Apache-2.0](https://github.com/navervision/mlsd/blob/main/LICENSE) | [mlsd/](mlsd/) |

**Preprocessing**: resize to 512×512, append a 4th channel of ones, scale `(x/127.5)-1`, NCHW. **Decode**: sigmoid center map → 3×3 max NMS → displacement → endpoints (×2 to 512-space).

**Sample app**: [mlsd/](mlsd/) — image picker + line-segment overlay.

# OCR

### PP-OCRv5

[PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR) (PaddleOCR 2025) text detection + recognition running **fully on CompiledModel GPU**. Detects text regions and reads each line. **No autoregressive decoder** (recognition uses a CTC head), so both stages ride the GPU with no CPU/ONNX fallback — unlike VLM-based OCR (Florence-2/GOT-OCR) whose AR decoder must run on CPU. The sample runs OCR on a bundled image and overlays boxes + recognized text.

**On-device (Pixel 8a, Tensor G3 — verified):** detector **777/777** + recognizer **827/827** nodes on the LiteRT GPU delegate (`LITERT_CL`), ~9 ms each; bundled 3-line image read 3/3 correct ("Hello OCR 2026" / "PP-OCRv5 on GPU" / "LiteRT CompiledModel").

| Model | Download | Size | Input → Output | Placement |
| ----- | -------- | ---- | -------------- | --------- |
| Detection (DBNet) | [HF: litert-community/PP-OCRv5-LiteRT](https://huggingface.co/litert-community/PP-OCRv5-LiteRT) | 10 MB FP16 | image [1,3,640,640] → prob map [1,1,640,640] | CompiledModel GPU |
| Recognition (SVTR+CTC) | (same repo) | 17 MB FP16 | line [1,3,48,320] → logits [1,T,18385] | CompiledModel GPU |

**Pipeline**: image → detector.tflite (GPU) → DB box postprocess (CPU) → crop → recognizer.tflite (GPU) → CTC decode (CPU) → text.

**GPU compatibility**: the detector's DB-head `ConvTranspose2d` are rewritten to a GPU-clean **`ZeroStuffConvT2d`** (2D nearest-upsample × zero-stuff mask + flipped conv2d — the DAC/DA3 zero-stuff trick generalized to 2D; `TRANSPOSE_CONV` is Mali-rejected), and the recognizer's SVTR attention fused-QKV **5D reshape** is split into 4D. Per-graph tflite-vs-torch corr 1.0. Weights via the [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch) port (Apache-2.0). See [ppocr/scripts/](ppocr/scripts/).

**Sample app**: [ppocr/](ppocr/) — runs OCR on a bundled image, overlays detected boxes + recognized text.

**Original project**: [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | [Apache-2.0](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)

# Super Resolution

### Real-ESRGAN x4v3

<img src="https://github.com/user-attachments/assets/ba4466ad-ed23-4bbe-bffd-53b3b3cc3a4a" width="300">

Real-ESRGAN: Practical image restoration and upscaling. The General-x4v3 variant is a lightweight model (1.21M params) with excellent quality for 4x super resolution.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [real_esrgan_x4v3.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v1/real_esrgan_x4v3.tflite) | 4.7 MB | Float32 [1, 128, 128, 3] NHWC | Float32 [1, 512, 512, 3] NHWC | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | [BSD-3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) | [real-esrgan/](real-esrgan/) |

**Output format**: `[1, 512, 512, 3]` — 4x upscaled RGB image (0-1 range).

**Preprocessing**: RGB normalized to 0-1 (divide by 255). For images larger than 128x128, process as overlapping tiles and stitch.

# Style Transfer

### Fast Neural Style (4 styles)

Fast neural **style transfer** ([PyTorch examples](https://github.com/pytorch/examples/tree/main/fast_neural_style) `TransformerNet`, Johnson et al.): applies an artistic style to a photo — **4 styles** (candy / mosaic / rain_princess / udnie), each a **3.5 MB** fp16 graph. Runs **fully on the GPU** (`350/350` LITERT_CL on a Pixel 8a, **~9 ms** @ 256×256, device-vs-PyTorch corr **0.9998–0.9999** for all styles).

Converted via **litert-torch** with three numerically-exact re-authorings: (1) `ReflectionPad2d` → zero-pad (`GATHER_ND` → `PAD`); (2) the large conv activations (≈|5000|) lose fp16 precision on Mali (corr 0.34 at full residency) → **scale the conv weights down (InstanceNorm is scale-invariant → exact)** so the fp16 accumulation stays precise; (3) `InstanceNorm` → **SafeInstanceNorm** (down-scaled-domain spatial reduction, fp16-safe). Upsample is `interpolate(nearest)` (no ZeroStuff).

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [Fast-Neural-Style-LiteRT](https://huggingface.co/litert-community/Fast-Neural-Style-LiteRT) | 3.5 MB ×4 | Float32 [1, 3, 256, 256] NCHW (RGB 0-255) | [1, 3, 256, 256] (RGB 0-255) | [pytorch/examples](https://github.com/pytorch/examples) | [BSD-3-Clause](https://github.com/pytorch/examples/blob/main/LICENSE) | [neuralstyle/](neuralstyle/) |

**Preprocessing**: center-crop, resize to 256×256, RGB 0–255 (no normalization), NCHW. Output 0–255 RGB (clamp).

**Sample app**: [neuralstyle/](neuralstyle/) — image picker + 4 tappable style buttons.

### AnimeGANv2 (photo → anime)

[AnimeGANv2](https://github.com/bryandlee/animegan2-pytorch) (bryandlee, MIT): **photo-to-anime** stylization — **2 styles** (paprika = general anime, face_paint_512_v2 = anime face portrait), each ~4 MB fp16. Runs **fully on the GPU** (`685/685` LITERT_CL on a Pixel 8a, **~10 ms** @ 256×256, device-vs-PyTorch corr **0.99996**).

Converted via **litert-torch** with four numerically-exact re-authorings: `ReflectionPad2d` → zero-pad (`GATHER_ND` → `PAD`); **`GroupNorm(1)` → SafeGroupNorm** (native GroupNorm → `GATHER_ND`; manual 4D reduce over C,H,W in a down-scaled domain); **conv-weight scaling via GroupNorm scale-invariance** (keeps the large conv activations fp16-precise on Mali); bilinear `align_corners=True` → `False`. No transposed conv → no ZeroStuff.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [AnimeGANv2-LiteRT](https://huggingface.co/litert-community/AnimeGANv2-LiteRT) | ~4 MB ×2 | Float32 [1, 3, 256, 256] NCHW ([-1,1]) | [1, 3, 256, 256] ([-1,1]) | [bryandlee/animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch) | [MIT](https://github.com/bryandlee/animegan2-pytorch/blob/main/LICENSE) | [anime/](anime/) |

**Preprocessing**: center-crop, resize 256×256, RGB → [-1,1] (`x/127.5-1`), NCHW. Output [-1,1] → `(x+1)·127.5` clamp.

**Sample app**: [anime/](anime/) — image picker + 2 tappable style buttons (paprika / face).

# Image Restoration

### NAFNet (deblur)

NAFNet (Nonlinear Activation Free Network, ECCV 2022): image restoration — a U-Net of **NAFBlocks** with **no activation functions at all** (SimpleGate = channel-split multiply). The **GoPro-width32** variant removes motion blur. Pure CNN → runs **fully on the GPU** (`2179/2179` LITERT_CL on a Pixel 8a, ~42 ms at 256×256, device output **== PyTorch corr 1.0**).

Converted via **litert-torch** with three numerically-exact re-authorings: the custom `LayerNorm2d` → an **fp16-safe channel LayerNorm** (NAFNet's residual stream reaches |x|≈175, so the LayerNorm channel-sum `Σ_c(x−μ)²` ~15M **overflows fp16** (max 65504) on the Mali delegate — which computes in fp16 regardless of model dtype — giving a grid artifact; doing the reduction in a down-scaled `x/S` domain and rescaling is exact); the Simplified Channel Attention `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)`; and the upsample `PixelShuffle(2)` → depth-to-space `ZeroStuffConvT2d`.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [nafnet_fp16.tflite](https://huggingface.co/litert-community/NAFNet-GoPro-width32-LiteRT) (deblur) | 38 MB | Float32 [1, 3, 256, 256] NCHW | Float32 [1, 3, 256, 256] (RGB [0,1]) | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) | [MIT](https://github.com/megvii-research/NAFNet/blob/main/LICENSE) | [nafnet/](nafnet/) |
| [nafnet_sidd_width32_fp16.tflite](https://huggingface.co/litert-community/NAFNet-SIDD-width32-LiteRT) (denoise) | 62 MB | Float32 [1, 3, 256, 256] NCHW | Float32 [1, 3, 256, 256] (RGB [0,1]) | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) | [MIT](https://github.com/megvii-research/NAFNet/blob/main/LICENSE) | [nafnet/](nafnet/) |

**Preprocessing**: RGB normalized to 0-1 (divide by 255), NCHW planar. Output is the restored RGB image in [0,1].

**Sample app**: [nafnet/](nafnet/) — image picker showing input | restored (GoPro deblur; the same app runs the SIDD **denoise** model via `scripts/build_sidd.py`, device-verified corr 0.999999).

# Monocular Geometry Estimation

### MoGe-2 ViT-S

MoGe-2 (CVPR'25 Oral): Accurate monocular geometry estimation from a single image. Outputs an affine 3D point map, surface normals, confidence mask, and metric scale — all in a single forward pass. Based on DINOv2 ViT-S backbone with a multi-scale ConvStack decoder.

Converted via **litert-torch** with five GPU-compat patches: DINOv2 attention rewrite (4D slice instead of 5D stack+unbind), encoder output add instead of stack+sum, baked position embeddings (eliminates GATHER_ND from bicubic interpolation), replicate→zeros Conv2d padding, and bicubic→bilinear interpolation.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [moge.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v3/moge.tflite) | 136 MB | Float32 [1, 3, 448, 448] NCHW | Points [1,448,448,3] + Normal [1,448,448,3] + Mask [1,448,448,1] + Scale [1,1,1,1] | [microsoft/MoGe](https://github.com/microsoft/MoGe) | [MIT](https://github.com/microsoft/MoGe/blob/main/LICENSE) (DINOv2: Apache-2.0) | [moge/](moge/) |

**Preprocessing**: RGB normalized to 0-1 (divide by 255). NCHW planar layout. The DINOv2 encoder applies ImageNet normalization internally.

**Output format**: Four tensors:
- `points [1, 448, 448, 3]` — affine point map after exp remap (`xy * exp(z), exp(z)`)
- `normal [1, 448, 448, 3]` — L2-normalized surface normals (visualize as `(n + 1) / 2 * 255`)
- `mask [1, 448, 448, 1]` — sigmoid confidence (>0.5 = valid)
- `scale [1, 1, 1, 1]` — metric scale factor (multiply points/depth for metric units)

**Sample app**: [moge/](moge/) — Image picker + four visualization modes: normal map (RGB), depth heatmap (turbo colormap), 3D point cloud (touch-rotatable OpenGL ES), and geometry info overlay.

**Original project**: [microsoft/MoGe](https://github.com/microsoft/MoGe) | [MIT](https://github.com/microsoft/MoGe/blob/main/LICENSE)

### Depth Anything 3 ViT-S (Small)

![Depth Anything 3 Small — input | depth, on-device LiteRT GPU](https://huggingface.co/mlboydaisuke/Depth-Anything-3-Small-LiteRT/resolve/main/samples/avenue.png)

Depth Anything 3 (ByteDance-Seed, 2025): monocular depth from a single RGB image. DINOv2 ViT-S + RoPE backbone with a DPT/DualDPT depth head.

Converted via **litert-torch** with nine GPU-compat patches: RoPE data-dependent int → constant, fused-QKV → 4D attention, **LayerScale folded into the preceding Linear** (the LayerScale MUL otherwise mis-lays-out the token dim on the GPU delegate), baked bicubic `pos_embed`, **ConvTranspose2d → zero-stuff + Conv2d** (exact, since Pixel 8a rejects `TRANSPOSE_CONV`), `align_corners`→False, camera-token in-place assign → `cat` (avoids `SELECT_V2`). Processed at **native aspect** — a square letterbox drops fidelity from corr 0.9994 to 0.977 (padding leaks through global attention).

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [da3_small_gpu_fp16.tflite](https://huggingface.co/mlboydaisuke/Depth-Anything-3-Small-LiteRT) | 55 MB | Float32 [1, 3, 896, 504] NCHW | Depth [1, 1, 896, 504] | [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | [Apache-2.0](https://github.com/ByteDance-Seed/Depth-Anything-3) | [da3/](da3/) |

**Preprocessing**: resize to 504×896 (W×H), divide by 255, ImageNet mean/std `[0.485,0.456,0.406]`/`[0.229,0.224,0.225]`, NCHW planar.

**Fidelity**: Pearson corr **0.99948** vs the official PyTorch DA3-Small pipeline. FP16 is not a factor (FP32≡FP16); the residual ~0.05% is the DPT-head `align_corners=True→False` change, forced because the GPU delegate bans `align_corners=True` resize — an irreducible mobile-GPU constraint. Pixel 8a GPU ~1.8 s/image.

**Sample app**: [da3/](da3/) — loads a bundled image, runs CompiledModel GPU inference, shows input | depth.

**Original project**: [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | Apache-2.0

### Metric3D v2 ViT-S

Metric3D v2 (CVPR/TPAMI 2024): **metric** (absolute, in-meters) monocular depth from a single RGB image — a different output domain from Depth Anything (relative) / MoGe (affine) / DSINE (normals). DINOv2 ViT-S/14 + register tokens encoder with a **RAFT-DPT** iterative decoder (4 iters). Runs **fully on the GPU** (encoder *and* RAFT decoder) — `2447/2447` LITERT_CL on a Pixel 8a, ~44 ms, depth corr **0.96** vs the original.

Converted via **litert-torch** at a fixed 448×448. Encoder = the MoGe-2 DINOv2 ViT-S suite (fused-QKV→4D attention, LayerScale baked into Linear, baked pos-embed). RAFT decoder re-authoring: the **convex upsample (6/7-D)** → a **depth-to-space `ZeroStuffConvT2d`** (16 softmax-over-9 subpixel combines → fixed `ConvTranspose2d(96→6,k4,s4)`); the naive nearest-upsample + in-block mask gives correct desktop output but **corr 0.57 on Mali** (ML Drift `RESIZE_NEAREST` half-pixel differs at non-stride positions) — `ZeroStuffConvT2d` masks only stride-aligned positions and places the offset via the conv kernel. **GELU → accurate tanh approximation, not `x·sigmoid(1.702x)`**: at the coarse top of the 0.1–200 m log-depth bins the sigmoid error collapses depth corr to 0.51; the tanh form restores 0.96. `Token2Feature ConvTranspose2d` → `ZeroStuffConvT2d`; `elu`→SELECT-free; the DPT `ConvBlock`'s `inplace=True` leading ReLU mutates the residual (`relu(x)+convs`) and is replicated exactly.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [metric3d_fp16.tflite](https://huggingface.co/mlboydaisuke/Metric3D-v2-LiteRT) | 78 MB | Float32 [1, 3, 448, 448] NCHW | Depth [1, 1, 448, 448] (meters) | [YvanYin/Metric3D](https://github.com/YvanYin/Metric3D) | [BSD-2-Clause](https://github.com/YvanYin/Metric3D/blob/main/LICENSE) (DINOv2: Apache-2.0) | [metric3d/](metric3d/) |

**Preprocessing**: center-crop to square, resize to 448×448, ImageNet normalize in 0–255 scale `(px − [123.675,116.28,103.53]) / [58.395,57.12,57.375]`, NCHW planar. Output is canonical-camera metric depth; multiply by `fx/1000` for a calibrated camera.

**Sample app**: [metric3d/](metric3d/) — image picker + depth colormap with near/far metric range.

**Original project**: [YvanYin/Metric3D](https://github.com/YvanYin/Metric3D) | BSD-2-Clause

# GPU Compatibility Notes

CompiledModel GPU requires **all ops** to be GPU-compatible. Key constraints:

- All tensors must be 4D or less
- No dynamic dimensions (-1) in reshape
- Avoid: TOPK_V2, GATHER, GATHER_ND, CAST (float-int), GELU, PACK, SPLIT

**Proven conversion paths**:
1. **SavedModel → TFLiteConverter** — Eliminates PACK/SPLIT ops (used for YOLO11)
2. **Native Keras → from_keras_model()** — Full op control for ViT models
3. **litert-torch** — Only viable converter for Vision Transformers (ViT, TinyViT). onnx2tf breaks attention layers (corr≈0.3). See [docs/](docs/) for details.

**Common GPU-incompatible ops and fixes**:
- **GroupNorm** → Replace with manual 4D mean/var computation (`reshape(B*G, C//G, H, W)`)
- **Conv2d_WS** (weight standardization) → Pre-compute standardized weights, bake into regular Conv2d
- **F.normalize** → Manual `x / sqrt(sum(x*x) + eps)` to avoid div broadcast issues
- **GELU / QuickGELU** → `x * sigmoid(1.702 * x)` (SigmoidGELU approximation)
- **Swish / SiLU** → `x * sigmoid(x)`
- **`torch.bucketize`** → Pre-compute results for fixed input size, register as buffer
- **`padding='valid'` Conv2d** → Replace with `padding=0`
- **transformers `create_causal_mask`** → Monkey-patch with simple `torch.triu` mask for ONNX export
- **`scaled_dot_product_attention`** → Set `use_sdpa = False` to use manual matmul+softmax attention

> **Note**: litert-torch models use NCHW layout (PyTorch native). Large models (>150 MB) should be loaded from `filesDir` via `CompiledModel.create(path, options, null)` instead of APK assets.

# Text Generation (LLM)

> **Conversion recipes** (official `litert-torch` `export_hf`, no fork — blockwise int4 + OCTAV, `externalize_embedder`, simple chat templates): [`text-generation/`](text-generation/).

### Falcon3-3B-Instruct

[tiiuae/Falcon3-3B-Instruct](https://huggingface.co/tiiuae/Falcon3-3B-Instruct) converted to **LiteRT-LM** (`.litertlm`) for fully on-device chat. Dense `LlamaForCausalLM`, so it rides the official converter/runtime with no custom code. **int4 (blockwise-128) at parity with bf16** — GSM8K (n=100, greedy): LiteRT int4 **77%** ≈ MLX 4-bit 76% ≈ bf16 75%. Runs on **iPhone 17 Pro at ~27 tok/s** (Metal GPU).

| Type | Model | Size | Quant | Quality (GSM8K) | Source | License |
|---|---|---|---|---|---|---|
| LLM | [model.litertlm](https://huggingface.co/mlboydaisuke/Falcon3-3B-Instruct-LiteRT/resolve/main/model.litertlm) | ~1.7 GB | int4 blockwise-128 + int8 emb | 77% (≈ bf16 75%) | [tiiuae/Falcon3-3B-Instruct](https://huggingface.co/tiiuae/Falcon3-3B-Instruct) | [Falcon LLM License](https://falconllm.tii.ae/falcon-terms-and-conditions.html) |

**Run it**: via the [LiteRT-LM Swift sample app](https://github.com/john-rocky/swift-litert-lm) (model picker → *Falcon3-3B Instruct*), or any [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime. The `.litertlm` bundles the tokenizer and prompt template — no extra files. (This is an LLM `.litertlm` run by the LiteRT-LM runtime, not the `.tflite` / `CompiledModel` path above.)

**HF model**: [mlboydaisuke/Falcon3-3B-Instruct-LiteRT](https://huggingface.co/mlboydaisuke/Falcon3-3B-Instruct-LiteRT)

**Original project**: [tiiuae/Falcon3-3B-Instruct](https://huggingface.co/tiiuae/Falcon3-3B-Instruct) | [Falcon LLM License](https://falconllm.tii.ae/falcon-terms-and-conditions.html)

### Llama-3.2-3B-Instruct

Built with Llama. [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) converted to **LiteRT-LM** (`.litertlm`) for fully on-device chat. Dense `LlamaForCausalLM`, converted with the **official** litert-torch — no custom code. **int4 (blockwise-32) at parity** — GSM8K (n=100, greedy): LiteRT int4 **73%** == MLX 4-bit 73% (−5 pt vs bf16 78%). Runs on **iPhone 17 Pro at ~18.5 tok/s** (Metal GPU, loads in 8.8 s).

| Type | Model | Size | Quant | Quality (GSM8K) | Source | License |
|---|---|---|---|---|---|---|
| LLM | [model.litertlm](https://huggingface.co/mlboydaisuke/Llama-3.2-3B-Instruct-LiteRT/resolve/main/model.litertlm) | ~2.1 GB | int4 blockwise-32 + int8 emb (externalized) | 73% (== MLX-4bit, bf16 78%) | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) |

**Run it**: via the [LiteRT-LM Swift sample app](https://github.com/john-rocky/swift-litert-lm) (model picker → *Llama-3.2-3B Instruct*), or any [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime. The `.litertlm` bundles the tokenizer and prompt template — no extra files.

**iOS note**: exported with `externalize_embedder=True` so the embedding is its own section — a 28-layer 3B's single weights section otherwise exceeds the iOS ~2 GiB `mmap` limit (engine-create fails). This is the generic equivalent of Gemma's per-layer-embedding mmap; it also dedups the tied embedding (2.48 GB → 2.1 GB). Desktop/Mac load the un-split model fine.

**HF model**: [mlboydaisuke/Llama-3.2-3B-Instruct-LiteRT](https://huggingface.co/mlboydaisuke/Llama-3.2-3B-Instruct-LiteRT)

**Original project**: [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/)

### Ministral-3-3B-Instruct-2512

[mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) converted to **LiteRT-LM** (`.litertlm`) for fully on-device chat. Dense `Ministral3ForCausalLM` (YaRN RoPE); the source is multimodal, so only the text decoder is exported (vision tower dropped) — it then rides the existing converter/runtime with no custom code. **int4 (blockwise-32 + OCTAV) at parity** — GSM8K (n=100, greedy): LiteRT int4 **85%** (−4 pt vs bf16 89%). Runs on **iPhone 17 Pro at ~17.6 tok/s** (Metal GPU, loads in 7.6 s). Apache-2.0.

| Type | Model | Size | Quant | Quality (GSM8K) | Source | License |
|---|---|---|---|---|---|---|
| LLM | [model.litertlm](https://huggingface.co/mlboydaisuke/Ministral-3-3B-Instruct-2512-LiteRT/resolve/main/model.litertlm) | ~2.3 GB | int4 blockwise-32 + OCTAV + int8 emb (externalized) | 85% (−4 pt, bf16 89%) | [mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) | Apache-2.0 |

**Run it**: via the [LiteRT-LM Swift sample app](https://github.com/john-rocky/swift-litert-lm) (model picker → *Ministral-3-3B Instruct*), or any [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime. The `.litertlm` bundles the tokenizer and prompt template (Mistral `[INST]…[/INST]`, EOS `</s>` — not ChatML) — no extra files.

**iOS note**: exported with `externalize_embedder=True` so the embedding is its own section — the text decoder's single ~2.55 GiB weights section otherwise exceeds the iOS ~2 GiB `mmap` limit. Drops the main section to ~1.8 GiB (and dedups the tied embedding, 2.74 → 2.34 GB).

**HF model**: [mlboydaisuke/Ministral-3-3B-Instruct-2512-LiteRT](https://huggingface.co/mlboydaisuke/Ministral-3-3B-Instruct-2512-LiteRT)

**Original project**: [mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) | Apache-2.0

### SmolLM3-3B

[HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) converted to **LiteRT-LM** (`.litertlm`) for fully on-device chat. Dense `SmolLM3ForCausalLM` (GQA + **NoPE**, rotary disabled every 4th layer) — the per-layer NoPE lowers to generic ops through the official converter with no custom code. **int4 (blockwise-32 + OCTAV) — the tightest parity of the four** — GSM8K (n=100, greedy): LiteRT int4 **81%** == bf16 **81%** (0.0-pt drop). Runs on **iPhone 17 Pro at ~22.5 tok/s** (Metal GPU, loads in 7.7 s). Apache-2.0.

| Type | Model | Size | Quant | Quality (GSM8K) | Source | License |
|---|---|---|---|---|---|---|
| LLM | [model.litertlm](https://huggingface.co/mlboydaisuke/SmolLM3-3B-LiteRT/resolve/main/model.litertlm) | ~1.9 GB | int4 blockwise-32 + OCTAV + int8 emb (externalized) | 81% (== bf16 81%, 0-pt) | [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | Apache-2.0 |

**Run it**: via the [LiteRT-LM Swift sample app](https://github.com/john-rocky/swift-litert-lm) (model picker → *SmolLM3-3B*), or any [LiteRT-LM](https://github.com/google-ai-edge/litert-lm) runtime. The `.litertlm` bundles the tokenizer and prompt template (bare ChatML — SmolLM3 emits a short `<think>` block then the answer) — no extra files.

**iOS note**: exported with `externalize_embedder=True` (drops the main weights section to ~1.61 GiB, under the iOS ~2 GiB single-section `mmap` limit).

**HF model**: [mlboydaisuke/SmolLM3-3B-LiteRT](https://huggingface.co/mlboydaisuke/SmolLM3-3B-LiteRT)

**Original project**: [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | Apache-2.0

# License

MIT (sample apps). Model licenses follow their original projects.
