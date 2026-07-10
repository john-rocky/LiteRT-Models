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
  - [RF-DETR Nano](#rf-detr-nano)
  - [RT-DETRv2-S](#rt-detrv2-s)
  - [D-FINE-S](#d-fine-s)

- [**Multi-Object Tracking**](#multi-object-tracking)
  - [YOLO + DeepSORT (OSNet)](#yolo--deepsort-osnet)

- [**Video Action Recognition**](#video-action-recognition)
  - [MoViNet-A0 (streaming, Kinetics-600)](#movinet-a0-streaming-kinetics-600)

- [**Semantic Segmentation**](#semantic-segmentation)
  - [PIDNet-S (real-time, Cityscapes)](#pidnet-s-real-time-cityscapes)

- [**Pose Estimation**](#pose-estimation)
  - [YOLO26n-pose](#yolo26n-pose)
  - [RTMPose-s](#rtmpose-s)
  - [RTMW-m (whole-body, 133 keypoints)](#rtmw-m-whole-body-133-keypoints)
  - [RTMPose-Hand (21 keypoints)](#rtmpose-hand-21-keypoints)
  - [RTMPose-Animal (AP-10K, 17 keypoints)](#rtmpose-animal-ap-10k-17-keypoints)

- [**Document Dewarping**](#document-dewarping)
  - [DewarpNet](#dewarpnet)

- [**Lane Detection**](#lane-detection)
  - [Ultra-Fast-Lane-Detection](#ultra-fast-lane-detection)
  - [TwinLiteNet (drivable area + lanes)](#twinlitenet)

- [**Super-Resolution**](#super-resolution)
  - [EDSR (×4)](#edsr-4)

- [**Image Dehazing**](#image-dehazing)
  - [DehazeFormer-MCT](#dehazeformer-mct)

- [**Clothing Segmentation**](#clothing-segmentation)
  - [Cloth Segmentation (U²-Net)](#cloth-segmentation-u2net)

- [**Portrait Sketch**](#portrait-sketch)
  - [U²-Net Portrait](#u2-net-portrait)

- [**Face Liveness / Anti-Spoofing**](#face-liveness--anti-spoofing)
  - [Silent-Face (MiniFASNetV2)](#silent-face-minifasnetv2)

- [**Head Pose Estimation**](#head-pose-estimation)
  - [6DRepNet](#6drepnet)

- [**Camouflaged Object Detection**](#camouflaged-object-detection)
  - [SINet-V2](#sinet-v2)

- [**Crowd Counting**](#crowd-counting)
  - [DM-Count](#dm-count)

- [**Instance Segmentation**](#instance-segmentation)
  - [YOLACT-ResNet50](#yolact-resnet50)

- [**Segmentation**](#segmentation)
  - [MobileSAM](#mobilesam)
  - [SAM 2.1 (Hiera-Tiny)](#sam-21-hiera-tiny)
  - [EdgeTAM (SAM2)](#edgetam-sam2)
  - [EdgeTAM Video (SAM2 tracking)](#edgetam-video-sam2-tracking)

- [**Background Removal**](#background-removal)
  - [RMBG-1.4 (ISNet)](#rmbg-14-isnet)
  - [ormbg (open, Apache-2.0)](#ormbg-open-apache-20)
  - [DIS (high-precision cutout)](#dis-is-net-general-use)

- [**Portrait Matting**](#portrait-matting)
  - [MODNet (trimap-free)](#modnet-trimap-free)

- [**Inpainting**](#inpainting)
  - [LaMa-Dilated](#lama-dilated)
  - [MI-GAN (mobile inpainting / object removal)](#mi-gan-mobile-inpainting--object-removal)

- [**Zero-Shot Classification**](#zero-shot-classification)
  - [CLIP ViT-B/32](#clip-vit-b32)
  - [Places365 ResNet18 (scene recognition)](#places365-resnet18-scene-recognition)

- [**Dense Feature Visualization**](#dense-feature-visualization)
  - [DINOv2 ViT-S/14](#dinov2-vit-s14)

- [**Surface Normal Estimation**](#surface-normal-estimation)
  - [DSINE](#dsine)

- [**Speech Recognition**](#speech-recognition)
  - [Parakeet (FastConformer-CTC)](#parakeet-fastconformer-ctc)
  - [Whisper-tiny](#whisper-tiny)
  - [wav2vec2-CTC (fully-GPU, single-pass)](#wav2vec2-ctc-fully-gpu-single-pass)

- [**Text-to-Speech**](#text-to-speech)
  - [Kokoro-82M](#kokoro-82m)
  - [Matcha-TTS](#matcha-tts)

- [**Vision-Language Model**](#vision-language-model)
  - [SmolVLM-256M](#smolvlm-256m)

- [**Text Generation**](#text-generation)
  - [RWKV-7 World 0.1B](#rwkv-7-world-01b)

- [**Voice Assistant**](#voice-assistant)
  - [Whisper + SmolLM2 + Kokoro pipeline](#whisper--smollm2--kokoro-pipeline)

- [**Audio Codec**](#audio-codec)
  - [DAC 16kHz](#dac-16khz)
  - [Mimi (Kyutai 2024)](#mimi-kyutai-2024)

- [**Audio Classification**](#audio-classification)
  - [wav2vec2 Keyword Spotting](#wav2vec2-keyword-spotting)
  - [PANNs CNN14 Audio Tagging](#panns-cnn14-audio-tagging)

- [**Pitch Detection**](#pitch-detection)
  - [CREPE](#crepe)

- [**Audio Source Separation**](#audio-source-separation)
  - [TIGER-DnR (Dialog / Effects / Music)](#tiger-dnr-dialog--effects--music)

- [**Speaker Diarization**](#speaker-diarization)
  - [pyannote 3.1 stack (segmentation + WeSpeaker)](#pyannote-31-stack-segmentation--wespeaker)

- [**Speech Enhancement**](#speech-enhancement)
  - [CMGAN (noise suppression)](#cmgan-noise-suppression)

- [**Music Transcription**](#music-transcription)
  - [Basic Pitch (audio-to-MIDI)](#basic-pitch-audio-to-midi)

- [**Image Matching**](#image-matching)
  - [XFeat (local features)](#xfeat-local-features)

- [**Text-Prompted Segmentation**](#text-prompted-segmentation)
  - [CLIPSeg](#clipseg)

- [**Image tagging**](#image-tagging)
  - [RAM++ (Recognize Anything Plus)](#ram-recognize-anything-plus)

- [**Image quality**](#image-quality)
  - [NIMA (Neural Image Assessment)](#nima-neural-image-assessment)

- [**Image Classification**](#image-classification)
  - [Vision-RWKV (VRWKV-S)](#vision-rwkv-vrwkv-s)

- [**Fine-Grained Classification**](#fine-grained-classification)
  - [PlantNet-300K (1081 plant species)](#plantnet-300k-1081-plant-species)

- [**Face**](#face)
  - [3DDFA_V2 (3D face alignment)](#3ddfa_v2-3d-face-alignment)
  - [BiSeNet (face parsing)](#bisenet-face-parsing)
  - [HSEmotion (facial emotion recognition)](#hsemotion-facial-emotion-recognition)

- [**OCR**](#ocr)
  - [PP-OCRv5](#pp-ocrv5)

- [**Super Resolution**](#super-resolution)
  - [Real-ESRGAN x4v3](#real-esrgan-x4v3)

- [**Monocular Geometry Estimation**](#monocular-geometry-estimation)
  - [MoGe-2 ViT-S](#moge-2-vit-s)
  - [Depth Anything 3 ViT-S (Small)](#depth-anything-3-vit-s-small)
  - [Metric3D v2 ViT-S](#metric3d-v2-vit-s)

- [**Text Embedding (RAG)**](#text-embedding-rag)
  - [Qwen3-Embedding-0.6B](#qwen3-embedding-06b)
  - [Qwen3-Reranker-0.6B](#qwen3-reranker-06b)

- [**Text Generation (LLM)**](#text-generation-llm)
  - [Falcon3-3B-Instruct](#falcon3-3b-instruct)
  - [Llama-3.2-3B-Instruct](#llama-32-3b-instruct)
  - [Ministral-3-3B-Instruct-2512](#ministral-3-3b-instruct-2512)
  - [SmolLM3-3B](#smollm3-3b)

- [**Face Detection**](#face-detection)
  - [YuNet](#yunet)
  - [RTMPose-Face (WFLW, 98-point face alignment)](#rtmpose-face-wflw-98-point-face-alignment)

- [**Gaze Estimation**](#gaze-estimation)
  - [L2CS-Net](#l2cs-net)

- [**Saliency Prediction**](#saliency-prediction)
  - [UniSal](#unisal)

- [**Line Detection**](#line-detection)
  - [M-LSD-tiny](#m-lsd-tiny)

- [**Style Transfer**](#style-transfer)
  - [Fast Neural Style (4 styles)](#fast-neural-style-4-styles)

- [**Low-Light Enhancement**](#low-light-enhancement)
  - [CPGA-Net](#cpga-net)

- [**Image Restoration**](#image-restoration)
  - [NAFNet (deblur)](#nafnet-deblur)

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

## Shared components

Utilities that recur across the sample apps live as canonical sources in
[`common/`](common/README.md) (Kotlin: `CompiledModelRunner`, `ImageTensor`,
`RealtimeCameraPipeline`, `AudioCapture`, `MathOps`). Each app keeps a vendored
copy so it stays standalone; `python tools/sync_common.py --check` keeps the
copies identical to the canonical.

On the conversion side, the GPU-compatibility patches (fp16-safe norms,
zero-stuff ConvTranspose, zero-pad MaxPool, GELU rewrites, …) are packaged in
[`litert_gpu_toolkit/`](docs/LITERT_CONVERSION_GUIDE.md#litert_gpu_toolkit--canonical-patch-catalog)
— import them instead of re-implementing per script.

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

### RF-DETR Nano

RF-DETR (Roboflow 2025, an LW-DETR derivative): a **transformer** detector (windowed DINOv2 backbone +
deformable-attention DETR decoder) running **fully on CompiledModel GPU** — the first transformer/DETR
detector in this zoo to do so. Converted with **litert-torch** + a **2-graph split** (the two-stage
query selection `TOPK`/`GATHER` runs on the host between the graphs) + **SafeLayerNorm** (the projector
and decoder LayerNorms overflow Mali fp16). Device-verified on Pixel 8a: both graphs fully `LITERT_CL`
(Graph A `1381/1381`, Graph B `404/404`); runs **live camera at ~9 fps (~110 ms/frame)** — a transformer
detector entirely on the GPU — and reproduces the PyTorch detections at IoU 0.98–0.99.

| Model | Size (fp16) | Input | Outputs | Original Project | License | Sample App |
| ----- | ----------- | ----- | ------- | ---------------- | ------- | ---------- |
| RF-DETR-Nano (Graph A + Graph B) | 48.6 MB + 7.6 MB | Float32 [1, 3, 384, 384] NCHW | enc_class[1,576,91] / enc_coord[1,576,4] / memory[1,576,256] → boxes[1,300,4] / logits[1,300,91] | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | [Apache-2.0](https://github.com/roboflow/rf-detr/blob/main/LICENSE) | [rfdetr/](rfdetr/) |

**Output format**: Graph B gives `boxes` (cxcywh, normalized 0-1) + `logits` (91 = COCO id space). Host
applies sigmoid + score threshold + cxcywh→xyxy + per-class NMS.

**Preprocessing**: square resize to 384×384, RGB, ImageNet mean/std normalization. See
[litert-community/RF-DETR-Nano-LiteRT](https://huggingface.co/litert-community/RF-DETR-Nano-LiteRT).

### RT-DETRv2-S

RT-DETRv2 (Baidu 2024, `PekingU/rtdetr_v2_r18vd`): a real-time **transformer** detector (ResNet18-vd
backbone + hybrid AIFI/CCFM encoder + plain deformable-attention DETR decoder) running **fully on
CompiledModel GPU**. Converted with **litert-torch** + a **2-graph split** (two-stage `TOPK`/`GATHER` on
the host). The on-device gate here was **not** an fp16 wall but a Mali bug where a 3D *token* tensor
`[1,N,256]` that fans out inside the graph is silently corrupted — fixed by emitting only the two clean
leaves (`enc_class` + `memory_raw`) and moving the per-token tail (`enc_output` + box head) to the host
on the 300 selected tokens (exact, since per-token ops commute with gather). Device-verified on Pixel 8a:
both graphs fully `LITERT_CL` (Graph B `704/704`); reproduces the PyTorch detections at IoU 0.98–1.00
(COCO val giraffe 7/7, cats 6/6). **Still-image** demo — RT-DETR's 8400-token / 80×80 deformable decoder
is ~350 ms of GPU compute (GATHER-free tent-matmul), so ~615 ms/frame, not real-time.

| Model | Size (fp16) | Input | Outputs | Original Project | License | Sample App |
| ----- | ----------- | ----- | ------- | ---------------- | ------- | ---------- |
| RT-DETRv2-S (Graph A + Graph B) | 33.8 MB + 7.7 MB | Float32 [1, 3, 640, 640] NCHW | enc_class[1,8400,80] / memory_raw[1,8400,256] → boxes[1,300,4] / logits[1,300,80] | [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | [Apache-2.0](https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE) | [rtdetr/](rtdetr/) |

**Output format**: Graph B gives `boxes` (cxcywh, normalized 0-1) + `logits` (80 = contiguous COCO id
0–79). Host applies sigmoid + score threshold + cxcywh→xyxy + light NMS.

**Preprocessing**: square resize to 640×640, RGB, [0,1] rescale only (no ImageNet normalization). See
[litert-community/RT-DETRv2-S-LiteRT](https://huggingface.co/litert-community/RT-DETRv2-S-LiteRT).

### D-FINE-S

D-FINE (USTC 2024, `ustc-community/dfine-small-coco`) — the **SOTA real-time DETR** — running **fully on
CompiledModel GPU**. HGNetV2 backbone + hybrid AIFI/CCFM encoder + an **FDR** (Fine-grained Distribution
Refinement) decoder. Converted with **litert-torch** + the same **2-graph split** as RT-DETRv2 (host topk +
per-token tail). D-FINE was previously **parked** as a "FDR decoder fp16 wall" — but that was a misdiagnosis:
the real cause was the same Mali 3D-token fan-out bug (the raw `memory` output was silently garbage), and
with clean memory the FDR decoder is perfect. Device-verified on Pixel 8a: Graph A `511/511` + Graph B
`850/850` LITERT_CL, real-image detections at IoU 0.99–1.00 (still-image; deformable decoder GPU-compute-bound).

| Model | Size (fp16) | Input | Outputs | Original Project | License | Sample App |
| ----- | ----------- | ----- | ------- | ---------------- | ------- | ---------- |
| D-FINE-S (Graph A + Graph B) | 13.0 MB + 8.8 MB | Float32 [1, 3, 640, 640] NCHW | enc_class[1,8400,80] / memory_raw[1,8400,256] → boxes[1,300,4] / logits[1,300,80] | [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | [Apache-2.0](https://github.com/Peterande/D-FINE/blob/main/LICENSE) | [dfine/](dfine/) |

**Output format**: Graph B gives `boxes` (cxcywh, normalized 0-1) + `logits` (80 = contiguous COCO id
0–79). Host applies sigmoid + score threshold + cxcywh→xyxy + light NMS.

**Preprocessing**: square resize to 640×640, RGB, [0,1] rescale only (no ImageNet normalization). See
[litert-community/D-FINE-S-LiteRT](https://huggingface.co/litert-community/D-FINE-S-LiteRT).

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

# Video Action Recognition

### MoViNet-A0 (streaming, Kinetics-600)

The first **video-input** model in this zoo: it recognises *actions* across a stream of camera frames (not a single image), one frame at a time, with constant memory. [MoViNet-A0](https://arxiv.org/abs/2103.11511) streaming variant (Google Research), trained on **Kinetics-600** (600 action classes), running **fully on CompiledModel GPU**.

MoViNet is a causal 3D CNN whose temporal convolutions and global-average-pools each keep a small buffer of the recent past, so it can be fed one frame at a time and sharpens its prediction as more frames of the same action arrive. The stock streaming graph carries that history in **5D** state tensors `[1, T, H, W, C]`, which the GPU delegate cannot compile (all tensors must be ≤4D). So the model is re-authored as a **single-frame, 4D-only functional forward** with every recurrent buffer threaded explicitly through the graph I/O.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| MoViNet-A0 stream | [movinet_a0_stream.tflite](https://huggingface.co/litert-community/MoViNet-A0-Stream-LiteRT) | 15 MB | frame [1, 3, 172, 172] NCHW + 46 state tensors | logits [1, 600] + 27 state tensors | [Atze00/MoViNet-pytorch](https://github.com/Atze00/MoViNet-pytorch) ([google-research/movinet](https://github.com/tensorflow/models/tree/master/official/projects/movinet)) | [Apache-2.0](https://github.com/Atze00/MoViNet-pytorch/blob/main/LICENSE) | [movinet/](movinet/) |

**I/O** (47 inputs / 28 outputs) — `input[0]` = current RGB frame (NCHW, 0..1); `input[1..28]` = 28 temporal-conv stream buffers `[1,C,H,W]`; `input[29..44]` = 16 streaming avg-pool running sums `[1,C,1,1]`; `input[45]` = `inv_count` (1/frame-number); `input[46]` = constant `1.0`. `output[0]` = Kinetics-600 logits; `output[1..11]` = current per-conv frames; `output[12..27]` = fresh per-frame means. The **stream-buffer shift register and pool running-sum accumulation are done host-side** — the graph consumes recurrent state but only emits fresh tensors.

**Conversion** (`movinet/scripts/build_movinet.py`, litert-torch): temporal depthwise convs (kernel 3/5 over the stream buffer) become a per-channel weighted sum of the buffered frames; streaming pools become `avg = (running_sum + mean) * inv_count`; the tf-`same` residual average pool is reformulated as `count_include_pad=True` + a constant boundary-correction mask so it lowers to `AVERAGE_POOL_2D + MUL`. Result: **all float32, 0 tensors of rank > 4, 0 banned ops, 0 composites** — matches the original PyTorch model bit-for-bit (corr 0.99999999999). Keeping the recurrent state in-graph tripped three silent Mali `CompiledModel` bugs (input-passed-through-to-output loses its compute use; a `state + tensor` output reads zero; a conv output that is both consumed and emitted has its emitted copy corrupted ~2.5× → fp16 blow-up over frames), so all state plumbing is host-side and each emitted stream frame is decoupled from its compute use by a multiply against the runtime `1.0` input. Device GPU (Pixel 8a) locks onto "jumping jacks" within a few frames.

**Sample app**: [movinet/](movinet/) — live camera → per-frame inference → top-5 Kinetics-600 action bars. Tap to restart the classification window.

# Semantic Segmentation

### PIDNet-S (real-time, Cityscapes)

Real-time **semantic segmentation** running fully on the LiteRT `CompiledModel` GPU. [PIDNet-S](https://arxiv.org/abs/2206.02066) (CVPR 2023) segments a road scene into the **19 Cityscapes classes** (road, sidewalk, building, car, person, sky, …) at ~17 FPS on a Pixel 8a. PIDNet is a three-branch CNN (P: detail, I: context, D: boundary) — a **pure CNN that converts to a fully GPU-compatible graph with zero patches**.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| PIDNet-S | [pidnet_s.tflite](https://huggingface.co/litert-community/PIDNet-S-Cityscapes-LiteRT) | 30 MB | Float32 [1, 3, 1024, 1024] NCHW (ImageNet-norm) | Float32 [1, 19, 128, 128] logits | [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet) | [MIT](https://github.com/XuJiacong/PIDNet/blob/main/LICENSE) | [pidnet/](pidnet/) |

**Preprocessing**: RGB, resize to 1024×1024, ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), NCHW. **Postprocessing**: argmax over the 19 channels per pixel → Cityscapes-colored label map (1/8 res) → upscale.

**Conversion** (`pidnet/scripts/build_pidnet.py`, litert-torch): PIDNet has no attention, no dynamic shapes at a fixed input, and `align_corners=False` on every bilinear resize, so it converts with **zero GPU patches** — `CONV_2D` ×75, `RESIZE_BILINEAR` ×11, `AVERAGE_POOL_2D`, `ADD`/`MUL`/`SUB`/`SUM`, `LOGISTIC`; **0 tensors of rank > 4, 0 banned ops**. CPU-exact vs PyTorch (corr 0.99999999999, 100% argmax); device Mali GPU (fp16) agrees at 97% of pixels with correct classes (~59 ms/frame at 1024²). The trained weights are loaded from an ONNX mirror whose initializer names match the original repo's PyTorch keys.

**Sample app**: [pidnet/](pidnet/) — live camera → PIDNet-S GPU → Cityscapes-colored segmentation overlay.

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


### RTMPose-Animal (AP-10K, 17 keypoints)

[RTMPose](https://github.com/open-mmlab/mmpose) (mmpose, Apache-2.0) **animal pose** trained on **AP-10K**: 17 animal keypoints (eyes, nose, neck, tail root, and the four limbs) for pets / wildlife. The **same model family** as RTMPose-s above — only the config/checkpoint change to AP-10K, and the two on-device Mali fixes (SafeRMSNorm + GAU broadcast-reduce) transfer **unchanged**. Runs **fully on the GPU** (`333/333` LITERT_CL on a Pixel 8a, **~5 ms**, device-vs-PyTorch SimCC corr **0.999**, 17/17 keypoints).

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [rtm_animal_fp16.tflite](https://huggingface.co/litert-community/RTMPose-Animal-AP10K-LiteRT) | 27.5 MB | Float32 [1, 3, 256, 256] NCHW | simcc_x [1,17,512], simcc_y [1,17,512] | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) | [Apache-2.0](https://github.com/open-mmlab/mmpose/blob/main/LICENSE) | [rtmanimal/](rtmanimal/) |

**Output**: output[0] = simcc_x, output[1] = simcc_y; each keypoint = `argmax` over its 1D SimCC (bins = pixels × 2). **Preprocessing**: center-crop to square, resize 256×256, mmpose mean/std (RGB, 0-255).

**Sample app**: [rtmanimal/](rtmanimal/) — image picker + 17-keypoint AP-10K animal skeleton.


# Document Dewarping

### DewarpNet

Real-time **document dewarping / rectification** running fully on the LiteRT `CompiledModel` GPU. [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) (ICCV 2019) flattens a photographed, curved/folded document — the core of a document scanner. Two CNNs (WCNet UNet → BMNet DenseNet) predict a backward-mapping grid on the GPU; the `grid_sample` unwarp is a tiny host-side step. First document-processing model in the zoo.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| DewarpNet | [dewarp.tflite](https://huggingface.co/litert-community/DewarpNet-LiteRT) | 189 MB | Float32 [1, 3, 256, 256] NCHW (BGR, /255) | Float32 [1, 2, 128, 128] backward map | [cvlab-stonybrook/DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) | [MIT](https://github.com/cvlab-stonybrook/DewarpNet/blob/master/LICENSE) | [dewarp/](dewarp/) |

**Preprocessing**: BGR, resize 256×256, `x/255` (no mean/std), NCHW. **Unwarp (host-side)**: blur the backward map (3×3), resize to the image size, then bilinear `grid_sample(image, map)` → the flattened document.

**Conversion** (`dewarp/scripts/build_dewarp.py`, litert-torch): pure CNN → fully GPU-compatible (**371/371 nodes on the delegate, 1 partition**; device corr 0.999866, ~24 ms) with two exact patches — `ConvTranspose2d` → ZeroStuffConvT2d (Mali rejects `TRANSPOSE_CONV`) and `Hardtanh(0,1)` → `relu(x)-relu(x-1)` (Mali rejects `RELU_0_TO_1`). CPU-exact vs PyTorch (corr 0.9999999999).

**Sample app**: [dewarp/](dewarp/) — live camera → DewarpNet GPU → flattened document.

# Lane Detection

### Ultra-Fast-Lane-Detection

Real-time **lane detection** running fully on the LiteRT `CompiledModel` GPU. [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection) (ECCV 2020) reformulates lane detection as fast **row-wise classification**: the ResNet18 network runs on the GPU, and a tiny host-side arg/expectation decode turns the grid into lane points. First lane-detection model in the zoo; an ADAS building block.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| Ultra-Fast-Lane-Detection (ResNet18, CULane) | [ufld.tflite](https://huggingface.co/litert-community/Ultra-Fast-Lane-Detection-LiteRT) | 178 MB | Float32 [1, 3, 288, 800] NCHW (RGB, ImageNet-norm) | Float32 [1, 201, 18, 4] (griding+1, rows, lanes) | [cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection) | [MIT](https://github.com/cfzd/Ultra-Fast-Lane-Detection/blob/master/LICENSE) | [ufld/](ufld/) |

**Preprocessing**: RGB, resize 800×288, `x/255` then ImageNet-normalize, NCHW. **Decode (host-side)**: per lane & row anchor, softmax over the 200 grid cells → expectation column (drop if argmax = "no lane" index 200); map column → x via `linspace(0,799,200)`, row anchor → y.

**Conversion** (`ufld/scripts/build_ufld.py`, litert-torch): pure CNN → fully GPU-compatible (**41/41 nodes on the delegate, 1 partition**; device corr 0.999982, ~20 ms) with one patch — the ResNet18 stem `MaxPool2d(padding=1)` `-inf` PADV2 → 0-pad + unpadded maxpool (exact post-ReLU). CPU-exact vs PyTorch (corr 0.9999999999996).

**Sample app**: [ufld/](ufld/) — live camera → UFLD GPU → per-lane points overlaid.

### TwinLiteNet

Real-time **drivable-area + lane-line segmentation** running fully on the LiteRT `CompiledModel` GPU. [TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet) (2023) is an ultra-light ESPNet-based network with two segmentation heads — the ADAS "where can I drive" + "where are the lanes" building block. Only 3.1 MB.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| TwinLiteNet | [twinlite.tflite](https://huggingface.co/litert-community/TwinLiteNet-LiteRT) | 3.1 MB | Float32 [1, 3, 360, 640] NCHW (RGB, /255) | 2× [1, 2, 360, 640] (drivable area + lane line) | [chequanghuy/TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet) | [MIT](https://github.com/chequanghuy/TwinLiteNet/blob/main/LICENSE) | [twinlite/](twinlite/) |

**Preprocessing**: RGB, resize 640×360, `x/255`, NCHW. **Decode**: `argmax` over the 2 classes of each head → drivable-area mask + lane mask.

**Conversion** (`twinlite/scripts/build_twinlite.py`, litert-torch): pure CNN → fully GPU-compatible (**270/270 nodes on the delegate, 1 partition**; device corr 0.99997/0.99998, ~44 ms) with one patch — `ConvTranspose2d` → ZeroStuffConvT2d (Mali rejects `TRANSPOSE_CONV`). CPU-exact vs PyTorch (corr 1.0).

**Sample app**: [twinlite/](twinlite/) — live camera → TwinLiteNet GPU → drivable area (green) + lanes (red).

# Super-Resolution

### EDSR (×4)

Real-time **×4 single-image super-resolution** running fully on the LiteRT `CompiledModel` GPU. [EDSR](https://arxiv.org/abs/1707.02921) (CVPR 2017 winner) upscales a low-res image 4× with sharp detail. First super-resolution model in the zoo.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| EDSR-base (×4) | [edsr.tflite](https://huggingface.co/litert-community/EDSR-x4-LiteRT) | 7.7 MB | Float32 [1, 3, 128, 128] NCHW (RGB, /255) | Float32 [1, 3, 512, 512] (RGB 0–1) | [eugenesiow/edsr-base](https://huggingface.co/eugenesiow/edsr-base) | [Apache-2.0](https://github.com/eugenesiow/super-image/blob/main/LICENSE) | [edsr/](edsr/) |

**Preprocessing**: RGB, `x/255`, NCHW. **Output**: clamp 0–1, ×255.

**Conversion** (`edsr/scripts/build_edsr.py`, litert-torch): pure CNN, but the **PixelShuffle** upsampler lowers to rank-5/6 reshapes the Mali delegate rejects (the classic super-resolution wall). Exact fix — **PixelShuffle(r) ≡ a fixed-weight `ConvTranspose2d(stride=r)`** → ZeroStuffConvT2d. Result: **68/68 nodes on the delegate, 1 partition**; device corr 0.999946, ~23 ms. CPU-exact vs PyTorch (corr 1.0). This patch also unblocks other PixelShuffle SR models.

# Image Dehazing

### DehazeFormer-MCT

Real-time **image dehazing** with the network fully on the LiteRT `CompiledModel` GPU. [DehazeFormer](https://github.com/IDKiro/DehazeFormer) (TIP 2023, MCT curve-mapping variant trained on a mixed dataset for real-world haze) removes fog / haze / smoke and restores contrast. The 256×256 network predicts 72 per-pixel curve parameters; the curves are applied to the **full-resolution** frame host-side, so output resolution is independent of network resolution.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| DehazeFormer-MCT (mixed) | [dehazeformer_base.tflite](https://huggingface.co/litert-community/DehazeFormer-MCT-LiteRT) | 17 MB | Float32 [1, 3, 256, 256] NCHW (RGB, [-1,1]) | Float32 [1, 72, 256, 256] curve params | [IDKiro/DehazeFormer](https://github.com/IDKiro/DehazeFormer) | [MIT](https://github.com/IDKiro/DehazeFormer/blob/main/LICENSE) | [dehaze/](dehaze/) |

**Preprocessing**: RGB, `x/255*2-1`, NCHW at 256×256. **Decode (host-side)**: trilinear curve lookup per full-res pixel — `out[c] = Σᵢ trilinear(curve[c][i], depth=xᵢ, y, x)`, then `clamp(-1,1)*0.5+0.5` (exact official grid_sample mapping, replica corr 1.0).

**Conversion** (`dehaze/scripts/build_dehaze.py`, litert-torch): Swin-style windowed attention re-authored with the established recipes — window partition/reverse ≤4D, qkv channel slices, baked relative-position bias, reflect pads → slice+concat (litert-torch lowers `reflection_pad2d` to banned `GATHER_ND`, **including padding=0 reflect convs**), SKFusion 5D→4D pairwise softmax, Conv+PixelShuffle → ZeroStuffConvT2d. ⭐New Mali finding: a single `MEAN` over C·H·W (1.5M elements) **overflows the fp16 accumulator** → NaN; RLN global norm + SKFusion global pool re-authored as **hierarchical means** (equal-window `avg_pool` stages, mathematically identical). Result: **2042/2042 nodes on the delegate, 1 partition**; device corr 0.999998, E2E vs the official pipeline corr 0.999997, ~255 ms/frame. Desktop corr vs PyTorch 1.0000000.

**Sample app**: [dehaze/](dehaze/) — live camera → DehazeFormer GPU + host curve mapping → dehazed frame full-screen, tap to compare.

**Sample app**: [edsr/](edsr/) — live camera → EDSR GPU → ×4 super-resolved center region.

# Clothing Segmentation

### Cloth Segmentation (U²-Net)

Real-time **clothing segmentation** running fully on the LiteRT `CompiledModel` GPU. [cloth-segmentation](https://github.com/levindabhi/cloth-segmentation) is a U²-Net trained on iMaterialist-Fashion to segment **upper-body / lower-body / full-body clothing** — the building block for virtual try-on and fashion apps.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| Cloth Segmentation (U²-Net) | [clothseg.tflite](https://huggingface.co/litert-community/Cloth-Segmentation-U2Net-LiteRT) | 176 MB | Float32 [1, 3, 768, 768] NCHW (RGB, [-1,1]) | Float32 [1, 4, 768, 768] (argmax → clothing class) | [levindabhi/cloth-segmentation](https://github.com/levindabhi/cloth-segmentation) | [MIT](https://github.com/levindabhi/cloth-segmentation/blob/main/LICENSE) | [clothseg/](clothseg/) |

**Preprocessing**: RGB, resize 768×768, `(x/255 - 0.5)/0.5`, NCHW. **Decode**: `argmax` over the 4 classes → 0 background, 1 upper body, 2 lower body, 3 full body.

**Conversion** (`clothseg/scripts/build_clothseg.py`, litert-torch): pure CNN → fully GPU-compatible (**254/254 nodes on the delegate, 1 partition**; device corr 0.999798, ~88 ms) with one defensive patch — `align_corners=True` → `False`. CPU-exact vs PyTorch (corr 1.0). ⚠ Strip the `module.` prefix when loading the checkpoint.

**Sample app**: [clothseg/](clothseg/) — live camera → U²-Net GPU → clothing segments (upper/lower/full).

# Portrait Sketch

### U²-Net Portrait

Real-time **portrait sketch generation** running fully on the LiteRT `CompiledModel` GPU. The [U²-Net](https://github.com/xuebinqin/U-2-Net) portrait model turns a face photo into a **hand-drawn pencil line portrait** — a fun creative / AR filter.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| U²-Net Portrait | [portrait.tflite](https://huggingface.co/litert-community/U2Net-Portrait-Sketch-LiteRT) | 176 MB | Float32 [1, 3, 512, 512] NCHW (RGB, ImageNet-norm) | Float32 [1, 1, 512, 512] (0–1) | [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net) | [Apache-2.0](https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE) | [portrait/](portrait/) |

**Preprocessing**: RGB, resize 512×512, `x/255` then ImageNet-normalize, NCHW. **Decode**: min-max normalize the output, then invert (`1−x`) for dark strokes on white paper.

**Conversion** (`portrait/scripts/build_portrait.py`, litert-torch): pure CNN → fully GPU-compatible (**893/893 nodes on the delegate, 1 partition**; device corr 0.998683, ~12 ms) with one defensive patch — `align_corners=False`. CPU-exact vs PyTorch (corr 1.0).

**Sample app**: [portrait/](portrait/) — live camera → U²-Net GPU → live pencil portrait.

# Face Liveness / Anti-Spoofing

### Silent-Face (MiniFASNetV2)

Real-time **face liveness / anti-spoofing** running fully on the LiteRT `CompiledModel` GPU. [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) detects **presentation attacks** — a printed photo or a replayed screen shown to the camera — so a live face passes and a fake is rejected. The anti-fraud building block for face login / e-KYC. Tiny (1.85 MB).

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| Silent-Face (MiniFASNetV2) | [silentface.tflite](https://huggingface.co/litert-community/Silent-Face-Anti-Spoofing-LiteRT) | 1.85 MB | Float32 [1, 3, 80, 80] NCHW (BGR, /255, face crop) | Float32 [1, 3] softmax (class 1 = live) | [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) | [Apache-2.0](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/LICENSE) | [liveness/](liveness/) |

**Preprocessing**: face crop (~2.7× the face box), BGR, resize 80×80, `x/255`, NCHW. **Decode**: softmax; class 1 = live, 0 & 2 = spoof (print / replay); live score = `output[1]`.

**Conversion** (`liveness/scripts/build_silentface.py`, litert-torch): pure CNN → fully GPU-compatible (**168/168 nodes on the delegate, 1 partition**; device corr 1.0, ~5 ms) with zero patches — PReLU lowers to GPU-clean relu ops. CPU-exact vs PyTorch (corr 1.0).

**Sample app**: [liveness/](liveness/) — live camera → MiniFASNetV2 GPU → LIVE / SPOOF verdict.

# Head Pose Estimation

### 6DRepNet

Real-time **6-DoF head pose estimation** running fully on the LiteRT `CompiledModel` GPU. [6DRepNet](https://github.com/thohemp/6DRepNet) (ICIP 2022) regresses a continuous 6D rotation from a face crop — yaw / pitch / roll for driver-monitoring, AR, and attention. RepVGG (deploy) backbone.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| 6DRepNet | [6drepnet.tflite](https://huggingface.co/litert-community/6DRepNet-HeadPose-LiteRT) | 157 MB | Float32 [1, 3, 224, 224] NCHW (RGB, ImageNet-norm, face crop) | Float32 [1, 6] (6D rotation) | [thohemp/6DRepNet](https://github.com/thohemp/6DRepNet) | [MIT](https://github.com/thohemp/6DRepNet/blob/master/LICENSE) | [sixdrepnet/](sixdrepnet/) |

**Preprocessing**: face crop, resize 224×224, RGB, ImageNet-normalize, NCHW. **Decode (host-side)**: Gram-Schmidt the 6D → 3×3 rotation matrix → Euler `pitch=atan2(R21,R22)`, `yaw=atan2(-R20,√(R00²+R10²))`, `roll=atan2(R10,R00)`.

**Conversion** (`sixdrepnet/scripts/build_6drepnet.py`, litert-torch): deploy-mode RepVGG (plain convs) → fully GPU-compatible (**36/36 nodes on the delegate, 1 partition**; device corr 0.9993, ~21 ms) with zero patches. Use the deploy weights (fused `rbr_reparam`). CPU-exact vs PyTorch (corr 1.0).

**Sample app**: [sixdrepnet/](sixdrepnet/) — live camera → 6DRepNet GPU → 3D head-pose axes.

# Camouflaged Object Detection

### SINet-V2

Real-time **camouflaged object detection** running fully on the LiteRT `CompiledModel` GPU. [SINet-V2](https://github.com/GewelsJI/SINet-V2) (TPAMI 2022) finds objects that **blend into their background** — hidden animals, concealed items, defect/polyp-style targets — where ordinary segmentation fails.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| SINet-V2 (Res2Net-50) | [sinet.tflite](https://huggingface.co/litert-community/SINet-V2-Camouflage-LiteRT) | 100 MB | Float32 [1, 3, 352, 352] NCHW (RGB, ImageNet-norm) | Float32 [1, 1, 352, 352] sigmoid | [GewelsJI/SINet-V2](https://github.com/GewelsJI/SINet-V2) | [Apache-2.0](https://github.com/GewelsJI/SINet-V2) | [sinet/](sinet/) |

**Preprocessing**: RGB, resize 352×352, ImageNet-normalize, NCHW. **Output**: sigmoid map, high = concealed object; resize + threshold/overlay.

**Conversion** (`sinet/scripts/build_sinet.py`, litert-torch): pure CNN → fully GPU-compatible (**2447/2447 nodes on the delegate, 1 partition**; device corr 0.994) with two patches — ZeroPadMaxPool for the Res2Net stem + `align_corners=False`. CPU-exact vs PyTorch (corr 0.997).

**Sample app**: [sinet/](sinet/) — live camera → SINet-V2 GPU → concealed objects highlighted.

# Crowd Counting

### DM-Count

Real-time **crowd counting** running fully on the LiteRT `CompiledModel` GPU. [DM-Count](https://github.com/cvlab-stonybrook/DM-Count) (NeurIPS 2020) regresses a person **density map** whose sum is the crowd size — it counts hundreds of people where detector-based counting saturates.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| DM-Count (VGG19, UCF-QNRF) | [dmcount.tflite](https://huggingface.co/litert-community/DM-Count-Crowd-LiteRT) | 86 MB | Float32 [1, 3, 512, 512] NCHW (RGB, ImageNet-norm) | Float32 [1, 1, 64, 64] density map | [cvlab-stonybrook/DM-Count](https://github.com/cvlab-stonybrook/DM-Count) | [MIT](https://github.com/cvlab-stonybrook/DM-Count/blob/master/LICENSE) | [crowdcount/](crowdcount/) |

**Preprocessing**: RGB, resize 512×512, ImageNet-normalize, NCHW. **Output**: non-negative density map at 1/8 resolution; `sum(map)` = estimated person count, normalize per-frame for the heatmap overlay.

**Conversion** (`crowdcount/scripts/build_dmcount.py`, litert-torch): pure CNN (VGG19 + conv regression head) → fully GPU-compatible (**30/30 nodes on the delegate, 1 partition**; device corr 0.99998, count within 0.4%) with **one exact rewrite** — the mid-graph `F.upsample_bilinear` (align_corners=True `RESIZE_BILINEAR`, banned on the delegate) is a linear operator, re-authored as two constant-matrix multiplies (→ `FULLY_CONNECTED`; the constant must be on the RHS — the delegate rejects `BATCH_MATMUL` with a constant LHS). Desktop corr vs PyTorch 1.000000.

**Sample app**: [crowdcount/](crowdcount/) — live camera → DM-Count GPU → density heatmap + live person count.

# Instance Segmentation

### YOLACT-ResNet50

Real-time **instance segmentation** (per-object COCO masks) running fully on the LiteRT `CompiledModel` GPU. [YOLACT](https://arxiv.org/abs/1904.02689) (ICCV 2019): the network (ResNet50 + FPN + protonet + heads) runs on the GPU; the lightweight decode (NMS + linear-combination masks) runs host-side — the RF-DETR raw-head pattern. First instance-segmentation model in the zoo.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| YOLACT-ResNet50 | [yolact.tflite + priors.bin](https://huggingface.co/litert-community/YOLACT-ResNet50-LiteRT) | 125 MB | Float32 [1, 3, 550, 550] NCHW (BGR) | loc [1,19248,4] + conf [1,19248,81] + mask [1,19248,32] + proto [1,138,138,32] | [dbolya/yolact](https://github.com/dbolya/yolact) | [MIT](https://github.com/dbolya/yolact/blob/master/LICENSE) | [yolact/](yolact/) |

**Preprocessing**: BGR, resize 550×550, `(x - [103.94,116.78,123.68]) / [57.38,57.12,58.40]` (no /255), NCHW. **Decode (host-side)**: SSD box decode vs the baked 19248 priors (variances [0.1,0.2]) → per-class NMS (IoU 0.5) → lincomb masks `sigmoid(proto @ coeff)` cropped to each box.

**Conversion** (`yolact/scripts/build_yolact.py`, litert-torch): base YOLACT (no deformable conv) is a pure CNN → fully GPU-compatible (**138/138 nodes on the delegate, 1 partition**; device corr 0.99999–1.0 on all 4 outputs, ~41 ms) with **one patch** — the ResNet50 stem `MaxPool2d(padding=1)` lowers to a `-inf` PADV2 (rejected by Mali), replaced by a 0-pad + unpadded maxpool (exact post-ReLU); the scripted FPN is made traceable by disabling YOLACT's JIT. The 3D `[1,19248,C]` head outputs survive the Mali delegate. CPU-exact vs PyTorch (corr 1.0).

**Sample app**: [yolact/](yolact/) — live camera → YOLACT GPU → colored instance masks + boxes + COCO labels.

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

### SAM 2.1 (Hiera-Tiny)

The **full** SAM 2.1 (Meta) — the Hiera hierarchical ViT, not the distilled EdgeTAM — running entirely on **CompiledModel GPU**. Tap a point, get a mask. The heavy Hiera image encoder runs once per image; the tiny mask decoder runs per tap. Also the subject of a cross-framework benchmark: **[LiteRT vs MLX on the same Apple GPU](sam2/BENCHMARK.md)**.

Converted with **litert-torch** from the `transformers` `Sam2Model`. The SAM 2 mask decoder converts unchanged; the Hiera encoder needs three numerically-exact rewrites (parity held at corr 1.0 after each): **bake the windowed positional embedding** (constant for a fixed 1024² input — removes the bicubic `GATHER_ND` and the tiled `BROADCAST_TO`), **4-D window partition/unpartition** (the upstream 6-D `view`+`permute` becomes split-H→transpose→split-W; ML Drift rejects >4-D tensors), and **4-D multi-scale attention** (the fused 5-D `qkv` reshape becomes a channel-wise q/k/v slice). Result: `banned ops = NONE`, `>4-D tensors = 0` for both graphs.

> ⚠ **Keep the batch dim in attention.** A rank-3 attention (`q/k/v` shaped `[heads, N, d]`) compiles, delegates every node, passes the op gate and matches PyTorch on the host — yet ML Drift **silently mis-computes it** (corr 0.265 vs CPU on a Pixel 8a; still 0.473 with fp32 GPU compute forced, so it is a correctness bug, not an fp16 wall). See [GPU Compatibility Notes](#gpu-compatibility-notes).

| Model | Download Link | Size | Input | Output | API |
| ----- | ------------- | ---- | ----- | ------ | --- |
| Encoder | [sam2_encoder.tflite](https://huggingface.co/mlboydaisuke/SAM2-hiera-tiny-LiteRT) | 80 MB | Float32 [1, 3, 1024, 1024] NCHW | Float32 [1, 4194304] (ie \| fpn0 \| fpn1) | CompiledModel GPU |
| Decoder | [sam2_decoder.tflite](https://huggingface.co/mlboydaisuke/SAM2-hiera-tiny-LiteRT) | 17 MB | Float32 [1, 4194816] (ie \| sparse \| fpn0 \| fpn1) | Masks [1, 3, 256, 256] | CompiledModel GPU |

**Preprocessing**: resize to 1024×1024, divide by 255, ImageNet mean/std `[0.485,0.456,0.406]`/`[0.229,0.224,0.225]`, NCHW planar. The point→token prompt encoder runs in Kotlin/Swift from `sam2_prompt.bin`.

**Fidelity**: converted graphs match the PyTorch model at corr **1.0** (mask IoU 1.0/0.997/1.0). Device (all `fullyGPU`, mask foreground ≈ the 64.9k-px reference): Pixel 8a GPU enc **610 ms** / dec **76 ms**; iPhone 17 Pro (Metal) enc **248 ms** / dec **16 ms**.

**Sample apps**: [sam2/](sam2/) (Android, tap-to-segment + headless benchmark), [sam2-ios/](sam2-ios/) (iOS, LiteRT `CompiledModel` C API on Metal), [sam2-mlx-ios/](sam2-mlx-ios/) (iOS, a full **mlx-swift** port of the MLX SAM 2 image path, corr 1.0 vs the Python reference — used as the MLX side of the benchmark). Conversion: [sam2/scripts/convert_sam2.py](sam2/scripts/convert_sam2.py).

**Original project**: [facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) ([facebookresearch/sam2](https://github.com/facebookresearch/sam2)) | [Apache-2.0](https://github.com/facebookresearch/sam2/blob/main/LICENSE)

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

# Background Removal

### RMBG-1.4 (ISNet)

RMBG-1.4: High-quality background removal based on ISNet (U2-Net variant). Pure CNN architecture — 44M params, runs on CompiledModel GPU. Outputs alpha matte for clean foreground extraction.

Converted via **litert-torch** from [briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4). Output is sigmoid-activated (0-1 mask), no post-processing needed.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [rmbg14.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v1/rmbg14.tflite) | 176 MB | Float32 [1, 3, 1024, 1024] NCHW | Float32 [1, 1, 1024, 1024] | [briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) | [bria-rmbg-1.4](https://bria.ai/bria-huggingface-model-license-agreement/) | [rmbg/](rmbg/) |

**Preprocessing**: RGB normalized as `(pixel/255 - 0.5)`. NCHW planar layout.

### ormbg (open, Apache-2.0)

[ormbg](https://huggingface.co/schirrmacher/ormbg): a fully **open, Apache-2.0** background-removal model (an ISNet trained for photorealistic subject cut-out) — the permissively-licensed alternative to the non-commercial RMBG-1.4. Pure CNN, fully on CompiledModel GPU, ~10 ms/frame on a Pixel 8a.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [ormbg.tflite](https://huggingface.co/litert-community/ormbg-LiteRT) | 176 MB | Float32 [1, 3, 1024, 1024] NCHW (RGB, /255) | Float32 [1, 1, 1024, 1024] alpha | [schirrmacher/ormbg](https://huggingface.co/schirrmacher/ormbg) | [Apache-2.0](https://huggingface.co/schirrmacher/ormbg) | [ormbg/](ormbg/) |

**Preprocessing**: RGB, `x / 255` (no mean/std). **Output**: raw alpha matte — min-max normalize per frame before compositing.

### DIS (IS-Net, general-use)

[DIS](https://github.com/xuebinqin/DIS) (ECCV 2022): high-accuracy **dichotomous image segmentation** — cuts out the main object with fine structure detail (thin stems, petals, wires) for e-commerce product photos and graphics. IS-Net, fully on CompiledModel GPU, ~11 ms/frame on a Pixel 8a.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| DIS (IS-Net general-use) | [dis.tflite](https://huggingface.co/litert-community/DIS-ISNet-LiteRT) | 176 MB | Float32 [1, 3, 1024, 1024] NCHW (RGB, x/255−0.5) | Float32 [1, 1, 1024, 1024] alpha | [xuebinqin/DIS](https://github.com/xuebinqin/DIS) | [Apache-2.0](https://github.com/xuebinqin/DIS) | [dis/](dis/) |

**Preprocessing**: RGB, resize 1024×1024, `x/255 − 0.5`, NCHW. **Output**: sigmoid alpha (0–1).

**Conversion** (`dis/scripts/build_dis.py`, litert-torch): pure CNN → fully GPU-compatible (**247/247 nodes on the delegate, 1 partition**; device max|diff| 0.00034, ~11 ms) with one defensive patch — `align_corners=False`. CPU-exact vs PyTorch (max|diff| 0.0).

**Sample app**: [dis/](dis/) — live camera → DIS GPU → high-precision cutout.

**Conversion** (`ormbg/scripts/build_ormbg.py`, litert-torch): pure CNN → fully GPU-compatible (**246/246 nodes on the delegate, 1 partition**; device corr 0.999881, ~10 ms) with one defensive patch — `align_corners=True` → `False` on the bilinear upsamples. CPU-exact vs PyTorch (corr 0.9999999999).

**Output format**: Sigmoid mask (0-1). Apply as alpha channel to original image for transparent background.

# Portrait Matting

### MODNet (trimap-free)

Real-time **portrait matting** running fully on the LiteRT `CompiledModel` GPU. [MODNet](https://arxiv.org/abs/2011.11961) (AAAI 2022) predicts a **soft alpha matte** for a person — no trimap, no green screen — for background blur/replace (video calls, virtual backgrounds). ~79 ms/frame on a Pixel 8a. Distinct from RMBG background removal: MODNet targets soft human alpha (hair detail).

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| MODNet | [modnet.tflite](https://huggingface.co/litert-community/MODNet-LiteRT) | 26 MB | Float32 [1, 3, 512, 512] NCHW ([-1,1]) | Float32 [1, 1, 512, 512] alpha | [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet) | [Apache-2.0](https://github.com/ZHKKKe/MODNet/blob/master/LICENSE) | [modnet/](modnet/) |

**Preprocessing**: RGB, resize 512×512, normalize to [-1,1] (`(pixel/255 - 0.5)/0.5`), NCHW. **Output**: soft alpha matte 0–1; composite `fg·α + bg·(1-α)`.

**Conversion** (`modnet/scripts/build_modnet.py`, litert-torch): pure CNN (MobileNetV2 backbone), 2 re-authoring patches → fully GPU-compatible (**0 tensors of rank > 4, 0 banned ops**): (1) SE block `Linear`→`1×1 conv` (the 2D-reshape confuses NCHW↔NHWC), (2) **fp16-safe hierarchical-mean InstanceNorm** — MODNet's IBNorm runs InstanceNorm over up to 512² spatial, whose variance `sum(dd²)` overflows fp16 on Mali (matte degrades, corr 0.94); computing the mean via a cascade of `/2` avg-pools (magnitude-bounded, exact) restores GPU corr **0.99994** with clean edges. CPU-exact vs PyTorch (corr 0.99999999999).

**Sample app**: [modnet/](modnet/) — live camera → MODNet GPU → foreground composited over a replaceable background (tap to change).

# Inpainting

### LaMa-Dilated

LaMa-Dilated: Large Mask Inpainting with dilated convolutions. Draw a mask over unwanted objects and the model fills in the region naturally. Based on [LaMa](https://github.com/advimman/lama) with FFT blocks replaced by dilated convolutions for GPU compatibility.

Pre-converted TFLite from [Qualcomm AI Hub](https://aihub.qualcomm.com/models/lama_dilated). Pure CNN, 361 ops, all GPU-native.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [lama_dilated.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v1/lama_dilated.tflite) | 174 MB | Float32 [1, 512, 512, 3] + [1, 512, 512, 1] NHWC | Float32 [1, 512, 512, 3] NHWC | [advimman/lama](https://github.com/advimman/lama) | [Apache-2.0](https://github.com/advimman/lama/blob/main/LICENSE) | [lama/](lama/) |

**Preprocessing**: Image RGB normalized to 0-1 (divide by 255). Mask is single channel 0-1 (1 = area to inpaint).

**Sample app**: [lama/](lama/) — Image picker + finger drawing mask + inpainting with before/after toggle.

### MI-GAN (mobile inpainting / object removal)

[MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) (Picsart AI Research, **ICCV 2023**, MIT): a "magic eraser" — paint over an object and it is removed and inpainted. A mobile-designed StyleGAN-style generator (separable convs, nearest-upsample, **no norm**) — far smaller/faster than LaMa above. Verified **fully on the GPU** (`509/509` LITERT_CL on a Pixel 8a, **~6 ms** at 512×512, device-vs-PyTorch corr **0.99998**, **16.3 MB** fp16).

Converted via **litert-torch** with **no re-authoring** — the inference generator is already GPU-clean (depthwise-separable conv, `nn.Upsample(nearest)` + FIR-filter grouped conv, leaky-ReLU clamp → `MAXIMUM`/`MINIMUM`, no normalization). The FFT-free, norm-free generator lane.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [migan_fp16.tflite](https://huggingface.co/litert-community/MI-GAN-512-Places2-LiteRT) | 16.3 MB | Float32 [1, 4, 512, 512] NCHW (`concat(mask−0.5, rgb·mask)`) | Float32 [1, 3, 512, 512] ([−1,1]) | [Picsart-AI-Research/MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) | [MIT](https://github.com/Picsart-AI-Research/MI-GAN/blob/main/LICENSE) | [migan/](migan/) |

**I/O**: input `concat(mask−0.5, rgb·mask)` (rgb ∈ [−1,1], mask = 1 keep / 0 erase); composite back as `rgb·mask + out·(1−mask)`. **Preprocessing**: center-crop, resize 512×512.

**Sample app**: [migan/](migan/) — image picker + finger-paint mask + on-device erase.


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

### Places365 ResNet18 (scene recognition)

ResNet18 trained on [Places365](http://places2.csail.mit.edu/) (CSAILVision, MIT): **scene/place recognition** across **365 categories** (beach, kitchen, forest, office, restaurant, …) — a distinct task from object classification (it answers *what kind of place* a photo is). Pure CNN → runs **fully on the GPU** (`61/61` LITERT_CL on a Pixel 8a, **~2 ms**, fp16 **22.8 MB**, device-vs-PyTorch corr **1.0**, top-1 match).

Converted via **litert-torch** with two numerically-exact re-authorings: the global `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)`, and the ResNet stem `MaxPool2d(3,s2,p1)` → **zero-pad + valid max-pool** (PyTorch's max-pool pads with `-inf` → a `PADV2` the Mali delegate won't delegate; since the pool follows a ReLU, a 0-pad is exactly equivalent and emits a delegatable `PAD`).

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [places_fp16.tflite](https://huggingface.co/litert-community/Places365-ResNet18-LiteRT) | 22.8 MB | Float32 [1, 3, 224, 224] NCHW | Logits [1, 365] | [CSAILVision/places365](https://github.com/CSAILVision/places365) | [MIT](https://github.com/CSAILVision/places365/blob/master/LICENSE) | [places365/](places365/) |

**Preprocessing**: center-crop, resize to 224×224, /255, ImageNet mean/std, NCHW. Output 365-class scene logits; softmax + argmax for top-k.

**Sample app**: [places365/](places365/) — image picker + top-5 scene categories.


# Dense Feature Visualization

### DINOv2 ViT-S/14

Run the self-supervised [DINOv2](https://github.com/facebookresearch/dinov2) ViT-S/14 backbone fully on the LiteRT `CompiledModel` GPU and visualize its **dense patch features** — a top-3 PCA of the tokens mapped to RGB. Semantically similar patches (object parts vs background) land near each other in feature space, so they share a color and the object "pops out" with no labels or segmentation. The first self-supervised-backbone / feature-visualization demo in the zoo.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| DINOv2 ViT-S/14 | [dinov2_s_fp16.tflite](https://huggingface.co/litert-community/DINOv2-ViT-S14-LiteRT) | 45 MB | Float32 [1, 3, 448, 448] NCHW (ImageNet-norm) | Float32 [1, 1024, 384] patch tokens | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | [Apache-2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) | [dinov2/](dinov2/) |

**Preprocessing**: resize 448×448, ImageNet normalization, NCHW. **Decode (host-side)**: top-3 PCA of the 1024×384 token matrix (power iteration on the 384×384 covariance) → per-patch RGB → upscaled overlay.

**Conversion** (`dinov2/scripts/build_dinov2.py`, litert-torch): the proven ViT recipes — fused-qkv attention decomposed to 4D `[1,heads,N,d]` (C12), **SafeLayerNorm** (deviation scaled by 1/64 before squaring so the fp16 variance doesn't overflow on DINOv2's massive activations), **LayerScale** (`ls1`/`ls2`) baked into the projections, and **tanh-GELU** (`0.5x(1+tanh(…))`) — the sigmoid-GELU approximation drifts to feature corr 0.968 over 12 blocks, tanh → 0.99999. The pos_embed is baked at a fixed 448 grid by timm at model creation, so there is no runtime interpolation (no `GATHER_ND`). Result: **864/864 nodes on the delegate, 1 partition**, ~8 ms; device fp16 patch features vs desktop fp32 corr 0.996.

**Sample app**: [dinov2/](dinov2/) — pick a photo → image and its DINOv2 feature-PCA side by side.

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

### Parakeet (FastConformer-CTC)

NVIDIA Parakeet (`parakeet-tdt_ctc-110m`, the CTC branch): the 17-layer FastConformer encoder + CTC head run **fully on the LiteRT CompiledModel GPU** on a Pixel 8a — the first big global-attention transformer in this zoo to survive the Mali fp16 path end to end. On-device transcript matches PyTorch exactly (real-frame logits corr 0.99997), 3105/3105 ops on LITERT_CL (1 partition), ~330 ms GPU + ~70 ms host mel ≈ 0.4 s end-to-end per 16 s window (device-app measured).

Converted via **litert-torch**: RelPositionMultiHeadAttention re-authored as manual ≤4D matmuls, GLU→`a·sigmoid(b)`, masking folded into a GPU-clean additive attention bias for the fixed window, CTC `ConvASRDecoder` fused into the graph. Key fix: the subsampling front-end emits very large pre-norm activations (|x|≈7000), so the LayerNorm variance is reduced entirely in a down-scaled domain (never rebuilding the large variance, which overflows fp16 on Mali → blank output).

| Model | Download Link | Size | Input | Output | API |
| ----- | ------------- | ---- | ----- | ------ | --- |
| Encoder + CTC | [parakeet_ship_fp16.tflite](https://huggingface.co/litert-community/Parakeet-tdt-ctc-110m-LiteRT) | 226 MB | mel [1, 80, 1601] + frame mask [1, 201] | CTC logits [1, 201, 1025] | CompiledModel GPU |

**Preprocessing**: 16 kHz mono → NeMo log-mel (80 bins, preemphasis 0.97, 512-pt FFT, slaney filterbank, per-feature norm), computed in Kotlin. Audio up to 16 s is padded to the fixed window.

**Decoding**: greedy CTC (drop blank + repeats) + SentencePiece detokenize (1024 pieces), on the host.

**Sample app**: [parakeet/](parakeet/) — Microphone recording + bundled sample + transcription display.

**Original project**: [NVIDIA NeMo / parakeet-tdt_ctc-110m](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)

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

### wav2vec2-CTC (fully-GPU, single-pass)

[wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) (Facebook, Apache-2.0) running **fully on the CompiledModel GPU**. Unlike Whisper's encoder–decoder, the **CTC** head needs **no autoregressive decoder** — it's one GPU graph, a **single forward pass** (`997/997` LITERT_CL on a Pixel 8a, **~22 ms** for a 10 s clip, device-vs-PyTorch corr **0.99998**, **exact** transcription). CTC greedy decode runs on the host. **Zero FFT** — raw 16 kHz waveform → 1D-conv feature extractor → 12-layer transformer → CTC head.

Re-authorings (all numerically-equivalent): GELU → tanh-GELU; feature-extractor `GroupNorm` → 4D reshape `(B,G,C//G,T)` mean/var (kills `GATHER_ND`; wav2vec2's GroupNorm is per-channel-over-time, so fp16-precise on Mali — unlike a `GroupNorm(1)` joint reduction, which fp16-walls); fold the `pos_conv` weight-norm; bidirectional mask → None.

| Model | Download Link | Size | Input | Output | API |
| ----- | ------------- | ---- | ----- | ------ | --- |
| wav2vec2-CTC | [w2v2_ctc_fp16.tflite](https://huggingface.co/litert-community/wav2vec2-base-960h-CTC-LiteRT) | 190 MB FP16 | waveform [1, 160000] @ 16 kHz | logits [1, 499, 32] | CompiledModel GPU |

**Preprocessing**: mono 16 kHz, zero-mean / unit-variance, padded/truncated to 10 s. **Decoding**: CTC greedy (argmax per frame → collapse repeats → drop blanks) in Kotlin.

**Sample app**: [asr/](asr/) — "Hold to Talk" mic + bundled sample clip + transcription display.

**Original project**: [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) | [Apache-2.0](https://github.com/facebookresearch/fairseq/blob/main/LICENSE)


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

# Text Generation

### RWKV-7 World 0.1B

The first **autoregressive language model running its full forward pass on the LiteRT `CompiledModel` GPU delegate** (RNN mode, host-side state). [RWKV-7](https://github.com/BlinkDL/RWKV-LM) is an RNN: one token per step with a fixed-size recurrent state, so the whole model fits a single static GPU graph — no KV cache growth, no dynamic shapes, no CPU fallback for any op. (The earlier Qwen3 embedding/reranker ships were encoders; this is generation.)

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| RWKV-7 World 0.1B (step) | [rwkv7_step_fp16.tflite](https://huggingface.co/litert-community/RWKV-7-World-0.1B-LiteRT) | 282 MB | Float32 `x_emb[1,768]`, `att_shift[12,768]`, `ffn_shift[12,768]`, `wkv[144,64,64]` | Float32 `logits[1,65536]` + 3 updated states | [BlinkDL/rwkv-7-world](https://huggingface.co/BlinkDL/rwkv-7-world) | [Apache-2.0](https://huggingface.co/BlinkDL/rwkv-7-world) | [rwkv7/](rwkv7/) |

**Host side**: token embedding row lookup from a memory-mapped fp16 table (~100 MB, GATHER is GPU-banned), greedy argmax over the 65536 logits, and recycling the three recurrent states into the next step. Prefill = the same step loop over the prompt. Tokenizer: RWKV World greedy longest-match trie (Kotlin port, fixture-tested against the Python reference).

**Conversion** (`rwkv7/scripts/build_rwkv7_step.py`, litert-torch): wkv7 recurrence at T=1 re-authored as plain 4D BMM/elementwise; `GroupNorm(heads)` → manual per-head mean/var; `F.normalize` → `x·rsqrt(Σx²+eps)`; `softplus` → branch-free `relu(z)+log1p(exp(-|z|))` (stock lowering emits GREATER+SELECT); export inputs `.clone()`d. Result: **1863/1863 nodes on the delegate, 1 partition**, ~18 ms/token (fp16, Pixel 8a). Step-vs-GPT-mode parity corr 1.0000000; device 30-token greedy generation tracks desktop fp32 (28/30 identical, 2 near-tie rank-2 picks, prefill corr 0.99995).

**Sample app**: [rwkv7/](rwkv7/) — prompt/chat UI with streaming tokens, greedy decoding, tok/s stats.

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

### PANNs CNN14 Audio Tagging

[PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) **CNN14** (`Cnn14_mAP=0.431`) general sound-event tagging — predicts probabilities over the **527 AudioSet classes** (speech, music, instruments, animals, vehicles, alarms, household sounds…) for ~10 s of audio. Multi-label, so several tags can be high at once. The CNN body runs **fully on CompiledModel GPU**; only the log-mel front-end is host-side (it overflows fp16). Distinct from wav2vec2 keyword-spotting (fixed speech commands) — this is open-domain environmental sound tagging.

```
waveform[320000] --[Kotlin log-mel]--> logmel[1,1,1001,64] --[GPU CNN14]--> probs[527] (sigmoid)
```

**On-device (Pixel 8a, Tensor G3 — verified):** CNN body **45/45** nodes on the LiteRT GPU delegate (`LITERT_CL`), **1 partition** (single graph, no CPU fallback); ~124 ms GPU + ~99 ms host log-mel ≈ **0.22 s** per 10 s clip; bundled-clip self-test → top tag "Speech" (matches PyTorch).

| Stage | Download | Size | Input → Output | Placement |
| ----- | -------- | ---- | -------------- | --------- |
| log-mel | (Kotlin `MelSpectrogram.kt`) | — | waveform [320000] → logmel [1,1,1001,64] | CPU |
| CNN14 | [HF: litert-community/PANNs-CNN14-AudioSet-LiteRT](https://huggingface.co/litert-community) | 162 MB FP16 | logmel [1,1,1001,64] → probs [1,527] | CompiledModel GPU |

**Why the log-mel is host-side**: PANNs' spectrogram is a torchlibrosa *DFT-as-Conv1d*, so there is **no FFT op** and the raw-audio graph is almost GPU-clean — the only blocker is the STFT centering reflect-pad (one `GATHER_ND`, removable with `pad_mode='constant'`, corr 1.0). But litert-torch lowers the 1024-tap DFT-conv wrongly (fp32 corr ≈ 0.19) and `|STFT|²` (~1e6) overflows fp16 on Mali → NaN. So the spectral front-end runs on the CPU in Kotlin (Whisper/Kokoro pattern), matched to torchlibrosa exactly (host log-mel vs torch corr 1.000000, max\|d\| 0.0017). The CNN body (`bn0` + 6 conv blocks + pooling + 2 FC + sigmoid) is a pure CNN and converts at **corr 1.000000 in fp32 and fp16** (op-check banned NONE, >4D 0). Mel basis exported to `assets/mel_basis.bin` [64,513]; periodic Hann + radix-2 FFT in Kotlin. See [panns/scripts/](panns/scripts/).

**Sample app**: [panns/](panns/) — bundled-clip self-test on launch + **Record 10 s & tag** button with a top-tags bar chart.

**Original project**: [qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) | code [Apache-2.0](https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/LICENSE), weights [CC-BY-4.0](https://zenodo.org/record/3987831)

# Pitch Detection

### CREPE

[CREPE](https://github.com/marl/crepe) monophonic pitch (f0) estimation running **fully on CompiledModel GPU**. A 1024-sample (16 kHz) window → activations over **360 pitch bins** (20 cents each); the host decodes to a frequency + nearest note. The sample is a **real-time tuner** — it listens to the mic and shows the live note and how many cents flat/sharp you are.

```
frame[1,1024] (16 kHz, per-frame zero-mean/unit-var) --[GPU CNN]--> activations[1,360] --[host]--> Hz → note
```

**On-device (Pixel 8a, Tensor G3 — verified):** CNN **49/49** nodes on the LiteRT GPU delegate (`LITERT_CL`), **1 partition** (single graph, no CPU fallback); ~75 ms/frame (full model); self-test (synthesized 440 Hz) → **A4, 440.4 Hz**.

| Model | Download | Size | Input → Output | Placement |
| ----- | -------- | ---- | -------------- | --------- |
| CREPE (full) | [HF: litert-community/CREPE-pitch-LiteRT](https://huggingface.co/litert-community) | 44.5 MB FP16 | frame [1,1024] → activations [1,360] | CompiledModel GPU |

**GPU compatibility**: the whole network is a **pure CNN** — 6× {zero-pad → Conv2d → ReLU → BatchNorm → MaxPool} + permute/reshape (≤4D) + Linear + sigmoid. No banned ops; per-frame normalization keeps activations ~O(1) so there is no fp16-on-Mali precision issue (banned NONE, >4D 0, fp16 tflite-vs-torch corr **1.000000**). The 44.5 MB model is bundled in assets. Decode: `cents = 20·bin + 1997.379…`, `Hz = 10·2^(cents/1200)`, activation-weighted around the peak; nearest note from `midi = 69 + 12·log2(Hz/440)`. See [crepe/scripts/](crepe/scripts/).

**Sample app**: [crepe/](crepe/) — 440 Hz self-test on launch + live mic tuner (note + cents gauge, `AudioSource.UNPROCESSED`).

**Original project**: [marl/crepe](https://github.com/marl/crepe) (ICASSP 2018) | [MIT](https://github.com/marl/crepe/blob/master/LICENSE); PyTorch weights via [torchcrepe](https://github.com/maxrmorrison/torchcrepe) (MIT)

# Audio Source Separation

### TIGER-DnR (Dialog / Effects / Music)

[TIGER](https://github.com/JusperLee/TIGER) (ICASSP 2025) **cinematic sound separation** running **fully on
CompiledModel GPU**: split any clip (movie scene, game, vlog) into **Dialogue / Sound effects / Music** stems
on the phone. Three sibling ~1.4 M-param band-split TIGER graphs (dialog / effect / music, trained on the
openly-built [DnR](https://github.com/darius522/dnr-utils) dataset) each process a 12.06 s 44.1 kHz chunk; per
DnR convention each graph contributes one stem. The **STFT runs inside the GPU graph** (windowed DFT as one
`Conv1d`); the host does only reflect-pad, iSTFT and overlap-add.

```
wav[1,534016] --[GPU: DFT-conv STFT → 57-band split → 8 weight-tied freq/frame UConv+MHSA iters → complex masks]--> (real, imag)[1,3,1025,1040] --[host iSTFT+OLA]--> stems
```

**On-device (Pixel 8a, Tensor G3 — verified):** **23 974 / 23 974** nodes on the LiteRT GPU delegate
(`LITERT_CL`), **1 partition** — the largest single graph in this zoo (10× NAFNet) — device-vs-PyTorch waveform
corr **0.99987**; ~4.5 s per 12.06 s chunk per stem-graph.

| Model | Download | Size | Input → Output | Placement |
| ----- | -------- | ---- | -------------- | --------- |
| TIGER-DnR (dialog/effect/music) | [HF: litert-community/TIGER-DnR-LiteRT](https://huggingface.co/litert-community) | 16.1 MB FP16 × 3 | wav [1,534016] → spec real+imag [1,3,1025,1040] | CompiledModel GPU |

**GPU compatibility**: no RNN / gather / dense warp, but heavy exact re-authoring: folded-batch `Conv1d` →
4D `(1,k)`-`Conv2d`; per-sample `GlobLN` → per-position chained-mean SafeNorm; chunk length chosen so T=1040
is divisible by 16 → every adaptive pool is a uniform `AVERAGE_POOL_2D` and every nearest resize an exact
integer-repeat (on-stride `RESIZE_NEAREST`); non-uniform band axis via constant one-hot/averaging
`FULLY_CONNECTED`; MHSA → per-head batch-1 3D BMM (1/√d folded into Q); PReLU → `relu(x) − w·relu(−x)`;
6-D mask view → static channel slices. Two device-only Mali fixes: **norm eps 1e-8/1e-5 underflows to 0 in
fp16** (silent bands → 0/0 = NaN that spreads across time; eps=1e-4 is exact-equivalent) and the mask head's
**dim-1 broadcast MUL** (`[1,1,bw,T] × [1,3,bw,T]`) mis-executes (all stems became source 0; rewritten as
per-source same-shape arithmetic). banned NONE, >4D 0, fp16 tflite-vs-torch corr 0.99991.

**Sample app**: [tiger/](tiger/) — pick an audio/video clip (or record 15 s) → separate → play each stem.

**Original project**: [JusperLee/TIGER](https://github.com/JusperLee/TIGER) (MIT) | weights
[JusperLee/TIGER-DnR](https://huggingface.co/JusperLee/TIGER-DnR) (Apache-2.0)

# Speaker Diarization

### pyannote 3.1 stack (segmentation + WeSpeaker)

On-device **"who spoke when"**: record a conversation (or pick a clip) → per-speaker timeline +
per-speaker playback. The [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
recipe (MIT) ported to Android — the single biggest-demand audio pipeline on HF (~8M downloads/month)
with no Android/LiteRT port until now.

```
pcm 16 kHz → [10 s windows]—[PyanNet powerset seg, ONNX CPU]→ local speakers
           → solo audio per (window, speaker) —[kaldi fbank+CMN, Kotlin]→ [1,500,80]
           —[WeSpeaker ResNet34, GPU]→ 256-d embeddings —[AHC clustering, host]→ timeline
```

**On-device (Pixel 8a, Tensor G3 — verified):** embedding **108/108** nodes `LITERT_CL`
(1 partition), **~1.2 ms**/window, device-vs-PyTorch cosine **0.99997**; segmentation ONNX
corr 1.0 / argmax agreement 100% vs PyTorch. End-to-end mirror of the app pipeline separates a
male/female test conversation correctly (2 speakers, correct turns).

| Model | Download | Size | Input → Output | Placement |
| ----- | -------- | ---- | -------------- | --------- |
| WeSpeaker ResNet34 embedding | [HF: litert-community/Speaker-Diarization-LiteRT](https://huggingface.co/litert-community) | 13.4 MB FP16 | fbank [1,500,80] → embedding [1,256] | CompiledModel GPU |
| pyannote segmentation-3.0 | same repo | 5.9 MB ONNX | wav [1,1,160000] → powerset [1,589,7] | onnxruntime CPU |

**GPU compatibility**: the WeSpeaker ResNet34 is a pure CNN with no maxpool stem (stride-2 convs) —
converts with **zero re-authoring** except the StatsPool std (down-scaled unbiased variance,
fp16-safe). The segmentation BiLSTM has no Mali GPU kernel → onnxruntime CPU (tiny model, the
Silero-VAD pattern). The kaldi-fbank front-end (hamming 25/10 ms, 80 mel, ×2¹⁵, CMN) is ported to
Kotlin with precomputed mel banks, verified against `torchaudio.compliance.kaldi.fbank` (corr 1.0).

**Sample app**: [diarization/](diarization/) — record up to 120 s or pick a clip → colored
per-speaker timeline, talk-time summary, per-speaker playback.

**Original projects**: [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio) (MIT) |
[segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (MIT) |
[WeSpeaker](https://github.com/wenet-e2e/wespeaker) weights
[pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) (CC-BY-4.0)

# Speech Enhancement

### CMGAN (noise suppression)

[CMGAN](https://github.com/ruizhecao96/CMGAN) (TASLP 2024) **speech enhancement** running **fully on
CompiledModel GPU**: record in a noisy place (or pick a clip) and A/B the denoised result. One
1.83 M-param dual-path conformer per 2 s 16 kHz chunk; the **STFT and mag^0.3 power compression run
inside the GPU graph** — the host does only reflect-pad, un-compress, iSTFT, overlap-add.

```
wav[1,32400] --[GPU: DFT-conv STFT → mag^0.3 → dense encoder → 4×(time+freq conformer) → mask+complex decoders]--> (real, imag)[1,1,321,201] --[host: mag^(1/0.3) + iSTFT + OLA]--> denoised
```

**On-device (Pixel 8a, Tensor G3 — verified):** **1 651 / 1 651** nodes on the LiteRT GPU delegate
(`LITERT_CL`), **1 partition**; **~20 ms per 2 s chunk** (RTF ≈ 0.01); SI-SNR **+7.2 dB** on a
6.6 dB noisy sample (PyTorch +9.6 dB), device-vs-torch wav corr 0.997.

| Model | Download | Size | Input → Output | Placement |
| ----- | -------- | ---- | -------------- | --------- |
| CMGAN (VoiceBank-DEMAND) | [HF: litert-community/CMGAN-LiteRT](https://huggingface.co/litert-community) | 4.2 MB FP16 | wav [1,32400] → spec real+imag [1,1,321,201] | CompiledModel GPU |

**GPU compatibility**: the phase path **cancels algebraically** (`mask·mag·cos∠x ≡ mask·x_r` — no
atan2/cos/sin in-graph); Shaw relative positional embedding (Embedding-lookup GATHER) baked to a
constant + applied via a 2D `FULLY_CONNECTED` and the pad/reshape **skew** realignment; conformer
folded batches → batch-1 4D with channel-LN / 1×1-conv Linears / `(1,k)` depthwise; `mag^0.3` →
`exp(0.3·ln(·))` (POW banned); SPConvTranspose 5-D view → exact 4D reshape chain; InstanceNorm →
Safe spatial norm, BatchNorm (eval) → constant scale/shift, all eps ≥ 1e-4 (fp16 min-normal), no
dim-1 broadcasts. fp16 tflite-vs-torch corr 0.999999.

**Sample app**: [cmgan/](cmgan/) — record noisy audio (unprocessed mic) or pick a clip → A/B
Noisy vs Enhanced playback.

**Original project**: [ruizhecao96/CMGAN](https://github.com/ruizhecao96/CMGAN) (MIT), trained on
VoiceBank-DEMAND

# Music Transcription

### Basic Pitch (audio-to-MIDI)

[Basic Pitch](https://github.com/spotify/basic-pitch) (Spotify, ICASSP 2022) **music transcription**
running **fully on CompiledModel GPU — including the conv-based CQT front-end**: play an instrument
(or sing) and see the notes on a piano roll. Re-authored from the official ONNX (bit-exact torch
reimplementation, corr 1.000000), 0.84 MB fp32.

**On-device (Pixel 8a, Tensor G3 — verified):** **241/241** nodes `LITERT_CL` (1 partition),
**~4.4 ms** per 2 s window; note-event F1@0.5 **0.98** vs reference, per-frame argmax agreement 98%.
Two device-only fp16 fixes: post-log clamp (recovers log(0)=-inf from the fp16-flushed 1e-10 floor,
desktop no-op) and per-bin CQT norm folded into per-octave kernel copies (exact; device contour
0.845 → 0.982).

| Model | Download | Size | Input → Output | Placement |
| ----- | -------- | ---- | -------------- | --------- |
| Basic Pitch nmp | [HF: litert-community/Basic-Pitch-LiteRT](https://huggingface.co/litert-community) | 0.84 MB FP32 | wav [1,43844] → contour/note/onset posteriorgrams | CompiledModel GPU |

**Sample app**: [basicpitch/](basicpitch/) — record or pick a clip → piano roll + note events.

**Original project**: [spotify/basic-pitch](https://github.com/spotify/basic-pitch) (Apache-2.0)

# Image Matching

### XFeat (local features)

[XFeat](https://github.com/verlab/accelerated_features) (CVPR 2024) **local feature extraction +
matching** running **fully on CompiledModel GPU**: pick two photos of the same scene and see the
matched keypoints — the building block for AR, panorama stitching, SLAM and registration.

**On-device (Pixel 8a, Tensor G3 — verified):** **72/72** nodes `LITERT_CL` (1 partition),
**~0.4 ms** per 640×480 image, device-vs-PyTorch corr **0.9999**, fp16 **1.4 MB**. Host does
instance-norm, keypoint decode (8×8-cell logits + dustbin), NMS, bilinear descriptor sampling and
mutual-nearest-neighbor matching. Re-authoring: InstanceNorm host-side (fp16 spatial-reduction
overflow) and `_unfold2d` space-to-depth → an exact one-hot `Conv2d(1,64,k=8,s=8)`.

| Model | Download | Size | Input → Output | Placement |
| ----- | -------- | ---- | -------------- | --------- |
| XFeat | [HF: litert-community/xfeat-litert](https://huggingface.co/litert-community) | 1.4 MB FP16 | gray [1,1,480,640] → feats/keypoints/heatmap | CompiledModel GPU |

**Sample app**: [xfeat/](xfeat/) — pick two photos → side-by-side match lines.

**Original project**: [verlab/accelerated_features](https://github.com/verlab/accelerated_features) (Apache-2.0)

# Text-Prompted Segmentation

### CLIPSeg

[CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined) (CVPR 2022) **open-vocabulary
segmentation**: type what to segment ("a cat", "the sky") and get a mask — no fixed class list. CLIP
text + vision encoders run on the CompiledModel **GPU**; the tiny decoder runs on **CPU** (its
4-head/head_dim-16 attention fp16-miscomputes on Mali — the vision encoder's 12-head/head_dim-64
attention survives at 0.998).

**On-device (Pixel 8a — verified):** text **761/761** GPU (~8.7 ms) + vision **613/613** GPU
(~8.2 ms) + decoder CPU (exact); end-to-end device-vs-PyTorch logits corr **0.99998**, mask IoU
**0.9986**. Re-authoring: qkv-3D-BMM attention, quick-GELU, baked interpolated pos-embed,
⭐`safe_ln_up` (up-scaled LayerNorm so the eps stays fp16-normal), `convT4x4` exact ConvTranspose.

| Model | Download | Size | Input → Output | Placement |
| ----- | -------- | ---- | -------------- | --------- |
| CLIPSeg rd64 (text+vision+decoder) | [HF: litert-community/CLIPSeg-rd64-LiteRT](https://huggingface.co/litert-community) | 76+147 MB FP16 + 3 MB decoder | image + prompt → mask [352,352] | GPU + CPU |

**Sample app**: [clipseg/](clipseg/) — pick image, type prompt, red mask overlay.

**Original project**: [CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined) (Apache-2.0)

# Image tagging

### RAM++ (Recognize Anything Plus)

[RAM++](https://github.com/xinyu1205/recognize-anything) (Apache-2.0) **open-vocabulary multi-label
tagging**: a photo in, the recognized tags out (from a 4,585-tag vocabulary; per-tag sigmoid, no
fixed class head). Swin-L encoder stages 0-2 and the Query2Label tag head run on the CompiledModel
**GPU**; the last Swin stage and the 479 MB frozen tag bank run on **CPU**.

**On-device (Pixel 8a, Tensor G3 — verified):** Swin 0-2 GPU (corr 0.998) + stage-3/reweight CPU
(exact) + tag head GPU (corr 0.9987, ~270 ms); sample photo → 14 tags in **~2 s**, all correct.
⭐**New Mali finding**: Swin-L stage 3 fp16-miscomputes on the GPU delegate — not head_dim (stage 2
shares head_dim 32) and not overflow (fp16-round sim = 0.99999997), but **fp16 matmul accumulation**
in the deep, high-magnitude (absmax 847) blocks; the 6144-wide fc2 / 48-head attention accumulate in
fp16, so those 2 blocks go to CPU. Reweight bakes the tag bank once as fp16 (229 MB, not 686 MB).

| Model | Download | Size | Input → Output | Placement |
| ----- | -------- | ---- | -------------- | --------- |
| RAM++ (Swin 0-2 / stage-3 / reweight / tag head) | [HF: litert-community/RAM-Plus-LiteRT](https://huggingface.co/litert-community) | ~769 MB FP16 (4 graphs) | image [1,3,384,384] → tags | GPU + CPU |

**Sample app**: [ram/](ram/) — pick a photo (or the bundled sample) → recognized tags.

**Original project**: [xinyu1205/recognize-anything](https://github.com/xinyu1205/recognize-anything) (Apache-2.0)

# Image quality

### NIMA (Neural Image Assessment)

[NIMA](https://github.com/idealo/image-quality-assessment) (idealo, Apache-2.0) scores a photo's
quality **1-10**. Two MobileNet models — **aesthetic** (AVA) and **technical** (TID2013) — each
predict a 10-bin score distribution (the score is its mean). Both run **fully on the CompiledModel
GPU** — a pure CNN, so it converts straight through `tf.lite` with no re-authoring.

**On-device (Pixel 8a, Tensor G3 — verified):** both models ~173 ms on the GPU delegate;
tflite-vs-Keras score parity **0.999998** (aesthetic) / **0.999915** (technical). 10-bin distribution
is the graph output; the 1-10 mean is host-side.

| Model | Download | Size | Input → Output | Placement |
| ----- | -------- | ---- | -------------- | --------- |
| NIMA aesthetic + technical | [HF: litert-community/NIMA-LiteRT](https://huggingface.co/litert-community) | 6.4 MB FP16 each | image [1,224,224,3] → dist [10] | GPU |

**Sample app**: [nima/](nima/) — pick a photo → aesthetic + technical score.

**Original project**: [idealo/image-quality-assessment](https://github.com/idealo/image-quality-assessment) (Apache-2.0)

# Image Classification

### Vision-RWKV (VRWKV-S)

The first **RWKV-style vision backbone running fully on the LiteRT `CompiledModel` GPU** — the vision companion to the RWKV-7 language model in [Text Generation](#rwkv-7-world-01b). [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV) (ICLR 2025) swaps softmax self-attention for a **bidirectional WKV** linear-attention scan; this is the VRWKV-S ImageNet-1K classifier (80.1% top-1). ~28 ms/inference on a Pixel 8a.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| VRWKV-S (ImageNet-1K) | [vrwkv_s_fp16.tflite](https://huggingface.co/litert-community/Vision-RWKV-S-LiteRT) | 48 MB | Float32 `image[1,3,224,224]` NCHW (ImageNet-norm) + `dist[1,1,196,196]` | Float32 `logits[1,1000]` | [OpenGVLab/Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV) | [Apache-2.0](https://github.com/OpenGVLab/Vision-RWKV/blob/master/LICENSE) | [vrwkv/](vrwkv/) |

**Preprocessing**: resize (short edge 256) → center-crop 224 → ImageNet normalization, NCHW. The second input is the constant token-distance matrix `dist[t,i] = |t-i|` (see below).

**Conversion** (`vrwkv/scripts/build_vrwkv.py`, litert-torch): the bidirectional WKV (a CUDA kernel) is re-authored exactly — for the fixed 196-token grid it is a per-channel decay-biased attention `softmax_i(k[c,i] − (decay[c]/T)|t−i| + (first[c]/T)δ) · v`, i.e. plain 4D `softmax` + `matmul`, **no sequential scan**. ⭐The `[C,T,T]` decay bias `w·dist` would be const-folded into a 59 MB-per-block flatbuffer constant (an unshippable 1.5 GB model that fp16 can't shrink), so the token-distance matrix is fed as a **runtime input** (`eye = relu(1 − dist)`) and the bias is computed live → 48 MB. VRWKV-S is post-norm (LayerScale baked into the following norm); q-shift is pad+slice+concat (≤4D). Result: **1371/1371 nodes on the delegate, 1 partition**; device fp16 top-1 matches desktop fp32 (logits corr 0.9989).

**Sample app**: [vrwkv/](vrwkv/) — pick a photo → top-5 ImageNet predictions.

# Fine-Grained Classification

### PlantNet-300K (1081 plant species)

Identify **1081 plant species** from a photo, fully on the LiteRT `CompiledModel` GPU. A [PlantNet-300K](https://github.com/plantnet/PlantNet-300K) (NeurIPS 2021) ResNet18 — the first fine-grained classifier in this zoo. ~16 ms/frame on a Pixel 8a.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| PlantNet-300K ResNet18 | [plantnet.tflite](https://huggingface.co/litert-community/PlantNet-300K-ResNet18-LiteRT) | 47 MB | Float32 [1, 3, 224, 224] NCHW (ImageNet-norm) | Float32 [1, 1081] logits | [plantnet/PlantNet-300K](https://github.com/plantnet/PlantNet-300K) ([cpoisson/plantnet300k-resnet18](https://huggingface.co/cpoisson/plantnet300k-resnet18)) | [Apache-2.0](https://huggingface.co/cpoisson/plantnet300k-resnet18) | [plantnet/](plantnet/) |

**Preprocessing**: RGB, center-crop → resize 224×224, ImageNet normalization, NCHW. **Labels**: class index `i` → the `i`-th species when PlantNet-300K species-id strings are sorted (torchvision `ImageFolder` order).

**Conversion** (`plantnet/scripts/build_plantnet.py`, litert-torch): plain torchvision ResNet18 → fully GPU-compatible (**37/37 nodes on the delegate, 1 partition**; device corr 0.99999, top-1 match) with **one patch** — the ResNet stem `MaxPool2d(padding=1)` lowers to a `-inf` PADV2 (`PADV2: src has wrong size` on Mali), replaced by an explicit 0-pad + unpadded maxpool (exact post-ReLU). CPU-exact vs PyTorch (corr 0.99999999999).

**Sample app**: [plantnet/](plantnet/) — camera → PlantNet-300K GPU → top-5 species (Latin names).



# Face

### 3DDFA_V2 (3D face alignment)

[3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) (ECCV 2020, MIT) fits a **3D morphable face model**
to a photo: a MobileNetV1 regresses **62 3DMM parameters** (pose + 40 shape + 10 expression) on the
CompiledModel **GPU**; the 68 3D face landmarks (and a dense mesh) are reconstructed from the BFM
bases host-side. A pure CNN — converts through litert-torch with no re-authoring.

**On-device (Pixel 8a, Tensor G3 — verified):** fp16 tflite-vs-PyTorch 62-param corr **0.999999**,
reconstructed landmarks match to **0.02 px**; 68 landmarks in well under a second. Kotlin-port gotchas:
the model wants **cv2 BGR** input, the BFM bases are **interleaved** (`reshape(3,-1, order='F')`), and
`android.media.FaceDetector` needs an even width.

| Model | Download | Size | Input → Output | Placement |
| ----- | -------- | ---- | -------------- | --------- |
| 3DDFA_V2 MobileNetV1 | [HF: litert-community/3DDFA-V2-LiteRT](https://huggingface.co/litert-community) | 6.3 MB FP16 | crop [1,3,120,120] → 62 params → 68 landmarks | GPU |

**Sample app**: [tddfa/](tddfa/) — pick a frontal-face photo → 68 3D landmarks.

**Original project**: [cleardusk/3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) (MIT)

### BiSeNet (face parsing)

Real-time **face parsing** running fully on the LiteRT `CompiledModel` GPU. [BiSeNet](https://arxiv.org/abs/1808.00897) ([zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)) segments a face into the **19 CelebAMask-HQ classes** (skin, brows, eyes, nose, lips, ears, hair, hat, glasses, neck, cloth, …) for AR / beauty / makeup. ~22 ms/frame on a Pixel 8a. Pure CNN (ResNet18 backbone).

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| BiSeNet | [faceparsing.tflite](https://huggingface.co/litert-community/BiSeNet-Face-Parsing-LiteRT) | 53 MB | Float32 [1, 3, 512, 512] NCHW (ImageNet-norm) | Float32 [1, 19, 512, 512] logits | [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) | [MIT](https://github.com/zllrunning/face-parsing.PyTorch/blob/master/LICENSE) | [faceparsing/](faceparsing/) |

**Conversion** (`faceparsing/scripts/build_faceparsing.py`, litert-torch): 3 re-authoring patches → fully GPU-compatible (**74/74 nodes on the delegate, 1 partition**; device corr 0.99999, argmax 99.96% vs PyTorch): (1) `align_corners=True`→`False`; (2) global `avg_pool2d(x, x.size()[2:])`→`mean([2,3])` (Mali rejects a full-spatial-kernel `AVERAGE_POOL_2D`); (3) **ZeroPadMaxPool** — the ResNet stem `MaxPool2d(padding=1)` lowers to a `-inf` PADV2 (`PADV2: src has wrong size` on Mali), replaced by an explicit 0-pad + unpadded maxpool (exact since the input is post-ReLU ≥ 0). These are **on-device-only** rejections — the op inventory is clean and CPU parity is 1.0, but the GPU delegate won't compile without them. CPU-exact vs PyTorch (corr 0.99999999999).

**Sample app**: [faceparsing/](faceparsing/) — front camera → BiSeNet GPU → 19-class CelebAMask face-part overlay.

### HSEmotion (facial emotion recognition)

Recognize the **8 AffectNet emotions** (anger, contempt, disgust, fear, happiness, neutral, sadness, surprise) from a face, fully on the LiteRT `CompiledModel` GPU. [HSEmotion](https://github.com/av-savchenko/face-emotion-recognition) (EmotiEffLib, Apache-2.0) is an EfficientNet-B0 fine-tuned on AffectNet — the first emotion classifier in this zoo. ~2 ms/inference on a Pixel 8a.

| Model | Download Link | Size | Input | Output | Original Project | License | Sample App |
| ----- | ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| HSEmotion EfficientNet-B0 | [hsemotion_b0_fp16.tflite](https://huggingface.co/litert-community/HSEmotion-B0-LiteRT) | 8 MB | Float32 [1, 3, 224, 224] NCHW (ImageNet-norm) | Float32 [1, 8] emotion logits | [av-savchenko/face-emotion-recognition](https://github.com/av-savchenko/face-emotion-recognition) | [Apache-2.0](https://github.com/av-savchenko/face-emotion-recognition/blob/main/LICENSE) | [hsemotion/](hsemotion/) |

**Preprocessing**: detect + crop the face (the app uses the built-in `android.media.FaceDetector`), resize 224×224, ImageNet normalization, NCHW.

**Conversion** (`hsemotion/scripts/build_hsemotion.py`, litert-torch): the released weights are an **old-timm pickle** whose forward is broken under current timm, so the state dict is lifted into a fresh timm `tf_efficientnet_b0` (`classifier.0.*`→`classifier.*`) with a working forward. ⭐The one GPU fix — the **SqueezeExcite global mean** `x.mean((2,3))` over the 112×112 stem map is a single fp16 reduction whose partial sum overflows 65504 → **all-NaN device output** (the delegate computes it in fp16 even for an fp32 graph); replaced by a **hierarchical mean** (`avg_pool2d` over equal-size tiling windows ≤ 49 elements — mathematically identical, fp16-safe). Result: **342/342 nodes on the delegate, 1 partition**; device fp16 top-1 matches desktop fp32 (logits corr 0.99997). Desktop fp16 CPU corr vs PyTorch 1.0.

**Sample app**: [hsemotion/](hsemotion/) — pick a face photo → detected face + emotion distribution.

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

### GFPGAN v1.4 (Blind Face Restoration)

GFPGAN (TencentARC): restores degraded / low-quality faces using a StyleGAN2 generative facial prior, running **fully on CompiledModel GPU**. Converted with **litert-torch**; the StyleGAN2 `ModulatedConv2d` (a 5D runtime-weight conv, doubly GPU-banned) is rewritten to an exact 4D form (modulation → input channel-scale + constant conv; demod → a constant `(c_out×c_in)` matmul + `RSQRT`). The demod sum `Σ s²·Wsq` overflows Mali fp16 (style vectors reach |s|~1000 → ~2.3e6 ≫ 65504 → the decoder collapses to a flat color), fixed by normalizing the style by its per-image max before squaring — the scale cancels exactly against the demod, so the device output matches desktop fp32. The app detects the face with **YuNet** and FFHQ-aligns it before restoration (the StyleGAN prior mangles the mouth on off-template crops). Device-verified on Pixel 8a: `551/551 LITERT_CL`, fully GPU, ~1.2 s/face.

| Model | Size (fp16) | Input | Output | Original Project | License | Sample App |
| ----- | ----------- | ----- | ------ | ---------------- | ------- | ---------- |
| GFPGAN v1.4 | 431 MB | Float32 [1, 3, 512, 512] NCHW, [-1,1] | Float32 [1, 3, 512, 512] NCHW, [-1,1] | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) | [Apache-2.0](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE) | [gfpgan/](gfpgan/) |

**Output format**: `[1, 3, 512, 512]` NCHW restored face in [-1,1] → denormalize `(x+1)*127.5`.

**Preprocessing**: detect 5 face landmarks (YuNet), similarity-warp to the facexlib 512 template, then normalize to [-1,1] (`x/127.5 - 1`). See [litert-community/GFPGAN-v1.4-LiteRT](https://huggingface.co/litert-community/GFPGAN-v1.4-LiteRT).

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


# Face Detection

### YuNet

[YuNet](https://github.com/ShiqiYu/libfacedetection) (ShiqiYu/libfacedetection, BSD-3-Clause): a tiny, fast **face detector** (faces + 5 landmarks). At **0.076 M params / 0.3 MB fp16** it is the **smallest model in this repo**. Runs **fully on the GPU** (`146/146` LITERT_CL on a Pixel 8a, **~4 ms** at 640×640, device-vs-PyTorch corr **0.9999**).

Pure CNN (depthwise-separable `ConvDPUnit`) + a **nearest-upsample** neck (→ `RESIZE_NEAREST_NEIGHBOR`, no transposed conv); non-padded `MaxPool` (no `PADV2`). No re-authoring — banned ops NONE, ≤4D. The head's `permute/reshape/sigmoid` per stride is baked in (12 outputs: cls/obj/bbox/kps × strides {8,16,32}); decode (priors + center/exp box + landmarks + NMS) runs in the app.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [yunet_fp16.tflite](https://huggingface.co/litert-community/YuNet-Face-LiteRT) | 0.3 MB | Float32 [1, 3, 640, 640] NCHW (BGR, 0-255) | 12 × (cls/obj/bbox/kps per stride) | [ShiqiYu/libfacedetection](https://github.com/ShiqiYu/libfacedetection) | [BSD-3-Clause](https://github.com/ShiqiYu/libfacedetection/blob/master/LICENSE) | [yunet/](yunet/) |

**Decode**: score = `cls·obj`; box = `center + exp(wh)·stride`; 5 landmarks; NMS (IoU 0.45). **Preprocessing**: letterbox to 640×640, **BGR, 0-255 (no normalization)**.

**Sample app**: [yunet/](yunet/) — image picker + face boxes + 5 landmarks.


### RTMPose-Face (WFLW, 98-point face alignment)

[RTMPose](https://github.com/open-mmlab/mmpose) (mmpose, Apache-2.0) **face alignment** trained on **WFLW**: **98 dense facial landmarks** (contour, eyebrows, eyes, nose, mouth, pupils) — the dense complement to YuNet's 5 points (detect a face, then align). The **same model family** as RTMPose-s above; only the config/checkpoint change to WFLW, and the two Mali fixes (SafeRMSNorm + GAU broadcast-reduce) transfer **unchanged**. Runs **fully on the GPU** (`333/333` LITERT_CL on a Pixel 8a, **~4 ms**, device-vs-PyTorch SimCC corr **0.9995**).

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [rtm_face_fp16.tflite](https://huggingface.co/litert-community/RTMPose-Face-WFLW-LiteRT) | 33.6 MB | Float32 [1, 3, 256, 256] NCHW | simcc_x [1,98,512], simcc_y [1,98,512] | [open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) | [Apache-2.0](https://github.com/open-mmlab/mmpose/blob/main/LICENSE) | [rtmface/](rtmface/) |

**Output**: output[0] = simcc_x, output[1] = simcc_y; each landmark = `argmax` over its 1D SimCC (bins = pixels × 2). **Preprocessing**: center-crop to a face, resize 256×256, mmpose mean/std (RGB, 0-255).

**Sample app**: [rtmface/](rtmface/) — image picker + 98-point face mesh.


# Gaze Estimation

### L2CS-Net

[L2CS-Net](https://github.com/Ahmednull/L2CS-Net) (Ahmednull, MIT): **gaze estimation** — predicts where a centered face is looking (yaw/pitch), for attention/AR/accessibility. ResNet50 backbone trained on Gaze360. Runs **fully on the GPU** (`139/139` LITERT_CL on a Pixel 8a, **~3 ms**, fp16 **47.9 MB**, device-vs-PyTorch corr **0.9999**).

Converted via **litert-torch** with the two ResNet fixes: the stem `MaxPool2d(3,s2,p1)` → **zero-pad + valid max-pool** (PyTorch's max-pool pads with `-inf` → a `PADV2` the Mali delegate won't delegate; since the pool follows a ReLU, a 0-pad is exactly equivalent → `PAD`), and the global `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)`. The angle-bin softmax is baked in.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [gaze_fp16.tflite](https://huggingface.co/litert-community/L2CS-Gaze360-LiteRT) | 47.9 MB | Float32 [1, 3, 448, 448] NCHW | yaw [1,90], pitch [1,90] (softmax bins) | [Ahmednull/L2CS-Net](https://github.com/Ahmednull/L2CS-Net) | [MIT](https://github.com/Ahmednull/L2CS-Net/blob/main/LICENSE) | [gaze/](gaze/) |

**Output / decode**: 90 angle bins spanning [-180,180]° (4° each); the gaze angle is the softmax expectation `Σ p_i·i · 4 − 180`. **Preprocessing**: center-crop to a (centered) face, resize 448×448, /255, ImageNet mean/std, NCHW.

**Sample app**: [gaze/](gaze/) — image picker + gaze-direction arrow.


# Saliency Prediction

### UniSal

[UniSal](https://github.com/rdroste/unisal) (rdroste, Apache-2.0): **visual saliency** — predicts a heatmap of *where humans look* in an image. MobileNetV2 encoder + bilinear decoder, **3.71 M params**. Runs **fully on the GPU** (`158/158` LITERT_CL on a Pixel 8a, **~3 ms** at 256×256, device-vs-PyTorch corr **0.9998**, **6.5 MB** fp16).

Three numerically-exact GPU fixes: the MobileNetV2 strided subsample `x[..., ::2, ::2]` → `F.avg_pool2d(x, 1, 2)` (same pixels, avoids `GATHER_ND`); the 16 Gaussian prior maps **baked** to constants (size-only; avoids `GATHER_ND`/`BROADCAST_TO`); and the 41×41 Gaussian-smoothing `replicate`-pad → 0-pad. For static images the Bypass-RNN path is used + the SALICON domain pinned.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [unisal_fp16.tflite](https://huggingface.co/litert-community/UniSal-Saliency-LiteRT) | 6.5 MB | Float32 [1, 3, 256, 256] NCHW | saliency [1, 1, 256, 256] | [rdroste/unisal](https://github.com/rdroste/unisal) | [Apache-2.0](https://github.com/rdroste/unisal/blob/master/LICENSE) | [saliency/](saliency/) |

**Preprocessing**: center-crop, resize 256×256, /255, ImageNet mean/std, NCHW. The app min-max normalizes the saliency and overlays a jet heatmap.

**Sample app**: [saliency/](saliency/) — image picker + saliency heatmap overlay.


# Line Detection

### M-LSD-tiny

[M-LSD](https://github.com/navervision/mlsd) (NAVER, AAAI 2022): light-weight real-time **line segment detection** — straight line segments for building edges, document borders, wireframes, and room layout. The **tiny** variant (MobileNetV2 backbone, 0.62M params) runs **fully on the GPU** (`99/99` LITERT_CL on a Pixel 8a, **~2 ms**, device-vs-PyTorch corr **0.997**). At **1.4 MB** fp16 it is the **smallest model in this zoo**.

Converted via **litert-torch** with a single re-authoring: the decoder's `F.interpolate(bilinear, align_corners=True)` → `align_corners=False` (the delegate bans `align_corners=True`). MobileNetV2 has no max-pool (strided convs → no `PADV2`) and the upsample is `RESIZE_BILINEAR` (not a transposed conv) → fully GPU-clean. The output is a "TP map" (center heatmap + displacement); the decode (sigmoid + NMS + displacement → endpoints) runs in the app.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [mlsd_fp16.tflite](https://huggingface.co/litert-community/M-LSD-tiny-LiteRT) | 1.4 MB | Float32 [1, 4, 512, 512] NCHW (RGB + ones) | tpMap [1, 9, 256, 256] | [navervision/mlsd](https://github.com/navervision/mlsd) | [Apache-2.0](https://github.com/navervision/mlsd/blob/main/LICENSE) | [mlsd/](mlsd/) |

**Preprocessing**: resize to 512×512, append a 4th channel of ones, scale `(x/127.5)-1`, NCHW. **Decode**: sigmoid center map → 3×3 max NMS → displacement → endpoints (×2 to 512-space).

**Sample app**: [mlsd/](mlsd/) — image picker + line-segment overlay.


# Style Transfer

### Fast Neural Style (4 styles)

Fast neural **style transfer** ([PyTorch examples](https://github.com/pytorch/examples/tree/main/fast_neural_style) `TransformerNet`, Johnson et al.): applies an artistic style to a photo — **4 styles** (candy / mosaic / rain_princess / udnie), each a **3.5 MB** fp16 graph. Runs **fully on the GPU** (`350/350` LITERT_CL on a Pixel 8a, **~9 ms** @ 256×256, device-vs-PyTorch corr **0.9998–0.9999** for all styles).

Converted via **litert-torch** with three numerically-exact re-authorings: (1) `ReflectionPad2d` → zero-pad (`GATHER_ND` → `PAD`); (2) the large conv activations (≈|5000|) lose fp16 precision on Mali (corr 0.34 at full residency) → **scale the conv weights down (InstanceNorm is scale-invariant → exact)** so the fp16 accumulation stays precise; (3) `InstanceNorm` → **SafeInstanceNorm** (down-scaled-domain spatial reduction, fp16-safe). Upsample is `interpolate(nearest)` (no ZeroStuff).

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [Fast-Neural-Style-LiteRT](https://huggingface.co/litert-community/Fast-Neural-Style-LiteRT) | 3.5 MB ×4 | Float32 [1, 3, 256, 256] NCHW (RGB 0-255) | [1, 3, 256, 256] (RGB 0-255) | [pytorch/examples](https://github.com/pytorch/examples) | [BSD-3-Clause](https://github.com/pytorch/examples/blob/main/LICENSE) | [neuralstyle/](neuralstyle/) |

**Preprocessing**: center-crop, resize to 256×256, RGB 0–255 (no normalization), NCHW. Output 0–255 RGB (clamp).

**Sample app**: [neuralstyle/](neuralstyle/) — image picker + 4 tappable style buttons.


# Low-Light Enhancement

### CPGA-Net

[CPGA-Net](https://github.com/Shyandram/CPGA-Net-Pytorch) (Shyandram, IJPRAI, MIT): **low-light image enhancement** (brighten dark photos) via Channel Prior + Gamma Correction. At **0.025 M params / 0.1 MB fp16** it is the **smallest model in this repo**. Runs **fully on the GPU** (`135/135` LITERT_CL on a Pixel 8a, **~2 ms** at 256×256, device-vs-PyTorch corr **0.99999**).

Three numerically-exact GPU fixes: the gamma correction `x^γ` → `exp(γ·log x)` (avoids the banned `POW`); the CBAM/gamma global pools → `mean(3).mean(2)` and `F.max_pool2d(x,(H,W))`; the dark/bright channel prior stays as `REDUCE_MAX`/`REDUCE_MIN`. The guided-filter post-process is disabled.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [cpga_fp16.tflite](https://huggingface.co/litert-community/CPGA-Net-LowLight-LiteRT) | 0.1 MB | Float32 [1, 3, 256, 256] NCHW ([0,1]) | enhanced [1, 3, 256, 256] ([0,1]) | [Shyandram/CPGA-Net-Pytorch](https://github.com/Shyandram/CPGA-Net-Pytorch) | [MIT](https://github.com/Shyandram/CPGA-Net-Pytorch/blob/main/LICENSE) | [lowlight/](lowlight/) |

**Preprocessing**: center-crop, resize 256×256, RGB scaled to [0,1], NCHW.

**Sample app**: [lowlight/](lowlight/) — image picker + enhanced view (press-and-hold to compare).


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


# GPU Compatibility Notes

CompiledModel GPU requires **all ops** to be GPU-compatible. Key constraints:

- All tensors must be 4D or less
- No dynamic dimensions (-1) in reshape
- Avoid: TOPK_V2, GATHER, GATHER_ND, CAST (float-int), GELU, PACK, SPLIT
- ⚠ **Never collapse the batch dim of an attention / batched-matmul chain.** A rank-3 SDPA (`q/k/v` as `[heads, N, d]`) compiles, delegates every node, passes the op gate and matches the host exactly — yet ML Drift **silently returns wrong values** (SAM 2.1 mask decoder: corr **0.265** vs CPU on a Pixel 8a; still **0.473** with fp32 GPU compute forced, so it is a correctness bug, not an fp16 wall). Keep tensors at rank 4 (`[1, heads, N, d]`) — that also ran ~20% faster here. **Full GPU residency + a clean op gate + desktop parity do not imply correctness**; only a numeric GPU-vs-CPU check on device catches this.

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

# Text Embedding (RAG)

### Qwen3-Embedding-0.6B

[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (2025 SOTA, Apache-2.0)
re-authored to run **entirely on CompiledModel GPU** for on-device semantic search / RAG retrieval.
Last-token pooling = a **single forward** (no generation, no KV cache), so it is a plain `.tflite` —
not a `.litertlm`. Token embedding is done **host-side** (GATHER is GPU-banned), GQA heads are
`cat`-repeated (a broadcast matmul emits the Mali-rejected `BROADCAST_TO`), and a **max-normalized
RMSNorm** fixes the deep-stack fp16 overflow that otherwise collapses the 28-layer output to 0.

**On-device (Pixel 8a, Tensor G3 — verified):** all **3264/3264 nodes on the GPU delegate** (zero CPU
fallback), ~390 ms per embedding, output cosine **0.9997** vs the HF fp32 reference; semantic ranking
correct (*"What is the capital of China?"* → *"…Beijing"* 0.77, unrelated docs <0.1).

| Model | Download | Size | Input → Output | Placement |
| ----- | -------- | ---- | -------------- | --------- |
| Qwen3-Embedding-0.6B | [HF: litert-community/Qwen3-Embedding-0.6B-LiteRT](https://huggingface.co/litert-community/Qwen3-Embedding-0.6B-LiteRT) | 881 MB FP16 + 310 MB embed table | inputs_embeds [1,128,1024] → hidden [1,128,1024] → 1024-d embedding | GPU |

**Sample app**: [text-embedding/](text-embedding/) — type a query → cosine-ranked documents. Also
upstreamed as [litert-samples #196](https://github.com/google-ai-edge/litert-samples/pull/196).

**Original project**: [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (Apache-2.0)

### Qwen3-Reranker-0.6B

[Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) (2025 SOTA, Apache-2.0)
re-authored to run **entirely on CompiledModel GPU** — the reranking half of on-device RAG
(embed → retrieve → **rerank**). It scores a `(query, document)` pair by `P("yes")` relevance in a
single forward pass. Same Qwen3-0.6B graph as the embedder; the only difference is a baked **2-logit
head** (tied-embedding rows for `"no"`=2152 / `"yes"`=9693) → `[1,256,2]`, host softmax → P(yes).

**On-device (Pixel 8a, Tensor G3 — verified):** all nodes on the GPU delegate, `P(yes)` parity
**ref 0.9995 / dev 0.9994**; query *"capital of China?"* → Beijing doc **0.999**, all others **0.000**.

| Model | Download | Size | Input → Output | Placement |
| ----- | -------- | ---- | -------------- | --------- |
| Qwen3-Reranker-0.6B | [HF: litert-community/Qwen3-Reranker-0.6B-LiteRT](https://huggingface.co/litert-community/Qwen3-Reranker-0.6B-LiteRT) | 882 MB FP16 + 310 MB embed table | inputs_embeds [1,256,1024] → logits [1,256,2] → P(yes) | GPU |

**Sample app**: [text-reranking/](text-reranking/) — query + candidate docs → P(yes) ranking. Also
upstreamed as [litert-samples #197](https://github.com/google-ai-edge/litert-samples/pull/197).

**Original project**: [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) (Apache-2.0)

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
