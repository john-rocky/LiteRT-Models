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

- [**Pose Estimation**](#pose-estimation)
  - [YOLO26n-pose](#yolo26n-pose)

- [**Segmentation**](#segmentation)
  - [MobileSAM](#mobilesam)

- [**Background Removal**](#background-removal)
  - [RMBG-1.4 (ISNet)](#rmbg-14-isnet)

- [**Inpainting**](#inpainting)
  - [LaMa-Dilated](#lama-dilated)

- [**Zero-Shot Classification**](#zero-shot-classification)
  - [CLIP ViT-B/32](#clip-vit-b32)

- [**Surface Normal Estimation**](#surface-normal-estimation)
  - [DSINE](#dsine)

- [**Speech Recognition**](#speech-recognition)
  - [Whisper-tiny](#whisper-tiny)

- [**Text-to-Speech**](#text-to-speech)
  - [Kokoro-82M](#kokoro-82m)

- [**Vision-Language Model**](#vision-language-model)
  - [SmolVLM-256M](#smolvlm-256m)

- [**Voice Assistant**](#voice-assistant)
  - [Whisper + SmolLM2 + Kokoro pipeline](#whisper--smollm2--kokoro-pipeline)

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

# Pose Estimation

### YOLO26n-pose

Real-time human pose estimation with Ultralytics YOLO26n-pose. 17 COCO keypoints + skeleton overlay, runs on CompiledModel GPU. Three input modes in the sample app: live camera, picked image, picked video — all share the same pose decoder.

Converted via **litert-torch** by wrapping the head with `end2end=False, export=True, format='tflite'`. This bypasses the default end-to-end NMS-free path (which compiles to GPU-incompatible `TOPK_V2`/`GATHER`) and exposes the legacy one-to-many head output. Ultralytics' default ONNX → onnx2tf path breaks on the YOLO26 backbone (`model.2/m.0/Add` channel mismatch), so the conversion goes PyTorch → litert-torch directly, the same path used for MobileSAM, RMBG, and DSINE in this repo.

| Download Link | Size | Input | Output | Original Project | License | Sample App |
| ------------- | ---- | ----- | ------ | ---------------- | ------- | ---------- |
| [yolo26n_pose.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/yolo26n_pose.tflite) | 12 MB | Float32 [1, 3, 384, 384] NCHW | Float32 [1, 56, 3024] | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) | [yolo-pose/](yolo-pose/) |

**Output format**: `[1, 56, N]` — `4 bbox (cx, cy, w, h) + 1 person conf + 17 keypoints * 3 (x, y, vis)`. Bbox is YOLO **xywh center format** (the legacy one-to-many head emits xywh, not xyxy). Bbox and keypoint xy values are in input image pixel space (0..384); person confidence and per-keypoint visibility are sigmoid-activated (0..1). Requires NMS post-processing.

**Preprocessing**: RGB normalized to 0-1 (divide by 255), planar NCHW layout. No ImageNet mean/std.

**Sample app**: [yolo-pose/](yolo-pose/) — Camera / Image / Video mode toggle, skeleton overlay matching either FILL_CENTER (camera) or FIT_CENTER (image/video).

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

Runs on **ONNX Runtime with NNAPI EP fallback to XNNPACK CPU**. Phonemization is pure Kotlin/Java (no NDK): English uses the CMU Pronouncing Dictionary (126k entries) with ARPABET → Misaki IPA mapping (verified bit-identical to misaki for in-vocab inputs); Japanese uses kuromoji-ipadic morphological analyzer with katakana → IPA lookup (yōon and long vowel handling).

| Model | Download Link | Size | Input | Output | API |
| ----- | ------------- | ---- | ----- | ------ | --- |
| TTS | [model_fp16.onnx](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_fp16.onnx) | 163 MB | input_ids [1, seq] int64 + style [1, 256] + speed [1] | waveform [1, samples] @ 24 kHz | ONNX Runtime |
| Voices | [voices/*.bin](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main/voices) | 510 KB each | — | Style vectors [N, 1, 256] | — |

**Bundled voices**: af_heart, am_michael, bf_emma (English), jf_alpha, jm_kumo (Japanese). Add more from [the HF voices folder](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main/voices).

**Phonemizer assets**: `cmudict.txt` (3.3 MB plain text, generated by `scripts/build_cmudict.py`), `kokoro_vocab.json` (vocab mapping IPA characters to Kokoro token IDs).

**Sample app**: [kokoro/](kokoro/) — Free-form text input with auto-language detection, voice picker, preset phrase fallback, AudioTrack PCM_FLOAT playback.

**Original project**: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | [Apache-2.0](https://huggingface.co/hexgrad/Kokoro-82M)

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

Full on-device conversational pipeline running entirely offline on a Pixel-class device:

```
mic ─► Whisper-tiny STT (TFLite GPU)
     ─► SmolLM2-135M chat (ONNX CPU, text-only via SmolVLM decoder)
     ─► EnglishPhonemizer (CMU dict + ARPABET → IPA)
     ─► Kokoro-82M TTS (ONNX, NNAPI EP)
     ─► AudioTrack streaming playback
```

**Streaming TTS** drops time-to-first-audio from ~4.5 s to **~1.5 s** on Pixel 8a: the LM token-by-token callback detects sentence boundaries, each completed sentence is phonemized and synthesized while the LM keeps generating, and audio chunks are pushed to a player thread via a blocking queue. Each chunk plays via a one-shot MODE_STATIC AudioTrack, decoupling the LM/TTS producer from playback duration.

**Per-stage on Pixel 8a** (short replies):
- STT: ~700 ms
- LM: ~1000-1500 ms
- TTS total: ~1100 ms (sentence-chunked)
- End-to-end: ~5 s for a typical Q&A turn

| Component | Model | Size |
| --------- | ----- | ---- |
| STT encoder | [whisper_encoder.tflite](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/whisper_encoder.tflite) | 33 MB |
| STT decoder | [whisper_decoder.onnx](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/whisper_decoder.onnx) | 199 MB |
| LM decoder | [smolvlm_decoder.onnx](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/smolvlm_decoder.onnx) | 515 MB |
| LM embeddings | [embed_tokens.bin](https://github.com/john-rocky/LiteRT-Models/releases/download/v2/embed_tokens.bin) | 108 MB |
| TTS | [model_fp16.onnx](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_fp16.onnx) | 163 MB |
| TTS voices | [voices/*.bin](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main/voices) | 510 KB each |

**Sample app**: [voiceassistant/](voiceassistant/) — Hold-to-talk button, transcript display, streaming response display, sentence-by-sentence audio playback. English-only for the MVP.

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

# License

MIT (sample apps). Model licenses follow their original projects.
