# CLIP Zero-Shot Classification

Zero-shot image classification using OpenAI CLIP ViT-B/32 with LiteRT CompiledModel GPU.

## Features

- **Zero-shot classification** — classify images into any of 96 pre-defined labels
- CLIP ViT-B/32 image encoder (87.8M params) with pre-computed text embeddings
- Softmax-normalized confidence scores with top-10 results display
- Text prompt template: "a photo of a {label}" for optimal accuracy

## Setup

1. Download models from [Releases](https://github.com/john-rocky/LiteRT-Models/releases/tag/v2):
   - `clip_image_encoder.tflite` (352 MB)
   - `labels.txt` + `text_embeddings.bin` (already in `app/src/main/assets/`)
2. Push to device (too large for APK assets):
   ```bash
   adb push clip_image_encoder.tflite /data/local/tmp/
   adb shell "run-as com.clip cp /data/local/tmp/clip_image_encoder.tflite /data/data/com.clip/files/"
   ```
3. Open this directory in Android Studio and run

## Architecture

| File | Description |
|------|-------------|
| `CLIPClassifier.kt` | CompiledModel GPU image encoder + cosine similarity with pre-computed text embeddings |
| `MainActivity.kt` | Image picker + classification results with confidence bars |
| `convert_clip.py` | Converts CLIP ViT-B/32 via litert-torch, pre-computes text embeddings |

## Model Details

- **Image Encoder Input**: `[1, 3, 224, 224]` float32 NCHW, CLIP normalization (mean=[122.77, 116.75, 104.09], std=[68.50, 66.63, 70.32])
- **Image Encoder Output**: `[1, 512]` float32 L2-normalized embedding
- **Text Embeddings**: `[96, 512]` float32, pre-computed with prompt template "a photo of a {label}"
- **Classification**: Cosine similarity → softmax (temperature=100) → top-K results

### Conversion Notes

- Converted via **litert-torch** (ViT requires litert-torch; onnx2tf breaks attention)
- SigmoidGELU patch applied (matches OpenAI CLIP's QuickGELU natively)
- `--export_text_encoder` flag available for ONNX text encoder (custom label support)
