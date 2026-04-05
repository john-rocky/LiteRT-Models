# SmolVLM Vision-Language Model

On-device image understanding using SmolVLM-256M with LiteRT CompiledModel GPU vision encoder + ONNX Runtime CPU language model.

## Features

- **Ask questions about images** — "Describe this image", "What is this?", custom prompts
- First on-device VLM with LiteRT GPU-accelerated vision encoder
- Streaming text generation (token-by-token UI update)
- SigLIP vision encoder (86M) + SmolLM2 language model (163M)
- 64 visual tokens per image (efficient pixel shuffle compression)

## Setup

1. Download models from [Releases](https://github.com/john-rocky/LiteRT-Models/releases/tag/v2):
   - `smolvlm_vision.tflite` (357 MB)
   - `smolvlm_decoder.onnx` (515 MB)
   - `embed_tokens.bin` (108 MB)
   - `vocab.json` + `config.json` (already in `app/src/main/assets/`)
2. Push to device:
   ```bash
   adb push smolvlm_vision.tflite /data/local/tmp/
   adb push smolvlm_decoder.onnx /data/local/tmp/
   adb push embed_tokens.bin /data/local/tmp/
   adb shell "run-as com.smolvlm cp /data/local/tmp/smolvlm_vision.tflite /data/data/com.smolvlm/files/"
   adb shell "run-as com.smolvlm cp /data/local/tmp/smolvlm_decoder.onnx /data/data/com.smolvlm/files/"
   adb shell "run-as com.smolvlm cp /data/local/tmp/embed_tokens.bin /data/data/com.smolvlm/files/"
   ```
3. Open this directory in Android Studio and run

## Architecture

| File | Description |
|------|-------------|
| `VLMInference.kt` | Vision encoder (TFLite GPU) + LM autoregressive decoder (ONNX CPU) with repetition penalty |
| `MainActivity.kt` | Image picker + prompt input + streaming response display |
| `convert_smolvlm.py` | Converts SigLIP+connector via litert-torch, SmolLM2 via ONNX, exports vocab/embeddings |

## Model Details

- **Vision Encoder Input**: `[1, 3, 512, 512]` float32 NCHW, normalized to [-1, 1]
- **Vision Encoder Output**: `[1, 64, 576]` float32 visual tokens
- **LM Decoder Input**: `[1, seq_len, 576]` float32 embeddings (visual + text merged)
- **LM Decoder Output**: `[1, seq_len, 49280]` float32 logits
- **Generation**: Greedy decoding with repetition penalty 1.2x, max 128 tokens

### Conversion Notes

| Issue | Fix |
|-------|-----|
| `torch.bucketize` (SigLIP position embeddings) | Pre-computed position IDs for fixed 512x512 |
| `padding='valid'` Conv2d | Replaced with `padding=0` |
| transformers v5.5 `create_causal_mask` | Monkey-patched with simple triu mask |
| GELU | SigmoidGELU approximation |
| GPT-2 byte encoding in vocab (`Ġ`=space) | Decoded on device with char mapping |

**Original project**: [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | [Apache-2.0](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct/blob/main/LICENSE)
