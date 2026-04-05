# Whisper Speech Recognition

On-device speech-to-text using OpenAI Whisper-tiny with LiteRT CompiledModel GPU encoder + ONNX Runtime CPU decoder.

## Features

- **First LiteRT GPU Whisper implementation** — encoder runs on GPU via CompiledModel
- Microphone recording (up to 30s) with real-time duration display
- Audio file picker (any Android-supported format)
- 10 language support (en, ja, zh, ko, fr, de, es, it, pt, ru)
- Mel spectrogram computed in pure Kotlin (FFT + mel filterbank)

## Setup

1. Download models from [Releases](https://github.com/john-rocky/LiteRT-Models/releases/tag/v2):
   - `whisper_encoder.tflite` (33 MB)
   - `whisper_decoder.onnx` (199 MB)
   - `vocab.json` + `mel_filters.bin` + `config.json` (already in `app/src/main/assets/`)
2. Push to device (too large for APK assets):
   ```bash
   adb push whisper_encoder.tflite /data/local/tmp/
   adb shell "run-as com.whisper cp /data/local/tmp/whisper_encoder.tflite /data/data/com.whisper/files/"
   adb push whisper_decoder.onnx /data/local/tmp/
   adb shell "run-as com.whisper cp /data/local/tmp/whisper_decoder.onnx /data/data/com.whisper/files/"
   ```
3. Open this directory in Android Studio and run

## Architecture

| File | Description |
|------|-------------|
| `MelSpectrogram.kt` | Cooley-Tukey FFT + mel filterbank in pure Kotlin. 16kHz audio → [80, 3000] |
| `WhisperTranscriber.kt` | Encoder (TFLite GPU) + decoder autoregressive loop (ONNX CPU, greedy decoding) |
| `MainActivity.kt` | Microphone recording via AudioRecord + audio file picker via MediaCodec + language selector |
| `convert_whisper.py` | Converts Whisper via litert-torch (encoder) + torch.onnx (decoder), exports vocab and mel filters |

## Model Details

- **Encoder Input**: `[1, 80, 3000]` float32 log-mel spectrogram (80 mel bins, 30s audio)
- **Encoder Output**: `[1, 1500, 384]` float32 audio features
- **Decoder Input**: `[1, seq_len]` int64 token IDs + `[1, 1500, 384]` audio features
- **Decoder Output**: `[1, seq_len, 51865]` float32 logits

### Conversion Notes

- Encoder converted via **litert-torch** with SigmoidGELU patch (same as CLIP/MobileSAM)
- Decoder exported to ONNX with `MultiHeadAttention.use_sdpa = False` (manual attention for ONNX compatibility)
- No KV-cache — full sequence recomputed each step (acceptable for tiny's 4 decoder layers)
- Mel spectrogram preprocessing fully in Kotlin (no native/C++ dependency)

**Original project**: [openai/whisper](https://github.com/openai/whisper) | [MIT](https://github.com/openai/whisper/blob/main/LICENSE)
