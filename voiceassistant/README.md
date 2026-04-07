# Voice Assistant

Full on-device conversational pipeline: **Whisper STT → SmolLM2 chat → Kokoro TTS**, all running on a Pixel-class device with no network.

## Pipeline

```
mic ─┐
     ▼
  AudioRecord (16kHz mono PCM)
     │
     ▼
  MelSpectrogram (Kotlin FFT)
     │
     ▼
  Whisper-tiny encoder (TFLite GPU)
     │
     ▼
  Whisper-tiny decoder (ONNX CPU, autoregressive)
     │  text
     ▼
  SmolLM2-135M-Instruct (ONNX CPU, autoregressive)
     │  response text
     ▼
  English phonemizer (CMU dict + ARPABET → IPA)
     │  Kokoro vocab token IDs
     ▼
  Kokoro-82M (ONNX CPU, single graph)
     │  24 kHz mono float PCM
     ▼
  AudioTrack (PCM_FLOAT)
     │
     ▼
  speaker
```

## Setup

1. Build the dependent module outputs first (each only needed once):
   ```bash
   cd whisper/scripts   && python convert_whisper.py
   cd ../../smolvlm/scripts && python convert_smolvlm.py
   cd ../../kokoro/scripts  && python convert_kokoro.py && python build_cmudict.py
   ```
   This populates `whisper/scripts/output/`, `smolvlm/scripts/output/`, and `kokoro/scripts/output/` with all model files.

2. Copy the small Whisper mel-filter binary into this module's assets (it is gitignored as `*.bin` per repo convention):
   ```bash
   cp whisper/scripts/output/mel_filters.bin voiceassistant/app/src/main/assets/
   ```

3. Stage the model files on the device (~1 GB total):
   ```bash
   adb push whisper/scripts/output/whisper_encoder.tflite /data/local/tmp/
   adb push whisper/scripts/output/whisper_decoder.onnx   /data/local/tmp/
   adb push smolvlm/scripts/output/smolvlm_decoder.onnx   /data/local/tmp/
   adb push smolvlm/scripts/output/embed_tokens.bin       /data/local/tmp/
   adb push kokoro/scripts/output/model_fp16.onnx          /data/local/tmp/
   adb shell mkdir -p /data/local/tmp/kokoro_voices
   adb push kokoro/scripts/output/voices/. /data/local/tmp/kokoro_voices/
   ```

4. Open this directory in Android Studio and **build & run** to install `com.voiceassistant`. The first launch will fail with a "model not found" error — that is expected.

5. Move the staged files into the app's private files dir:
   ```bash
   bash scripts/install_to_device.sh
   ```

6. Relaunch the VoiceAssistant app, grant microphone permission, tap **Hold to talk**, speak, tap again to stop. The transcript appears, the response begins streaming token-by-token, and the assistant starts speaking the first sentence within ~1.5 s of recording stop.

## Architecture

| File | Description |
|------|-------------|
| `MainActivity.kt` | Mic recording, voice picker, transcript / response display, AudioTrack playback |
| `VoiceAssistant.kt` | Pipeline orchestrator: Whisper → LM → phonemizer → Kokoro |
| `WhisperTranscriber.kt` | Whisper-tiny encoder (TFLite GPU) + decoder (ONNX CPU) |
| `MelSpectrogram.kt` | Pure Kotlin FFT + mel filterbank |
| `LanguageModel.kt` | SmolLM2 decoder (text-only, no visual tokens) with greedy generation + repetition penalty |
| `KokoroSynthesizer.kt` | Kokoro StyleTTS2 single-graph ONNX, NNAPI EP with CPU fallback |
| `EnglishPhonemizer.kt` | CMU dict (126k entries) + ARPABET → Misaki IPA → Kokoro vocab |
| `JapanesePhonemizer.kt` | kuromoji + katakana → Misaki IPA (currently unused — pipeline is English only) |

## Asset filenames

To avoid name collisions when bundling assets from three modules in one app:
- `whisper_config.json`, `whisper_vocab.json`, `mel_filters.bin` (Whisper)
- `llm_config.json`, `llm_vocab.json` (SmolLM2)
- `kokoro_vocab.json`, `cmudict.txt` (Kokoro phonemizer)

## Notes

- **English-only chat for MVP.** Whisper transcribes in English, the LM responds in English, the English phonemizer produces phonemes. Adding Japanese would require detecting input language from Whisper, swapping the LM prompt template (or chat instructions), and routing to `JapanesePhonemizer` and a Japanese voice.
- **Response quality**: SmolLM2-135M-Instruct is a tiny LM. Expect short, sometimes inaccurate responses. The pipeline architecture is what matters for the demo — swap to a larger LM (Qwen 0.5B, Phi-3 mini) for better answers.
- **Latency budget on Pixel 8a (estimated)**: Whisper ~3s, LM ~3-8s for short response, Kokoro ~1.5s for 3s of audio. Total ~10s end-to-end. Streaming the LM tokens into the TTS chunk-by-chunk would dramatically reduce perceived latency but is not implemented in MVP.
- **Models stored in app private dir**: ~1 GB total. Pixel 8a has plenty of storage but be aware of the disk footprint.

## Why bundled instead of cross-module references

Each module in this repo is a standalone Gradle project. There is no shared library or multi-module setup. The voice assistant is a fresh Gradle project that bundles its own copy of each model wrapper class. This is intentional — keeps each module self-contained and independently runnable.
