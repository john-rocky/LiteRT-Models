# Voice Assistant

Full on-device conversational pipeline: **Silero VAD → Whisper STT → SmolLM2 chat → Kokoro TTS**, all running on a Pixel-class device with no network. Hands-free with VAD-driven turn taking and barge-in.

## Pipeline

```
mic ─┐
     ▼
  AudioRecord (VOICE_COMMUNICATION, 16 kHz mono PCM)
     │
     ▼  AEC / NS / AGC (platform effects)
     │
     ▼  512-sample chunks (32 ms each)
     │
  Silero VAD v5 (ONNX CPU, stateful LSTM, ~1 ms/chunk)
     │  speech probability per chunk
     ▼
  SegmentTracker (hysteresis: start 0.5 / end 0.35, min_speech 250 ms,
     │             min_silence 600 ms) → speech segment events
     │
     ▼  on SPEECH_START: start capture (with 5-chunk preroll)
     │  on SPEECH_END:   submit captured PCM to the turn pipeline
     │
  Whisper-tiny encoder (TFLite GPU) → decoder (ONNX CPU)
     │  text
     ▼
  SmolLM2-135M-Instruct (ONNX CPU, streaming, cancellable)
     │  response text, token-by-token
     ▼
  English phonemizer (CMU dict + ARPABET → IPA)
     │  Kokoro vocab token IDs, sentence-by-sentence
     ▼
  Kokoro-82M (ONNX CPU, single graph)
     │  24 kHz mono float PCM
     ▼
  AudioTrack (PCM_FLOAT, MODE_STATIC, hard-stoppable on barge-in)
     │
     ▼
  speaker
```

While the assistant is replying, the mic loop and VAD keep running. A new
SPEECH_START event during THINKING/SPEAKING is treated as **barge-in**: the
in-flight LM generation, TTS synthesis, and AudioTrack playback are all
cancelled, and the new utterance is captured as the next turn.

## Setup

1. Build the dependent module outputs first (each only needed once):
   ```bash
   cd whisper/scripts   && python convert_whisper.py
   cd ../../smolvlm/scripts && python convert_smolvlm.py
   cd ../../kokoro/scripts  && python convert_kokoro.py && python build_cmudict.py
   ```
   This populates `whisper/scripts/output/`, `smolvlm/scripts/output/`, and `kokoro/scripts/output/` with all model files.

2. Download Silero VAD (~2.3 MB ONNX, no conversion needed):
   ```bash
   python voiceassistant/scripts/download_silero_vad.py
   ```
   Output: `voiceassistant/scripts/output/silero_vad.onnx`. The model is also
   published as a [GitHub Release asset](https://github.com/john-rocky/LiteRT-Models/releases)
   so end users do not need a Python toolchain — `silero_vad.onnx` can be
   downloaded directly from the release page.

3. Copy the small Whisper mel-filter binary into this module's assets (it is gitignored as `*.bin` per repo convention):
   ```bash
   cp whisper/scripts/output/mel_filters.bin voiceassistant/app/src/main/assets/
   ```

4. Stage the model files on the device (~1 GB total):
   ```bash
   adb push whisper/scripts/output/whisper_encoder.tflite      /data/local/tmp/
   adb push whisper/scripts/output/whisper_decoder.onnx        /data/local/tmp/
   adb push smolvlm/scripts/output/smolvlm_decoder.onnx        /data/local/tmp/
   adb push smolvlm/scripts/output/embed_tokens.bin            /data/local/tmp/
   adb push kokoro/scripts/output/model_fp16.onnx               /data/local/tmp/
   adb push voiceassistant/scripts/output/silero_vad.onnx       /data/local/tmp/
   adb shell mkdir -p /data/local/tmp/kokoro_voices
   adb push kokoro/scripts/output/voices/. /data/local/tmp/kokoro_voices/
   ```

5. Open this directory in Android Studio and **build & run** to install `com.voiceassistant`. The first launch will fail with a "model not found" error — that is expected.

6. Move the staged files into the app's private files dir:
   ```bash
   bash scripts/install_to_device.sh
   ```

7. Relaunch the VoiceAssistant app, grant microphone permission, tap **Listen** once, and speak. The mic stays open until you tap **Stop listening**:
   - You speak a sentence → 600 ms of trailing silence → the captured audio is sent through STT → LM → TTS automatically.
   - The assistant starts replying. While it is replying, the mic and VAD keep running.
   - **Barge-in**: start speaking again mid-reply. The current TTS playback is hard-stopped and your new utterance becomes the next turn.

## Architecture

| File | Description |
|------|-------------|
| `MainActivity.kt` | Mic loop with VAD-driven turn taking, Listen toggle UI, AudioTrack playback with hard-stoppable barge-in, AEC/NS/AGC effects |
| `VoiceAssistant.kt` | Pipeline orchestrator: Whisper → LM → phonemizer → Kokoro, with cancellation predicate for barge-in |
| `VadDetector.kt` | Silero VAD v5 ONNX wrapper, holds the `[2, 1, 128]` LSTM state across 32 ms chunks |
| `SegmentTracker.kt` | Hysteresis state machine over per-chunk VAD probabilities, emits SPEECH_START / SPEECH_END events |
| `WhisperTranscriber.kt` | Whisper-tiny encoder (TFLite GPU) + decoder (ONNX CPU) |
| `MelSpectrogram.kt` | Pure Kotlin FFT + mel filterbank |
| `LanguageModel.kt` | SmolLM2 decoder (text-only) with greedy generation + repetition penalty + cancellation between tokens |
| `KokoroSynthesizer.kt` | Kokoro StyleTTS2 single-graph ONNX, NNAPI EP with CPU fallback |
| `EnglishPhonemizer.kt` | CMU dict (126k entries) + ARPABET → Misaki IPA → Kokoro vocab |
| `JapanesePhonemizer.kt` | kuromoji + katakana → Misaki IPA (currently unused — pipeline is English only) |

## Asset filenames

To avoid name collisions when bundling assets from three modules in one app:
- `whisper_config.json`, `whisper_vocab.json`, `mel_filters.bin` (Whisper)
- `llm_config.json`, `llm_vocab.json` (SmolLM2)
- `kokoro_vocab.json`, `cmudict.txt` (Kokoro phonemizer)

## Models in `filesDir`

Pushed by `scripts/install_to_device.sh`:

| File | Size | Purpose |
|------|------|---------|
| `whisper_encoder.tflite` | 32 MB  | STT encoder (LiteRT GPU) |
| `whisper_decoder.onnx`   | 199 MB | STT decoder (ORT CPU)    |
| `smolvlm_decoder.onnx`   | 540 MB | LM decoder (ORT CPU)     |
| `embed_tokens.bin`       | 114 MB | LM token embeddings      |
| `model_fp16.onnx`        | 163 MB | Kokoro TTS (ORT NNAPI)   |
| `silero_vad.onnx`        | 2.3 MB | Silero VAD v5 (ORT CPU)  |
| `voices/*.bin`           | 510 KB each | Kokoro style vectors |

## Notes

- **VAD model is small but pushed via adb anyway.** Even though `silero_vad.onnx` is only ~2.3 MB and would fit comfortably in `assets/`, it lives in `filesDir` next to the other models. Consistency: every model used by this module is loaded the same way, and it ships from the same GitHub Release page so end users have one download flow regardless of file size.
- **Barge-in needs decent echo cancellation** to avoid the assistant interrupting itself with its own TTS bleed-through. We use `MediaRecorder.AudioSource.VOICE_COMMUNICATION` plus `AcousticEchoCanceler` / `NoiseSuppressor` / `AutomaticGainControl` effects on the mic session. Quality is device-dependent — Pixel-class devices work well; lower-end devices may need a longer warmup or higher VAD threshold during playback.
- **English-only chat for MVP.** Whisper transcribes in English, the LM responds in English, the English phonemizer produces phonemes. Adding Japanese would require detecting input language from Whisper, swapping the LM prompt template (or chat instructions), and routing to `JapanesePhonemizer` and a Japanese voice.
- **Response quality**: SmolLM2-135M-Instruct is a tiny LM. Expect short, sometimes inaccurate responses. The pipeline architecture is what matters for the demo — swap to a larger LM (Qwen 0.5B, Phi-3 mini) for better answers.
- **Latency budget on Pixel 8a**: STT ~700 ms, LM ~1000-1500 ms, TTS ~1100 ms (sentence-chunked). End-to-end ~5 s for a typical Q&A turn. Time-to-first-audio is ~1.5 s thanks to sentence-chunked TTS streaming.
- **Models stored in app private dir**: ~1 GB total. Pixel 8a has plenty of storage but be aware of the disk footprint.

## Why bundled instead of cross-module references

Each module in this repo is a standalone Gradle project. There is no shared library or multi-module setup. The voice assistant is a fresh Gradle project that bundles its own copy of each model wrapper class. This is intentional — keeps each module self-contained and independently runnable.
