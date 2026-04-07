# Kokoro Text-to-Speech

On-device text-to-speech using Kokoro-82M v1.0 (StyleTTS2-based) via ONNX Runtime CPU.

## Features

- **82M parameter neural TTS** — small yet competitive with much larger models
- **Bilingual** — English (American/British) and Japanese voices included
- **Free-form text input** — type any English or Japanese text, on-device phonemization
- **24 kHz mono output** via AudioTrack streaming
- **Single ONNX graph** — no model splitting, single inference call
- **Apache 2.0 weights** (model itself is Apache; this Android wrapper is BSD-style)
- **Pixel 8a CPU**: RTF 0.60 (verified, 3.90s of audio in 2357ms)

## Setup

1. Run the conversion script to download model weights, precompute demo phonemes, and build the CMU dictionary asset:
   ```bash
   cd scripts
   pip install huggingface_hub "misaki[en,ja]" numpy
   python convert_kokoro.py
   python build_cmudict.py
   ```
2. Copy the small assets into the app:
   ```bash
   cp scripts/output/demo_phrases.json app/src/main/assets/
   cp scripts/output/vocab.json        app/src/main/assets/
   ```
3. Stage the model and voices on the device (too large for the APK):
   ```bash
   adb push scripts/output/model_fp16.onnx /data/local/tmp/
   adb shell mkdir -p /data/local/tmp/kokoro_voices
   adb push scripts/output/voices/. /data/local/tmp/kokoro_voices/
   ```
4. Open this directory in Android Studio and **build & run** to install `com.kokoro`. The first launch will fail with a "model not found" error — that is expected.
5. Move the staged files into the app's private files dir:
   ```bash
   bash scripts/install_to_device.sh
   ```
6. Relaunch the Kokoro app.

## Architecture

| File | Description |
|------|-------------|
| `KokoroSynthesizer.kt` | Single ORT session: phoneme IDs + style vector + speed → 24 kHz waveform. Tries NNAPI EP first, falls back to CPU XNNPACK |
| `Phonemizer.kt` | Common interface for text → vocab IDs |
| `EnglishPhonemizer.kt` | CMU dict (134k entries, 845 KB gz) → ARPABET → Misaki IPA → vocab IDs |
| `JapanesePhonemizer.kt` | kuromoji-ipadic morphological analyzer → katakana pronunciation → IPA → vocab IDs |
| `MainActivity.kt` | Free-form text input, phrase picker, voice picker, AudioTrack playback |
| `convert_kokoro.py` | Downloads model + voices from HuggingFace, runs misaki to precompute phoneme token IDs for demo phrases |
| `build_cmudict.py` | Downloads CMU Pronouncing Dictionary, normalizes, gzips into assets |
| `install_to_device.sh` | Post-install adb script that moves staged model + voices into the app's filesDir |

## Model Details

- **Source**: [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)
- **Variant**: `model_fp16.onnx` (163 MB)
- **Inputs**:
  - `input_ids` `[1, seq_len]` int64 — phoneme token IDs with BOS=0 and EOS=0
  - `style` `[1, 256]` float32 — voice style vector at `voices[seq_len]`
  - `speed` `[1]` float32 — speed multiplier (1.0 = normal)
- **Output**: `[1, samples]` float32 mono PCM at 24 kHz

### Voices bundled by default

| ID | Language | Gender |
|---|---|---|
| `af_heart` | English (US) | Female |
| `am_michael` | English (US) | Male |
| `bf_emma` | English (UK) | Female |
| `jf_alpha` | Japanese | Female |
| `jm_kumo` | Japanese | Male |

Add more from [voices/](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/tree/main/voices) by editing `DEFAULT_VOICES` in `convert_kokoro.py`.

## Phonemizer

Kokoro takes IPA phoneme token IDs as input, not raw text. Two phonemization paths
are provided:

1. **Preset phrases** (default fallback): `assets/demo_phrases.json` ships token IDs
   precomputed offline by [misaki](https://github.com/hexgrad/misaki). Bit-identical to
   misaki output.

2. **Free-form text** (live, on-device): pure Kotlin/Java phonemizers, no NDK required.
   - **English**: CMU Pronouncing Dictionary (845 KB gzipped, 126k entries) lookup,
     ARPABET → Misaki IPA mapping with stress digits handled, OOV words skipped.
     Verified bit-identical to misaki for in-vocab inputs (e.g. "Hello world.").
   - **Japanese**: [kuromoji-ipadic](https://www.atilika.com/en/kuromoji/) gradle dep
     (~10 MB JAR with built-in dictionary), uses each morpheme's `pronunciation` field
     (closer to actual sound than literal `reading`, e.g. は particle → ワ), then
     katakana → Misaki IPA via lookup table with yōon and long vowel handling.

Both phonemizers fall back gracefully on unknown words (skip + log warning).
The MainActivity auto-detects language from the selected voice prefix
(`af_/am_/bf_/bm_` → English, `jf_/jm_` → Japanese).

## Notes

- Inference runs on CPU via ONNX Runtime XNNPACK. NNAPI EP can be enabled in
  `KokoroSynthesizer.kt` for benchmarking — Pixel 8a may route to GPU/NPU.
- StyleTTS2 has a transformer text encoder + duration predictor + HiFi-GAN style
  vocoder. The vocoder is CNN-heavy and would benefit from a TFLite GPU split if
  CPU inference proves insufficient. See task tracker for the conversion plan.
- Voice `.bin` files contain a 3D array `[N_lengths, 1, 256]` where the first axis
  is indexed by token length — the model expects a length-conditioned style.

**Original project**: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Apache 2.0
