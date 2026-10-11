# EMA Lightning — Turkish TTS on LiteRT (CPU + GPU)

Type Turkish, tap **Speak**, hear it: an 8.6M-parameter text-to-speech model (a 4-step flow-matching DiT and a
HiFi-GAN decoder, one voice, 48 kHz) running on the phone through LiteRT `CompiledModel`. The acoustic graphs run on
the CPU, the decoder on the GPU; the integer bookkeeping between them (word timeline, aligner window, decoder masks,
the four Euler steps) runs in Kotlin. No network after the one-time model download.

| | |
|---|---|
| Model | [litert-community/EMA-Lightning-LiteRT](https://huggingface.co/litert-community/EMA-Lightning-LiteRT) (from [canberkkkkkk/ema-lightning](https://huggingface.co/canberkkkkkk/ema-lightning), Apache-2.0) |
| Graphs | `ema_text_multi_fp32` · `ema_align_multi_fp32` · `ema_step_multi_fp32` · `ema_decoder_multi_fp16` — 27.3 MB, every length bucket a signature |
| App | `com.emalightning` — text in, 48 kHz audio out, `files/last.wav` saved |
| Measured | Galaxy S26, LiteRT 2.2.0, 2026-10-11: one 3.5 s sentence in 56.5 ms (CPU acoustic graphs + GPU fp16 decoder), first audio 25.5 ms — see the model card |

## Run it with your own text

1. Build and install: `./gradlew installDebug` (Android Studio works too; `local.properties` needs `sdk.dir`).
2. Open the app and tap **Download the model (27 MB)** once (or push the four files with
   `scripts/install_to_device.sh <dir>` after downloading them from the Hub).
3. Type a Turkish sentence and tap **Speak**. The status line shows the speech length, the synthesis time, the time to
   the first audio window and the RTF; the pill turns **● PLAYING** when sound leaves the speaker.

Write numbers, dates and abbreviations out in words ("on dört otuz", not "14:30"): the Rust text normalizer the
Python package uses (normalizer-tr) is not in this app. Letters-only sentences are read exactly as the package reads
them. Turkish only, one voice.

## Pipeline

```
letters --text (CPU)--> h [L,224], frames per letter
        --Kotlin: frames per word, frame timeline, letter / frame positions, the aligner window-->
        --align (CPU)--> one condition per frame
        --Kotlin Euler x4: step (CPU)(x, cond, step table) -> v;  x1 = x + (1-t) v;  x = (1-t') noise' + t' x1-->
        latents [T,64] --decoder (GPU fp16), windows of 25 then 100 frames with 8 of context--> 48 kHz audio
```

Buckets: letters 32 / 64 / 128 / 256, frames 48 / 96 / 192 / 384 / 512, decoder windows 48 / 120 frames; the host
picks the signature by the piece's length (a piece is at most 250 letters; longer text is cut at sentence, clause,
then word boundaries, as the package does).

## Recording harness

The same code path a finger drives, from adb:

```
adb shell am start -n com.emalightning/.MainActivity --es text "Merhaba, bu ses telefonunuzun üzerinde üretiliyor." --ez autoplay true
adb shell run-as com.emalightning cat files/run.json      # the sentence, letters / frames / buckets, ms per stage, PLAYING time
adb shell run-as com.emalightning cat files/last.wav > last.wav
```

`logcat` carries `EMA_READY`, `EMA_SYNTH`, `EMA_PLAYING`, `EMA_DONE`. The activity shows over the keyguard and keeps
the screen on (debug / recording convenience).

## Files

- `app/src/main/java/com/emalightning/EmaSynthesizer.kt` — the four graphs and the host loop
- `app/src/main/java/com/emalightning/EmaFrontend.kt` — the alphabet step, the chunker, letter / word indices
- `app/src/main/java/com/emalightning/MainActivity.kt` — the screen, playback, run JSON, the download
- `app/src/main/assets/config.json`, `ema_step_table.bin` — vocabulary, buckets, stage lengths; the four steps' modulation vectors
- `scripts/install_to_device.sh` — push the graphs by adb instead of downloading in the app

Conversion recipe and measurements: [hf-to-litertlm/ema_tts_work](https://github.com/john-rocky/hf-to-litertlm/tree/main/ema_tts_work).
