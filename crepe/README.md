# CREPE — On-Device Pitch Detection / Real-Time Tuner

[CREPE](https://github.com/marl/crepe) monophonic pitch (f0) estimation on Android, running **fully on the LiteRT `CompiledModel` GPU (ML Drift)**. Feed a 1024-sample (16 kHz) window and it returns activations over **360 pitch bins** (20 cents each, ~C1–B7); the host decodes them to a frequency and the nearest musical note. The sample is a **real-time tuner**: it listens to the mic and shows the detected note + how many cents flat/sharp you are.

```
frame[1,1024] (16 kHz, per-frame zero-mean/unit-var) --[GPU CNN]--> activations[1,360] --[host]--> Hz → note
```

## On-device (Pixel 8a, Tensor G3 — verified)

CNN **49/49** nodes on the LiteRT GPU delegate (`LITERT_CL`), **1 partition** (single graph, no CPU fallback); ~75 ms/frame (full model); self-test (synthesized 440 Hz sine) → **A4, 440.4 Hz**. fp16 tflite-vs-PyTorch corr **1.000000**.

## Model

| Stage | File | Size | Input → Output | Placement |
| ----- | ---- | ---- | -------------- | --------- |
| frame norm | (Kotlin) | — | mic → frame[1,1024], zero-mean/unit-var | CPU |
| CREPE (full) | `crepe_full_fp16.tflite` | 44.5 MB FP16 | frame[1,1024] → activations[1,360] | CompiledModel GPU |
| decode | (Kotlin) | — | 360 bins → Hz → note + cents | CPU |

The whole network is a **pure CNN** — 6× {zero-pad → Conv2d → ReLU → BatchNorm → MaxPool} + permute/reshape (≤4D) + Linear + sigmoid. No banned ops, and per-frame normalization keeps activations ~O(1) so there is no fp16-on-Mali precision issue (banned NONE, >4D 0, corr 1.0). The 44.5 MB model is **bundled in assets** (no device push needed).

## Decode

CREPE bin → cents: `cents = 20·bin + 1997.3794…`; `Hz = 10·2^(cents/1200)`. The pitch is the activation-weighted average over ±4 bins around the peak (torchcrepe `weighted_argmax`); the peak activation is the confidence (the tuner shows "—" below 0.5). Nearest note from `midi = 69 + 12·log2(Hz/440)`.

## Build & run

```bash
cd scripts/
~/clipconv/bin/python build_crepe.py      # -> crepe_full_fp16.tflite (+ self-test 220/440/880 Hz)
cp crepe_full_fp16.tflite ../app/src/main/assets/
cd .. && ./gradlew :app:installDebug
```

On launch the app self-tests on a synthesized 440 Hz tone (→ A4), then **Start tuner** opens the mic and shows the live note + cents gauge. Uses `AudioSource.UNPROCESSED` (no AGC/NS) for accurate pitch; the capture loop drains backlog each cycle so the reading stays current.

## Original project & license

[marl/crepe](https://github.com/marl/crepe) (Kim et al., ICASSP 2018) — [MIT](https://github.com/marl/crepe/blob/master/LICENSE). PyTorch weights via [torchcrepe](https://github.com/maxrmorrison/torchcrepe) (MIT).
