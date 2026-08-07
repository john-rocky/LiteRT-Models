# Inflect-Nano-v2 — LiteRT, dynamic length + exact streaming

[Inflect-Nano-v2](https://huggingface.co/owensong/Inflect-Nano-v2) (4.0M params,
VITS-family end-to-end TTS, English, fixed male voice, 24 kHz, Apache-2.0)
converted to **LiteRT CPU/XNNPACK** graphs with a **dynamic sequence length**
and **exact intra-sentence streaming**. Built for small-CPU targets
(Raspberry Pi class).

The HF repo ships a PyTorch checkpoint + runtime (the "ONNX only" impression is
wrong — `onnx/` holds just a README). This port re-authors the VITS inference
graph in TF (weights loaded from `model.pth`) and converts with the official
`TFLiteConverter`, keeping both sequence axes dynamic.

## Graphs

| Graph | Inputs | Outputs | fp32 | fp16 |
| ----- | ------ | ------- | ---- | ---- |
| `inflect_text_encoder.tflite` | tokens [1,N] int32 | m_p [1,N,128], logs_p [1,N,128], logw [1,N,1] | 3.5 MB | 1.8 MB |
| `inflect_decoder.tflite` | z_p [1,T,128] | wav [1,256·T] @ 24 kHz | 12.6 MB | 6.4 MB |

Host glue: `durations = ceil(exp(logw)/speed)`; expand `m_p`/`logs_p` with
`np.repeat`; `z_p = m_p + randn·exp(logs_p)·variation` (noise generated host-side
for reproducibility); decoder → waveform. use_sdp=false in this checkpoint, so
the duration predictor is deterministic convs — no spline flows anywhere.

## Verification (vs. PyTorch reference, same inputs/noise)

| Check | Result |
| ----- | ------ |
| text encoder (m_p / logs_p / logw) | maxerr ≤ 2.3e-6 |
| decoder wav, golden sentence | **corr 1.000000**, maxerr 2.6e-5 |
| dynamic lengths | N = 49 / 165 / 217, T = 134 / 430 / 527 on the same graphs |
| streaming vs full decode | **corr 1.000000** (maxerr ≤ 1e-6) |
| fp16 decoder | corr 0.999879 |

## Speed (Mac M-series, 4 threads, XNNPACK)

| Sentence | N | T | audio | encoder | decoder | RTF |
| -------- | - | - | ----- | ------- | ------- | --- |
| golden | 165 | 430 | 4.59 s | 3 ms | 88 ms | **0.020** |
| short | 49 | 134 | 1.43 s | 1 ms | 28 ms | 0.020 |
| long | 217 | 527 | 5.62 s | 3 ms | 99 ms | 0.018 |

In a `python:3.12-slim` **linux/arm64** container (same aarch64
`ai-edge-litert` 2.1.6 wheel the Pi uses; Apple-Silicon-native, so a
functional-identity check, not a Pi performance proxy): waveform corr
**1.000000** vs the Mac output, streaming corr 1.000000, RTF 0.028,
time-to-first-audio ~60 ms.

## Measured on a real Raspberry Pi 5 (2026-08-06)

Pi 5 8 GB, Raspberry Pi OS 64-bit, Python 3.13.5, `ai-edge-litert` 2.1.6,
4 threads. `vcgencmd get_throttled` = 0x0 before/after each fp32 run.

| Sentence | N tokens | audio | encoder | decoder | sentence | RTF | TTFA |
| -------- | -------- | ----- | ------- | ------- | -------- | --- | ---- |
| [0] | 75 | 2.07 s | 2.7 ms | 220.4 ms | 223.1 ms | 0.108 | 218.8 ms |
| [1] | 121 | 3.21 s | 4.9 ms | 351.1 ms | 356.0 ms | 0.111 | 221.2 ms |
| [2] | 269 | 7.19 s | 22.2 ms | 788.3 ms | 810.5 ms | 0.113 | 239.0 ms |

Overall **RTF 0.111** (fp32), streaming exact (corr 1.000000) and full-waveform
ref-corr **1.000000** — Piper-class speed (their Piper baseline: RTF 0.10,
147 ms/phrase). ⚠ fp16 is speed-identical but showed a real quality break on
one bench sentence (ref-corr 0.388) — the flow layers are fp16-sensitive;
**deploy fp32**.

### GPU (v3dv WebGPU) status — measured on the Pi 5, 2026-08-06

With Mesa built from git (blogpost Step 7; v3dv Vulkan 1.3, driver 26.2.99,
`V3D_WEBGPU_OVERRIDE=1`): the static-chunk decoder
(`out/inflect_decoder_static228.tflite`) **compiles and runs fully accelerated**
(`is_fully_accelerated=True`) with output corr 0.9904 vs CPU (fp16-class
divergence; the flow layers are precision-sensitive — listen before adopting).
Dynamic graphs do not compile (static shapes required). The known gpu-backend
OOM did not reproduce. CPU remains the recommended deployment (faster on this
board per the blogpost's own guidance).

### One-command benchmark (Raspberry Pi)

```bash
python scripts/bench.py                 # fp32, 4 threads (= Pi 5's 4×A76)
python scripts/bench.py --precision fp16 --write-wavs
```

Needs only `numpy` + `ai-edge-litert` — inputs are pre-tokenized in
`scripts/bench_inputs.npz` (regenerate with `make_bench_inputs.py` on a machine
with espeak). Reports encoder/decoder latency, RTF, streaming
time-to-first-audio, and waveform-correlation identity checks (full decode vs
the bundled Mac reference, streamed vs full).

## Streaming (exact)

The decoder (flow + HiFi-GAN generator) is **fully convolutional with no
normalization layers**, so overlap-discard chunking is *exact*: 100-frame
chunks (+64 frames context each side) reproduce the full decode at
corr 1.000000. First-chunk latency 25–32 ms on this machine — time-to-first-
audio is encoder + one chunk (~35 ms here). This is the model to use when true
sub-sentence streaming matters (compare: KittenTTS's AdaIN statistics make its
chunked mode approximate).

## Files

| File | Purpose |
| ---- | ------- |
| `scripts/extract_weights.py` | Weight + golden-intermediate dump from the torch checkpoint (torch venv) |
| `scripts/build_inflect_tflite.py` | TF re-authoring (rel-pos attention, flow, generator) + conversion (TF venv) |
| `scripts/verify_inflect_litert.py` | End-to-end LiteRT check: accuracy, dynamic lengths, streaming, timing |
| `scripts/tf_dyn_gate.py` | Smoke test: dynamic-axis Conv1D/ConvT/attention/BiLSTM through TFLiteConverter |
| `scripts/bench.py` + `bench_inputs.npz` | One-command device benchmark (numpy + ai-edge-litert only) |
| `scripts/say.py` | Drop-in `say(text)` / `stream(text)` synthesis module + CLI (Piper replacement; mirrors upstream inference.py sentence handling) |

## Conversion notes

- **litert-torch dynamic export is a dead end** (0.9.2): beyond the known
  dynamic-LSTM wall, even a plain conv stack exported with `torch.export.Dim`
  bakes the trace length into internal RESHAPEs and fails at any other length;
  `F.embedding` doesn't lower with a symbolic axis at all. The TF/Keras →
  `TFLiteConverter` path handles all of it (shape-computed reshapes, fused
  dynamic LSTM) — that finding is what unlocked both this port and KittenTTS.
- VITS's relative-position attention (window 4) uses pad/reshape "skew" tricks;
  in TF they convert fine. (Under torch.export they generate unprovable
  divisibility/stride guards — band-mask reformulation fixes that, kept here
  for reference in git history.)
- `tf_keras` (Keras 2) is required: Keras 3 models leave READ_VARIABLE resource
  ops in the converted graph.

## Text frontend / licensing

English phonemization via the repo's `inflect_vits_frontend` = **espeak-ng
(GPL-3.0)** + num2words. Same note as KittenTTS: run espeak as a separate
process, or swap a DeepPhonemizer-based G2P for a GPL-free stack. Model:
Apache-2.0 (BigVGAN/VITS third-party notices in the upstream repo).
