# KittenTTS nano 0.8 — LiteRT, dynamic length + streaming

[KittenTTS](https://github.com/KittenML/KittenTTS) nano (15M params, StyleTTS2 +
ISTFTNet + mini-ALBERT, 8 voices, 24 kHz) converted to **LiteRT CPU/XNNPACK**
graphs with a **dynamic sequence length** — any sentence length runs on the same
graphs, no padding buckets. Built for small-CPU targets (Raspberry Pi class).

Upstream ships **ONNX only** (Apache-2.0). This port re-authors the model in
TF/Keras from the ONNX weights and converts with the official
`TFLiteConverter`, which emits **fused dynamic-length TFLite LSTM kernels** for
the five BiLSTMs — the piece that torch-path conversions cannot keep dynamic
(see "Why TF re-authoring" below).

## Graphs

| Graph | Inputs | Outputs | fp32 | fp16 |
| ----- | ------ | ------- | ---- | ---- |
| `kitten_predictor.tflite` | input_ids [1,N] int32, style [1,256], speed [1] | d [1,N,256], t_en [1,N,128], durations [N] int32 | 33.8 MB | 17.0 MB |
| `kitten_prosody.tflite` | en [1,T,256], style [1,256] | f0 [1,2T], n [1,2T], har [1,120T+1,22] | 3.3 MB | 1.7 MB |
| `kitten_vocoder.tflite` | asr [1,T,128], f0, n, har, style | wav [1,600T] @ 24 kHz | 26.4 MB | 13.4 MB |

Host glue between graphs is ~10 lines of numpy: `en = repeat(d, durations)`,
`asr = repeat(t_en, durations)` (equivalent to the in-graph `Loop` alignment of
the ONNX — verified bit-exact), then slice `har`/`f0`/`n` per vocoder call.
`style` comes from `voices.npz` exactly as in the pip package
(`voices[voice][min(len(text), 399)]`).

## Verification (vs. the official ONNX, same inputs)

The reference model is **stochastic** (SineGen draws a random initial harmonic
phase and additive noise every run), so the fair bar is the ONNX's own
run-to-run variability. All numbers on the golden sentence, Mac M-series CPU:

| Comparison | log-mel corr | spec-conv |
| ---------- | ------------ | --------- |
| ONNX vs ONNX (two runs, same inputs) | 0.98327 | 0.0949 |
| **LiteRT fp32 vs ONNX (deterministic)** | **0.98388** | 0.1242 |
| LiteRT fp16 vs ONNX (deterministic) | 0.98231 | — |

i.e. the port sits **inside the model's intrinsic noise floor**. Predicted
durations are bit-identical to the ONNX output. The decoder/vocoder chain is
float-exact in isolation (corr 1.000000 against the deterministic reference
when fed the reference harmonics). int8 dynamic-range quantization was tried
and rejected (log-mel corr 0.913, durations change).

Raw-waveform correlation is *not* a meaningful metric here: sub-0.5 % f0
differences de-correlate the waveform via accumulated sine phase while being
inaudible and spectrally identical.

## Speed (Mac M-series, 4 threads, XNNPACK)

| Sentence | N tokens | frames | audio | predictor | prosody | vocoder | RTF |
| -------- | -------- | ------ | ----- | --------- | ------- | ------- | --- |
| golden | 85 | 182 | 4.55 s | 16 ms | 8 ms | 52 ms | **0.017** |
| short | 27 | 80 | 2.00 s | 7 ms | 4 ms | 25 ms | 0.018 |
| long | 112 | 195 | 4.88 s | 19 ms | 8 ms | 53 ms | 0.016 |

In a `python:3.12-slim` **linux/arm64** container (same
aarch64 `ai-edge-litert` 2.1.6 wheel the Pi uses; Apple-Silicon-native, so a
functional-identity check, not a Pi performance proxy): durations still
bit-identical, log-spec corr ≥ 0.9955, overall RTF 0.035.

## Measured on a real Raspberry Pi 5 (2026-08-06)

Pi 5 8 GB, Raspberry Pi OS 64-bit, Python 3.13.5, `ai-edge-litert` 2.1.6,
4 threads. `vcgencmd get_throttled` = 0x0 before/after each run (no
undervoltage/throttling during measurement).

| Sentence | N tokens | audio | predictor | prosody | vocoder | sentence | RTF |
| -------- | -------- | ----- | --------- | ------- | ------- | -------- | --- |
| [0] | 41 | 2.77 s | 59.9 ms | 20.2 ms | 403.8 ms | 483.8 ms | 0.174 |
| [1] | 63 | 4.08 s | 87.0 ms | 30.1 ms | 610.5 ms | 727.6 ms | 0.179 |
| [2] | 138 | 7.03 s | 188.8 ms | 53.2 ms | 1141.7 ms | 1383.7 ms | 0.197 |

Overall **RTF 0.187** (fp32; durations bit-OK, log-spec corr ≥ 0.9954). fp16 is
speed-identical on the Pi (XNNPACK unpacks fp16 weights to fp32 compute) with
slightly lower corr — deploy fp32. For reference, the same model on ONNX RT was
measured at RTF 0.30 on this device class, so the LiteRT path is ~1.6× faster.

### GPU (v3dv WebGPU) status — measured on the Pi 5, 2026-08-06

With Mesa built from git (blogpost Step 7; v3dv Vulkan 1.3, driver 26.2.99,
`V3D_WEBGPU_OVERRIDE=1`): the static-chunk vocoder
(`out/kitten_vocoder_static80.tflite`) **compiles and runs fully accelerated**
(`is_fully_accelerated=True`) with output corr 0.9997 vs CPU (fp16-class
divergence). The dynamic vocoder does not compile (the GPU delegate requires
static shapes — same as the container pre-check), and the fused-LSTM
predictor/prosody graphs are CPU-only. The known gpu-backend OOM did not
reproduce on these graphs. Per the blogpost's own guidance the CPU is faster on
this board; the all-CPU config above is the recommended deployment.

### One-command benchmark (Raspberry Pi)

```bash
python scripts/bench.py                 # fp32, 4 threads (= Pi 5's 4×A76)
python scripts/bench.py --precision fp16 --write-wavs
```

Needs only `numpy` + `ai-edge-litert` — inputs are pre-tokenized in
`scripts/bench_inputs.npz` (regenerate with `make_bench_inputs.py` on a machine
with espeak), so the device needs no phonemizer. Reports per-graph latency,
per-sentence synthesis latency, RTF, and an output-identity check (durations
bit-compare + log-spectrogram correlation vs the bundled Mac fp32 reference).

**Deployment note — LSTM state:** the fused TFLite LSTM kernels keep their
hidden state in *variable tensors that persist across `invoke()`*. Call
`interpreter.reset_all_variables()` (after `allocate_tensors()`) before every
utterance, or the second synthesis on a reused interpreter is corrupted.

## Streaming

- **Sentence-level (exact, recommended)** — synthesize per sentence and play
  while the next sentence synthesizes; this is the same granularity the
  official pip package uses (`chunk_text`). First-audio latency = one short
  sentence (≈36 ms here, ≈0.4–0.7 s RPi-5-scale).
- **Intra-sentence chunked vocoder (approximate)** — the vocoder alone can be
  run on overlapping frame chunks (`verify_kitten_litert.py stream_vocoder`,
  ~1 s chunks, 20-frame overlap, first chunk 19 ms). It is *not* exact because
  StyleTTS2's AdaIN InstanceNorms take statistics over the whole utterance:
  chunked output measures log-mel corr 0.970 against the full decode. Use
  sentence-level unless latency demands force this mode.

## Files

| File | Purpose |
| ---- | ------- |
| `scripts/make_goldens.py` | Reference inputs/intermediates/wav from the official ONNX (onnxruntime), plus the full weight dump |
| `scripts/kitten_tf.py` | The TF re-authoring: ALBERT, text encoder, duration encoder, prosody (AdainResBlk1d), SineGen+STFT harmonics, ISTFTNet decoder |
| `scripts/build_kitten_tflite.py` | Builds + converts the three graphs (fp32) |
| `scripts/check_stages.py`, `scripts/check_decoder.py` | Per-module numeric verification against the goldens |
| `scripts/verify_kitten_litert.py` | End-to-end LiteRT check: accuracy, dynamic lengths, streaming, timing |
| `scripts/bench.py` + `bench_inputs.npz` | One-command device benchmark (numpy + ai-edge-litert only) |
| `scripts/say.py` | Drop-in `say(text)` / `stream(text)` synthesis module + CLI (Piper replacement; frontend tokenization verified identical to the official package) |
| `scripts/dump_graph.py`, `scripts/inspect_onnx.py` | ONNX archaeology used to reverse the architecture |

Reproduce: `make_goldens.py` (torch/ORT venv) → `build_kitten_tflite.py` (TF
venv: `tensorflow`, `tf-keras`, `ai-edge-litert`) → `verify_kitten_litert.py`.

## Why TF re-authoring (and not litert-torch / onnx2tf)

- `litert-torch` (0.9.2) cannot export a **dynamic-length LSTM**: torch.export's
  LSTM decomposition specializes the time axis (documented in the Kokoro port).
  Worse, even conv-only graphs exported with a dynamic axis bake the example
  length into internal RESHAPEs, so the resulting model only runs at the trace
  length. Dynamic shapes effectively require the TF converter path today.
- The TF path also dodges the litert-torch constant-dedup bug that corrupts the
  ISTFTNet cos/sin inverse-DFT ConvTranspose pair — here the iSTFT converts
  cleanly in-graph.
- `onnx2tf` is not used in this repo (accuracy hazards on attention models).

Conversion notes that generalize: ONNX exporters value-deduplicate identical
initializers *and* CSE-merge whole InstanceNorm nodes (two AdaINs normalizing
the same tensor shared one node here) — map norm parameters by graph
connectivity, not by module name. `tf.nn.leaky_relu` defaults to α=0.2 while
torch defaults to 0.01. SineGen's `rad % 1` matters because unvoiced frames
carry small *negative* f0. TFLite has no `ATAN` builtin — `atan2` is the
supported route to the harmonic STFT phase.

## Text frontend / licensing

The model consumes espeak-ng IPA phoneme IDs (same 178-symbol table as
Kokoro/StyleTTS2). The pip package phonemizes with **espeak-ng (GPL-3.0)** —
fine as a separate process, or swap in the Apache-licensed DeepPhonemizer
LiteRT G2P from the [kokoro/](../kokoro/) module for a GPL-free stack.

Model weights: Apache-2.0 (KittenML). Deterministic SineGen: the random initial
phase / noise of the reference are fixed to zero (the reference itself produces
a different waveform every run; zeroing selects one deterministic sample —
same approach as the Kokoro port).
