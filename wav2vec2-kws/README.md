# wav2vec2 — Keyword Spotting on-device (LiteRT GPU, all-GPU)

[wav2vec2](https://huggingface.co/facebook/wav2vec2-base) keyword spotting
([`superb/wav2vec2-base-superb-ks`](https://huggingface.co/superb/wav2vec2-base-superb-ks), Apache-2.0)
running **fully on the LiteRT CompiledModel GPU** (ML Drift). Classifies 1 s of 16 kHz audio into 12
Speech-Commands labels (yes / no / up / down / left / right / on / off / stop / go / _unknown_ /
_silence_). The demo classifies a bundled clip on launch and lets you record from the mic.

Unlike the spectral audio models, wav2vec2 has **no FFT anywhere** — the raw 16 kHz waveform goes
straight into a 1D-conv feature extractor (no mel step, in or out of the graph), so the entire model
rides the GPU delegate.

## On-device (Pixel 8a, Tensor G3 — verified)

| graph | nodes on GPU | time |
|---|---|---|
| **frontend** (waveform → features) | `134/134` LITERT_CL | ~2 ms |
| **head** (features → logits) | `893/893` LITERT_CL | ~17 ms |

End-to-end ~19 ms for a 1 s clip (**RTF ≈ 0.02**). Validated on real speech (macOS `say` of all 10
keywords → 10/10 correct; device-vs-CPU logits corr 0.9995).

## Why two graphs (and how it splits)

The model is op-clean for the GPU after re-authoring, **but the full 1008-node graph exceeds the Mali
shader-compile limit** (it fails to compile fused — 2 partitions — even though every op is supported).
Splitting at the conv-frontend / transformer-encoder boundary makes each half compile (134/134 +
893/893). Both halves run on the GPU; the frontend's `feat[1,49,768]` feeds the head.

- **frontend** = `feature_extractor` (7 strided 1D convs + GroupNorm) + `feature_projection`.
- **head** = `encoder` (12 transformer layers + conv positional embedding) + the **weighted-layer-sum**
  over all 13 hidden states (this checkpoint uses `use_weighted_layer_sum`) + `projector` + mean-pool +
  `classifier`.

## Re-authoring → GPU-clean (parity corr 1.0)

| op | rewrite |
|---|---|
| `nn.GELU` ×20 | tanh-GELU `0.5x(1+tanh(√(2/π)(x+0.044715x³)))` |
| feature-extractor `GroupNorm` | GN4D — reshape `(B,G,C//G,T)`, mean/var over `(2,3)` (kills GATHER_ND) |
| pos-conv `weight_norm` (kernel-128 grouped) | fold to a static weight (no runtime `_weight_norm`) |
| encoder `create_bidirectional_mask` | return `None` (fixed-length, no padding → SDPA full attention) |
| weighted-layer-sum | accumulate incrementally `acc += w[i]·hᵢ` with **baked** `softmax(layer_weights)` constants (stack-13 + runtime `w[i]` gathers split the Mali partition) |

The transformer residual peaks at `|x|≈3.2`, so there is no fp16-precision issue — the whole model is
fp16-exact on the GPU (no CPU fallback).

## Files

| File | Description |
|------|-------------|
| `Wav2Vec2Kws.kt` | Loads the 2 GPU graphs, chains frontend → head, returns the top keyword |
| `MainActivity.kt` | Classifies a bundled clip on launch + a mic Record button (AudioRecord 1 s) |

## Setup

1. Build the two tflites with `scripts/build_w2v2_split.py` (or get them from Hugging Face —
   [litert-community/wav2vec2-keyword-spotting](https://huggingface.co/litert-community/wav2vec2-keyword-spotting)).
2. Build/install the app, then push the models into its private storage:
   ```bash
   ./scripts/install_to_device.sh <dir-with-the-tflites>
   ```
   (The 9 MB + 181 MB tflites are too big to bundle; `test_audio.bin` is bundled.)
3. Launch **Wav2Vec2 KWS** — it compiles the GPU shaders (~10 s first launch), classifies the bundled
   clip, then the Record button captures 1 s from the mic and classifies it.

## Conversion

- `scripts/build_w2v2.py` — the single-graph re-authoring + op-check + parity (shows it is op-clean).
- `scripts/build_w2v2_split.py` — the 2-graph deployment split (frontend + head, fp16), parity corr 1.0.

**Original project**: [superb/wav2vec2-base-superb-ks](https://huggingface.co/superb/wav2vec2-base-superb-ks) | [Apache-2.0](https://huggingface.co/facebook/wav2vec2-base)
