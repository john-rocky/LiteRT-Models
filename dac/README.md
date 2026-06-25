# DAC — Neural Audio Codec on-device (LiteRT GPU)

[Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) (DAC, 16 kHz) running
on the LiteRT CompiledModel **GPU** (ML Drift) path. Compresses 1 s of audio to **12×50 = 600
integer codes (~43:1)** and reconstructs it — round-trips **faster than real-time** on a Pixel 8a,
with the two convolutional halves fully GPU-resident.

The demo app round-trips a bundled clip and plays **original vs. reconstructed** so you can A/B by ear.

## On-device (Pixel 8a, Tensor G3 — verified)

| stage | nodes on GPU | time (warm) |
|---|---|---|
| **encoder** (audio → latent) | `367/367` LITERT_CL | ~0.30 s |
| **decoder** (latent → audio) | `398/398` LITERT_CL | ~0.35 s |
| **RVQ** (codes ↔ latent) | CPU (~1 ms) | — |

End-to-end **RTF ≈ 0.82** (faster than real-time). Reconstruction is bit-faithful to the PyTorch
DAC (numpy/tflite round-trip corr **1.0**).

## How it splits (and why)

A neural codec doesn't ride the GPU delegate as-is — two walls, both solved here:

1. **Decoder `ConvTranspose1d`** → the real DAC's odd stride-5 transposed conv **fails converter
   legalization** (`mhlo.convolution` lhs_dilation=5), and even strides emit `TRANSPOSE_CONV`
   which Mali rejects. **Fix: `ZeroStuffConvT1d`** — nearest-upsample (in 2D → `RESIZE_NEAREST`) ×
   constant mask + flipped-weight `conv1d`, numerically exact (corr 1.0), all GPU-clean ops.
2. **RVQ** (codes ↔ latent) uses `EMBEDDING_LOOKUP` + int64 indices, which Mali rejects
   (`CAST: INT64 not supported`). **Fix: run the RVQ on CPU** ([`DacRVQ.kt`], ~1 ms) and keep only
   the float conv encoder/decoder on the GPU. This is the standard codec deployment split.

So: `audio → encoder.tflite (GPU) → z → RVQ.encode (CPU) → codes → RVQ.decode (CPU) → z_q →
decoder.tflite (GPU) → audio`.

## Files

| File | Description |
|------|-------------|
| `DacCodec.kt` | Loads encoder + decoder on CompiledModel GPU, runs the round-trip |
| `DacRVQ.kt` | Residual VQ (codes ↔ latent) on CPU — 12 codebooks, validated bit-exact vs torch |
| `MainActivity.kt` | Loads a clip, round-trips, plays original vs. reconstructed (AudioTrack) |

## Setup

1. Get the two tflites — from Hugging Face **mlboydaisuke/DAC-16kHz-LiteRT** *(upload pending)*,
   or build with `scripts/convert_dac_encoder.py` + `scripts/convert_dac_deconly.py`.
2. Build/install the app, then push the models into its private storage:
   ```bash
   ./scripts/install_to_device.sh <dir-with-the-tflites>
   ```
   (The 43 MB + 105 MB models are too big to bundle; `dac_rvq.bin` + `test_audio.bin` are bundled.)
3. Launch **DAC Codec** — it compiles the GPU shaders (~10 s first launch), round-trips, and plays.

## Conversion

- `scripts/convert_dac_encoder.py` — encoder (strided Conv1d + Snake1d), GPU-clean as-is.
- `scripts/convert_dac_deconly.py` — decoder with `ZeroStuffConvT1d` (ConvTranspose1d → GPU-clean).
- `scripts/dac_rvq_validate_export.py` — validates the numpy RVQ vs torch (codes 100%, corr 1.0) and
  exports `dac_rvq.bin` for the app.

> Note: `ZeroStuffConvT1d` is a model-op rewrite (a patch), so this is hosted on a personal HF
> namespace, not as a patch-free "clean" sample. License: MIT (Descript DAC).
