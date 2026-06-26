# Mimi — Kyutai 2024 Neural Codec on-device (LiteRT, hybrid GPU/CPU)

[Mimi](https://huggingface.co/kyutai/mimi) (the Kyutai/Moshi streaming neural codec, 24 kHz, 12.5 Hz
frame rate) running on-device. The heavy **SEANet convolutional halves run on the LiteRT
CompiledModel GPU** (ML Drift); the two 8-layer Transformers and the split RVQ run on CPU. A 2 s clip
round-trips **faster than real-time** on a Pixel 8a.

The demo app round-trips a bundled clip and plays **original vs. reconstructed** so you can A/B by ear.

## On-device (Pixel 8a, Tensor G3 — verified)

| stage | placement | nodes |
|---|---|---|
| **enc_conv** (audio → features) | GPU | `189/189` LITERT_CL |
| **enc_tx** (encoder-transformer + downsample) | CPU | XNNPACK |
| **RVQ** encode/decode (32 codebooks) | CPU | — |
| **dec_tx** (upsample + decoder-transformer) | CPU | XNNPACK |
| **deconly** (features → audio) | GPU | `220/220` LITERT_CL |

encode ≈ 0.49 s · decode ≈ 0.18 s for a 2 s clip → **RTF ≈ 0.35** (faster than real-time).
Reconstruction sits at the codec's own quality floor (device-vs-input corr 0.935 = the PyTorch Mimi
reference). Codes: 32 × 25 = **800 ints** for 2 s.

## Why hybrid (and the C33 result)

Every op in all four graphs is GPU-clean (re-authored — see below) and the convs are fp16-exact on
Mali (decoder-only fed the exact transformer output = **48 dB SNR**). But the **decoder transformer's
residual stream reaches |x| = 27**, and the Mali GPU delegate's internal fp16 compute loses precision
there — full-GPU decode drops to ~12 dB SNR on real speech. The transformer behaves **identically
standalone and fused** on device (corr 0.70 either way), so this is fp16 **precision**, not a fusion
collapse: **the Matcha "C33" transformer-fusion bug does NOT generalize** to Mimi's own transformer
implementation (it is diffusers-`BasicTransformerBlock`-specific).

The transformers are tiny (8 layers × 512, seq ~50), so running them on CPU is trivial and exact,
while the heavy SEANet convs stay on the GPU. RVQ goes on CPU for the usual reason (Euclidean argmin +
int64 indices that Mali rejects). See `docs/LITERT_CONVERSION_GUIDE.md` for the full write-up.

```
audio →[GPU enc_conv]→ feat →[CPU enc_tx]→ emb →[CPU RVQ.encode]→ codes
      →[CPU RVQ.decode]→ emb →[CPU dec_tx]→ conv_in →[GPU deconly]→ audio
```

## Re-authoring (all GPU-clean, parity ~1.0)

| op | rewrite |
|---|---|
| `nn.GELU` (erf, banned) | tanh-GELU `0.5x(1+tanh(√(2/π)(x+0.044715x³)))` (MUL/ADD/TANH) |
| `MimiRotaryEmbedding` | baked const cos/sin + `rotate_half` (kills GATHER_ND) |
| causal/sliding mask | baked const additive bias `(1,1,S,S)` (kills CUMSUM/EQUAL/SELECT_V2) |
| `MimiLayerScale` | bake γ into the preceding Linear (o_proj / fc2) |
| `ConvTranspose1d` (upsample, depthwise) | grouped `ZeroStuffConvT1d` (kills TRANSPOSE_CONV) |
| `MimiConv1d` causal pad | bake constant `F.pad` (the int64 `.item()` breaks tracing) |
| `nn.ELU` (→ SELECT) | `relu(x) − relu(1 − exp(min(x,0)))` (SELECT-free, exact) |
| downsample `replicate` pad | edge-slice SLICE+CONCAT (tflite PAD is constant-only) |

## Files

| File | Description |
|------|-------------|
| `MimiCodec.kt` | Loads 2 GPU + 2 CPU CompiledModels, runs the hybrid round-trip |
| `MimiRvq.kt` | Split RVQ (1 semantic + 31 acoustic) on CPU — Euclidean argmin, validated vs torch |
| `MainActivity.kt` | Loads a clip, round-trips, plays original vs. reconstructed (AudioTrack) |

## Setup

1. Build the graphs + RVQ weights with `scripts/build_hybrid_graphs.py` +
   `scripts/mimi_rvq_validate_export.py`, or get them from Hugging Face
   [**mlboydaisuke/Mimi-LiteRT**](https://huggingface.co/mlboydaisuke/Mimi-LiteRT).
2. Build/install the app, then push the models into its private storage:
   ```bash
   ./scripts/install_to_device.sh <dir-with-the-files>
   ```
   (The 4 tflites + 69 MB `mimi_rvq.bin` are too big to bundle; `test_audio.bin` is bundled.)
3. Launch **Mimi Codec** — it compiles the GPU shaders (~10 s first launch), round-trips, and plays.

## Conversion

- `scripts/build_mimi.py` — full re-authoring + op-check + parity + the C33 standalone/tapped graphs.
- `scripts/build_hybrid_graphs.py` — the deployment split (2 GPU conv graphs + 2 CPU transformer graphs).
- `scripts/mimi_rvq_validate_export.py` — validates the split RVQ vs torch (encode codes / decode exact),
  exports `mimi_rvq.bin`.

> Note: the conversion uses model-op rewrites (patches) + a GPU/CPU placement split, so this is hosted
> on a personal HF namespace, not as a patch-free "clean" sample. License: same as upstream Mimi (CC-BY).
