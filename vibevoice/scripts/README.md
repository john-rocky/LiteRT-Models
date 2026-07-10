# Building the VibeVoice LiteRT graphs + assets

The app loads four `.tflite` graphs and three binary assets from the device `filesDir`. They are
too big to bundle in the APK, so they are built here and pushed with `install_to_device.sh`.

## Prerequisites

```bash
# 1. The VibeVoice source (standalone modeling code) and the checkpoint
git clone https://github.com/microsoft/VibeVoice.git
huggingface-cli download microsoft/VibeVoice-Realtime-0.5B --local-dir VibeVoice-Realtime-0.5B

# 2. A conversion env (Python 3.10) with the LiteRT toolchain + the repo pin
pip install "torch==2.6.0" "transformers==4.51.3" diffusers accelerate safetensors soundfile
pip install ai-edge-litert litert-torch ai-edge-quantizer      # graph conversion
```

## Build

```bash
python build_vibevoice.py --stage all \
    --src   ./VibeVoice \
    --ckpt  ./VibeVoice-Realtime-0.5B \
    --out   ./out
```

This produces, in `--out`:

| File | What |
|---|---|
| `vv_base_lm_kv_fp16.tflite` | 4-layer Qwen2 text LM, one AR step (packed KV cache) |
| `vv_tts_lm_kv_fp16.tflite` | 20-layer Qwen2 TTS LM, one AR step |
| `vv_diffhead_fp16.tflite` | 4-layer DDPM diffusion head, one denoise step |
| `vv_decoder_fp16.tflite` | σ-VAE conv decoder (fixed 128 frames) |
| `embed_tokens.f16` | fp16 token-embedding table (host lookup) |
| `glue.f32` | acoustic connector + type embedding + EOS classifier weights |
| `voice_en-Emma_woman.bin` | precomputed prompt KV cache = the voice preset |

Individual stages: `--stage {decoder,head,backbone,assets}`.

## Install

```bash
./install_to_device.sh ./out
```

Each conversion prints the eager-vs-tflite parity (all `corr 1.0`) and asserts no GPU-banned ops.
The `VibeVoiceSynthesizer.kt` host loop was validated against the original torch model end-to-end
(the σ-VAE decoder reproduces the reference waveform from real latents at `corr 1.0`), and the
Kotlin DPM-Solver++ port matches diffusers exactly. See `../README.md` for the op-rewrite catalog
and `../../docs/LITERT_CONVERSION_GUIDE.md`.
