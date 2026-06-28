# wav2vec2-CTC — On-device speech recognition (LiteRT GPU, fully-GPU, single-pass)

[wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) (Facebook, Apache-2.0) **speech
recognition** running **fully on the LiteRT CompiledModel GPU**. Unlike encoder–decoder ASR (Whisper), the CTC
head needs **no autoregressive decoder** — it's **one GPU graph, a single forward pass** (~22 ms / 10 s on a
Pixel 8a). CTC greedy decode runs on the host.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **997 / 997** LITERT_CL (full residency, single graph) |
| inference | **~22 ms** (10 s clip) |
| fp16 size | 189.8 MB |
| accuracy | device-vs-PyTorch corr **0.99998**, **exact** transcription |

```
waveform[1,160000] (16 kHz, zero-mean/unit-var) →[GPU: wav2vec2-CTC]→ logits[1,499,32] →[host CTC]→ text
```

Verified transcription (LibriSpeech sample), device == PyTorch == ground truth:
> *MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL*

## How it converts (litert-torch) — all numerically-equivalent

Raw 16 kHz waveform → 1D-conv feature extractor → 12-layer transformer → CTC `lm_head` (32 vocab). **Zero FFT.**
The transformer residual peaks at |x|≈3.2, so there is no fp16-on-Mali precision issue. Three re-authorings:

1. **GELU → tanh-GELU** (exact GELU lowers to the banned `Erf`).
2. **Feature-extractor `GroupNorm` → 4D reshape `(B,G,C//G,T)` + mean/var over `(2,3)`** (kills `GATHER_ND`);
   wav2vec2's GroupNorm is per-channel-over-time (a small reduction), so it's fp16-precise on Mali.
3. **Fold the `pos_conv` weight-norm** into a static weight (the runtime `g·v/‖v‖` splits the Mali partition),
   and `create_bidirectional_mask → None` for a fixed-length, no-padding clip (the all-valid mask is a no-op).

Result: banned ops NONE, all tensors ≤4D, **no FFT**, tflite-vs-torch corr 1.0, device-vs-torch corr 0.99998.

## Build & run

```bash
python scripts/build_w2v2_ctc.py all     # W2V_SEC=10 -> w2v2_ctc_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-w2v2_ctc_fp16.tflite>   # large model -> filesDir
```

The first launch fails with "Model not found" until the model is pushed. **Hold to Talk** (mic) or **Transcribe
sample clip**. Input: mono 16 kHz, zero-mean/unit-variance, padded/truncated to 10 s.

Model: `litert-community/wav2vec2-base-960h-CTC-LiteRT` (Hugging Face). Upstream:
[facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) (Apache-2.0).
