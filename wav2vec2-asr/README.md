# wav2vec2 ASR — LiteRT CompiledModel GPU

English speech recognition with **wav2vec2-base-960h** (char-CTC) running **fully on the LiteRT
`CompiledModel` GPU** — and with **zero FFT anywhere**: the raw 16 kHz waveform goes straight
into the 1D-conv feature extractor, no mel/fbank even on the host.

```
16 kHz PCM ─► [GPU] conv frontend (waveform → feat 50 Hz) ─► [GPU] 12-layer
transformer + lm_head ─► greedy char-CTC (host)
```

Two GPU graphs (the fused graph exceeds the Mali whole-graph shader-compile limit — same
finding as the wav2vec2 KWS ship): frontend 9 MB + head 180 MB. Pixel 8a: 448 ms + 391 ms
per 16 s window (RTF ≈ 0.05), valid-region logits corr 0.9928 vs PyTorch.

Greedy char decode without an LM has the model's known spelling quirks on hard words —
this sample favors the simplest possible all-GPU pipeline over the last WER point.

## Run

1. Download both tflites from
   [litert-community/wav2vec2-base-960h-LiteRT](https://huggingface.co/litert-community/wav2vec2-base-960h-LiteRT).
2. `./scripts/install_to_device.sh <dir-with-the-tflites>`
3. Build + install (`./gradlew :app:installDebug`), launch **wav2vec2 ASR**, record from the
   mic or transcribe the bundled sample (JFK 1961 inaugural address, public domain).

## Files

- `Wav2Vec2Asr.kt` — two chained CompiledModels + greedy char-CTC (blank 0, `|` → space).
- `assets/tokens.txt` — 32-entry char vocab, index-ordered.
