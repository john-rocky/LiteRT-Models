# Zipformer ASR — LiteRT CompiledModel GPU

English speech recognition with the **Zipformer medium CR-CTC** encoder (k2/icefall,
LibriSpeech, 64 M params, WER 2.12 / 4.62 greedy) running **fully on the LiteRT
`CompiledModel` GPU**. First Zipformer architecture in this zoo.

```
16 kHz PCM ─► ZipformerFbank (kaldi 80-mel, host) ─► [GPU] Conv2dSubsampling +
Zipformer2 (6 stacks) + CTC linear ─► greedy CTC + BPE detok (host)
```

- Fixed 16 s window: fbank `[1,1600,80]` + 4 additive attention-bias masks
  (`0` real / `-1000` pad) at the internal frame rates `[1,796] [1,398] [1,199] [1,100]`.
- Output: raw CTC logits `[1,398,500]` at 25 Hz (blank id 0, BPE-500).
- Pixel 8a: GPU compile 1.8 s, **156 ms** run+readback per 16 s window (RTF ≈ 0.01),
  transcripts identical to the PyTorch reference on the test sweep.

## Run

1. Download `zipformer_ctc_fp16.tflite` (132 MB) from
   [litert-community/Zipformer-medium-CR-CTC-LiteRT](https://huggingface.co/litert-community/Zipformer-medium-CR-CTC-LiteRT).
2. `./scripts/install_to_device.sh <dir-with-the-tflite>` (pushes it into the app's filesDir).
3. Build + install this project (`./gradlew :app:installDebug`), launch **Zipformer ASR**,
   then record from the mic or transcribe the bundled sample (JFK 1961 inaugural address,
   U.S. National Archives, public domain).

## Files

- `ZipformerFbank.kt` — `torchaudio.compliance.kaldi.fbank` port (povey window,
  snip_edges=false reflect padding, high_freq −400, dither 0, **[-1,1] input scale, no CMN**);
  verified vs torchaudio (log-domain max|diff| 0.0026, corr 1.0).
- `ZipformerAsr.kt` — CompiledModel GPU runner (5 inputs resolved by capacity) + greedy CTC.
- `assets/` — `mel80_257.bin`, `povey400.bin` (precomputed fronts), `tokens.txt` (BPE-500),
  `sample.wav`.

Model/conversion details: the model card above and the root README's Speech Recognition
section.
