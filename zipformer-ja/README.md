# Japanese Zipformer ASR — LiteRT CompiledModel GPU

Japanese speech recognition with
[japanese-zipformer-base](https://huggingface.co/reazon-research/japanese-zipformer-base-k2-rs35kh-bpe)
(reazon-research, ReazonSpeech, 96.5M, avg CER 11.46 %) running **fully on the LiteRT
`CompiledModel` GPU** — with **zero FFT anywhere**: raw waveform → wav2vec2-style conv
frontend → Zipformer encoder → CTC, all in one GPU graph. First Japanese ASR in this zoo.

```
16 kHz PCM ─► [GPU] conv frontend (stride 320 → 50 Hz) + Zipformer2 (6 stacks) +
CTC linear ─► greedy CTC + BPE detok (host)
```

- Fixed 16 s window with a 0.5 s zero lead pad (upstream convention); 4 additive
  attention-bias masks (`0` real / `-1000` pad) at rates `[1,799] [1,400] [1,200] [1,100]`.
- Output: raw CTC logits `[1,799,3004]` at 50 Hz — **blank id 0** (icefall convention).
- Pixel 8a: GPU compile 2.4 s, **621 ms** per 16 s window (RTF ≈ 0.04), transcripts
  identical to the PyTorch reference on the test sweep.
- Greedy decode without an LM is phonetically exact; occasional kanji homophone swaps
  (選挙 → 占拠) are the expected no-LM behavior.

## Run

1. Download `ja_zipformer_ctc_fp16.tflite` (197 MB) from
   [litert-community/japanese-zipformer-base-LiteRT](https://huggingface.co/litert-community/japanese-zipformer-base-LiteRT).
2. `./scripts/install_to_device.sh <dir-with-the-tflite>`
3. Build + install (`./gradlew :app:installDebug`), launch **Japanese Zipformer ASR**,
   record from the mic or transcribe the bundled sample.

The bundled sample is a reading of the Preamble of the Constitution of Japan
([Wikimedia Commons](https://commons.wikimedia.org/wiki/File:%E6%97%A5%E6%9C%AC%E5%9B%BD%E6%86%B2%E6%B3%95%E5%89%8D%E6%96%87.ogg), CC0).

## Files

- `JaZipformerAsr.kt` — CompiledModel GPU runner (5 inputs resolved by capacity) +
  greedy CTC + BPE-3004 detokenize.
- `assets/tokens.txt` — vocab, index-ordered; `assets/sample.wav` — CC0 sample.

## Conversion

`scripts/build_ja_zipformer.py` — re-authors `reazon-research/japanese-zipformer-base`
(wav2vec2-style raw-waveform frontend + Zipformer2 + CTC) for CompiledModel GPU and
converts via litert-torch. Prereq: the HF snapshot at `scripts/model/` (modeling code +
weights), e.g. `hf download reazon-research/japanese-zipformer-base --local-dir scripts/model`.

### Converting your own fine-tuned checkpoint

A fine-tune of the reazon model saved with `save_pretrained()` converts the same way —
the modeling code still comes from the base snapshot in `scripts/model/`, while config and
weights load from your directory (defaults reproduce the official ship exactly):

```bash
ZIPJA_MODEL_DIR=/path/to/your_saved_model ZIPJA_WAV=my_test.wav \
python scripts/build_ja_zipformer.py all
```

If the fine-tune changed the vocab, set `ZIPJA_VOCAB=N` (default 3004) and swap the app's
`assets/tokens.txt` to match (blank id 0 convention unchanged).
