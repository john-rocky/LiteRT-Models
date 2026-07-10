# Speaker Diarization — on-device "who spoke when" (pyannote 3.1 stack, GPU embeddings)

On-device **speaker diarization**: record a conversation (or pick a clip) and get a per-speaker
timeline + per-speaker playback. The stack is the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
recipe (MIT) ported to Android:

| Stage | Model | Runtime |
| :-- | :-- | :--: |
| Segmentation (speech + local speakers, powerset) | [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (PyanNet SincNet+BiLSTM, 1.5 M, MIT) | onnxruntime **CPU** |
| Speaker embedding | [WeSpeaker ResNet34](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) (6.6 M, CC-BY-4.0) | LiteRT CompiledModel **GPU** |
| kaldi fbank + CMN, clustering, stitching | — | Kotlin host |

The BiLSTM has no Mali GPU kernel, so segmentation stays on CPU (tiny + fast); the heavy embedding
CNN runs **fully on the GPU**.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| embedding nodes on GPU | **108 / 108** LITERT_CL (full residency, 1 partition) |
| embedding inference | **~1.2 ms** per 5 s window, fp16 13.4 MB |
| embedding accuracy | device-vs-PyTorch cosine **0.99997** |
| segmentation (ONNX CPU) | corr 1.0 / argmax agreement 100% vs PyTorch, 5.9 MB |

## Pipeline (simplified pyannote 3.1)

Sliding 10 s windows (5 s hop) → powerset argmax → per-(window, local-speaker) units with ≥1 s of
solo speech → WeSpeaker embedding of the unit's concatenated solo audio (tile-padded to the fixed
5.015 s / 500-frame fbank window, CMN'd) → agglomerative clustering (centroid linkage, cosine
distance, threshold 0.7046 from the 3.1 config) → windows stitched by their central regions →
merged per-speaker segments.

The kaldi-fbank front-end (hamming, 25/10 ms, 80 mel, dither 0, pre-emphasis 0.97, snip edges,
×2¹⁵ scaling) is ported to Kotlin with precomputed mel banks (`assets/mel80_257.bin`), verified
against `torchaudio.compliance.kaldi.fbank` (corr 1.0, max|d| 4e-4 log-domain).

## Build & run

```bash
python scripts/build_diar.py      # -> wespeaker_emb_fp16.tflite + pyannote_seg30.onnx
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-models>
```

The first launch fails with "Model not found" until the models are pushed. **Record conversation**
(up to 120 s) or **pick a clip**, then read the timeline and tap ▶ per speaker.

Models: `litert-community` (Hugging Face). Upstream: [pyannote.audio](https://github.com/pyannote/pyannote-audio)
(MIT); WeSpeaker weights CC-BY-4.0.
