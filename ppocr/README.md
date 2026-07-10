# PP-OCRv5 — Text Detection + Recognition on-device (LiteRT GPU, fully GPU)

<p align="center"><img src="https://huggingface.co/litert-community/PP-OCRv5-LiteRT/resolve/main/hero.png" width="380" alt="PP-OCRv5 on-device OCR on a Pixel 8a"></p>


[PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR) (PaddleOCR 2025) text detection + recognition
running **fully on the LiteRT CompiledModel GPU** (ML Drift). Detects text regions in an image and
reads each line. **No autoregressive decoder** (recognition uses a CTC head), so unlike VLM-based OCR
both stages ride the GPU with no CPU/ONNX fallback. The demo runs OCR on a bundled image and overlays
the detected boxes + recognized text.

## On-device (Pixel 8a, Tensor G3 — verified)

| stage | nodes on GPU | time |
|---|---|---|
| **detection** (DBNet) | `777/777` LITERT_CL | ~9 ms |
| **recognition** (SVTR+CTC, per line) | `827/827` LITERT_CL | ~9 ms / line |

Full pipeline (det + 3 lines rec + host postprocess) ~340 ms; all three lines of the bundled image read
correctly. Both tflites are GPU-resident; DB box extraction and CTC decode run on CPU (trivial).

## How it splits (and why it's fully GPU)

```
image[1,3,640,640] →[GPU detector]→ prob map → [CPU: threshold + connected components + unclip] → boxes
   → crop+resize each box[1,3,48,320] →[GPU recognizer]→ CTC logits → [CPU: CTC greedy decode] → text
```

Two GPU blockers, both re-authored to GPU-clean equivalents (per-graph tflite-vs-torch corr **1.0**):

1. **Detector `ConvTranspose2d`** (DB head, 2× upsample) → **`ZeroStuffConvT2d`** — 2D nearest-upsample ×
   stride zero-stuff mask + flipped conv2d (the DAC/DA3 zero-stuff trick generalized to 2D). `TRANSPOSE_CONV`
   is rejected by Mali; this is `RESIZE_NEAREST` + `MUL` + `CONV_2D`, numerically exact.
2. **Recognizer SVTR attention** fused-QKV **5D reshape** → split q/k/v into 4D (numerically identical).

The recognizer head is **CTC** (not autoregressive), so there is no KV-cache decoder to push to CPU —
the whole recognizer runs on the GPU. This is what makes a fully-GPU OCR possible (a VLM-OCR like
Florence-2/GOT would force the decoder onto CPU/ONNX).

## Files

| File | Description |
|------|-------------|
| `PpocrDetector.kt` | Detector on CompiledModel GPU + DB postprocess (threshold, connected components, unclip) |
| `PpocrRecognizer.kt` | Recognizer on CompiledModel GPU + CTC greedy decode (18385-char dict) |
| `MainActivity.kt` | Runs OCR on a bundled image, overlays boxes + recognized text |

## Setup

1. Build the two tflites with `scripts/build_det.py` + `scripts/build_rec.py` (or get them from Hugging
   Face — [litert-community/PP-OCRv5-LiteRT](https://huggingface.co/litert-community/PP-OCRv5-LiteRT)).
2. Build/install the app, then push the models into its private storage:
   ```bash
   ./scripts/install_to_device.sh <dir-with-the-tflites>
   ```
   (`test_image.png` + `ppocrv5_dict.txt` are bundled.)
3. Launch **PP-OCRv5** — it compiles the GPU shaders (~10 s first launch), detects + reads the text.

## Conversion

Weights come from the **PaddleOCR2Pytorch** PyTorch port (Apache-2.0, pure-torch reimplementation of
PaddleOCR, no PaddlePaddle dependency); the converted PyTorch weights are on Hugging Face
[JoyCN/PaddleOCR-Pytorch](https://huggingface.co/JoyCN/PaddleOCR-Pytorch).

- `scripts/build_det.py` — detector + `ZeroStuffConvT2d` (ConvTranspose2d → GPU-clean), op-check, parity.
- `scripts/build_rec.py` — recognizer + 4D-QKV attention, op-check, parity. CTC decode is host-side.

**Original project**: [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (PP-OCRv5) | [Apache-2.0](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)
