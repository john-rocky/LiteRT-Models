# LR-ASPP MobileNetV3 — Semantic Segmentation on-device (LiteRT GPU, fully GPU)

[Lite R-ASPP](https://arxiv.org/abs/1905.02244) with a MobileNetV3-Large backbone (torchvision
`lraspp_mobilenet_v3_large`, COCO-VOC 21 classes) running **fully on the LiteRT CompiledModel GPU**
(ML Drift). A pure-CNN real-time semantic segmentation model — labels every pixel as one of 21 PASCAL-VOC
classes (person, dog, car, chair, …). The demo segments a bundled image and any image picked from the
gallery, overlaying the class colormap.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **242 / 242** LITERT_CL (full residency) |
| inference | **~5 ms** (512×512) |
| fp16 size | 6.7 MB |
| accuracy | device-vs-PyTorch corr **0.99998**, argmax agreement **99.85%** |

```
image[1,3,512,512] (ImageNet-normalized) →[GPU: MobileNetV3 + Lite R-ASPP]→ logits[1,512,512,21]
```

The per-pixel `argmax` (class id) runs on the CPU — trivial.

## How it converts (and why it's fully GPU)

Pure CNN — a single re-authoring. The MobileNetV3 Squeeze-Excite blocks and the Lite R-ASPP scale branch use
**`AdaptiveAvgPool2d(1)`** (global average pool over H×W). A single multi-axis pool is mis-computed / can
overflow on the Mali delegate, so each is replaced with **`mean(3).mean(2)`** (two single-axis means,
numerically identical). Everything else is already GPU-clean: MobileNetV3's `Hardswish`/`Hardsigmoid` map to
the native `HARD_SWISH` builtin, and the R-ASPP `F.interpolate` already uses `align_corners=False`.

Result: banned ops NONE, all tensors ≤4D, tflite-vs-torch corr **1.0**, device-vs-torch corr **1.0**.

## Files

| file | what |
|---|---|
| `app/src/main/java/com/lraspp/LrasppSegmenter.kt` | model wrapper (ImageNet norm → CompiledModel GPU → per-pixel argmax) |
| `app/src/main/java/com/lraspp/MainActivity.kt` | image picker + VOC colormap overlay |
| `scripts/build_lraspp.py` | conversion: re-author + op-check + fp16 + parity |
| `scripts/install_to_device.sh` | push `lraspp_fp16.tflite` into the app's `filesDir` |

## Build & run

```bash
python scripts/build_lraspp.py all       # produces lraspp_fp16.tflite (weights auto-downloaded by torchvision)
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-lraspp_fp16.tflite>
```

The first launch fails with "Model not found" until the install script has been run.

Model: `mlboydaisuke/LRASPP-MobileNetV3-LiteRT` (Hugging Face). Upstream:
[torchvision](https://github.com/pytorch/vision) `lraspp_mobilenet_v3_large` (BSD-3-Clause).
