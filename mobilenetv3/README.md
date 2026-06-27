# MobileNetV3-Large — ImageNet Classification on-device (LiteRT GPU, fully GPU)

MobileNetV3-Large (torchvision, ImageNet-1k, 1000 classes) running **fully on the LiteRT CompiledModel GPU**
(ML Drift). A standard supervised image classifier (top-1/top-5 ImageNet labels) — distinct from the repo's
CLIP zero-shot classifier. The demo classifies a bundled image and any image picked from the gallery, showing
the top-5 predictions.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **233 / 233** LITERT_CL (full residency) |
| inference | **~4 ms** (224×224) |
| fp16 size | 11.2 MB |
| accuracy | device-vs-PyTorch corr **0.99995**, top-1 match (bundled beagle → "beagle") |

```
image[1,3,224,224] (ImageNet-normalized) →[GPU: MobileNetV3-Large]→ logits[1,1000]
```

## How it converts (and why it's fully GPU)

Pure CNN — a single re-authoring: the MobileNetV3 Squeeze-Excite blocks and the final classifier pool use
`AdaptiveAvgPool2d(1)` (global average pool), each replaced with `mean(3).mean(2)` (two single-axis means;
a single multi-axis pool is mis-computed on the Mali delegate). Everything else is already GPU-clean —
MobileNetV3's `Hardswish`/`Hardsigmoid` map to the native `HARD_SWISH` builtin. Softmax + top-k run on the CPU.

Result: banned ops NONE, all tensors ≤4D, tflite-vs-torch corr **1.0**, device-vs-torch corr **1.0**.

## Files

| file | what |
|---|---|
| `app/src/main/java/com/mnv3/ImagenetClassifier.kt` | model wrapper (ImageNet norm → CompiledModel GPU → top-k) |
| `app/src/main/java/com/mnv3/MainActivity.kt` | image picker + top-5 predictions |
| `app/src/main/assets/imagenet_classes.json` | 1000 ImageNet class names |
| `scripts/build_mnv3.py` | conversion: re-author + op-check + fp16 + parity |
| `scripts/install_to_device.sh` | push `mnv3_fp16.tflite` into the app's `filesDir` |

## Build & run

```bash
python scripts/build_mnv3.py all       # weights auto-downloaded by torchvision; produces mnv3_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-mnv3_fp16.tflite>
```

The first launch fails with "Model not found" until the install script has been run.

**Preprocessing**: center-crop to square, resize to 224×224, divide by 255, ImageNet mean/std, NCHW planar.

Model: `mlboydaisuke/MobileNetV3-Large-ImageNet-LiteRT` (Hugging Face). Upstream:
[torchvision](https://github.com/pytorch/vision) `mobilenet_v3_large` (BSD-3-Clause).
