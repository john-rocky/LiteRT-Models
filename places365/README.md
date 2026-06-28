# Places365 — Scene Recognition on-device (LiteRT GPU, fully GPU)

ResNet18 trained on [Places365](http://places2.csail.mit.edu/) (CSAILVision, MIT) — **scene/place
recognition** across **365 categories** (beach, kitchen, forest, office, restaurant, …). Distinct from object
classification (it answers *what kind of place* a photo is). Runs **fully on the LiteRT CompiledModel GPU**.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **61 / 61** LITERT_CL (full residency) |
| inference | **~2 ms** (224×224) |
| fp16 size | 22.8 MB |
| accuracy | device-vs-PyTorch corr **1.0**, top-1 match (bundled beach → "beach") |

```
image[1,3,224,224] (ImageNet-normalized) →[GPU: ResNet18]→ logits[1,365]
```

## How it converts (litert-torch) — two numerically-exact re-authorings

Pure CNN, but a plain ResNet needs two fixes for the Mali delegate (both exact):

1. **Global `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)`** (two single-axis means; a single multi-axis pool is
   mis-computed on the Mali delegate).
2. **ResNet stem `MaxPool2d(3, s2, p1)` → zero-pad + valid max-pool.** PyTorch's max-pool pads with `-inf`,
   which litert-torch lowers to a `PADV2` op the Mali delegate **won't delegate** (it splits the graph into CPU
   partitions and fails to compile). Because the pool follows a ReLU (inputs ≥ 0), padding with **0** is
   exactly equivalent and emits a delegatable `PAD` → full GPU residency. (Reusable for any ResNet-style stem.)

Result: banned ops NONE, all tensors ≤4D, tflite-vs-torch corr **1.0**, device-vs-torch corr **1.0**.

## Files

| file | what |
|---|---|
| `app/src/main/java/com/places365/PlacesClassifier.kt` | model wrapper (ImageNet norm → CompiledModel GPU → top-k) |
| `app/src/main/java/com/places365/MainActivity.kt` | image picker + top-5 scenes |
| `app/src/main/assets/places365_classes.json` | 365 scene category names |
| `scripts/build_places.py` | conversion: load weights + the two re-authorings + op-check + fp16 + parity |
| `scripts/install_to_device.sh` | push `places_fp16.tflite` into the app's `filesDir` |

## Build & run

```bash
python scripts/build_places.py all     # downloads resnet18_places365 weights + labels; produces places_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-places_fp16.tflite>
```

The first launch fails with "Model not found" until the install script has been run.

**Preprocessing**: center-crop to square, resize to 224×224, /255, ImageNet mean/std, NCHW. Output 365-class
logits; softmax + argmax for top-k.

Model: `mlboydaisuke/Places365-ResNet18-LiteRT` (Hugging Face). Upstream:
[CSAILVision/places365](https://github.com/CSAILVision/places365) (MIT).
