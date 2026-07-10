# L2CS-Net — Gaze Estimation on-device (LiteRT GPU, fully GPU)

[L2CS-Net](https://github.com/Ahmednull/L2CS-Net) (Ahmednull, MIT) **gaze estimation** — predicts where a
(centered) face is looking (yaw/pitch), **fully on the LiteRT CompiledModel GPU**. ResNet50 backbone trained on
Gaze360. Useful for attention/AR/accessibility.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **139 / 139** LITERT_CL (full residency) |
| inference | **~3 ms** (448×448) |
| fp16 size | 47.9 MB |
| accuracy | device-vs-PyTorch corr **0.9999**, gaze angle within ~0.1° |

```
face[1,3,448,448] (ImageNet-normalized) →[GPU: ResNet50]→ yaw[1,90], pitch[1,90] (softmax over angle bins)
```

The 90 bins span [-180,180]° at 4° each; the gaze angle is the softmax expectation `deg = Σ p_i·i · 4 − 180`
(decoded + softmax baked in). The decode + arrow run in the app.

## How it converts (litert-torch) — two numerically-exact re-authorings

Pure CNN (ResNet50 + 2 FC heads). Same two fixes as a plain ResNet:

1. **ResNet stem `MaxPool2d(3, s2, p1)` → zero-pad + valid max-pool.** PyTorch's max-pool pads with `-inf`,
   which litert-torch lowers to a `PADV2` op the Mali delegate **won't delegate** (it splits the graph and
   fails to compile). Because the pool follows a ReLU (inputs ≥ 0), padding with **0** is exactly equivalent and
   emits a delegatable `PAD` → full GPU residency.
2. **Global `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)`** (two single-axis means).

The softmax over the angle bins is baked into the graph. Result: banned ops NONE, all tensors ≤4D,
tflite-vs-torch corr **1.0**, device-vs-torch corr **0.9999**.

## Files

| file | what |
|---|---|
| `app/src/main/java/com/gaze/GazeEstimator.kt` | model wrapper + bin-expectation decode (→ yaw/pitch degrees) |
| `app/src/main/java/com/gaze/MainActivity.kt` | image picker + gaze-direction arrow overlay |
| `scripts/build_gaze.py` | conversion: load L2CS-Net weights + the two fixes + op-check + fp16 + parity |
| `scripts/model.py` | the L2CS-Net model definition (MIT, from Ahmednull/L2CS-Net) |

## Build & run

```bash
# weights: L2CSNet_gaze360.pkl from HF (tianfxc/l2cs) or the L2CS-Net release
python scripts/build_gaze.py all       # produces gaze_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-gaze_fp16.tflite>
```

The first launch fails with "Model not found" until the model is pushed.

**Preprocessing**: center-crop to a (centered) face, resize to 448×448, /255, ImageNet mean/std, NCHW. Decode:
softmax expectation over the 90 bins → yaw/pitch degrees → gaze direction.

Model: `litert-community/L2CS-Gaze360-LiteRT` (Hugging Face). Upstream:
[Ahmednull/L2CS-Net](https://github.com/Ahmednull/L2CS-Net) (MIT).
