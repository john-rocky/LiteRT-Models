# MobileNetV4 — ImageNet classification on-device (LiteRT GPU, fully GPU)

[MobileNetV4](https://arxiv.org/abs/2404.10518) (Google, ECCV 2024 — the latest mobile backbone) ImageNet-1k
classifier (`mobilenetv4_conv_medium`, timm, Apache-2.0), running **fully on the LiteRT CompiledModel GPU**.
Pure CNN (UIB blocks) → guaranteed GPU residency.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **186 / 186** LITERT_CL (full residency) |
| inference | **~4 ms** (256×256) |
| fp16 size | 19.7 MB |
| accuracy | device-vs-PyTorch corr **0.99997**, top-1 match |

```
image[1,3,256,256] (ImageNet-normalized) →[GPU: MobileNetV4-Conv-Medium]→ logits[1,1000]
```

## How it converts (litert-torch) — one re-authoring

Pure CNN. The only GPU fix is the global `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)` (two single-axis means;
the Mali multi-axis-pool fix). MobileNetV4's conv stem has **no max-pool** (strided convs → no `PADV2`), and the
conv variant has no attention → fully GPU-clean. Result: banned ops NONE, all tensors ≤4D, tflite-vs-torch corr
**1.0**, device-vs-torch corr **0.99997**.

## Build & run

```bash
python scripts/build_mnv4.py all   # weights auto-download via timm; produces mnv4_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-mnv4_fp16.tflite>
```

The first launch fails with "Model not found" until the model is pushed.

**Preprocessing**: center-crop, resize 256×256, /255, ImageNet mean/std, NCHW. Output: 1000-class logits →
softmax + top-k in the app.

Model: `mlboydaisuke/MobileNetV4-Conv-Medium-ImageNet-LiteRT` (Hugging Face). Upstream:
[timm](https://github.com/huggingface/pytorch-image-models) `mobilenetv4_conv_medium.e500_r256_in1k` (Apache-2.0).
