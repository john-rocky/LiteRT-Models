# UniSal — On-device visual saliency prediction (LiteRT GPU, fully GPU)

[UniSal](https://github.com/rdroste/unisal) (rdroste, Apache-2.0) **visual saliency prediction** — a heatmap of
**where humans look** in an image — running **fully on the LiteRT CompiledModel GPU**. MobileNetV2 encoder +
bilinear-upsample decoder, **3.71 M params / 6.5 MB fp16**.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **158 / 158** LITERT_CL (full residency) |
| inference | **~3 ms** (256×256) |
| fp16 size | 6.5 MB |
| accuracy | device-vs-PyTorch corr **0.9998** |

```
image[1,3,256,256] (ImageNet mean/std) →[GPU: UniSal]→ saliency[1,1,256,256] (higher = more attended)
```

UniSal is a unified image+video model; for static images the **Bypass-RNN** path is used and the **SALICON**
domain is pinned (its domain-specific BatchNorm / smoothing fold to constants).

## How it converts (litert-torch) — three numerically-exact fixes

1. **Strided subsample `x[..., ::2, ::2]` → `F.avg_pool2d(x, 1, 2)`** — the MobileNetV2 strided slice lowers to
   `GATHER_ND` (banned); a kernel-1 stride-2 average-pool selects the exact same pixels → `AVERAGE_POOL_2D`.
2. **Bake the 16 Gaussian prior maps** — `_get_gaussian_maps` (meshgrid + exp) emits `GATHER_ND` + `BROADCAST_TO`;
   the maps depend only on the (fixed) feature size, so precompute them once and concatenate the constant.
3. **`F.pad(mode="replicate")` → 0-pad** for the 41×41 Gaussian smoothing (replicate → `GATHER_ND`; the smoothing
   is essential — it suppresses border artifacts — so it's kept, with a 0-pad).

Result: banned ops NONE, all tensors ≤4D, tflite-vs-torch corr **1.0**, device-vs-torch corr **0.9998**. The
final spatial log-softmax / normalization runs in the app.

## Build & run

```bash
# weights training_runs/pretrained_unisal/weights_best.pth ship in the unisal repo (Apache-2.0)
python scripts/build_unisal.py all     # produces unisal_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-unisal_fp16.tflite>
```

The first launch fails with "Model not found" until the model is pushed.

**Preprocessing**: center-crop, resize 256×256, /255, ImageNet mean/std, NCHW. The app min-max normalizes the
saliency and overlays a jet heatmap.

Model: `litert-community/UniSal-Saliency-LiteRT` (Hugging Face). Upstream:
[rdroste/unisal](https://github.com/rdroste/unisal) (Apache-2.0).
