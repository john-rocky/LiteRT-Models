# NAFNet — Image Restoration (deblur) on-device (LiteRT GPU, fully GPU)

[NAFNet](https://github.com/megvii-research/NAFNet) (Nonlinear Activation Free Network, ECCV 2022, MIT)
image restoration running **fully on the LiteRT CompiledModel GPU** (ML Drift). NAFNet is a U-Net of
**NAFBlocks** with **no activation functions at all** (SimpleGate = channel-split multiply), so the whole
network is a clean CNN that rides the GPU delegate. This demo runs **NAFNet-GoPro-width32** (motion deblur)
on a bundled blurry image and on any image picked from the gallery, showing input | restored.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **2179 / 2179** LITERT_CL (full residency) |
| inference | **~42 ms** (256×256) |
| fp16 size | 38 MB |
| accuracy | device output **== PyTorch (corr 1.000000)**; re-authoring is numerically exact |

```
image[1,3,256,256] (RGB [0,1]) →[GPU: NAFNet U-Net]→ restored[1,3,256,256]
```

## How it converts (and the SafeLayerNorm device fix)

Pure CNN, three numerically-exact re-authorings:

1. **`LayerNorm2d` (custom autograd Function) → fp16-safe channel LayerNorm.** NAFNet's residual stream grows
   large (|x|≈175 at the bottleneck), so the LayerNorm channel reductions `Σ_c x` (~90k over 512 channels) and
   `Σ_c (x−μ)²` (~15M) **overflow fp16 (max 65504)** on the Mali delegate — which computes in fp16 regardless
   of the model dtype. The device output then looks ~right (corr 0.98, dominated by the input image) but the
   learned residual is destroyed (corr 0.016) → a periodic **grid artifact**. Fix: do the reductions in a
   down-scaled domain (`x/S`, S=128) so the sums stay < 65504, then rescale variance and `(x−μ)` back —
   numerically exact (LayerNorm is scale-invariant). → device corr **1.0**.
2. **Simplified Channel Attention `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)`** (two single-axis means; a single
   multi-axis global pool is mis-computed / overflows on Mali).
3. **Upsample `Conv2d(1×1) + PixelShuffle(2)` → Conv2d + depth-to-space `ZeroStuffConvT2d`** (PixelShuffle
   lowers to a 6D reshape; ZeroStuffConvT2d is `RESIZE_NEAREST` + `MUL` + `CONV_2D`, exact).

Result: banned ops NONE, all tensors ≤4D, tflite-vs-torch corr **1.0**, device-vs-torch corr **1.0**.

## Files

| file | what |
|---|---|
| `app/src/main/java/com/nafnet/NafnetRestorer.kt` | model wrapper ([0,1] RGB → CompiledModel GPU → restored) |
| `app/src/main/java/com/nafnet/MainActivity.kt` | image picker + input/restored view |
| `scripts/build_nafnet.py` | conversion: re-author + op-check + fp16 + parity |
| `scripts/install_to_device.sh` | push `nafnet_fp16.tflite` into the app's `filesDir` |

## Build & run

```bash
# 1) Convert (download NAFNet-GoPro-width32.pth from HF nyanko7/nafnet-models first)
python scripts/build_nafnet.py all       # produces nafnet_fp16.tflite

# 2) Build + install the app, then push the model
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-nafnet_fp16.tflite>
```

The first launch fails with "Model not found" until the install script has been run.

## Denoising variant (NAFNet-SIDD)

The same NAFNet architecture trained on SIDD does real-image **denoising**. Build it with
`scripts/build_sidd.py` (downloads `NAFNet-SIDD-width32.pth` from HF `nyanko7/nafnet-models`), then push
`sidd_fp16.tflite` as the app's model (rename to `nafnet_fp16.tflite`) — the same `input | restored` UI shows
noisy → denoised. Device-verified on a Pixel 8a: `2179/2179` LITERT_CL, ~46 ms, device-vs-torch corr
**0.999999**, fp16 62.5 MB. Model:
[litert-community/NAFNet-SIDD-width32-LiteRT](https://huggingface.co/litert-community/NAFNet-SIDD-width32-LiteRT).

Models: `litert-community/NAFNet-GoPro-width32-LiteRT` (deblur) / `litert-community/NAFNet-SIDD-width32-LiteRT`
(denoise) (Hugging Face). Upstream: [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) (MIT).
