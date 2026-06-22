# Depth Anything 3 (Small) — Monocular Depth, LiteRT GPU

On-device **monocular depth** with [Depth Anything 3 — Small](https://huggingface.co/depth-anything/DA3-SMALL)
(ByteDance-Seed, Apache-2.0), running fully on the mobile **GPU** via LiteRT `CompiledModel` (ML Drift).

- DINOv2 ViT-S + RoPE backbone, DPT/DualDPT depth head
- Input `[1,3,896,504]` NCHW (native portrait aspect), ImageNet-normalized; output `[1,1,896,504]` depth
- FP16, **55 MB**, **~1.8 s / image on Pixel 8a GPU**
- **corr 0.99948** vs the official PyTorch DA3-Small pipeline

## Setup

1. Download the model from
   [`mlboydaisuke/Depth-Anything-3-Small-LiteRT`](https://huggingface.co/mlboydaisuke/Depth-Anything-3-Small-LiteRT):
   - `da3_small_gpu_fp16.tflite` (55 MB) → put it in `app/src/main/assets/`
2. Open this directory in Android Studio and run (or `./gradlew :app:installDebug`).
3. Tap **Select image** to pick any photo from the library (portrait works best — the model is built for
   ~9:16; other aspects are letterboxed and the depth is cropped back, so they work but at lower effective
   resolution). Inference is ~1.8 s/image on Pixel 8a GPU — on-demand, not live (DA3-Small ViT is too heavy
   for real-time on a phone GPU; ~3 fps even at 448).

## Architecture

| File | Description |
|------|-------------|
| `app/src/main/java/com/da3/MainActivity.kt` | photo-library picker (EXIF-aware) → inference → input \| depth |
| `app/src/main/java/com/da3/DA3Predictor.kt` | `CompiledModel` GPU inference + ImageNet preprocessing + turbo colormap |
| `scripts/convert_da3.py` | the GPU-clean conversion (needs the upstream DA3 source — see header) |

## Preprocessing (must match the model)

```
resize to 504×896 (W×H)  →  x/255  →  (x - mean)/std
mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]   # ImageNet, RGB, NCHW
```

## Conversion notes (GPU-clean, `litert-torch`)

DA3 is not GPU-clean out of the box. `scripts/convert_da3.py` applies exact, GPU-clean rewrites:
checkpoint key-prefix fix · RoPE data-dependent int → const · fused-QKV → 4D attention · **LayerScale
folded into the preceding Linear** (else the token dim is mis-laid-out: `fully_connected {1,1,N,C} vs
{N,1,1,C}`) · **pos_embed bicubic baked to a constant** (interpolating a constant → `GATHER_ND` /
runtime-less `RESIZE_BILINEAR`) · **ConvTranspose2d → zero-stuff + Conv2d** (exact ~1e-7; Pixel 8a rejects
`TRANSPOSE_CONV`) · camera-token in-place assign → `cat` (else `SELECT_V2`).

**Native aspect matters.** DA3 runs at the image's native aspect; a square letterbox drops fidelity to corr
0.977 (padding leaks through global attention). This build is fixed to 896×504 — re-convert at your aspect
(`python scripts/convert_da3.py <image> <H> <W>`) for other shapes.

**Honest residual.** corr 0.99948 (not 1.0). FP16 is not the cause (FP32≡FP16). The ~0.05 % is the DPT-head
`align_corners=True→False` change, forced because the GPU delegate bans `align_corners=True` resize — an
irreducible mobile-GPU constraint, not a bug. Structure and edge sharpness are visually identical.
