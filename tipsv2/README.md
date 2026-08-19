# TIPSv2-B/14 + DPT — depth · normals · segmentation, LiteRT GPU

One backbone, three dense tasks, one GPU graph. [TIPSv2](https://huggingface.co/google/tipsv2-b14)
(Google DeepMind, CVPR 2026, Apache-2.0) is a DINOv2-style ViT-B/14 vision-language backbone; the
[`tipsv2-b14-dpt`](https://huggingface.co/google/tipsv2-b14-dpt) release adds three DPT heads on the
frozen backbone — **metric depth** and **surface normals** (NYU Depth V2) and **ADE20K semantic
segmentation** (150 classes). This module runs all of it fully on the mobile **GPU** via LiteRT
`CompiledModel` (ML Drift).

![TIPSv2 — input | depth | normals | ADE20K seg, on-device LiteRT GPU](https://huggingface.co/litert-community/TIPSv2-B14-DPT-LiteRT/resolve/main/hero.png)

- Input `[1,3,448,448]` NCHW, RGB in **[0,1]** (no ImageNet normalization — TIPSv2 convention)
- Outputs `[1,1,448,448]` depth (metres) · `[1,3,448,448]` unit normals · `[1,150,256,256]` seg logits
- FP16, **318 MB**, **~0.9 s / image on Pixel 8a GPU** (all three heads; 1434/1434 ops on `LITERT_CL`)
- Parity vs the official PyTorch model: depth corr 0.999998 · normals 0.999999 · seg argmax 99.96 %

## Setup

1. Download `tipsv2_b14_dpt_fp16.tflite` (318 MB) from
   [`litert-community/TIPSv2-B14-DPT-LiteRT`](https://huggingface.co/litert-community/TIPSv2-B14-DPT-LiteRT).
2. Open this directory in Android Studio and run (or `./gradlew :app:installDebug`).
3. Stage the model into the app's private storage (too large for APK assets):
   ```bash
   ./scripts/install_to_device.sh path/to/tipsv2_b14_dpt_fp16.tflite
   ```
4. Tap **Select image** — the app shows input | depth, normals | segmentation, and a legend of the
   classes found. Arbitrary aspect ratios are letterboxed into the 448 square and cropped back.

## Architecture

| File | Description |
|------|-------------|
| `app/src/main/java/com/tipsv2/MainActivity.kt` | photo-library picker (EXIF-aware) → inference → four panels + ADE20K legend |
| `app/src/main/java/com/tipsv2/TipsPredictor.kt` | `CompiledModel` GPU inference; depth (Spectral), normals (xyz→RGB), seg (host argmax + ADE20K palette) |
| `scripts/build_tipsv2.py` | re-authored GPU-clean model, parity vs HF, litert-torch convert, fp16, device harness |
| `scripts/dump_ref.py` | dumps the official `google/tipsv2-b14-dpt` outputs used as the parity reference |
| `scripts/install_to_device.sh` | `adb push` + `run-as cp` into `files/` |

## Preprocessing (must match the model)

```
resize/letterbox to 448×448  →  x/255        # RGB, NCHW; NO mean/std normalization
```

Post-processing: depth is metric (the head's bins span 0.001–10 m); the app colours inverse depth
with a 2–98 percentile clip. Normals are unit vectors (x→R, y→G, z→B via `(n+1)/2`). Segmentation
logits come out at the DPT head's native 256×256 grid — `argmax` over the 150 channels host-side,
then nearest-upscale (the official pipeline bilinearly upsamples the logits first; argmax agreement
between the two is 99.96 %).

## Conversion notes (GPU-clean, `litert-torch`)

`scripts/build_tipsv2.py` re-authors the model from the HF weights with exact rewrites (all proven
in this zoo) and converts it in one go:

- **Backbone** (DINOv2-style ViT-B/14, 1 register token, 1026 tokens): fused-qkv attention → 4D
  per-head matmuls (the 5D head split is rejected) · LayerScale γ baked into `attn.proj` / `mlp.fc2`
  · **SafeLayerNorm** (deviation pre-scaled by 1/64 before squaring so the fp16 variance cannot
  overflow) · exact GELU → tanh-GELU (no `ERF` kernel; the only non-exact rewrite) · the 448 input
  matches the native 32×32 pos_embed grid, so there is no runtime interpolation.
- **DPT heads**: readout `cat(patch, cls.expand) @ W` → `patch @ W_a + cls @ W_b` (exact, avoids
  `BROADCAST_TO`) · `ConvTranspose2d(k=s)` → zero-stuff + `Conv2d` (exact; Pixel 8a rejects
  `TRANSPOSE_CONV`) · the fusion blocks' bilinear ×2 with `align_corners=True` (banned on the GPU)
  → **two constant-RHS matmuls** `U·X·Uᵀ` (exact, unlike DA3's `align_corners` flip).
- **Depth-head fp16 range fold.** The depth decoder's activations reach **~1e8** at the logits
  (fp16 max 65504) — on the GPU it returned a constant 3.46 m everywhere while normals/seg were
  fine. The decoder is a ReLU/affine chain ending in a scale-invariant normalisation
  (`relu(l)/Σrelu(l)`), so power-of-2 scales are folded into its weights/biases (convs 1/4096 …
  1/64 per level, each fusion `out_conv` ×1/4, `project` 1/32, head 1/16 — the residual adds see
  matching scales) keeping every stage ≲100. Bit-exact in fp32; device depth corr 0.99986 after.

Device fp16 vs desktop fp32 (Pixel 8a): depth corr 0.99986 · normals 0.99990 · seg argmax 99.3 %.

## Model

| Model | Size | Input | Output | Original | License |
|---|---|---|---|---|---|
| [tipsv2_b14_dpt_fp16.tflite](https://huggingface.co/litert-community/TIPSv2-B14-DPT-LiteRT) | 318 MB | Float32 `[1,3,448,448]` NCHW, [0,1] | depth `[1,1,448,448]` · normals `[1,3,448,448]` · seg `[1,150,256,256]` | [google/tipsv2-b14-dpt](https://huggingface.co/google/tipsv2-b14-dpt) | Apache-2.0 |
