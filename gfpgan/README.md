# GFPGAN — Blind Face Restoration (LiteRT CompiledModel GPU)

On-device [GFPGAN v1.4](https://github.com/TencentARC/GFPGAN) (Apache-2.0): restores degraded /
low-quality faces using a StyleGAN2 generative facial prior. Runs **fully on the GPU** via LiteRT
`CompiledModel` (ML Drift).

| | |
|---|---|
| Model | GFPGAN v1.4 (clean arch: U-Net encoder + StyleGAN2 decoder with SFT) |
| Source | TencentARC/GFPGAN, weights `GFPGANv1.4.pth` (`params_ema`) |
| License | Apache-2.0 |
| Input | `[1, 3, 512, 512]` **NCHW**, RGB, normalized to `[-1, 1]` (aligned face) |
| Output | `[1, 3, 512, 512]` **NCHW**, restored face in `[-1, 1]` |
| Size | fp16 431 MB, GPU-clean (banned ops: none, >4D tensors: none) |

## Conversion (GPU compatibility)

Converted with **litert-torch** (NCHW preserved). The one real re-authoring is the StyleGAN2
`ModulatedConv2d`, whose original form is doubly GPU-incompatible: it builds a **5D** weight
`(b, c_out, c_in, k, k)` at runtime from the style vector and convolves with that runtime filter
(GPU CONV requires a **constant** filter). It is rewritten to a mathematically exact 4D form
(verified corr 1.000000):

- **modulation** — `conv(x, W·style) == conv(x · style_per_in_channel, W_const)` (conv is linear),
  so the style becomes an input channel-scale and the filter stays constant.
- **demodulation** — `rsqrt(Σ_{i,k}(W[o,i,k]·s[i])² + eps) == rsqrt((s²) @ Wsqᵀ + eps)` where
  `Wsq[o,i] = Σ_k W[o,i,k]²` is a constant `(c_out × c_in)` matrix — a small matmul + `RSQRT`.

Everything else in the clean arch is already GPU-friendly: upsampling is `RESIZE_BILINEAR`
(`align_corners=False`, no `TRANSPOSE_CONV`), no `GroupNorm`, noise fixed to stored buffers
(`randomize_noise=False`) for a deterministic graph. Build script: `build_gfpgan.py`.

## Run

1. Get `gfpgan_fp16.tflite` (Hugging Face `litert-community/GFPGAN-v1.4-LiteRT`, or build it).
2. Push it into the app's private storage (too large to bundle):
   ```bash
   ./scripts/install_to_device.sh <dir-with-the-tflite>
   ```
3. Build & install, then pick a face photo. The app detects the face (YuNet), FFHQ-aligns it, and
   shows a before/after slider.

## Face alignment (required for quality)

GFPGAN's StyleGAN prior mangles the mouth on off-template crops, so the app aligns first:
[YuNet](https://github.com/ShiqiYu/libfacedetection) (`yunet_fp16.tflite`, GPU ~4 ms) gives 5
landmarks, then a least-squares similarity transform warps the face to facexlib's 512 template
(`FaceAligner.kt`). Falls back to a center crop if no face is found. Push both `gfpgan_fp16.tflite`
and `yunet_fp16.tflite` (the install script handles both).

## fp16 note (Mali)

The StyleGAN2 modulated-conv **demod overflows fp16** — style vectors reach |s|~1000 so `Σ s²·Wsq`
hits ~2.3e6 (fp16 max 65504) → `rsqrt(inf)=0` → the decoder collapses to a flat color (it still
compiles 551/551 and runs). Fix: normalize the style by its per-image max before squaring — the
scale cancels exactly against the demod (`build_gfpgan.py`, verified device-vs-desktop identical).
