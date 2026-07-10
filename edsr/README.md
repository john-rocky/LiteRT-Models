# EDSR (×4) — Super-resolution (LiteRT GPU)

Real-time **×4 single-image super-resolution** running fully on the LiteRT `CompiledModel`
GPU. [EDSR](https://arxiv.org/abs/1707.02921) (CVPR 2017 winner) upscales a low-res image
4× with sharp detail. First super-resolution model in the zoo. ~23 ms/frame on a Pixel 8a.

- **Model:** [eugenesiow/edsr-base](https://huggingface.co/eugenesiow/edsr-base) (super-image, DIV2K) · Apache-2.0
- **HF:** [litert-community/EDSR-x4-LiteRT](https://huggingface.co/litert-community/EDSR-x4-LiteRT)
- **Input:** `[1, 3, 128, 128]` NCHW, RGB, `x/255`
- **Output:** `[1, 3, 512, 512]` NCHW, RGB 0–1 (clamp, ×255)
- **Size:** 7.7 MB · pure CNN

## GPU conversion

EDSR is a pure CNN, but its **PixelShuffle** sub-pixel upsampler lowers to rank-5/6
reshapes that the Mali delegate rejects (the classic super-resolution wall). The exact
fix: **PixelShuffle(r) ≡ a fixed-weight grouped-identity `ConvTranspose2d(stride=r)`**,
then converted with **ZeroStuffConvT2d** (nearest-upsample + stride zero-stuff mask +
flipped conv). Result: **68/68 nodes on the delegate, 1 partition**; device corr 0.999946,
~23 ms. CPU-exact vs PyTorch (corr 1.0). This patch also unblocks other PixelShuffle SR
models (Real-ESRGAN, etc.).

## Build & run

```bash
cd edsr/
./gradlew :app:installDebug
```

The 7.7 MB `edsr.tflite` is bundled in `app/src/main/assets/` (build it with
`scripts/build_edsr.py`; not committed). The center region of the camera is super-resolved
×4 and shown full-screen (live preview in the corner).

## Regenerate the model

```bash
pip install torch litert-torch super-image numpy
python scripts/build_edsr.py    # loads the Apache-2.0 EDSR-base ×4 weights + PixelShuffle patch
cp edsr.tflite app/src/main/assets/
```

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
- Fixed 128→512 graph — tile larger images into 128 blocks, or resize the region to 128.
