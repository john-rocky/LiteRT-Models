# MI-GAN — On-device image inpainting / object removal (LiteRT GPU, fully GPU)

[MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) (Picsart AI Research, **ICCV 2023**, MIT) — a "magic
eraser": paint over an object, and it is removed and the region inpainted, **fully on the LiteRT CompiledModel
GPU**. A mobile-designed StyleGAN-style generator (separable convs, nearest-upsample, **no norm**).

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **509 / 509** LITERT_CL (full residency) |
| inference | **~6 ms** (512×512) |
| fp16 size | 16.3 MB |
| accuracy | device-vs-PyTorch corr **0.99998**, no NaN |

```
in[1,4,512,512] = concat(mask-0.5, rgb·mask)  →[GPU: MI-GAN]→  out[1,3,512,512] (inpainted, [-1,1])
```

## I/O contract

- **Input** (4 ch): `concat(mask − 0.5, rgb · mask)`, where `rgb` ∈ [−1,1] (`pixel/127.5 − 1`) and `mask` is
  **1 = keep, 0 = erase** (the hole to fill).
- **Output** (3 ch): the generated image in [−1,1]. Composite back as `rgb·mask + out·(1−mask)` so only the
  erased region changes.

## How it converts (litert-torch) — clean in one shot

The MI-GAN **inference** generator (`migan_inference.py`, the re-parametrized mobile model) is already
GPU-friendly: depthwise-separable `Conv2d`, `nn.Upsample(nearest)` + a fixed FIR-filter grouped conv, leaky-ReLU
with gain/clamp (→ `MAXIMUM`/`MINIMUM`), and **no normalization layers** (StyleGAN-style). No re-authoring was
needed: banned ops NONE, all tensors ≤4D, tflite-vs-torch corr **1.0**, device-vs-torch corr **0.99998**. (This
is the FFT-free, norm-free generator lane — contrast the InstanceNorm/GroupNorm generators that need the
SafeNorm + conv-scaling fixes.)

## Build & run

```bash
# weights migan_512_places2.pt from the MI-GAN repo's Google-Drive (or HF mirror d4rkk3y/migan for 256)
python scripts/build_migan.py all     # MG_RES=512; produces migan_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-migan_fp16.tflite>
```

The first launch fails with "Model not found" until the model is pushed.

**App**: paint over an object (finger), tap **✨ Erase**, the region is inpainted on device. **Reset** restores
the loaded image. **Preprocessing**: center-crop, resize 512×512.

Model: `litert-community/MI-GAN-512-Places2-LiteRT` (Hugging Face). Upstream:
[Picsart-AI-Research/MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) (MIT).
