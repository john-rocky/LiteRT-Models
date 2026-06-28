# CPGA-Net — On-device low-light image enhancement (LiteRT GPU, fully GPU)

[CPGA-Net](https://github.com/Shyandram/CPGA-Net-Pytorch) (Shyandram, IJPRAI, MIT) — **low-light image
enhancement** (brighten dark photos) via Channel Prior + Gamma Correction, running **fully on the LiteRT
CompiledModel GPU**. At **0.025 M params / 0.1 MB fp16** it is the **smallest model in this repo**.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **135 / 135** LITERT_CL (full residency) |
| inference | **~2 ms** (256×256) |
| fp16 size | **0.1 MB** |
| accuracy | device-vs-PyTorch corr **0.99999**, no NaN |

```
image[1,3,256,256] (RGB [0,1]) →[GPU: CPGA-Net]→ enhanced[1,3,256,256] ([0,1])
```

## How it converts (litert-torch) — three numerically-exact fixes

1. **Gamma correction `x^γ` → `exp(γ·log x)`** — `torch.pow` lowers to the banned `POW`; the algebraic identity
   `x^γ = exp(γ·log x)` is exact (the base is clamped to [1e-9, 1]) and uses native `EXP`/`LOG`.
2. **CBAM / gamma global pools** — `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)`, `AdaptiveMaxPool2d(1)` →
   `F.max_pool2d(x, (H,W))` (the Mali multi-axis-pool fix; max-pool avoids the `amax` converter gap).
3. The dark/bright **channel prior** (`max`/`min` over RGB) stays as `REDUCE_MAX`/`REDUCE_MIN` (GPU-clean).

Result: banned ops NONE, all tensors ≤4D, tflite-vs-torch corr **1.0**, device-vs-torch corr **0.99999**. The
guided-filter post-process is disabled (`isdgf=False`).

## Build & run

```bash
# weights weights/enhance_color-llie-ResCBAM_g.pkl ship in the repo (MIT)
python scripts/build_cpga.py all      # produces cpga_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-cpga_fp16.tflite>
```

The first launch fails with "Model not found" until the model is pushed.

**App**: shows the enhanced image; **press-and-hold** to compare with the original. **Preprocessing**:
center-crop, resize 256×256, RGB scaled to [0,1].

Model: `litert-community/CPGA-Net-LowLight-LiteRT` (Hugging Face). Upstream:
[Shyandram/CPGA-Net-Pytorch](https://github.com/Shyandram/CPGA-Net-Pytorch) (MIT).
