# Fast Neural Style Transfer on-device (LiteRT GPU, fully GPU)

Fast neural **style transfer** (the [PyTorch examples](https://github.com/pytorch/examples/tree/main/fast_neural_style)
`TransformerNet`, Johnson et al.) — applies an artistic style to a photo, **fully on the LiteRT CompiledModel
GPU**. Four styles (**candy / mosaic / rain_princess / udnie**), each a **3.5 MB** fp16 graph; tap to switch.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **350 / 350** LITERT_CL (full residency) |
| inference | **~9 ms** (256×256) |
| fp16 size | 3.5 MB per style |
| accuracy | device-vs-PyTorch corr **0.9998–0.9999** (all 4 styles) |

```
image[1,3,256,256] (RGB 0-255) →[GPU: TransformerNet]→ stylized[1,3,256,256] (RGB 0-255)
```

## How it converts (litert-torch) — three numerically-exact re-authorings

1. **ReflectionPad2d → zero-pad.** `nn.ReflectionPad2d` lowers to `GATHER_ND` (reflect indices, banned).
   Folding a zero-pad into each conv removes it (border-only cosmetic difference).
2. **Large conv activations → conv-weight scaling (InstanceNorm scale-invariance).** The conv outputs reach
   ≈ |5000|, where the Mali delegate's fp16 conv accumulation loses precision → garbage (device corr 0.34,
   full residency — *residency ≠ correctness*). Because each conv is followed by an `InstanceNorm` (which is
   **scale-invariant**), scaling those conv weights/bias down so the output is ≈ |10| is **exact** (the
   InstanceNorm output is unchanged) and keeps the fp16 accumulation precise → corr 1.0.
3. **InstanceNorm → SafeInstanceNorm.** Its spatial mean/variance over 256×256 overflows fp16; reducing in a
   down-scaled domain (two single-axis means) is fp16-safe and exact (same class as SafeLayerNorm).

The upsample is `F.interpolate(nearest)` (not a transposed conv) → no ZeroStuff. Result: banned ops NONE, all
tensors ≤4D, tflite-vs-torch corr **1.0**, device-vs-torch corr **0.9999**.

## Build & run

```bash
git clone https://github.com/pytorch/examples   # neural_style/transformer_net.py + saved_models
python scripts/build_style.py all               # ST_STYLE=candy (set per style); produces style_<style>_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-the-4-style-tflites>
```

The first launch fails with "Model not found" until the styles are pushed.

**Preprocessing**: center-crop to square, resize to 256×256, RGB **0–255** (no normalization), NCHW. Output is
0–255 RGB (clamp).

Models: `litert-community/Fast-Neural-Style-LiteRT` (Hugging Face). Upstream:
[pytorch/examples](https://github.com/pytorch/examples/tree/main/fast_neural_style) (BSD-3-Clause).
