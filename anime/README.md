# AnimeGANv2 — Photo → Anime on-device (LiteRT GPU, fully GPU)

[AnimeGANv2](https://github.com/bryandlee/animegan2-pytorch) (bryandlee, MIT) photo-to-anime stylization,
running **fully on the LiteRT CompiledModel GPU**. Two styles — **paprika** (general anime) and
**face_paint_512_v2** (anime face portrait), each a ~4 MB fp16 graph; tap to switch.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **685 / 685** LITERT_CL (full residency) |
| inference | **~10 ms** (256×256) |
| fp16 size | ~4 MB per style |
| accuracy | device-vs-PyTorch corr **0.99996** (both styles) |

```
image[1,3,256,256] (RGB → [-1,1]) →[GPU: AnimeGANv2 Generator]→ stylized[1,3,256,256] ([-1,1])
```

## How it converts (litert-torch) — four numerically-exact re-authorings

1. **`ReflectionPad2d` → zero-pad** (reflect lowers to `GATHER_ND`; zero-pad = `PAD`; border-only difference).
2. **`GroupNorm(num_groups=1)` → SafeGroupNorm.** The native GroupNorm lowering emits `GATHER_ND`; the manual
   4D form reduces over (C,H,W) via three single-axis means in a down-scaled domain (fp16-safe, exact).
3. **Conv-weight scaling (GroupNorm scale-invariance).** The conv activations reach large magnitudes where the
   Mali delegate's fp16 conv accumulation loses precision → garbage at full residency. Because each conv is
   followed by a (scale-invariant) `GroupNorm`, scaling those conv weights down so the output is ≈ |10| is
   **exact** (the norm output is unchanged) and keeps the fp16 accumulation precise (same fix as Fast Neural
   Style).
4. **`F.interpolate(bilinear, align_corners=True)` → `align_corners=False`** (the delegate bans align_corners).

The upsample is bilinear (no transposed conv → no ZeroStuff). Result: banned ops NONE, all tensors ≤4D,
tflite-vs-torch corr **1.0**, device-vs-torch corr **0.99996**.

## Build & run

```bash
# weights via torch.hub (bryandlee/animegan2-pytorch): paprika, face_paint_512_v2
python scripts/build_anime.py all   # AN_STYLE=paprika (set per style); produces anime_<style>_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-the-style-tflites>
```

The first launch fails with "Model not found" until the styles are pushed.

**Preprocessing**: center-crop to square, resize to 256×256, RGB scaled to **[-1, 1]** (`x/127.5 - 1`), NCHW.
Output is [-1, 1] → `(x+1)·127.5` clamped to [0, 255].

Models: `litert-community/AnimeGANv2-LiteRT` (Hugging Face). Upstream:
[bryandlee/animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch) (MIT).
