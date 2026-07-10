# MODNet — Trimap-free portrait matting (LiteRT GPU)

Real-time **portrait matting** running fully on the **LiteRT `CompiledModel` GPU**
delegate. [MODNet](https://arxiv.org/abs/2011.11961) (AAAI 2022) predicts a **soft
alpha matte** for a person — no trimap, no green screen — for background
blur/replace (video calls, virtual backgrounds). ~79 ms/frame on a Pixel 8a.

- **Model:** [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet) · Apache-2.0 · MobileNetV2 backbone
- **Input:** `[1, 3, 512, 512]` NCHW, RGB, normalized to `[-1, 1]` (`(x/255 - 0.5)/0.5`)
- **Output:** `[1, 1, 512, 512]` soft alpha matte in `[0, 1]`
- **Size:** 26 MB · ~6.5 M params · pure CNN

## GPU conversion

MODNet is a pure CNN (MobileNetV2 low-res branch + high-res + fusion branches) with
`align_corners=False` interpolation. Two re-authoring patches make it a fully
GPU-compatible graph (**0 tensors of rank > 4, 0 banned ops**):

1. **SE block `Linear` → `1×1 conv`.** The stock squeeze-excite does
   `pool → Linear → view(b,c,1,1) → x * w`; the 2D-Linear→4D-reshape confuses the
   NCHW↔NHWC layout and fails to convert (`mul` broadcast mismatch). Expressing the
   FC as 1×1 convs on the pooled `[b,c,1,1]` tensor is mathematically identical and
   NCHW-clean.
2. **fp16-safe `InstanceNorm` (hierarchical mean).** MODNet's IBNorm runs
   `InstanceNorm2d` over up to 512×512 spatial; on the Mali GPU (fp16) the variance
   `sum(dd²)` overflows (≫ 65504) and the matte degrades badly (halos, blotchy
   interior — corr 0.94). Computing the spatial mean via a cascade of `/2`
   average-pools (each stage averages 4 values → magnitude-bounded, exact for
   power-of-2 sizes) and `dd·rsqrt(mean(dd²)+eps)` fixes it — GPU corr **0.99994**,
   clean edges, matching the fp32 reference.

CPU-exact vs PyTorch (corr 0.99999999999); device Mali GPU corr 0.99994.

## Build & run

```bash
cd modnet/
./gradlew :app:installDebug
```

The 26 MB `modnet.tflite` is bundled in `app/src/main/assets/` (build it with
`scripts/build_modnet.py`; the file is not committed). Point the front/back camera at
a person — the foreground is composited over a replaceable background. Tap the screen
to cycle background colors.

## Regenerate the model

```bash
pip install torch litert-torch huggingface_hub
git clone https://github.com/ZHKKKe/MODNet.git
MODNET_REPO=./MODNet python scripts/build_modnet.py
cp modnet.tflite app/src/main/assets/
```

`scripts/build_modnet.py` loads the trained MODNet weights (HF mirror
`DavG25/modnet-pretrained-models`), applies the two patches, and converts with
litert-torch.

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
