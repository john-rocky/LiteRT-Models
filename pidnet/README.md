# PIDNet-S — Real-time semantic segmentation (LiteRT GPU)

Real-time **semantic segmentation** running fully on the **LiteRT `CompiledModel`
GPU** delegate. [PIDNet-S](https://arxiv.org/abs/2206.02066) (CVPR 2023) segments a
road scene into the **19 Cityscapes classes** (road, sidewalk, building, car,
person, sky, …) at ~17 FPS on a Pixel 8a.

- **Model:** [XuJiacong/PIDNet](https://github.com/XuJiacong/PIDNet) · MIT · 78.8% mIoU (Cityscapes val)
- **Input:** `[1, 3, 1024, 1024]` NCHW, RGB, ImageNet-normalized
- **Output:** `[1, 19, 128, 128]` class logits (1/8 res; argmax + upscale in the app)
- **Size:** 30 MB · ~7.6 M params · pure CNN

## GPU conversion

PIDNet is a three-branch CNN (P: detail, I: context, D: boundary) — no attention,
no dynamic shapes at a fixed input size, and `align_corners=False` on every bilinear
resize. So it converts to a **fully GPU-compatible graph with zero patches**:
`CONV_2D` ×75, `RESIZE_BILINEAR` ×11 (align_corners=False), `AVERAGE_POOL_2D`,
`ADD`/`MUL`/`SUB`/`SUM`, `LOGISTIC` — **0 tensors of rank > 4, 0 GPU-incompatible
ops**. The converted graph matches the original PyTorch model bit-for-bit on CPU
(corr 0.99999999999, 100% argmax); on the Mali GPU (fp16) it agrees with the fp32
reference at 97% of pixels with correct classes.

## Build & run

```bash
cd pidnet/
./gradlew :app:installDebug
```

The 30 MB `pidnet_s.tflite` is bundled in `app/src/main/assets/` (build it with
`scripts/build_pidnet.py`; the file is not committed). Point the camera at a street /
outdoor scene — the Cityscapes-colored segmentation is overlaid on the preview with
per-frame latency. (The model is trained on street scenes, so road/outdoor views
give the most meaningful labels.)

## Regenerate the model

```bash
pip install torch litert-torch onnx huggingface_hub
git clone https://github.com/XuJiacong/PIDNet.git
PIDNET_REPO=./PIDNet python scripts/build_pidnet.py
cp pidnet_s.tflite app/src/main/assets/
```

`scripts/build_pidnet.py` loads the trained PIDNet-S weights (from an ONNX mirror
whose initializer names match the original repo's PyTorch keys) and converts with
litert-torch.

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
- Colors follow the official Cityscapes palette (`CityscapesPalette.kt`).
