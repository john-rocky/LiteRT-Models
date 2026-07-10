# BiSeNet — Face parsing (LiteRT GPU)

Real-time **face parsing** running fully on the LiteRT `CompiledModel` GPU.
[BiSeNet](https://arxiv.org/abs/1808.00897) (zllrunning/face-parsing.PyTorch) segments
a face into the **19 CelebAMask-HQ classes** (skin, brows, eyes, nose, lips, ears,
hair, hat, glasses, neck, cloth, …) — for AR / beauty / makeup. ~22 ms/frame on a Pixel 8a.

- **Model:** [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) · MIT · ResNet18 backbone
- **Input:** `[1, 3, 512, 512]` NCHW, RGB, ImageNet-normalized
- **Output:** `[1, 19, 512, 512]` class logits (argmax → face-part map)
- **Size:** 53 MB · ~13.3 M params · pure CNN

## GPU conversion

BiSeNet is a pure CNN (ResNet18 + context path + feature-fusion). Three re-authoring
patches make it a **fully GPU-compatible graph — 74/74 nodes on the delegate, 1
partition** (device-verified corr 0.99999, argmax 99.96% vs PyTorch):

1. **`align_corners=True` → `False`** — the output upsamples use `align_corners=True`,
   the resize form the GPU delegate rejects (1.6% argmax change vs original).
2. **global `avg_pool2d(x, x.size()[2:])` → `mean([2,3])`** — the ARM/context/fusion
   modules do global pooling with a full-spatial kernel, which the Mali delegate
   rejects as an `AVERAGE_POOL_2D`; a MEAN reduce (like `AdaptiveAvgPool2d(1)`) is
   supported.
3. **zero-pad maxpool** — the ResNet stem `MaxPool2d(padding=1)` lowers to a PADV2
   with `-inf` padding (`PADV2: src has wrong size` on Mali); replacing it with an
   explicit 0-pad + unpadded maxpool is exact (the input is post-ReLU ≥ 0).

These are on-device-only rejections — the op inventory looks clean and CPU parity is
1.0, but the GPU delegate fails to compile without them. CPU-exact vs PyTorch
(corr 0.99999999999).

## Build & run

```bash
cd faceparsing/
./gradlew :app:installDebug
```

The 53 MB `faceparsing.tflite` is bundled in `app/src/main/assets/` (build it with
`scripts/build_faceparsing.py`; the file is not committed). The front camera feeds a
selfie; the 19-class face parsing is overlaid on the preview.

## Regenerate the model

```bash
pip install torch litert-torch huggingface_hub
git clone https://github.com/zllrunning/face-parsing.PyTorch.git fp
FP_REPO=./fp python scripts/build_faceparsing.py
cp faceparsing.tflite app/src/main/assets/
```

`scripts/build_faceparsing.py` loads the trained BiSeNet weights (HF mirror
`AI2lab/face-parsing.PyTorch`), applies the three patches, and converts with litert-torch.

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
