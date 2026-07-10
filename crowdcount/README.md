# DM-Count — Crowd counting (LiteRT GPU)

Real-time **crowd counting** running fully on the LiteRT `CompiledModel` GPU.
[DM-Count](https://github.com/cvlab-stonybrook/DM-Count) (NeurIPS 2020) regresses a person
**density map** — its sum is the crowd size — so it counts hundreds of people where
detector-based counting saturates.

- **Model:** [cvlab-stonybrook/DM-Count](https://github.com/cvlab-stonybrook/DM-Count) (UCF-QNRF) · MIT · VGG19
- **HF:** [litert-community/DM-Count-Crowd-LiteRT](https://huggingface.co/litert-community/DM-Count-Crowd-LiteRT)
- **Input:** `[1, 3, 512, 512]` NCHW, RGB, ImageNet-normalized
- **Output:** `[1, 1, 64, 64]` density map — `sum(map)` = person count
- **Size:** 86 MB · pure CNN

## GPU conversion

DM-Count is a pure CNN (VGG19 + conv regression head) → fully GPU-compatible (**30/30 nodes
on the delegate, 1 partition**; device corr 0.99998, count within 0.4% of PyTorch) with **one
exact rewrite**: the mid-graph `F.upsample_bilinear` (align_corners=True, banned as
`RESIZE_BILINEAR` on the delegate) is a linear operator, so it is re-authored as two
constant-matrix multiplies (→ `FULLY_CONNECTED`; the constant must be on the RHS — the
delegate rejects `BATCH_MATMUL` with a constant LHS). Desktop corr vs PyTorch is 1.000000.

## Build & run

```bash
cd crowdcount/
./gradlew :app:installDebug
```

The 86 MB `dmcount.tflite` is bundled in `app/src/main/assets/` (build it with
`scripts/build_dmcount.py`; not committed). Point the camera at a crowd — the density
heatmap is overlaid and the estimated person count is shown.

## Regenerate the model

```bash
pip install torch litert-torch numpy
cd scripts/
git clone https://github.com/cvlab-stonybrook/DM-Count.git dmcount-src   # weights via git-lfs
python build_dmcount.py
cp dmcount.tflite ../app/src/main/assets/
```

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
- Weights: `model_qnrf.pth` (UCF-QNRF) generalizes best; `model_nwpu.pth` is also bundled
  in the upstream repo if you want the NWPU-Crowd variant.
- The count is resolution-dependent: very small far-away heads beyond the 512×512 input
  resolution are undercounted, as with any fixed-input crowd counter.
