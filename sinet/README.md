# SINet-V2 — Camouflaged / concealed object detection (LiteRT GPU)

Real-time **camouflaged object detection** running fully on the LiteRT `CompiledModel` GPU.
[SINet-V2](https://github.com/GewelsJI/SINet-V2) (TPAMI 2022) finds objects that **blend into
their background** — hidden animals, concealed items, defect/polyp-style targets — where
ordinary segmentation fails.

- **Model:** [GewelsJI/SINet-V2](https://github.com/GewelsJI/SINet-V2) (COD10K) · Apache-2.0 · Res2Net-50
- **HF:** [litert-community/SINet-V2-Camouflage-LiteRT](https://huggingface.co/litert-community/SINet-V2-Camouflage-LiteRT)
- **Input:** `[1, 3, 352, 352]` NCHW, RGB, ImageNet-normalized
- **Output:** `[1, 1, 352, 352]` sigmoid map — high = concealed object
- **Size:** 100 MB · pure CNN

## GPU conversion

SINet-V2 is a pure CNN (Res2Net backbone + conv decoder) → fully GPU-compatible (**2447/2447
nodes on the delegate, 1 partition**; device corr 0.994) with **two patches**: ZeroPadMaxPool
for the Res2Net stem (`-inf` PADV2) and `align_corners=True` → `False`. CPU-exact vs PyTorch (corr 0.997).

## Build & run

```bash
cd sinet/
./gradlew :app:installDebug
```

The 100 MB `sinet.tflite` is bundled in `app/src/main/assets/` (build it with
`scripts/build_sinet.py`; not committed). Point the camera at a scene — concealed / hard-to-see
objects are highlighted in red.

## Regenerate the model

```bash
pip install torch litert-torch gdown numpy
git clone https://github.com/GewelsJI/SINet-V2.git sinet-src
python scripts/build_sinet.py    # fetches the Apache-2.0 Net_epoch_best.pth (Res2Net)
cp sinet.tflite app/src/main/assets/
```

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
- Res2Net stem needs ZeroPadMaxPool; the model uses `imagenet_pretrained=False` (full checkpoint loaded).
