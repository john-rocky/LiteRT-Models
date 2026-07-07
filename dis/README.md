# DIS (IS-Net, general-use) — High-precision object cutout (LiteRT GPU)

Real-time **dichotomous image segmentation** running fully on the LiteRT `CompiledModel`
GPU. [DIS](https://github.com/xuebinqin/DIS) (ECCV 2022) is a high-accuracy IS-Net that cuts
out the main object with **fine structure detail** (thin stems, petals, wires, handles) —
for e-commerce product photos and graphics. ~11 ms/frame on a Pixel 8a.

- **Model:** [xuebinqin/DIS](https://github.com/xuebinqin/DIS) `isnet-general-use` · Apache-2.0 · IS-Net
- **HF:** [litert-community/DIS-ISNet-LiteRT](https://huggingface.co/litert-community/DIS-ISNet-LiteRT)
- **Input:** `[1, 3, 1024, 1024]` NCHW, RGB, `x/255 - 0.5`
- **Output:** `[1, 1, 1024, 1024]` sigmoid alpha (0–1)
- **Size:** 176 MB · pure CNN

## GPU conversion

DIS is a pure CNN (IS-Net RSU blocks) → fully GPU-compatible (**247/247 nodes on the
delegate, 1 partition**; device max|diff| 0.00034, ~11 ms) with **one defensive patch**:
`align_corners=True` → `False`. CPU-exact vs PyTorch (max|diff| 0.0).

## Build & run

```bash
cd dis/
./gradlew :app:installDebug
```

The 176 MB `dis.tflite` is bundled in `app/src/main/assets/` (build it with
`scripts/build_dis.py`; not committed). The camera subject is cut out onto a studio
background using the predicted alpha.

## Regenerate the model

```bash
pip install torch litert-torch huggingface_hub numpy
git clone https://github.com/xuebinqin/DIS.git dis-src
python scripts/build_dis.py    # loads the Apache-2.0 isnet-general-use weights
cp dis.tflite app/src/main/assets/
```

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
- Preprocessing `x/255 - 0.5` (not /255). Output is a soft alpha — threshold/composite as needed.
