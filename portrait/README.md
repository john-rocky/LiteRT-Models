# U²-Net Portrait — Photo → pencil line drawing (LiteRT GPU)

Real-time **portrait sketch generation** running fully on the LiteRT `CompiledModel` GPU.
The [U²-Net](https://github.com/xuebinqin/U-2-Net) portrait model turns a face photo into a
**hand-drawn pencil line portrait** — a fun creative / AR filter. ~12 ms/frame on a Pixel 8a.

- **Model:** [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net) (`u2net_portrait`) · Apache-2.0 · U²-Net
- **HF:** [litert-community/U2Net-Portrait-Sketch-LiteRT](https://huggingface.co/litert-community/U2Net-Portrait-Sketch-LiteRT)
- **Input:** `[1, 3, 512, 512]` NCHW, RGB, `x/255` then ImageNet-normalized
- **Output:** `[1, 1, 512, 512]` in `[0,1]` → min-max normalize, invert (`1−x`) for dark strokes on white
- **Size:** 176 MB · pure CNN

## GPU conversion

U²-Net is a pure CNN → fully GPU-compatible (**893/893 nodes on the delegate, 1
partition**; device corr 0.998683, ~12 ms) with **one defensive patch**: `align_corners=True`
→ `False`. CPU-exact vs PyTorch (corr 1.0).

## Build & run

```bash
cd portrait/
./gradlew :app:installDebug
```

The 176 MB `portrait.tflite` is bundled in `app/src/main/assets/` (build it with
`scripts/build_portrait.py`; not committed). Center your face — a live pencil portrait is
drawn (live preview in the corner).

## Regenerate the model

```bash
pip install torch litert-torch huggingface_hub numpy
git clone https://github.com/xuebinqin/U-2-Net.git u2net-src
python scripts/build_portrait.py    # loads the Apache-2.0 u2net_portrait weights
cp portrait.tflite app/src/main/assets/
```

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
- Works best on a centered face. Output is a soft map — min-max normalize then invert for the sketch.
