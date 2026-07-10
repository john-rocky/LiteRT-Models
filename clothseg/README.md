# Cloth Segmentation (U²-Net) — LiteRT GPU

Real-time **clothing segmentation** running fully on the LiteRT `CompiledModel` GPU.
[cloth-segmentation](https://github.com/levindabhi/cloth-segmentation) is a U²-Net trained
on iMaterialist-Fashion to segment **upper-body / lower-body / full-body clothing** — the
building block for virtual try-on and fashion apps. ~88 ms/frame on a Pixel 8a.

- **Model:** [levindabhi/cloth-segmentation](https://github.com/levindabhi/cloth-segmentation) (iMaterialist-Fashion) · MIT · U²-Net
- **HF:** [litert-community/Cloth-Segmentation-U2Net-LiteRT](https://huggingface.co/litert-community/Cloth-Segmentation-U2Net-LiteRT)
- **Input:** `[1, 3, 768, 768]` NCHW, RGB, `(x/255 - 0.5)/0.5`
- **Output:** `[1, 4, 768, 768]` logits — argmax: 0 bg, 1 upper, 2 lower, 3 full body
- **Size:** 176 MB · pure CNN

## GPU conversion

U²-Net is a pure CNN → fully GPU-compatible (**254/254 nodes on the delegate, 1
partition**; device corr 0.999798, ~88 ms) with **one defensive patch**: `align_corners=True`
→ `False` on the bilinear upsamples. CPU-exact vs PyTorch (corr 1.0).

## Build & run

```bash
cd clothseg/
./gradlew :app:installDebug
```

The 176 MB `clothseg.tflite` is bundled in `app/src/main/assets/` (build it with
`scripts/build_clothseg.py`; not committed). Point the camera at a person — upper-body
clothing is shaded cyan, lower-body orange, full-body dresses magenta.

## Regenerate the model

```bash
pip install torch litert-torch huggingface_hub numpy
git clone https://github.com/levindabhi/cloth-segmentation.git
python scripts/build_clothseg.py    # loads the MIT U2NET cloth weights (strip module. prefix!)
cp clothseg.tflite app/src/main/assets/
```

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
- ⚠ The checkpoint has a `module.` prefix — strip it before `load_state_dict` (else random weights → garbage).
