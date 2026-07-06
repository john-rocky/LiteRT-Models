# ormbg — Open background removal (LiteRT GPU)

Real-time **background removal** running fully on the LiteRT `CompiledModel` GPU.
[ormbg](https://huggingface.co/schirrmacher/ormbg) is a fully **open, Apache-2.0**
foreground/alpha matte model (an ISNet trained for photorealistic subject cut-out) —
the permissively-licensed alternative to the non-commercial RMBG-1.4. ~10 ms/frame on
a Pixel 8a.

- **Model:** [schirrmacher/ormbg](https://huggingface.co/schirrmacher/ormbg) · Apache-2.0 · ISNet (RSU / U²-Net-style)
- **HF:** [litert-community/ormbg-LiteRT](https://huggingface.co/litert-community/ormbg-LiteRT)
- **Input:** `[1, 3, 1024, 1024]` NCHW, RGB, `x / 255`
- **Output:** `[1, 1, 1024, 1024]` alpha matte in `[0,1]` (min-max normalize per frame)
- **Size:** 176 MB · pure CNN

## GPU conversion

ormbg is a pure CNN (ISNet RSU blocks), so it converts fully GPU-compatible (**246/246
nodes on the delegate, 1 partition**; device corr 0.999881, ~10 ms) with **one
defensive patch**: `align_corners=True` → `False` on the bilinear upsamples (the GPU
delegate rejects `align_corners=True`). CPU-exact vs PyTorch (corr 0.9999999999).

## Build & run

```bash
cd ormbg/
./gradlew :app:installDebug
```

The 176 MB `ormbg.tflite` is bundled in `app/src/main/assets/` (build it with
`scripts/build_ormbg.py`; not committed). The camera view has its background replaced
with a studio color using the predicted alpha matte.

## Regenerate the model

```bash
pip install torch litert-torch huggingface_hub
python scripts/build_ormbg.py    # downloads schirrmacher/ormbg (Apache-2.0)
cp ormbg.tflite app/src/main/assets/
```

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
- Output is a raw matte — min-max normalize per frame before compositing.
