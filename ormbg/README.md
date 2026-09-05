# ormbg — Open background removal (LiteRT GPU)

Real-time **background removal** running fully on the LiteRT `CompiledModel` GPU.
[ormbg](https://huggingface.co/schirrmacher/ormbg) is a fully **open, Apache-2.0**
foreground/alpha matte model (an ISNet trained for photorealistic subject cut-out) —
the permissively-licensed alternative to the non-commercial RMBG-1.4. 246 ms/frame on a
Pixel 8a (2026-09-05, run + readback — see [INTEGRATION.md](INTEGRATION.md); the “~10 ms”
quoted here earlier timed only the asynchronous `run()`).

- **Model:** [schirrmacher/ormbg](https://huggingface.co/schirrmacher/ormbg) · Apache-2.0 · ISNet (RSU / U²-Net-style)
- **HF:** [litert-community/ormbg-LiteRT](https://huggingface.co/litert-community/ormbg-LiteRT)
- **Input:** `[1, 3, 1024, 1024]` NCHW, RGB, `x / 255`
- **Output:** `[1, 1, 1024, 1024]` alpha matte in `[0,1]` (min-max normalize per frame)
- **Size:** 176 MB · pure CNN

## GPU conversion

ormbg is a pure CNN (ISNet RSU blocks), so it converts fully GPU-compatible (**246/246
nodes on the delegate, 1 partition**; device corr 0.999881) with **one
defensive patch**: `align_corners=True` → `False` on the bilinear upsamples (the GPU
delegate rejects `align_corners=True`). CPU-exact vs PyTorch (corr 0.9999999999).

## Add it to your own app

[INTEGRATION.md](INTEGRATION.md) is the recipe for an existing app: the Gradle dependency, the
drop-in [`BgRemover.kt`](app/src/main/java/com/ormbg/BgRemover.kt), the model download with its
checksum, pre/post-processing, cancel and release, and the on-device check with the values it
should print. [`recipe.json`](recipe.json) carries the same facts in machine-readable form.

## Build & run

```bash
cd ormbg/
./gradlew :app:installGpuDebug          # camera demo (gpu flavor)
./gradlew :app:connectedGpuDebugAndroidTest   # integration check on the connected device
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

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.2.0`.
- Output is a raw matte — min-max normalize per frame before compositing.

### Converting your own fine-tuned checkpoint

Only the defensive align_corners patch touches the graph, so a checkpoint from the
official ORMBG trainer converts directly (defaults reproduce the official ship exactly):

```bash
ORMBG_CKPT=/path/to/my_ormbg.pth python scripts/build_ormbg.py
```

Output stays a `[1, 1, R, R]` sigmoid mask; `ORMBG_RES` changes the fixed input size
(default 1024 — the app's scaling must follow).
