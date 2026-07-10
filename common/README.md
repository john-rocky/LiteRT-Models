# common/ — canonical shared utilities

Each module in this repo is an independent Gradle project and must stay standalone
(a sample can be copied out and built on its own, with all of its code readable in
place). Shared Kotlin utilities are therefore distributed as **vendored copies**,
not as a shared Gradle module: the canonical source lives here, every module keeps
its own physical copy, and `tools/sync_common.py` keeps the copies byte-identical
(only the `package` line differs — the canonical carries the
`package __MODULE_PACKAGE__` placeholder).

```bash
python tools/sync_common.py --check   # CI-style drift check (exit 1 on divergence)
python tools/sync_common.py --apply   # push the canonical out to all copies
```

Rules:

1. **Fix bugs in the canonical first**, then `--apply`. Never patch a single
   module's copy in place — that is how the 4 diverged `MelSpectrogram.kt`
   variants happened.
2. **Model-specific parameters go in constructor arguments**, not edits to the
   copy (e.g. normalization mean/std, mel bin count).
3. Task-specific visualization (overlay views) is intentionally NOT shared —
   it is the part of a sample readers most want to see whole.

## Canonical files

| File | What it provides | Copies |
|---|---|---|
| `kotlin/RealtimeCameraPipeline.kt` | CameraX two-thread capture loop, RGBA `ImageProxy` → pooled `Bitmap` (incl. rotation), FPS counter | ~22 modules |
| `kotlin/CompiledModelRunner.kt` | CompiledModel GPU lifecycle (create from assets/file, buffers, run, close-all incl. `TensorBuffer`s); KDoc captures the undocumented gotchas (async `run`, no CPU fallback, buffer ping-pong) | new — adopt in next module |
| `kotlin/ImageTensor.kt` | Bitmap → NCHW/NHWC float tensor, parameterized mean/std / RGB-BGR / 0-1 vs 0-255, stretch + letterbox with coordinate `Mapping` back to source | new — adopt in next module |
| `kotlin/AudioCapture.kt` | 16 kHz mono `AudioRecord` daemon loop delivering normalized float chunks | new — adopt in next module |
| `kotlin/MathOps.kt` | sigmoid, in-place softmax, argmax, IoU, greedy NMS on flat xyxy arrays | new — adopt in next module |

Known divergence: `yolox/` carries a trimmed 238-line `RealtimeCameraPipeline`
variant that predates the canonical. Re-unify it only together with an on-device
re-verification of the yolox sample.

New canonical files (`CompiledModelRunner`, `ImageTensor`, `AudioCapture`,
`MathOps`) are extracted from the recurring per-module patterns but have not been
vendored anywhere yet — the first module that adopts them is their compile +
device verification. Existing modules migrate opportunistically, never in bulk.

## Planned (extract next)

- `MelSpectrogram.kt` — unify the 4 diverged copies (whisper / parakeet /
  voiceassistant / panns) behind explicit mel parameters; requires per-module
  device re-verification before switching.
