# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository nature

Despite the directory name `depthanything-android`, this repo is the **LiteRT-Models** zoo: a collection of standalone Android sample apps that each demonstrate one converted on-device model running on **LiteRT `CompiledModel` GPU (ML Drift)**, plus the Python conversion scripts that produced the model files. The DepthAnything V2 demo lives in `sample-app/`; the root `app/` is also a depth demo. Everything else is grouped by model family.

The README at the repo root is the canonical user-facing index of every model, its download URL, input/output shape, preprocessing, and which sample app demonstrates it. When adding or changing a model, update the README too.

## Repository layout

Each top-level directory listed below is **its own independent Gradle project** with its own `build.gradle.kts`, `gradlew`, `settings.gradle.kts`, and an `:app` module — there is no multi-module setup, no shared library, and no root Gradle build that includes them. Open the specific module's directory in Android Studio (or run its own `./gradlew`) to build it.

| Directory          | Android `applicationId`     | What it demos                                       |
|--------------------|-----------------------------|-----------------------------------------------------|
| `app/`             | `com.depthanything`         | Depth Anything V2 (root, Compose)                   |
| `sample-app/`      | `com.depthanything` (sample)| Depth Anything V2 with NCNN/native variants too     |
| `yolo/`            | `com.yolo`                  | YOLO11n / YOLO26n object detection                  |
| `yolo-pose/`       | —                           | YOLO pose estimation                                |
| `mobilesam/`       | `com.mobilesam`             | MobileSAM tap-to-segment (TFLite encoder + ONNX decoder) |
| `rmbg/`            | —                           | RMBG-1.4 background removal                         |
| `lama/`            | —                           | LaMa-Dilated inpainting                             |
| `clip/`            | —                           | CLIP ViT-B/32 zero-shot classification              |
| `dsine/`           | —                           | DSINE surface normal estimation                     |
| `whisper/`         | —                           | Whisper-tiny speech recognition                     |
| `kokoro/`          | —                           | Kokoro-82M TTS (EN+JA)                              |
| `smolvlm/`         | —                           | SmolVLM-256M vision-language model                  |
| `voiceassistant/`  | `com.voiceassistant`        | Whisper → SmolLM2 → Kokoro pipeline + Silero VAD    |
| `real-esrgan/`     | —                           | Real-ESRGAN x4v3 super resolution                   |
| `moge/`            | `com.moge`                  | MoGe-2 ViT-S monocular geometry (points, normals, depth, 3D) |
| `yolo-tracking/`   | `com.yolotracking`          | YOLO11n + OSNet x0.25 DeepSORT multi-object tracking |
| `musicgen/`        | —                           | (work in progress)                                  |

Each module's `app/src/main/java/com/<module>/` typically contains a `MainActivity.kt`, a model wrapper (e.g. `ObjectDetector.kt`, `MobileSAMSegmenter.kt`, `WhisperTranscriber.kt`), and a custom view for visualization. The voice assistant deliberately bundles its own copies of `WhisperTranscriber.kt`, `KokoroSynthesizer.kt`, etc. instead of cross-referencing other modules — see `voiceassistant/README.md` for the rationale (each module must remain standalone and independently runnable).

Shared at the repo root:
- `litert_gpu_toolkit/` — Python package wrapping litert-torch with the GPU-compatibility patches catalogued in `docs/LITERT_CONVERSION_GUIDE.md`. Entry point: `convert_for_gpu(model, dummy_input, output_path)`.
- `scripts/` — Older one-off conversion scripts for the Depth Anything V2 model (`convert_models.py`, `convert_keras_native.py`, etc.) plus `requirements.txt` for the conversion environment.
- `docs/LITERT_CONVERSION_GUIDE.md` — Practical findings on which converter (litert-torch / onnx2tf / SavedModel / native Keras) to use for each architecture, plus the catalog of GPU-incompatible op fixes (GELU → SigmoidGELU, GroupNorm → manual 4D, Conv2d_WS → baked weights, etc.).

## Build and run

There is **no root build**. Each module is built independently:

```bash
cd <module>/                # e.g. cd mobilesam/
./gradlew :app:assembleDebug
./gradlew :app:installDebug  # install to a connected device
```

The user builds and tests on a real device — do not run build/test commands speculatively to "verify" Android code (per global instructions). Just edit and let the user build.

`minSdk = 26`, `compileSdk = 35`, Kotlin JVM target 17, AGP 8.7.3, Kotlin 2.1.10, `arm64-v8a` only. Every module pins LiteRT `com.google.ai.edge.litert:litert:2.1.3`. Modules that also use ONNX add `com.microsoft.onnxruntime:onnxruntime-android:1.24.3`.

## Model files: how they get to the device

Model files are **never committed** (CoreML, `.tflite` over a few MB, build outputs, etc. are gitignored / excluded by user policy). Each sample app has one of two patterns:

1. **Small models** — bundled in `app/src/main/assets/` and loaded with `CompiledModel.create(context.assets, "model.tflite", options, null)`. `aaptOptions { noCompress += "tflite" }` is set so the file is mmappable.
2. **Large models (>~150 MB)** — staged via `adb push /data/local/tmp/...`, then moved into the app's private `filesDir` by an `install_to_device.sh` script (see `voiceassistant/scripts/install_to_device.sh`, `kokoro/scripts/install_to_device.sh`, etc.). The app loads them with the file-path overload `CompiledModel.create(path, options, null)`. The first launch is expected to fail with "model not found" before the install script has been run.

When working on a new module, follow whichever pattern its peers use. Don't introduce a third pattern.

## LiteRT CompiledModel constraints (must respect when converting / debugging)

`CompiledModel` with `Accelerator.GPU` requires **every** op in the graph to be GPU-compatible. There is no CPU fallback — an unsupported op makes the whole model fail to compile. The constraints and the standard fixes are documented in detail in `docs/LITERT_CONVERSION_GUIDE.md` and summarized in the root `README.md` ("GPU Compatibility Notes"). The short version:

- All tensors must be 4D or fewer; no dynamic dims (`-1`) in `Reshape`.
- Banned ops: `TOPK_V2`, `GATHER`, `GATHER_ND`, float↔int `CAST`, `GELU` (FlexErf), `PACK`, `SPLIT`, `Erf`, `RESIZE_BILINEAR` with `align_corners=True`.
- Vision Transformers **must** be converted with **litert-torch** (NCHW preserved). `onnx2tf` silently destroys ViT attention accuracy (corr ≈ 0.29 vs 0.99) because it transposes batch/spatial dims inside MatMul.
- Pure CNNs (YOLO, Real-ESRGAN) can use `onnx2tf` or the SavedModel path.
- Common rewrites: `nn.GELU` → `x * sigmoid(1.702 * x)`; `GroupNorm` → manual `reshape(B*G, C//G, H, W)` mean/var; `Conv2d_WS` → bake standardized weights into a regular `Conv2d`; `F.normalize` → manual `x / sqrt(sum(x*x) + eps)`; `torch.bucketize` → pre-compute for the fixed input size; `scaled_dot_product_attention` → manual matmul+softmax; transformers' `create_causal_mask` → monkey-patch with `torch.triu`.
- **On-device-only constraints** (not caught by desktop checker): `TRANSPOSE_CONV` rejected by Pixel 8a ML Drift delegate; `Conv2d` rejects constant-only inputs ("input must be a runtime tensor"); `nn.Upsample(scale_factor=N)` produces dynamic `RESIZE_BILINEAR` sizes; `LayerScale` MUL between 2D FC output and 3D gamma causes shape conflict. See `docs/LITERT_CONVERSION_GUIDE.md` MoGe-2 section for details.

Most of these patches are pre-packaged in `litert_gpu_toolkit` — prefer `convert_for_gpu(model, dummy_input, output_path)` over hand-rolling the patches again.

## Project conventions (from user's global CLAUDE.md)

- **Do not include "claude" in commit messages or as a committer.**
- Keep the repo small. Do not commit CoreML models, build outputs, or other large/derived files.
- **Build testing is done on the actual device** — do not run `./gradlew assemble` etc. as a validation step. Edit, then hand off.
- Code comments and UI strings are in **English**, even though the user often communicates in Japanese.
