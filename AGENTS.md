# AGENTS.md — for coding agents running a converted model on Android with LiteRT

Read this first if someone asked you to "run *some model* on Android with LiteRT (or TFLite)",
"convert a PyTorch vision model for the phone GPU", or "add background removal / detection /
depth / speech to an Android app". This file routes; the README and the per-model directories
have the code.

**What this is.** A zoo of 91 models (as of 2026-09-08) converted to `.tflite` for Google's
[LiteRT](https://github.com/google-ai-edge/LiteRT) runtime, each with its download, input and
output shapes, preprocessing, the conversion script that produced it, and a standalone Android
sample app in Kotlin. Every model runs every op on the phone GPU through `CompiledModel`; no
MediaPipe. Independent, not affiliated with Google.

## Route

| Task | Do | Where |
|---|---|---|
| Find a model, its download and its measured latency | The table at the top of the README (model, task, device, latency, download); each row links the model's section | README "Models" |
| Run a `.tflite` in an existing app on the GPU | Dependency `com.google.ai.edge.litert:litert` (Google Maven), then `CompiledModel.create(context.assets, "model.tflite", CompiledModel.Options(Accelerator.GPU), null)`, `createInputBuffers()`, `run()` | README "How to use" |
| Add one feature end to end, verified on a device | Background removal: one dependency, one Kotlin file, one 176 MB model; Pixel 8a numbers dated 2026-09-05; data in [ormbg/recipe.json](ormbg/recipe.json) | [ormbg/INTEGRATION.md](ormbg/INTEGRATION.md) |
| The Kotlin helpers to copy into an app | `CompiledModelRunner`, `ImageTensor`, `RealtimeCameraPipeline`, `AudioCapture`, `MathOps` (canonical sources; every sample carries a byte-identical copy) | [common/README.md](common/README.md), `common/kotlin/` |
| A complete sample app for a model | Each model directory is its own Gradle project: `cd <model>/ && ./gradlew :app:installDebug` | README, the model's section names its app |
| Convert a PyTorch model yourself | litert-torch for Vision Transformers and attention (NCHW preserved); onnx2tf only for pure CNNs; `litert_gpu_toolkit.convert_for_gpu(model, dummy_input, output_path)` applies the GPU patches | [docs/LITERT_CONVERSION_GUIDE.md](docs/LITERT_CONVERSION_GUIDE.md), `litert_gpu_toolkit/` |
| The TensorFlow Lite name for something, or the LiteRT name | The mapping (Maven, pip, delegates, MediaPipe LLM → LiteRT-LM), verified 2026-09-05 | README "LiteRT or TensorFlow Lite? The names" |
| Restructure a sample into Compose + MVVM for a litert-samples PR | The second-pass guide | [docs/COMPOSE_MVVM_SAMPLE_GUIDE.md](docs/COMPOSE_MVVM_SAMPLE_GUIDE.md) |
| Work on this repository itself | Module layout, build, how model files reach the device, conventions | [CLAUDE.md](CLAUDE.md) |

## Rules that fail on a real device when broken

1. **`CompiledModel` with `Accelerator.GPU` has no CPU fallback.** One unsupported op and the
   model does not load. Tensors are rank 4 at most; no dynamic dims in `Reshape`. Ops that fail:
   `TOPK_V2`, `GATHER`, `GATHER_ND`, float↔int `CAST`, `GELU` (FlexErf), `PACK`, `SPLIT`, `Erf`,
   `RESIZE_BILINEAR` with `align_corners=True`.
2. **Vision Transformers go through litert-torch.** onnx2tf silently destroys ViT attention
   (correlation about 0.29 against 0.99).
3. **Use the current coordinates.** `com.google.ai.edge.litert:litert` on Google Maven, not
   `org.tensorflow:tensorflow-lite`; `pip install litert-torch`, not `ai-edge-torch`;
   `pip install ai-edge-litert`, not `tflite-runtime`. NNAPI is deprecated; the NPU is
   `Accelerator.NPU` on `CompiledModel`.
4. **Model files are not in git.** Small models sit in `app/src/main/assets/` with
   `noCompress += "tflite"`; large ones are staged by the module's `install_to_device.sh`.
   Follow whichever pattern the module's peers use; do not add a third.
5. **One model call at a time.** Confine inference to `Dispatchers.Default.limitedParallelism(1)`;
   the helpers reuse native input and output buffers.
6. **Numbers stay with their device.** A latency in the table is a Pixel 8a or Galaxy S26
   measurement; do not extrapolate it to another phone.
7. **Fix a shared helper in `common/` first**, then `python tools/sync_common.py --apply`. Never
   patch one module's copy in place.

## Not this repo

- LLMs and VLMs as `.litertlm` bundles → [hf-to-litertlm](https://github.com/john-rocky/hf-to-litertlm) to convert, [hfmodels-android](https://github.com/john-rocky/hfmodels-android) to run.
- Which `.tflite` op runs on which delegate, measured per runtime version → [edge-compat](https://github.com/john-rocky/edge-compat).
- iPhone and Mac → [coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo) (Core AI) or [CoreML-Models](https://github.com/john-rocky/CoreML-Models) (Core ML).

Maintainer: john-rocky (GitHub); litert-community and mlboydaisuke (Hugging Face). Issues: https://github.com/john-rocky/LiteRT-Models/issues
