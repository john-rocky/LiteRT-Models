# LiteRT or TensorFlow Lite? The names

LiteRT is TensorFlow Lite renamed (2024-09-04). The `.tflite` format and the models did not change; the coordinates did. Verified 2026-11-11:

| Old name | New name | Note |
|---|---|---|
| `org.tensorflow:tensorflow-lite` (last 2.17.0) | `com.google.ai.edge.litert:litert` 2.3.2 — Google Maven, not Maven Central | `Interpreter` + `CompiledModel`, minSdk 24; the 1.4.x line is `Interpreter`-only, minSdk 21. The 2.3.x classes carry Kotlin metadata 2.4.0: a Kotlin 2.1.10 sample needs `-Xskip-metadata-version-check` |
| `org.tensorflow:tensorflow-lite-gpu` + `GpuDelegate` | `CompiledModel.Options(Accelerator.GPU)`; from 2.3.0 also `com.google.ai.edge.litert:litert-gpu` at the same version as `litert` | Without it, on 2.3.2 (Pixel 11 Pro): `Options(Accelerator.GPU)` throws `LiteRtException: Failed to compile model`, `Options(Accelerator.GPU, Accelerator.CPU)` runs on the CPU, and Logcat shows `GPU accelerator could not be loaded and registered`. In 2.0–2.2 the GPU accelerator is inside `litert`: the samples here pin 2.1.3–2.2.0, or 2.3.0 with `litert-gpu`. `GpuDelegate` is in `litert-gpu` up to 1.4.2 only |
| `NnApiDelegate` / Qualcomm `qnn-litert-delegate` | `Accelerator.NPU` on `CompiledModel` | NNAPI is deprecated from Android 15 |
| `pip install tflite-runtime` (last release 2023, wheels to Python 3.11) | `pip install ai-edge-litert` (cp310–cp314) | `ai_edge_litert.interpreter.Interpreter`, `ai_edge_litert.compiled_model.CompiledModel` |
| `pip install ai-edge-torch` | `pip install litert-torch` (`litert_torch.convert(model, sample_inputs)`) | most models in this zoo were converted with it |
| MediaPipe LLM Inference API | LiteRT-LM (`com.google.ai.edge.litertlm:litertlm-android` 0.19.1) | deprecated since `com.google.mediapipe:tasks-genai` 0.10.33: "Migrate to LiteRT LM instead." |

What moving from LiteRT 1.4.0's `Interpreter` C API to LiteRT 2.3.2's `CompiledModel` changes on a Galaxy S26 (latency and output agreement, three models, 2026-11-10): [interpreter-vs-compiledmodel/](../interpreter-vs-compiledmodel/).

Two `CompiledModel` GPU rules: every op in the graph must be GPU-compatible (no CPU fallback), and tensors are rank 4 at most. Full mapping with sources: [docs/LITERT_CONVERSION_GUIDE.md § Three PyTorch → Android routes](LITERT_CONVERSION_GUIDE.md#three-pytorch--android-routes-on-one-model-measured-2026-09-05) <!-- TODO(lane B, articles): replace this in-repo anchor with the published article URL -->
