# Add background removal (person segmentation) to an existing Android app, on the GPU

**Short answer.** Add one Gradle dependency, copy one Kotlin file, download one 176 MB model
file. You get a `[1024 × 1024]` alpha matte per image with `cutout()` / `composite()` helpers,
computed entirely on the GPU through LiteRT `CompiledModel`. Measured on a Pixel 8a: 246 ms
per image for the model, 361 ms end to end including pre/post-processing, 2566 ms
one-off load. Model and code are Apache-2.0. Everything below was run on 2026-09-05; the same facts
in machine-readable form are in [`recipe.json`](recipe.json).

The model is [ormbg](https://huggingface.co/schirrmacher/ormbg) (an ISNet trained for
photorealistic subject cut-out), converted to `.tflite` with litert-torch and published as
[`litert-community/ormbg-LiteRT`](https://huggingface.co/litert-community/ormbg-LiteRT). It is a
plain `.tflite` with only built-in ops — no custom ops, no MediaPipe, no ONNX detour — so the
standard LiteRT runtime loads it as is.

## 1. Dependency

`app/build.gradle.kts`:

```kotlin
android {
  defaultConfig { minSdk = 26 }
  androidResources { noCompress += listOf("tflite") }   // the model is memory-mapped out of the APK
}
dependencies {
  implementation("com.google.ai.edge.litert:litert:2.2.0")   // Google Maven (google())
}
```

- `com.google.ai.edge.litert:litert` is the current name of the runtime formerly published as
  `org.tensorflow:tensorflow-lite`. Do not add both; see the
  [naming table](../README.md#litert-or-tensorflow-lite-the-names) if your app still has the old
  coordinates.
- Kotlin 2.x (this module uses 2.3.21 with AGP 8.7.3); `arm64-v8a` is the ABI this was verified on.

## 2. Copy one file

[`app/src/main/java/com/ormbg/BgRemover.kt`](app/src/main/java/com/ormbg/BgRemover.kt) → your
package. Change the `package` line and nothing else: the pre- and post-processing in it is the
model contract, and the numbers in §6 hold for that code only. It depends on `android.graphics`
and the LiteRT artifact, nothing else.

| Call | What it does |
|---|---|
| `BgRemover.fromAssets(context, "ormbg.tflite")` | compiles the model for the GPU and runs one warm-up inference (call once, off the main thread; 2566 ms on a Pixel 8a) |
| `BgRemover.fromFile(context, path)` | same, from an absolute path — for a model downloaded at runtime into `filesDir` |
| `process(bitmap): Matte?` | segments one image of any size; `null` only if `cancel()` was called while it was running |
| `cancel()` | aborts the `process()` call in flight (from any thread); later calls run normally |
| `close()` | releases the model, its GPU buffers and the scratch bitmap |
| `Matte.alpha` / `.size` | the `1024 × 1024` matte in `[0, 1]`, row-major, 1 = subject; `rawMin` / `rawMax` are the model output range before normalization |
| `Matte.cutout(bitmap)` | the input with a transparent background, at the input's own size |
| `Matte.composite(bitmap, color)` | the input over a solid color |
| `Matte.downsample(n)` | nearest-neighbour `n × n` alpha for realtime compositing |
| `Matte.toMaskBitmap()` | the matte as an `ARGB_8888` bitmap (alpha = matte) for your own compositing |

## 3. Get the model

Not committed to git — download it into `app/src/main/assets/` and ignore it there:

```bash
curl -L -o app/src/main/assets/ormbg.tflite \
  https://huggingface.co/litert-community/ormbg-LiteRT/resolve/main/ormbg.tflite
shasum -a 256 app/src/main/assets/ormbg.tflite
# f6e359cf35c55e7bf216eb3141f6d76828667a5e8e67a929f53e9f7f38fae030  (176,240,148 bytes)
echo '*.tflite' >> .gitignore
```

Check the hash before anything else; a truncated download is the most common "the model is
broken". The file is ungated (no token needed) and Apache-2.0.

If 176 MB is too much for your APK, download it at first launch into `context.filesDir` and use
`BgRemover.fromFile(context, file.absolutePath)` — that loader is exercised by the same test (§6).
Those are the two model-delivery patterns used in this repository; there is no third.

## 4. Use it

One owner, one background thread, `process` off the main thread, `cancel` when the screen goes
away, `close` on teardown. With a ViewModel and coroutines:

```kotlin
class CutoutViewModel(app: Application) : AndroidViewModel(app) {
  private val modelThread = Dispatchers.IO.limitedParallelism(1)
  private val remover = viewModelScope.async(modelThread) { BgRemover.fromAssets(app) }

  fun removeBackground(photo: Bitmap, onDone: (Bitmap) -> Unit) =
    viewModelScope.launch {
      val cutout = withContext(modelThread) { remover.await().process(photo)?.cutout(photo) }
      if (cutout != null) onDone(cutout)
    }

  fun cancel() = runCatching { remover.getCompleted().cancel() }   // e.g. from onStop

  override fun onCleared() {
    remover.cancel()
    viewModelScope.launch(modelThread) { runCatching { remover.await().close() } }
  }
}
```

Without coroutines, a `Executors.newSingleThreadExecutor()` that owns the object works the same
way (that is how the demo in this directory does it). `Accelerator.GPU` compiles the whole graph
for the GPU or throws at creation — there is no partial delegation and no silent CPU fallback. Show
that exception; do not catch it into a CPU path, which would turn a 25× slowdown into a "working"
app.

## 5. Pre- and post-processing (what the file does for you)

- **Input:** the bitmap is stretched to `1024 × 1024` (no letterbox — the upstream inference script
  stretches too), split into RGB planes, scaled to `[0, 1]`, laid out NCHW: `[1, 3, 1024, 1024]`
  float32. No mean/std normalization.
- **Output:** `[1, 1, 1024, 1024]` sigmoid matte, then min-max normalized over the image, as the
  upstream `postprocess_image` does. The result maps back to your bitmap by plain scaling: `cutout`
  / `composite` scale an `ARGB_8888` mask bitmap (`toMaskBitmap()`, alpha = matte) to the bitmap's
  size and write it into the alpha channel. (An `ALPHA_8` mask drawn with `PorterDuff.Mode.DST_IN`
  silently does nothing on current Android — the test checks the cutout's alpha for that reason.)
  `Matte.rawMin`/`rawMax` let you spot a collapsed output (a near-zero range) without looking at
  pixels.
- **Cost split on a Pixel 8a:** model 246 ms, everything else (stretch, pixel loop,
  readback normalization) about 115 ms per `1024²` image. For a live camera loop, feed
  smaller frames and composite with `downsample(256)` like the demo does; the model time does not
  change with the input bitmap size.

## 6. Verify the integration

The check is an instrumented test that loads the model on the GPU, segments a fixed photo, and
compares the matte with recorded values. Copy
[`app/src/androidTest/java/com/ormbg/BgRemoverTest.kt`](app/src/androidTest/java/com/ormbg/BgRemoverTest.kt)
and its 24 KB fixture
[`app/src/androidTest/assets/person.jpg`](app/src/androidTest/assets/person.jpg) into your module
(fixing the package line), add the two test dependencies from
[`app/build.gradle.kts`](app/build.gradle.kts), connect a device, and run:

```bash
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=<your.package>.BgRemoverTest
# in this module the flavor is part of the task name: :app:connectedGpuDebugAndroidTest
```

Expected: `BUILD SUCCESSFUL`, 3 tests passed, and in logcat (tag `ormbg`):

```
Replacing 246 out of 246 node(s) with delegate (LITERT_CL) node, yielding 1 partitions for subgraph 0 (main).
RESULT device=Pixel 8a sdk=36 accel=GPU load_ms=2566 model_ms_median=246 process_ms_median=361 runs=20 raw_min=0.0000 raw_max=1.0000 mean=0.5208 fg_frac=0.5213 center=1.0000 corners=0.0001 thermal=0->0
```

What the test asserts, and why:

| Check | Value on the fixture | Tolerance |
|---|---|---|
| GPU compile succeeds | constructor returns | — (a failure is an exception naming the op) |
| raw output range | `0.0000 … 1.0000` | range > 0.5 |
| center 64×64 patch (subject) | `1.0000` | > 0.9 |
| four 64×64 corners (background) | `0.0001` | < 0.1 |
| foreground fraction (alpha > 0.5) | `0.5213` | ± 0.02 |
| mean alpha | `0.5208` | ± 0.02 |
| `cutout()` alpha at a corner / at the center | `0` / `255` | < 26 / > 230 |

The residency line comes from the runtime, not the test: with `Accelerator.GPU` there is no partial
delegation, so the line reads `N out of N` or creation fails. Latency is logged, not asserted — it
moves with thermal status (also logged, `thermal=before->after`; 0 means none).

**Host parity, for the numeric proof.** The test also writes the exact input tensor it fed the
model and the raw output into the app's `filesDir/ormbg_test/`. Keep the APK installed
(`-Pandroid.injected.androidTest.leaveApksInstalledAfterRun=true`), pull the two files, and compare
the device's fp16 GPU output with the host CPU fp32 output on the identical input:

```bash
adb exec-out run-as <your.package> cat files/ormbg_test/input_nchw.f32 > input_nchw.f32
adb exec-out run-as <your.package> cat files/ormbg_test/raw_out.f32   > raw_out.f32
pip install ai-edge-litert numpy
python scripts/verify_device_dump.py app/src/main/assets/ormbg.tflite input_nchw.f32 raw_out.f32
```

On the Pixel 8a: raw-output correlation 1.000000, max |diff| 0.00905 (sigmoid units), foreground-mask IoU 0.99997, and all six summary statistics identical to four digits (run of 2026-09-05). `cutout.png` in the same directory is the visual check.

## 7. Verified / unverified

| Device | OS | LiteRT | Date | Model (median of 20) | End to end | Load | Residency | Thermal |
|---|---|---|---|---|---|---|---|---|
| Pixel 8a (Tensor G3, Mali-G715) | Android 16, build CP1A.260505.005 | 2.2.0, `Accelerator.GPU` (LITERT_CL) | 2026-09-05 | 246 ms | 361 ms | 2566 ms | 246/246 nodes, 1 partition | status 0 → 0 |

Conditions: screen on, USB connected, battery 100 %, nothing else running, the values above from
the last of five runs of the test in a row (model median 241–249 ms across the five). `load` includes GPU shader compilation and one warm-up inference.

**Not verified by this recipe:**

- Android emulators — `CompiledModel` with `Accelerator.GPU` does not run there; the check is
  device-only.
- Any device other than the Pixel 8a above, including other Pixels, Samsung Galaxy (Adreno) and
  MediaTek devices. An earlier benchmark of the same model file on a Galaxy S26 (GPU and NPU) is on
  the [Hugging Face card](https://huggingface.co/litert-community/ormbg-LiteRT), taken with a
  different harness, not with this test.
- Android versions other than 16, LiteRT versions other than 2.2.0, the `npu` flavor of this module
  (Qualcomm devices only), and any change to the pre-processing (letterbox, other input sizes).
- Earlier text in this repository and on the model card quoted "~10 ms/frame" for this model on a
  Pixel 8a. Today's measurement, with the readback that waits for the GPU included, is
  246 ms; the `~10 ms` figure was a timing of `run()` alone, which only enqueues the work.

## 8. Provenance

| | |
|---|---|
| Converted by | [john-rocky](https://github.com/john-rocky), litert-torch 0.9.4 (torch 2.13.0), [`scripts/build_ormbg.py`](scripts/build_ormbg.py); re-run on 2026-09-05, output byte-identical to the published file (same sha256) |
| Conversion patch | one: `F.interpolate(align_corners=True)` → `False` on the bilinear upsamples (the GPU delegate rejects `align_corners=True`) |
| Recipe origin | this file, [john-rocky/LiteRT-Models `ormbg/`](https://github.com/john-rocky/LiteRT-Models/tree/main/ormbg) |
| Measurement report | §6–7 above; raw evidence = the `RESULT` and `Replacing …` logcat lines of the 2026-09-05 run on the Pixel 8a and `scripts/verify_device_dump.py` output |
| Commit | conversion script and sample: `0c48ff1` (2026-07-07); this recipe: the commit that adds this file |
| Maintainer | john-rocky |
