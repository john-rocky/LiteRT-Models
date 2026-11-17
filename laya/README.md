# Laya multilingual — Android text triage

This Android sample app runs the multilingual
[Laya](https://huggingface.co/litert-community/Laya-Multilingual-LiteRT) checkpoint with
LiteRT on the GPU, the Qualcomm Hexagon NPU, or the CPU. Enter a support message in
Japanese or English and ask a fixed set of questions on the phone: which team should
handle it, how urgent it is, or whether a post breaks a rule. It needs an arm64 phone
with Android 8.0 or newer and was verified on a Galaxy S26 and a Pixel 11 Pro.

Select **Email triage**, **Support intent**, or **Moderation**, choose a JA/EN example,
edit the text, and tap **Run**. Each question gets an answer, option probabilities,
confidence, and timing. The examples are invented. The question schemas are the upstream
presets unchanged.

## Download, build and install

Run the commands from this module directory. The build needs JDK 17, Android SDK platform
35 and build-tools 35.0.0. The install needs Android platform-tools (`adb`), the Hugging
Face CLI (`hf`) and Python 3.9+. Set `ANDROID_HOME` to your SDK directory, or point `AAPT`
at the `aapt` executable. `PYTHON` can select a private Python interpreter.

```bash
export HF_HUB_DISABLE_XET=1
export HF_HOME="$PWD/.cache/huggingface"
export LAYA_MODEL_DIR="$PWD/model-files"
hf download litert-community/Laya-Multilingual-LiteRT \
  laya_ml_s256_embeds_wfp16.tflite \
  laya_ml_act_head_fp32.tflite laya_ml_calibration.json \
  tokenizer.json tokenizer_config.json token_embeddings_fp16.bin token_embeddings.json \
  --local-dir "$LAYA_MODEL_DIR"
./scripts/fetch_npu_libs.sh /path/to/npu-libraries   # optional; see NPU below
./gradlew --no-daemon clean :app:assembleDebug :app:assembleRelease
./scripts/install_to_device.sh --assets "$LAYA_MODEL_DIR" --graph wfp16 --validate-only
export ANDROID_SERIAL=your-device-serial
adb -s "$ANDROID_SERIAL" install -r app/build/outputs/apk/debug/app-debug.apk
./scripts/install_to_device.sh --assets "$LAYA_MODEL_DIR" --graph wfp16
adb -s "$ANDROID_SERIAL" shell am start -n com.laya/.MainActivity
```

The model files are not needed to build either APK. The installer checks the files
before contacting the device and copies them into the app's private files with
`run-as com.laya`, so install the debug APK first. It uses `aapt` to verify that the APK
is the debuggable `com.laya` app.

## When it does not work

- Started before the model files are installed, the app names the first missing file and
  the installer script.
- Without the Qualcomm runtime libraries the NPU option is disabled, and the GPU and CPU
  paths are unchanged. See [NPU](#npu).
- A compile error appears inline with the runtime message, and **Use CPU** is an explicit
  user choice.
- No other phone was tested. The NPU ran only on the Galaxy S26.
- The app uses the 256-token window. S512 has not been run on a device.

## Model and requirements

- Upstream: [Convai Innovations Laya](https://huggingface.co/convaiinnovations/laya/tree/1c5edc17a7acd8701df6fc341c0d179f1c62c982/multilingual),
  revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, multilingual mmBERT-base checkpoint.
- Android: arm64-v8a, Android 8.0 / API 26 or newer, compile and target SDK 35.
- Runtime: LiteRT 2.2.0 `CompiledModel`, Kotlin and Material 1 Compose with MVVM.

The GPU path requests
`CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)`.
`wfp16` describes how the fully connected weights are stored. Graph inputs, outputs and
GPU arithmetic are float32. The NPU compiles the same graph on the device and computes
in fp16. The app remembers the selected accelerator.

## Model files

The WFP16 graph and the shared files total 679,274,443 bytes, and the optional FP32 graph
adds 500,970,372. Only one main graph is loaded per process. The installer manifest
checks hashes and exact sizes.

| File | Bytes | Purpose |
|---|---:|---|
| `laya_ml_s256_embeds_wfp16.tflite` | 250,889,040 | S256 main graph, FP16 weight storage |
| `laya_ml_s256_embeds_fp32.tflite` | 500,970,372 | Optional S256 main graph, FP32 weights |
| `laya_ml_act_head_fp32.tflite` | 795,816 | Shared action head |
| `laya_ml_calibration.json` | 9,074 | Temperatures and calibration provenance |
| `tokenizer.json` | 34,363,188 | BPE vocabulary, added tokens, merge ranks |
| `tokenizer_config.json` | 524 | Tokenizer metadata |
| `token_embeddings_fp16.bin` | 393,216,000 | FP16 table `[256000,768]` |
| `token_embeddings.json` | 801 | Table shape, hash, source revision |

`--graph fp32` installs the FP32 variant and `--graph both` installs both. To run the
FP32 storage, stop the app and launch it with `--es storage fp32`. The extra is named
`storage`, not `graph`. The three preset schemas are packaged in
`app/src/main/assets/presets.json`.

The release build is non-debuggable and signed with the local debug key. Install it with
`adb install -r` after the debug installer has copied the files. Sign with your own key
to distribute an app.

## How the app works

```text
State + preset questions → Kotlin BPE tokenizer and prompt builder
  → memory-mapped token embedding lookup → LiteRT main graph
  → raw-probability features + LiteRT action head
  → calibrated Kotlin decoder → immutable UI state → Compose cards
```

`MainActivity` observes immutable state, and `MainViewModel` owns the engine on a
serialized model dispatcher. One complete pass warms up before **Ready**. The status
header reports loading, compilation and launch-to-Ready time, and `LAYA_READY` logs the
same interval with the individual initialization timings.

Email uses subject and body fields, and support and moderation use one text field. Each
preset runs five question rows one after another. The header's token count sums their
real sequence lengths without padding. Each graph call has a static window of 256
positions, and state text is right-truncated after reserving the question and options.

Calibration is enabled by default. **T=1 (uncalibrated)** changes only option decoding.
Choice answers are labels, score answers are expected zero-based rubric levels, and
yes/no questions show a continuous true probability. Confidence is the contract's entropy
or binary confidence, not a measured correctness rate. See the
[host contract](docs/HOST_CONTRACT.md).

## NPU

The NPU needs Qualcomm runtime libraries that are not in git. Collect the ten files
listed in [scripts/fetch_npu_libs.sh](scripts/fetch_npu_libs.sh) and run
`scripts/fetch_npu_libs.sh <directory>` before building. They are two LiteRT GitHub
Release assets and the QAIRT libraries for the phone's Hexagon version, v81 for the
Galaxy S26.
[Running on the NPU](https://github.com/john-rocky/LiteRT-Models/blob/main/MODELS.md#running-on-the-npu)
in MODELS.md explains where each file comes from.

The first NPU launch compiles the graph on the phone in about 40 s and caches it. Later
launches load it in about 0.2 s. The 1 MB action head stays on the CPU in FP32.

The NPU computes in fp16, so the graph file carries three rewrites that give identical
results in fp32. Each large LayerNorm computes on its input scaled by a power of two,
with epsilon scaled by the square. The attention masks use −1e4 instead of −1e9. Layers
11 and 12 scale the rows of `Wi` that produce `gate` by 2^-4 and `Wo` by 2^4, so the
GeGLU product stays near 2e3.

## Reproduce the host and device tests

The JVM suite reads a separate local set of test data and does not run the model. Set
`LAYA_TEST_DATA` as described in [scripts/TEST_DATA.md](scripts/TEST_DATA.md), then run:

```bash
./gradlew --no-daemon :app:testDebugUnitTest
```

For a device test, also pass `--gate-rows FILE` to the installer with the S256 rows file,
1,018,117 bytes of validation data that are not part of the app download. Then run:

```bash
export ANDROID_SERIAL=your-device-serial
adb -s "$ANDROID_SERIAL" shell am force-stop com.laya
adb -s "$ANDROID_SERIAL" shell am start -W -n com.laya/.MainActivity \
  --ez gate true --es window 256 --es accel gpu --es storage wfp16
adb -s "$ANDROID_SERIAL" exec-out run-as com.laya cat files/laya_gate_gpu_256.json \
  > gate_gpu_wfp16_256.json
```

Wait for the `LAYA_GATE` log tag before pulling the report. Use `--es storage fp32` for
the other graph, `--es accel npu` or `--es accel cpu` for the other accelerators, and
force-stop the app between configurations. The output filename contains the accelerator
and the window. The `gate` extra is debug-only.

## Verified on

These checks compare the app with the upstream implementation, not task accuracy.

**Kotlin host, Apple M4 Max, JDK 17.** The host matched its captured Python references on
402/402 builder IDs and marker positions, 300/300 tokenizer stress strings, 402/402
answer dictionaries at T=1 and 402/402 calibrated ones, and 20/20 JSON serializations.
Embedding lookup matched NumPy on 402/402 rows with a maximum absolute error of 0.

**Graphs on Mac CPU, ai-edge-litert 2.1.6, four threads.** Both S256 graphs passed
201/201 rows with 81/81 choice and score argmax matches and a maximum probability error
of 0.0001 for FP32 and 0.0014 for WFP16.

**Galaxy S26 SM-S942Q, Android 16, LiteRT 2.2.0.** No other workload, screen on, thermal
status none before each run. The GPU placed 1809/1809 main ops and 4/4 act ops in one
partition each at explicit FP32. On the NPU, Hexagon v81 with JIT, the main graph became
one `DispatchDelegate` node. Each run has 201/201 exact on-device IDs and markers, 81/81
choice and score argmax matches and finite outputs.

| Accelerator (debug build, WFP16 storage) | Max Δp | Creation ms | Warm median [min, max] ms, 200 rows |
|---|---:|---:|---|
| GPU, FP32 precision | 0.0014 | 1343.173 | 51.903 [50.655, 55.978] |
| NPU | 0.0069 | 37089.553 first launch; 231.021 later launch | 36.460 [35.816, 38.251] |

The times cover the writes, run and readback of both graphs, without host preparation,
embedding lookup and decoding. On 2026-09-21, before the graph revision, the CPU measured
163.020 ms. The tokenizer loads in about 1.5 s in the debug build.

GPU fp16 arithmetic is not used. On the same phone FP16 with FP32 accumulation took
40.8 ms with one row at Δp 0.012 against a limit of 0.01, and FP16 took 28.2 ms and
changed one of 81 answers. Before the revision the graph returned no finite row on the
GPU at FP16 or on the NPU.

| UI build (JA email, 5 questions) | Launch→Ready (ms) | Run (ms) |
|---|---:|---:|
| Debug, NPU, JIT cache present | 2032.86 | 257.25, 268.02 |
| Debug, GPU FP32 (2026-09-21) | 3194.32, 3161.64 | 338.30, 346.72 |
| Non-debuggable release, GPU FP32 (2026-09-21) | 2296.08, 2298.29 | 301.33, 302.90 |

UI totals cover the invented Japanese email preset: five questions, 529 prompt tokens,
calibration enabled. Launch→Ready starts at `Activity.onCreate`. The NPU row is one
launch after the JIT cache was written.

**Pixel 11 Pro, Tensor G6, Android 17, LiteRT 2.2.0.** On 2026-11-16 the debug build with
WFP16 storage ran the same 201 rows on the GPU at explicit FP32 and on the CPU. Each run
has 201/201 exact on-device IDs and markers, 201/201 argmax matches, finite outputs and a
maximum probability difference of 0.0014. The NPU was not run there.

## License

This sample is [Apache-2.0](https://github.com/john-rocky/LiteRT-Models/blob/main/laya/LICENSE). Laya code, checkpoint and tokenizer assets declare
Apache-2.0, and mmBERT-base declares MIT. Attribution and license texts for ModernBERT,
Transformers and tokenizers are under [licenses/](licenses/README.md) and in
[NOTICE](https://github.com/john-rocky/LiteRT-Models/blob/main/laya/NOTICE), which also lists the conversion changes. Raw calibration datasets are
not included.
