# Laya multilingual — Android text triage

Enter a support message in Japanese or English and ask a fixed set of questions on the phone:
which team should handle it, how urgent it is, or whether a post breaks a rule. Select
**Email triage**, **Support intent**, or **Moderation**, choose a JA/EN example, edit the text,
and tap **Run**. Each question has an answer, option probabilities, confidence, and timing.
The examples are invented; the question schemas are the upstream presets unchanged.

## Model and requirements

- Model download: [litert-community/Laya-Multilingual-LiteRT](https://huggingface.co/litert-community/Laya-Multilingual-LiteRT).
- Upstream: [Convai Innovations Laya](https://huggingface.co/convaiinnovations/laya/tree/1c5edc17a7acd8701df6fc341c0d179f1c62c982/multilingual),
  revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, multilingual mmBERT-base checkpoint.
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile/target SDK 35.
- Runtime: LiteRT **2.2.0** `CompiledModel`; Kotlin and Material 1 Compose with MVVM.
- Build: JDK 17, Android SDK platform 35 and build-tools 35.0.0.

The GPU path explicitly requests
`CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)`.
`wfp16` describes fully connected weight storage; graph inputs, outputs, and requested GPU
arithmetic precision are float32. Both storage variants passed the Galaxy S26 GPU gate;
WFP16 storage is selected and CPU remains selectable. A GPU compile error appears inline with the runtime message; **Use CPU** is an
explicit user choice. The app remembers the selected accelerator.

## Download, build and install

Commands below run from this module directory. Model files are external and are not required to build either APK.
Install Android platform-tools (`adb`) and the Hugging Face CLI (`hf`) for device installation.
The installer uses Python 3.9+ and SDK Build Tools 35.0.0 `aapt` to verify that the APK
is the debuggable `com.laya` app. Set `ANDROID_HOME` to your SDK directory, or provide
the `AAPT` executable explicitly; `PYTHON` can select a private Python interpreter.

```bash
export HF_HUB_DISABLE_XET=1
export HF_HOME="$PWD/.cache/huggingface"
export LAYA_MODEL_DIR="$PWD/model-files"
hf download litert-community/Laya-Multilingual-LiteRT \
  laya_ml_s256_embeds_wfp16.tflite \
  laya_ml_act_head_fp32.tflite laya_ml_calibration.json \
  tokenizer.json tokenizer_config.json token_embeddings_fp16.bin token_embeddings.json \
  --local-dir "$LAYA_MODEL_DIR"
./gradlew --no-daemon clean :app:assembleDebug :app:assembleRelease
./scripts/install_to_device.sh --assets "$LAYA_MODEL_DIR" --graph wfp16 --validate-only
export ANDROID_SERIAL=your-device-serial
adb -s "$ANDROID_SERIAL" install -r app/build/outputs/apk/debug/app-debug.apk
./scripts/install_to_device.sh --assets "$LAYA_MODEL_DIR" --graph wfp16
adb -s "$ANDROID_SERIAL" shell am start -n com.laya/.MainActivity
```

The installer validates the selected files before contacting the device, stages them on the
device, and copies them into this app's private files with `run-as com.laya`. Install the debug
APK first because `run-as` needs a debuggable package. `--graph wfp16` or `--graph fp32` installs
one variant; `wfp16` is the default. Use `--graph both` to install both for a comparative gate.
The current default is WFP16 storage. For the alternate storage, stop this app and launch it
with `--es storage fp32`; the extra is named `storage`, not `graph`.

The non-debuggable release build uses the local debug signing key for sample profiling. It can
be installed with `adb install -r` over this sample after the debug installer has copied the
files; the files are retained. Use a separate production signing configuration when distributing
an application.

## External files

The selected WFP16 graph and shared assets total **679,274,893 bytes**. The optional FP32
graph adds 500,969,948 bytes (1,180,244,841 bytes with both); only one main graph is loaded per process. The embedding table is
memory-mapped read-only. Hashes and exact sizes are checked by the installer manifest.

| File | Bytes | Purpose |
|---|---:|---|
| `laya_ml_s256_embeds_wfp16.tflite` | 250,889,408 | S256 main graph; FP16 fully connected weight storage |
| `laya_ml_s256_embeds_fp32.tflite` | 500,969,948 | Optional S256 main graph; FP32 weight storage |
| `laya_ml_act_head_fp32.tflite` | 795,816 | Shared action head |
| `laya_ml_calibration.json` | 9,156 | Option-count temperatures and calibration provenance |
| `tokenizer.json` | 34,363,188 | Complete BPE vocabulary, added tokens, and merge ranks |
| `tokenizer_config.json` | 524 | Checkpoint tokenizer metadata |
| `token_embeddings_fp16.bin` | 393,216,000 | Little-endian FP16 table, shape `[256000,768]` |
| `token_embeddings.json` | 801 | Table shape, hash, source revision, and PAD policy |

For a debug numerical gate, also pass `--gate-rows FILE` to the installer. The current S256
gate file is 1,018,117 bytes and is validation data, not part of the app download. See
[test data](scripts/TEST_DATA.md). The three preset schemas are bundled in
`app/src/main/assets/presets.json`.

## Architecture and behavior

```text
State + preset questions → Kotlin BPE tokenizer and prompt builder
  → memory-mapped token embedding lookup → LiteRT main graph
  → raw-probability features + LiteRT action head
  → calibrated Kotlin decoder → immutable UI state → Compose cards
```

`MainActivity` observes immutable state, and `MainViewModel` owns the engine on a serialized
model dispatcher. A complete tokenizer → builder → embedding → graph → decoder pass warms
before **Ready**. The status header reports loading, compilation, and launch-to-Ready time.
Missing assets identify the first missing filename and the installer script.
The displayed launch interval starts at `Activity.onCreate` and ends after all five warm-up
questions; it includes tokenizer loading, table mapping, and model compilation, and excludes
operating-system process startup before the Activity callback. `LAYA_READY` logs the same
interval plus individual initialization timings.

Email uses subject and body fields; support and moderation each use one text field. JA/EN swaps
the invented prefill for the selected preset. Each preset runs five question rows sequentially.
The header's token count sums their real sequence lengths, including repeated state text and
special tokens but excluding padding. Each graph call has a static window of 256 positions;
state text is right-truncated after reserving the question and options. S512 is not exposed in
this screen.

Calibration is enabled by default. **T=1 (uncalibrated)** changes only option decoding; action
features and action probability use raw probabilities in both modes. Choice answers are labels;
score answers are expected zero-based rubric levels, with their legend; yes/no questions show
a continuous true probability. Confidence is the contract's entropy or binary confidence,
not a measured correctness rate. See the [host contract](docs/HOST_CONTRACT.md) for details.

Graph timings cover input writes, `run()`, and output readback for the main and action graphs.
Embedding lookup is measured separately. Request totals also include host preparation and
decoding. Compilation and automatic warm-up are initialization work.

## Reproduce host and device gates

The JVM suite uses a separate local fixture bundle and does not invoke model inference. Set
`LAYA_TEST_DATA` as described in [scripts/TEST_DATA.md](scripts/TEST_DATA.md), then run:

```bash
./gradlew --no-daemon :app:testDebugUnitTest
```

For a device gate, install the debug APK, models, and supplied S256 gate file, then use:

```bash
export ANDROID_SERIAL=your-device-serial
adb -s "$ANDROID_SERIAL" shell am force-stop com.laya
adb -s "$ANDROID_SERIAL" shell am start -W -n com.laya/.MainActivity \
  --ez gate true --es window 256 --es accel gpu --es storage wfp16
adb -s "$ANDROID_SERIAL" exec-out run-as com.laya cat files/laya_gate_gpu_256.json \
  > gate_gpu_wfp16_256.json
```

Wait for completion before pulling the report. Use `--es storage fp32` for the other graph or
`--es accel cpu` for CPU, and force-stop this app between configurations. The output filename
contains accelerator and window, so save each variant before starting another. Reports include
on-device token IDs, marker positions, numerical outputs, creation and per-row timings;
completion uses the `LAYA_GATE` log tag. The gate entry is debug-only; release launches the UI.

## Verified on

Host parity on Apple M4 Max, JDK 17: **402/402** builder IDs and marker positions, **300/300**
tokenizer stress strings, **402/402** official answer dictionaries at T=1, **402/402** calibrated
dictionaries, and **20/20** JSON serializations matched their captured Python references.
Embedding lookup matched NumPy for **402/402 rows / 118,554,624 float32 values**, maximum absolute
error **0**, including padded rows. All **65,536** binary16 bit patterns also matched NumPy's
float32 conversion. These are implementation parity checks, not task accuracy measurements.

Both S256 embedding-input graphs passed Mac CPU parity on **201/201** rows with **81/81**
choice/score argmax matches and finite outputs: maximum probability error versus captured
four-decimal dictionaries was **0.0001 FP32** and **0.0014 WFP16**. Runtime: ai-edge-litert
2.1.6 `CompiledModel`, four CPU threads. Device runtime is LiteRT 2.2.0.

Galaxy S26 SM-S942Q, Android 16, LiteRT 2.2.0, no other workload, screen on.
Both main and act graphs run in one full GPU partition: WFP16 main 1779/1779 ops,
FP32 main 1680/1680 ops, shared act 4/4 ops. GPU precision is explicitly FP32.
Each gate has 201/201 exact on-device IDs and markers, 81/81 choice/score argmax
matches and finite outputs. The probability comparison includes noul and act_probability.

| Accelerator × storage (debug build) | Max Δp | Creation ms | Cold main + act ms | Warm median [min, max] ms, 200 rows |
|---|---:|---:|---:|---|
| GPU FP32 arithmetic × WFP16 storage (selected) | 0.0014 | 1247.432 | 54.535 | 50.881 [49.980, 53.508] |
| GPU FP32 arithmetic × FP32 storage | 0.0001 | 1217.176 | 51.233 | 50.754 [49.771, 53.570] |
| CPU × WFP16 storage | 0.0014 | 304.975 | 97.603 | 163.020 [92.216, 190.245] |

These per-question times include main+act writes, run enqueue and readback, excluding
host preparation, embedding lookup and decode. Selected GPU debug embedding lookup
has warm median 14.566 ms [2.342, 22.772]; the tokenizer loads in 1520.317 ms and the
table maps in 0.220 ms. Table pages are accessed during lookup, not eagerly loaded by mmap.

| GPU WFP16 UI build | Launch→Ready, two fresh processes (ms) | First Run (ms) | Second Run in those processes (ms) |
|---|---:|---:|---:|
| Debug | 3194.32, 3161.64 | 338.30, 346.72 | 331.67, 335.93 |
| Non-debuggable release, isolated debug key | 2296.08, 2298.29 | 301.33, 302.90 | 300.03, 295.48 |

UI totals cover the invented Japanese email preset: five questions, 529 prompt tokens,
calibration enabled. Ready follows automatic full-pipeline warm-up. Launch→Ready starts
at Activity.onCreate, not at operating-system process creation. Two launches per build
are observations, not a latency guarantee. Release per-question buffer timing was not
measured separately; the 50.881 ms gate value above is from the debug build.

The Python host in the model repository (`laya_host.py`) passed the same 201-row check on Mac CPU
for all four graphs, with 81/81 argmax matches each. S512 has not been run on a device.

## License

This sample is [Apache-2.0](LICENSE). Laya code, checkpoint, and tokenizer assets declare
Apache-2.0; mmBERT-base declares MIT. ModernBERT, Transformers, and tokenizers attribution and
license texts are retained under [licenses/](licenses/README.md) and in [NOTICE](NOTICE).
Model conversions move token lookup to the host, freeze graph shapes and attention constants,
split the action head, and optionally store fully connected weights in FP16. Host logic is
ported to Kotlin. Calibration provenance remains in its JSON; raw calibration datasets are
not included in this module.
