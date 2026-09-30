# Julia-1 decisions — Android typed decisions on the GPU

Type a short text (a support ticket, an agent trace, a product review), pick a preset, and get
one answer per question on the phone: which team should handle it (`choice`), when the customer
needs a reply (`score`), whether a refund is requested (`noul`). Every question is one graph call;
the app reads the option probabilities the model returns and shows them as bars. The presets are
the app's own question sets; the example texts are invented.

## Model and requirements

- Model download: [litert-community/Julia-1-LiteRT](https://huggingface.co/litert-community/Julia-1-LiteRT)
  (revision `b92a0d21eb1aedad6afdc815e027a47896d16eb3`).
- Upstream: [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1), revision
  `a85b127321d580d65176c89ced8273f305745d85` (144.3M parameters, mmBERT-small encoder with a
  decision head, Apache-2.0).
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile/target SDK 35.
- Runtime: LiteRT **2.2.0** `CompiledModel`; Kotlin and Material 1 Compose with MVVM.
- Build: JDK 17, Android SDK platform 35 and build-tools 35.0.0.

The GPU path requests `CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)`
explicitly. The answers change under fp16 arithmetic (the NPU and the GPU's default precision;
see the model card), so the app offers **GPU FP32** and **CPU** only. A compile error appears
inline with the runtime message; **Use CPU** is an explicit user choice. The app remembers the
selected accelerator.

## Download, build and install

Commands below run from this module directory. Model files are external and are not required to
build the APK. Install Android platform-tools (`adb`) and the Hugging Face CLI (`hf`) for device
installation. The installer uses Python 3.9+ and SDK Build Tools 35.0.0 `aapt` to verify that the
APK is the debuggable `com.julia1` app. Set `ANDROID_HOME` to your SDK directory, or provide the
`AAPT` executable explicitly; `PYTHON` can select a private Python interpreter.

```bash
export HF_HUB_DISABLE_XET=1
export HF_HOME="$PWD/.cache/huggingface"
export JULIA1_MODEL_DIR="$PWD/model-files"
hf download litert-community/Julia-1-LiteRT \
  julia1_s512_fp32.tflite julia1_token_table_fp16.bin tokenizer.json \
  --local-dir "$JULIA1_MODEL_DIR"
./gradlew --no-daemon clean :app:assembleDebug
./scripts/install_to_device.sh --assets "$JULIA1_MODEL_DIR" --validate-only
export ANDROID_SERIAL=your-device-serial
adb -s "$ANDROID_SERIAL" install -r app/build/outputs/apk/debug/app-debug.apk
./scripts/install_to_device.sh --assets "$JULIA1_MODEL_DIR"
adb -s "$ANDROID_SERIAL" shell am start -n com.julia1/.MainActivity
```

The installer validates the files against `scripts/assets.json` (sizes and SHA-256) before
contacting the device, stages them on the device and copies them into this app's private files
with `run-as com.julia1`. Install the debug APK first because `run-as` needs a debuggable package.
Add `julia1_s1024_fp32.tflite` to the download and pass `--window both` to install the 1,024-token
graph as well: the app then answers requests longer than 512 tokens with it (compiled on first
use). Without it such a request shows the strict-encoding error instead of a truncated answer.

## External files

| File | Bytes | Purpose |
|---|---:|---|
| `julia1_s512_fp32.tflite` | 185,095,348 | Graph for a 512-token window, float32 weights |
| `julia1_token_table_fp16.bin` | 196,608,000 | Little-endian float16 token table, shape `[256000,384]` |
| `tokenizer.json` | 34,363,188 | The checkpoint's tokenizer, unchanged |
| `julia1_s1024_fp32.tflite` | 188,503,220 | Optional graph for a 1,024-token window |

The required three total **416,066,536 bytes**. The token table is memory-mapped read-only. For
the debug fixture gate, also pass `--fixtures DIR` with the files described in
[scripts/TEST_DATA.md](scripts/TEST_DATA.md). The three preset schemas are in
`app/src/main/assets/presets.json`.

## Architecture and behavior

```text
State text + preset questions → Kotlin BPE tokenizer and strict sequence builder
  → memory-mapped float16 token lookup → LiteRT graph (token_logits)
  → float64 softmax at the marker positions → immutable UI state → Compose cards
```

`MainActivity` observes immutable state, and `MainViewModel` owns the engine on a serialized
model dispatcher. A complete tokenizer → builder → lookup → graph → decoder pass over the preset
runs before **Ready**; the status header reports loading, compilation and launch-to-Ready time.
Missing files name the first missing filename and the installer script.

One question is one graph call with a static 512-token window: `<bos>`, `"{type} question:
{instructions}"`, `<eos>`, one `<mask>` marker followed by each option text, `<eos>`, the state,
`<eos>`. The options are the criteria descriptions themselves (choice values, rubric strings,
`[false, true]` descriptions or the literal words). Nothing is truncated: a request that does not
fit raises `EncodingException`, as the author's strict encoding does. See the
[host contract](docs/HOST_CONTRACT.md).

Answers follow the author's named-question API: `choice` is the option ID with the highest
probability, `score` is the expected zero-based rubric index (Σ i·pᵢ, shown with four decimals),
`noul` is the probability of true. The checkpoint ships no calibration, so probabilities are the
plain softmax at temperature 1. Each card shows the window used and the request's token count;
timings cover tokenizer, builder, lookup, graph call with readback, and decode.

## Reproduce host and device gates

The JVM suite uses a separate local fixture bundle and does not invoke model inference. Set
`JULIA1_TEST_DATA` as described in [scripts/TEST_DATA.md](scripts/TEST_DATA.md), then run:

```bash
./gradlew --no-daemon :app:testDebugUnitTest
```

For a device gate, install the debug APK, the model files and the fixtures (`--fixtures DIR`),
then use:

```bash
export ANDROID_SERIAL=your-device-serial
adb -s "$ANDROID_SERIAL" shell am force-stop com.julia1
adb -s "$ANDROID_SERIAL" shell am start -W -n com.julia1/.MainActivity \
  --ez gate true --es accel gpu --es window 512
adb -s "$ANDROID_SERIAL" exec-out run-as com.julia1 cat files/julia1_gate_gpu_512.json \
  > gate_gpu_512.json
```

Wait for the `JULIA1_GATE` summary line in logcat before pulling the report. Use
`--es accel cpu` for the CPU and force-stop the app between configurations. The report's
`encoding` block counts the on-device ids, markers and question types against the reference host
on all 2,100 fixture requests; its `rows` carry the marker logits and per-row timings of the 706
captured rows, in the layout `conversion/device_compare.py` in the model repository reads. The
gate entry is debug-only; release launches the UI.

## Verified on

Host parity on Apple M4 Max, JDK 17 (2026-09-30): **300/300** tokenizer stress strings,
**2,065/2,065** builder ids, markers and question types on the oracle requests that fit 512 tokens
and **35/35** strict rejections of the longer ones (**2,100/2,100** at 1,024), decoder
probabilities within **4.4e-16** of the Python arithmetic on 2,100 rows, **65,536/65,536** float16
bit patterns and **589,824/589,824** looked-up table values bit-exact with NumPy. These are
implementation parity checks, not task accuracy measurements.

Galaxy S26 SM-S942Q, Android 16, LiteRT 2.2.0, debug build, screen on, thermal status 0 before
and after the run, battery 32.6 °C to 37.8 °C (2026-09-30). GPU: 1,704/1,704 operators in one
`LITERT_CL` partition, precision explicitly FP32, compilation 801 ms. On the device the tokenizer
and builder reproduced the reference host's ids, markers and question types on **2,065/2,065**
requests and rejected the **35** that need more than 512 tokens (2.4 s for all 2,100). On the 706
captured rows (all 306 boundary rows, where the author's top probability is below 0.9, plus 400
others) every answer matched the author's runtime:

| Accelerator (S512 graph, float16 table) | Rows | Same argmax | Max Δp | Rows over 0.01 | Warm median [min, max] ms, 701 rows |
|---|---:|---:|---:|---:|---|
| GPU, FP32 precision | 706 | 706 | 0.0077 | 0 | 81.5 [56.8, 103.0] |

The per-row time covers the input writes, `run()` and the output readback; the table lookup
(about 12 ms per 512-token row in Kotlin) and the tokenizer are separate. The 0.0077 is the
float16 token table's effect, the same value the model card reports for the Python host on
desktop CPU. The Kotlin decoder's probabilities on the device logits differ from the Python
softmax by at most 2.2e-16. The tokenizer loads in about 1.5 s in the debug build.

| UI build (Support ticket preset, 3 questions) | Launch→Ready (ms) | Run (ms) |
|---|---:|---:|
| Debug, GPU FP32 | 2,557.10 | 229.72 |

UI totals cover the three preset questions (151 prompt tokens; the first card alone took 70.9 ms for a 51-token request). Launch→Ready splits into tokenizer load 1,482.5 ms, GPU compilation 851.7 ms and warm-up 210.9 ms; thermal status 0, battery 34.6 °C. Ready follows the automatic
full-pipeline warm-up; Launch→Ready starts at `Activity.onCreate`. These are observations from one
run, not a latency guarantee. The CPU path and the S1024 graph were not timed on the device.

## License

This sample is [Apache-2.0](LICENSE). The Julia-1 model artifacts are declared Apache-2.0 by
their README; mmBERT-small declares MIT. License texts and attribution are under
[licenses/](licenses/README.md) and in [NOTICE](NOTICE). The host logic is ported to Kotlin from
the model repository's `julia_litert.py`; the conversion itself is documented there.
