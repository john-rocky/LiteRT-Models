# GLiFormer Large NER on Android

Enter English text to find people, organizations, locations, products and dates on your phone.
The prefilled fictional example describes Mira Okafor unveiling a headset; tap **Extract** to see
highlighted entities, scores and Unicode code-point offsets. Choose GPU or the s128 CPU fallback,
and inspect the selected window and tokenize+lookup / graph / decode timings.

The app uses Kotlin preprocessing and decoding with LiteRT **2.2.0 CompiledModel**, Compose
Material 1 and MVVM. It requires arm64-v8a and Android 8.0 / API 26 or newer. Large models need
substantial memory: published native Galaxy S26 measurements reached **2.6 GB resident**,
**4.5 GB during compilation**, and **5.0 s to the first result**. Only one window is loaded at a time.

## Build, install and run

Use JDK 17, Android SDK platform 35 / build-tools 35.0.0, Android platform-tools and the Hugging
Face CLI. Set `ANDROID_HOME` to your SDK or configure `sdk.dir` in an untracked `local.properties`.
Open this directory in Android Studio or run these commands from its root:

```bash
export HF_HUB_DISABLE_XET=1
export MODEL_DIR=models/GLiFormer-Large-NER-LiteRT
hf download litert-community/GLiFormer-Large-NER-LiteRT \
  --include "gliformer_large_ner_s128_wfp16.tflite" "gliformer_large_ner_s256_*_wfp16.tflite" \
  "host_assets/*" "requirements-lock.txt" --local-dir "$MODEL_DIR"
./gradlew :app:assembleDebug
export ANDROID_SERIAL=your-device-serial
adb -s "$ANDROID_SERIAL" install -r app/build/outputs/apk/debug/app-debug.apk
./scripts/install_to_device.sh --serial "$ANDROID_SERIAL" "$MODEL_DIR"
adb -s "$ANDROID_SERIAL" shell am start -n com.gliformer/.MainActivity
```

Models come from the [GLiFormer Large NER LiteRT repository](https://huggingface.co/litert-community/GLiFormer-Large-NER-LiteRT).
The installer requires an explicit serial and checks every local source before contacting the
phone. It pushes through a dedicated `/data/local/tmp/gliformer_*` directory, copies into
`com.gliformer`'s private `files/` with `run-as`, and removes temporary files on exit. Install the
debug APK first: `run-as` requires a debuggable app. No models are bundled in the APK.

Default files are the s128 wfp16 graph, `word_embeddings_fp16.bin`, `tokenizer.json`,
`tokenizer_config.json`, `gliner_config.json` and `graph_contract_s128.json`. Graphs are at the
model repository root; the other files are under `host_assets/`. Add `--with-s256` for the s256
wfp16 encoder/head and contract. Add `--fp32-table` for the optional diagnostic table.

```bash
./scripts/install_to_device.sh --serial "$ANDROID_SERIAL" --with-s256 \
  "$MODEL_DIR"
./gradlew :app:assembleBenchmark
adb -s "$ANDROID_SERIAL" install -r app/build/outputs/apk/benchmark/app-benchmark.apk
```

Benchmark is non-debuggable, non-minified, and signed with the same local debug key. Existing
private model files survive replacement. Restore the debug APK to install more files or retrieve
private JSON with `run-as`. Release contains neither the gate Activity nor active first-tap
reporting. The wrapper defaults to a Gradle cache, signing key and temporary files under `.local/`;
build outputs and downloaded models are ignored by Git. SDK auto-installation is disabled.

## Execution and limits

| Window | Encoded tokens / word slots | Execution |
|---|---:|---|
| s128 | 128 / 48 | One full graph on GPU FP32, or selectable CPU |
| s256 | 256 / 256 | GPU FP32 encoder, CPU head, float32 host handoff |
| s512 | 512 / 512 | Same split when separately installed; device validation pending |

Both limits include the complete prompt and punctuation word slots. The smallest fitting window
is selected; oversized inputs are rejected without truncation. The installer supports s128 and
optional s256. GPU compilation explicitly uses
`CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)`; default GPU
precision does not produce the expected entities. Heads run on CPU because GPU head compilation
fails in the published validation. Stored wfp16 weights still use float32 inputs and outputs.

The app defaults to the **fp16 embedding table**, upcast to float32 during memory-mapped lookup.
It occupies 262,160,384 bytes on disk; the optional fp32 table occupies 524,320,768 bytes. Mapped
pages become resident on demand. Python `HostRuntime` defaults to fp32, so comparisons for this
app must explicitly use `HostRuntime(table="fp16")`. The fp32 references are separate diagnostics.
To use an installed fp32 table, launch MainActivity with `--es table fp32` in a fresh app process.

`MainActivity` observes immutable `UiState`; `MainViewModel` owns the extractor. The unchanged
extractor serializes every native call on one process worker and shares one Environment. It
releases previous models before switching windows. The complete tokenize → lookup → graph →
decode pipeline runs **12 untimed passes before Ready**, including after backend/window changes.
Graph timing includes input writes through output readback; it never stops at `run()` submission.

The Kotlin host follows GLiFormer 0.1.2 / GLiNER 0.2.29 and the published
[host contract](https://huggingface.co/litert-community/GLiFormer-Large-NER-LiteRT/blob/main/HOST_CONTRACT.md):
ordinary-token `[SEQ]`, schema and five entity markers, double separator, first-subtoken routing,
start/end pairing, inside checks, outside-neighbor scoring and flat overlap removal at threshold
0.5. Offsets are half-open Unicode code points. The UI converts them to UTF-16 with
`offsetByCodePoints` before highlighting. Supported quality is English NER with these five labels.

## Validation and device reports

See [scripts/TEST_DATA.md](scripts/TEST_DATA.md) to generate the external CPU references and run
all 15 JVM tests. Captured inputs match 80/80, including 400 routing/mask arrays; the matched-fp16
decoder matches Python/oracle span sets and Python ordering for 140/140 entries. Maximum score
difference is 0 versus Python on the same logits and 0.0006446243 versus the official oracle.
Without external data, five parity tests skip; those skips are not validation evidence.

The debug APK contains 289,090 bytes of input captures. Large float arrays and reference logits
are loaded from private `files/gate_fixtures/`, populated with `--fixtures test-data`:

```bash
./scripts/install_to_device.sh --serial "$ANDROID_SERIAL" --with-s256 \
  --fixtures test-data "$MODEL_DIR"
adb -s "$ANDROID_SERIAL" shell am force-stop com.gliformer
adb -s "$ANDROID_SERIAL" shell am start -W -n com.gliformer/.GliformerGateActivity \
  --es backend gpu --ei window 128 --es table fp16 --es report gate_s128_gpu.json
adb -s "$ANDROID_SERIAL" exec-out run-as com.gliformer cat files/gate_s128_gpu.json > gate_s128_gpu.json
```

Wait until the report status is final before retrieving it. For CPU, use `--es backend cpu` and a
new report basename. For the split gate use `--ei window 256 --ez forceWindow true`: it runs the
five inputs requiring s256 plus 60 short inputs. Each gate checks all 80 tokenizer captures and
the Unicode fixture, then runs five measured repetitions per selected input after 12 full warm-up
passes. It reports span equality, finite values, score deltas against both tables, phase timings,
battery/temperature and process memory. Acceptance is identical spans to matched Python and the
oracle, score error ≤1e-5 versus matched Python and ≤1e-3 versus oracle. No tolerance is applied
to tokenizer/routing values: they must be identical.

For first-tap measurements, install benchmark after staging files, launch MainActivity with the
screen on, wait for Ready, and tap Extract five times. `GLIFORMER_UI` logs Activity-onCreate-to-Ready;
`GLIFORMER_TAP` logs actual button-to-result, state publication and completed Compose draw times.
Use `adb -s "$ANDROID_SERIAL" logcat -d --pid=APP_PID` with this app's PID. Draw completion is not
display presentation. Benchmark JSON is private; restore debug to retrieve `files/first_tap/`.

## Measured Galaxy S26 results

Measured on 2026-09-26 with Samsung Galaxy S26 SM-S942Q / Android 16, LiteRT 2.2.0,
wfp16 graphs and the default fp16 table. The correctness gates use the **debug APK**;
the separate first-tap session uses the **non-debuggable benchmark APK**. Each gate starts a
fresh app process and warms the complete pipeline 12 times. Model files were already installed;
OS file caches and runtime compilation caches were not cleared.

Every gate matched all **80 tokenizer captures and 400 routing/mask arrays** element-wise,
including the supplementary-Unicode check. All selected inference span sets matched both the
official oracle and the matched-fp16 Python reference, and every output was finite. s128 uses
60 unique texts; s256 uses those 60 forced to s256 plus the five texts that require s256.

| Debug backend | Passing inputs | Max score difference vs matched fp16 | Max vs oracle | Tokenize+lookup / graph readback / decode (ms) |
|---|---:|---:|---:|---:|
| s128 GPU FP32 | 60/60 | 3.874302e-06 | 0.000646472 | 13.15 / 115.58 / 0.55 |
| s128 CPU (4 threads) | 60/60 | 1.907349e-06 | 0.000646174 | 5.30 / 188.95 / 0.20 |
| s256 GPU encoder + CPU head | 65/65 | 4.23193e-06 | 0.000646472 | 12.09 / 525.67 / 0.50 |

Phase timings are medians of per-input medians, five measured requests per input. They exclude
model loading, warm-up, reference comparisons and report writing. Graph timing includes input
writes, execution and output readback; the split path includes the float32 host handoff and CPU
head. The thresholds are 1e-5 versus matched Python and 1e-3 versus the oracle. The separate
fp32-table diagnostic reached 0.0002841949;
it does not gate the fp16-default app. These checks measure implementation agreement on this
corpus, not general NER accuracy. s512 device execution remains unmeasured.

| Debug backend | Load / 12 warm-up passes (s) | After warm-up: RSS / peak RSS / totalPSS (GB) | After gate: RSS / peak RSS / totalPSS (GB) |
|---|---:|---:|---:|
| s128 GPU FP32 | 3.158 / 1.088 | 2.465 / 4.389 / 4.214 | 2.447 / 4.389 / 4.232 |
| s128 CPU (4 threads) | 1.982 / 1.489 | 2.685 / 2.714 / 2.611 | 2.691 / 2.756 / 2.637 |
| s256 GPU encoder + CPU head | 4.185 / 3.994 | 4.342 / 4.354 / 5.655 | 4.352 / 4.415 / 5.675 |

GB means decimal bytes. RSS and peak RSS are `/proc/PID/status` VmRSS and VmHWM; totalPSS is
`dumpsys meminfo` and includes graphics accounting. They are different measures and must not be
added. The first host memory snapshot is taken after loading and warm-up, before measured
requests; the JSON also records app memory immediately after model loading. VmHWM is the
process high-water mark, not a continuously sampled PSS peak. The s256 pair stays resident for
both snapshots. High memory use can limit operation on smaller phones.

For comparison, the published native harness (not an APK variant) measured 82.3 ms warm s128
execution, 2,591,805,440 bytes resident, 4,542,996,480 bytes compile peak and 5,023 ms to its first
result. Those figures explain the roughly **2.6 GB resident / 4.5 GB compile peak / 5 s first-load**
memory profile; the current app measurements above include its Kotlin host and UI.

The **benchmark** session kept the screen on and used five actual Extract-button taps on the
prefilled fictional sentence, with GPU FP32, s128 and the fp16 table. Startup runs all 12 full
warm-up passes before Ready. Timings start inside the actual button handler; UI automation
locates and taps the visible button, and is outside those request timings.

| Benchmark measurement | Time |
|---|---:|
| Activity.onCreate → Ready, including warm-up | 4775.41 ms |
| First / fifth tap → extraction result | 168.06 / 154.81 ms |
| First / fifth tap → completed Compose draw | 190.47 / 167.15 ms |

Compose draw completion is not display presentation. These are one startup and five requests,
not a distribution of independent cold launches. Temperature and battery were recorded around
each session; backend timings are not temperature-matched comparisons.

| Session | Battery before → after | Temperature before → after |
|---|---:|---:|
| s128 GPU FP32 | 80 → 80% | 33.6 → 37.7 °C |
| s128 CPU (4 threads) | 80 → 80% | 36.0 → 39.2 °C |
| s256 GPU encoder + CPU head | 80 → 80% | 38.8 → 41.3 °C |
| Benchmark first-tap session | 80 → 80% | 40.7 → 40.5 °C |

The sample is Apache-2.0. [NOTICE](NOTICE) and [licenses/](licenses/) retain model and third-party
attribution, including the distinction between DeBERTa code lineage and base-checkpoint licensing.
