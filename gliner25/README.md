# GLiNER2.5 Small — Android entity extraction

Enter English text to find people, organizations, locations, products and dates on your phone.
Choose GPU or CPU and tap **Extract** for highlighted spans, confidence, Unicode offsets and timings.
For the prefilled sentence, “Maya Chen from Orvane Robotics demonstrated the Veltrix 9 in Lisbon
on March 12, 2025.”, the five entities are Maya Chen, Orvane Robotics, Veltrix 9, Lisbon and
March 12, 2025.

## Model and requirements

- Model: [litert-community/GLiNER2.5-Small-LiteRT](https://huggingface.co/litert-community/GLiNER2.5-Small-LiteRT),
  model/host assets from revision `fc3a084765b555f82b9fc4652a60d049e75ac8c5`.
- Upstream: [fastino/gliner2.5-small-v1](https://huggingface.co/fastino/gliner2.5-small-v1), Apache-2.0.
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile/target SDK 35.
- Runtime: LiteRT **2.2.0** `CompiledModel`; Material 1 Compose with MVVM.

**Explicit GPU FP32 precision is mandatory:** default GPU precision produces NaN on this graph.
The app uses `CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)`.
`wfp16` describes stored weights; host tensors and computation remain float32.

## Download, build and install

Use JDK 17, Android SDK platform 35 / build-tools 35.0.0, Android platform-tools (`adb`), and the
Hugging Face CLI (`hf`). Open this directory in Android Studio or build from its root:

```bash
export HF_HUB_DISABLE_XET=1
hf download litert-community/GLiNER2.5-Small-LiteRT \
  --local-dir "$HOME/Downloads/GLiNER2.5-Small-LiteRT"
./gradlew :app:assembleDebug
# Optional when multiple devices are connected:
# export ANDROID_SERIAL=your-device-serial
adb install app/build/outputs/apk/debug/app-debug.apk
./scripts/install_to_device.sh "$HOME/Downloads/GLiNER2.5-Small-LiteRT"
adb shell am start -n com.gliner25/.MainActivity
```

The install script's argument defaults to the download directory above. It validates all nine
files before contacting the device, pushes through `/data/local/tmp/gliner25/`, copies with
`run-as com.gliner25` into private `files/`, removes temporary files, and lists the installed files.
It uses plain `adb`, including its optional `ANDROID_SERIAL` selection. Install the debug APK first;
`run-as` requires a debuggable package. Model files are external and are not needed to build the APK.

## External files

The app loads graphs by path and memory-maps the embedding table. The installer flattens
`host_assets/` into the app's `files/` directory; all nine files total **407,515,254 bytes**.

| Downloaded file | Bytes | Purpose |
|---|---:|---|
| `gliner25_small_s128_wfp16.tflite` | 54,051,424 | 128 encoded slots / 48 text words |
| `gliner25_small_s256_wfp16.tflite` | 63,906,288 | 256 encoded slots / 192 text words |
| `gliner25_small_s512_wfp16.tflite` | 84,111,584 | 512 encoded slots / 384 text words |
| `host_assets/word_embeddings_fp32.bin` | 196,624,896 | Float32 `[128011,384]` embedding table |
| `host_assets/tokenizer.json` | 8,341,713 | SentencePiece Unigram vocabulary and normalization |
| `host_assets/sparse_decoder_fp32.safetensors` | 467,956 | 16 sparse-decoder tensors |
| `host_assets/graph_contract_s128.json` | 3,772 | 17 packed output slices |
| `host_assets/graph_contract_s256.json` | 3,807 | 17 packed output slices |
| `host_assets/graph_contract_s512.json` | 3,814 | 17 packed output slices |

The 3,152-byte checkpoint configuration is bundled as `app/src/main/assets/gliner_config.json`.

## Architecture and limits

```text
Text + five-label schema → Kotlin word splitter / Unigram tokenizer
  → padded masks and routing + memory-mapped embedding lookup
  → LiteRT dense graph (GPU FP32 or CPU) → unpack 17 float32 tensors
  → Kotlin sparse candidate pooling / scoring / decoding → spans and Compose highlights
```

The host ports follow gliner2 2.0.0 and the published
[host contract](https://huggingface.co/litert-community/GLiNER2.5-Small-LiteRT/blob/main/HOST_CONTRACT.md).
`MainActivity` observes immutable state; `MainViewModel` owns the helper and confined dispatcher.
One LiteRT Environment is shared per process. s128 loads at startup; s256/s512 compile on first use
and remain resident. Each window/backend receives an untimed warm-up. Graph timing includes the
first input-buffer write through output readback because `run()` is asynchronous.
Before Ready, the app shows Warming up while repeating the bundled sentence through tokenization,
embedding lookup, the graph and sparse decoding on its model worker; displayed extraction timings
still measure the actual request, and lazy windows retain their untimed graph-and-decode pass.
Sparse scoring uses flat buffers and three persistent workers plus the calling thread. Each dot
product and erf series retains its reduction order; primitive stable sorts preserve duplicate
and tie handling. There is no JNI, float16 host arithmetic or additional runtime dependency.

English and the five fixed labels are supported: person, organization, location, product, date.
The smallest fitting N/T window is chosen from 128/48, 256/192, 512/384. Both schema-plus-text encoded
length and text word count must fit; input above **512 encoded tokens / 384 words** is rejected
without truncation. Offsets are half-open Unicode code-point indices, not UTF-16 indices. Confidence
threshold is 0.5. Overlapping highlights prefer higher confidence. Arbitrary schemas, multilingual
quality, batching and NPU execution are outside the validated configuration.

## Reproduce parity and profiling

The debug APK bundles reference spans and captured tokenizer positions. Stop this app before
starting a new check so previous Activity extras are cleared:

```bash
adb shell am force-stop com.gliner25
adb shell am start -W -n com.gliner25/.MainActivity --ez gate true
# Extended set: append --es set f1; one backend: append --es accel GPU or --es accel CPU.
adb exec-out run-as com.gliner25 cat files/gate/gate_GPU.json > gate_GPU.json
```

The default checks ten texts, one warm-up + five timed runs. `--es set f1` checks 70 texts,
one warm-up + three timed runs, plus exact token IDs and text/query positions. Reports land in
private `files/gate/gate_GPU.json` / `gate_CPU.json`, or `gate_f1_GPU.json` / `gate_f1_CPU.json`.
Completion lines use `GLINER_GATE`. Span sets must match, confidence differences must be <= `5e-3`,
and every packed output must be finite. A normal launch without extras opens the interactive UI.

For decoder stages, add `--ez profile true` to the F1 command: this selects five timed runs and
records stage medians in JSON and `GLINER_GATE`. Profiling is otherwise off. To compare ART modes,
`./gradlew :app:assembleBenchmark` builds a non-debuggable, non-minified APK signed with the local
debug key. Install `app/build/outputs/apk/benchmark/app-benchmark.apk` with `adb install -r` over
this sample, then run the same profiling command. Read stage lines with `adb logcat -d -s GLINER_GATE:I`.
Restore the debug APK to export the full private JSON with `run-as`; model files stay installed.

The six JVM tests skip unless `-Pgliner.fixtures=/path/to/data` (or `-Dgliner.fixtures=...`) is set.
[scripts/TEST_DATA.md](scripts/TEST_DATA.md) describes downloaded host assets and regenerated packed
references, which are not distributed. Reports stay under `app/build/`. With fixtures: six pass,
80 files / 70 unique texts / 225 input checks, 195/195 decoder pairs. Maximum confidence difference
is `4.172325134277344e-7` versus Python on identical packed data, `0.002693772315979004` versus fp32.

## Verified on

First request after startup (2026-09-20, S26 SM-S942Q, LiteRT 2.2.0, GPU FP32, debug, screen on/unlocked, three cold app processes at 96% / 34.5 °C, s128 / 39 tokens / 12 startup passes): median tokenize+embed / graph-to-readback / decode **5.49 / 14.21 / 12.49 ms**; warm-up **0.507–0.586 s** (excludes compilation); **2/3** requests met both ≤6 ms tokenize+embed and ≤15 ms decode (one decode was 17.65 ms); a separate screen-off trial met 0/3.

**Samsung Galaxy S26 SM-S942Q, Android 16, LiteRT 2.2.0**, 2026-09-20. Explicit **GPU FP32**;
CPU graph uses four threads. Debug session started at **99% battery / 33.5 °C**, USB connected,
warm after earlier validation; GPU then CPU in one process. **70 inputs, one warm-up + five timed
runs each**, diagnostic timing enabled. Medians include all timed runs per window; compilation and
input inspection are excluded. Graph timing covers first input write through output readback.

| Backend | Window N / T | Inputs / runs | Tokenize + embed (ms) | Graph (ms) | Decode (ms) |
|---|---|---|---:|---:|---:|
| GPU FP32 | 128 / 48 | 60 / 300 | 3.526 | 11.895 | 9.518 |
| GPU FP32 | 256 / 192 | 5 / 25 | 8.826 | 26.837 | 13.497 |
| GPU FP32 | 512 / 384 | 5 / 25 | 19.083 | 92.171 | 12.778 |
| CPU | 128 / 48 | 60 / 300 | 1.666 | 14.898 | 6.022 |
| CPU | 256 / 192 | 5 / 25 | 5.356 | 42.315 | 8.050 |
| CPU | 512 / 384 | 5 / 25 | 11.533 | 128.694 | 8.849 |

Both backends pass **70/70 inputs and span sets / 400 spans**, all **420** warm-up/timed outputs
finite. Maximum confidence differences: **0.00269240140914917 GPU**, **0.002690911293029785 CPU**.
All windows compile; GPU delegation is 1120/1120 nodes at s128 and 1121/1121 at s256/s512, each
in one LITERT_CL partition. The debug session ended at 33.8 °C; battery was 99%.

Decode before → after the host optimization, same 70 texts and one warm-up + five timed runs:

| Build / session start before → after | s128 (ms) | s256 (ms) | s512 (ms) |
|---|---:|---:|---:|
| Debug, 31.6 → 33.5 °C | 58.258 → 9.518 | 68.842 → 13.497 | 71.447 → 12.778 |
| Non-debuggable, no minification, 33.5 → 33.8 °C | 15.611 → 10.070 | 19.110 → 12.306 | 19.384 → 12.845 |

The non-debuggable variant removed about 72–73% of baseline decode time in this sequential
comparison. After optimization, the debug build meets the 12 ms s128 / 20 ms s512 decode targets.
Non-debuggable profiling ended at 36.1 °C / 99% battery. These are host-decode timings, not total
application latency; CPU scheduling and temperature affect the other phases too.
