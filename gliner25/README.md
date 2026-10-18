# GLiNER2.5 Small and Multi — Android entity extraction

Enter text to find people, organizations, locations, products and dates on your phone.
Choose the model (Small for English, Multi for English and Japanese), the Multi word splitter,
GPU or CPU, and tap **Extract** for highlighted spans, confidence, Unicode offsets and timings.
For the prefilled Small sentence, “Maya Chen from Orvane Robotics demonstrated the Veltrix 9 in
Lisbon on March 12, 2025.”, the five entities are Maya Chen, Orvane Robotics, Veltrix 9, Lisbon
and March 12, 2025. Multi opens with the Japanese worked sentence of its card,
「星瀬澄香は京都で霧灯研究社の新製品『星糸端末』を紹介した。」, and finds 星瀬澄香 (person),
京都 (location), 霧灯研究社 (organization) and 星糸端末 (product); the names are invented.

## Model and requirements

- Small: [litert-community/GLiNER2.5-Small-LiteRT](https://huggingface.co/litert-community/GLiNER2.5-Small-LiteRT),
  model/host assets from revision `fc3a084765b555f82b9fc4652a60d049e75ac8c5`.
  Upstream: [fastino/gliner2.5-small-v1](https://huggingface.co/fastino/gliner2.5-small-v1), Apache-2.0.
- Multi: [litert-community/GLiNER2.5-Multi-LiteRT](https://huggingface.co/litert-community/GLiNER2.5-Multi-LiteRT),
  model/host assets from revision `7a2ae39a8271eb71f6166f5a81c8ae6965ad89be`.
  Upstream: [fastino/gliner2.5-multi-v1](https://huggingface.co/fastino/gliner2.5-multi-v1), Apache-2.0
  (encoder [microsoft/mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base), MIT).
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile/target SDK 35.
- Runtime: LiteRT **2.2.0** `CompiledModel`; Material 1 Compose with MVVM.

**Explicit GPU FP32 precision is mandatory:** default GPU precision produces NaN on the Small
graph and, per the Multi card, changes one of 70 span sets at s128.
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

For Multi, download the s128 graph and the host assets without the 768 MB float32 table, then
install them into `files/multi/` with `--multi`. Then choose **Multi** in the app.

```bash
hf download litert-community/GLiNER2.5-Multi-LiteRT gliner25_multi_s128_wfp16.tflite \
  --local-dir "$HOME/Downloads/GLiNER2.5-Multi-LiteRT"
hf download litert-community/GLiNER2.5-Multi-LiteRT --include "host_assets/*" \
  --exclude "host_assets/word_embeddings_fp32.bin" "host_assets/runtime/*" \
  --local-dir "$HOME/Downloads/GLiNER2.5-Multi-LiteRT"
./scripts/install_to_device.sh --multi "$HOME/Downloads/GLiNER2.5-Multi-LiteRT"
```

s128 holds 128 encoded tokens and 48 text slots: a Japanese sentence of up to 47 characters
before the appended period (the token limit can bind first), or English of up to 48 words and
punctuation marks, an appended period included. For longer text, add
`gliner25_multi_s256_wfp16.tflite gliner25_multi_s512_wfp16.tflite` to the first download command;
the installer copies them when they are present, and the app compiles them on first use.

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

Multi files go to `files/multi/`; the eight files of the s128 installation total
**592,962,252 bytes**, and the two optional larger graphs bring it to 1,054,379,244 bytes.

| Downloaded file (Multi) | Bytes | Purpose |
|---|---:|---|
| `gliner25_multi_s128_wfp16.tflite` | 191,876,560 | 128 encoded slots / 48 text slots |
| `gliner25_multi_s256_wfp16.tflite` (optional) | 211,168,608 | 256 encoded slots / 192 text slots |
| `gliner25_multi_s512_wfp16.tflite` (optional) | 250,248,384 | 512 encoded slots / 384 text slots |
| `host_assets/word_embeddings_fp16.bin` | 384,172,032 | Float16 `[250112,768]` table; selected rows upcast to float32 |
| `host_assets/tokenizer.json` | 16,035,853 | mDeBERTa SentencePiece Unigram vocabulary and normalization |
| `host_assets/sparse_decoder_fp32.safetensors` | 862,708 | 16 sparse-decoder tensors |
| `host_assets/config.json` | 3,151 | Checkpoint configuration |
| `host_assets/graph_contract_s128.json` | 3,956 | 17 packed output slices |
| `host_assets/graph_contract_s256.json` | 3,994 | 17 packed output slices |
| `host_assets/graph_contract_s512.json` | 3,998 | 17 packed output slices |

## Architecture and limits

```text
Text + five-label schema → Kotlin word splitter / Unigram tokenizer
  → padded masks and routing + memory-mapped embedding lookup
  → LiteRT dense graph (GPU FP32 or CPU) → unpack 17 float32 tensors
  → Kotlin sparse candidate pooling / scoring / decoding → spans and Compose highlights
```

The host ports follow gliner2 2.0.0 and the published host contracts
([Small](https://huggingface.co/litert-community/GLiNER2.5-Small-LiteRT/blob/main/HOST_CONTRACT.md),
[Multi](https://huggingface.co/litert-community/GLiNER2.5-Multi-LiteRT/blob/main/HOST_CONTRACT.md)).
`MainActivity` observes immutable state; `MainViewModel` owns the helper and confined dispatcher.
One LiteRT Environment is shared per process. s128 loads at startup; s256/s512 compile on first use
and remain resident. Switching models closes the previous model's graphs and table before the other
model loads. Each window/backend receives an untimed warm-up. Graph timing includes the
first input-buffer write through output readback because `run()` is asynchronous.
Before Ready, the app shows Warming up while repeating the model's bundled sentence through
tokenization, embedding lookup, the graph and sparse decoding on its model worker; displayed
extraction timings still measure the actual request, and lazy windows retain their untimed
graph-and-decode pass.
Sparse scoring uses flat buffers and three persistent workers plus the calling thread. Each dot
product and erf series retains its reduction order; primitive stable sorts preserve duplicate
and tie handling. There is no JNI, float16 host arithmetic or additional runtime dependency; the
Multi table's float16 rows are converted to float32 bit-exactly before the graph.

Small supports English and the five fixed labels: person, organization, location, product, date.
Multi supports the same labels in English with the `whitespace` splitter and in Japanese with the
`char` splitter. You choose the splitter; there is no language detection. `char` gives each
non-space character its own slot and keeps ASCII letter/digit runs together. As in gliner2 2.0.0,
Multi appends `.` to text that does not end in `.`, `!` or `?`, so a sentence ending in `。` uses
one more slot; highlights stop at the end of the typed text.
The smallest fitting N/T window is chosen from 128/48, 256/192, 512/384. Both schema-plus-text encoded
length and text word count must fit; input above **512 encoded tokens / 384 words** is rejected
without truncation. Offsets are half-open Unicode code-point indices, not UTF-16 indices. Confidence
threshold is 0.5. Overlapping highlights prefer higher confidence. Arbitrary schemas, other
languages, batching and NPU execution are outside the validated configuration.

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

For Multi, append `--es model multi`. The gate checks the two worked sentences of the card's
`host_assets/example.json` (English with `whitespace`, Japanese with `char`): exact encoded IDs and
first-subword / `[E]` positions, then one warm-up + five timed runs with the same span and
confidence rules. Reports land in `files/gate/gate_multi_GPU.json` / `gate_multi_CPU.json`.

```bash
adb shell am force-stop com.gliner25
adb shell am start -W -n com.gliner25/.MainActivity --ez gate true --es model multi --es accel GPU
adb exec-out run-as com.gliner25 cat files/gate/gate_multi_GPU.json > gate_multi_GPU.json
```

The debug APK also accepts `--es model small|multi`, `--es splitter whitespace|char` and
`--es text "<sentence>"`, which only places the sentence in the text field. `adb shell input text`
types ASCII only; for Japanese, quote the sentence for the device shell:

```bash
adb shell am start -n com.gliner25/.MainActivity --es model multi --es splitter char \
  --es text "'星瀬澄香は京都で霧灯研究社の新製品『星糸端末』を紹介した。'"
```

Each interactive extraction in the debug APK logs one `GLINER_RESULT` line with the text, model,
splitter, backend, window, timings and spans.

For decoder stages, add `--ez profile true` to the F1 command: this selects five timed runs and
records stage medians in JSON and `GLINER_GATE`. Profiling is otherwise off. To compare ART modes,
`./gradlew :app:assembleBenchmark` builds a non-debuggable, non-minified APK signed with the local
debug key. Install `app/build/outputs/apk/benchmark/app-benchmark.apk` with `adb install -r` over
this sample, then run the same profiling command. Read stage lines with `adb logcat -d -s GLINER_GATE:I`.
Restore the debug APK to export the full private JSON with `run-as`; model files stay installed.

The six Small JVM tests skip unless `-Pgliner.fixtures=/path/to/data` (or `-Dgliner.fixtures=...`) is set.
[scripts/TEST_DATA.md](scripts/TEST_DATA.md) describes downloaded host assets and regenerated packed
references, which are not distributed. Reports stay under `app/build/`. With fixtures: six pass,
80 files / 70 unique texts / 225 input checks, 195/195 decoder pairs. Maximum confidence difference
is `4.172325134277344e-7` versus Python on identical packed data, `0.002693772315979004` versus fp32.

The four Multi parity tests skip unless `-Pgliner.multiFixtures=/path/to/data` points at the Multi
download (captured inputs and packed outputs are optional, see TEST_DATA.md); four Multi host tests
need no data. With the 2026-10-02 Galaxy S26 GPU FP32 packed outputs of the conversion run: both
worked sentences match exactly; 80 captured inputs (35 English, 45 Japanese) match at all 225
fitting windows; decoding gives 70/70, 75/75 and 80/80 span sets at s128/s256/s512, with maximum
confidence difference `7.152557373046875e-7` versus Python on the same bytes and
`0.0014157295227050781` versus the official fp32 spans.

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

**GLiNER2.5 Multi, Samsung Galaxy S26 SM-S942Q, Android 16, LiteRT 2.2.0**, 2026-10-18. Debug APK,
s128 installation in `files/multi/`, airplane mode, thermal status 0 and battery 37.0–37.7 °C
throughout. CPU frequency limits varied: 2.23 GHz on both clusters (of 3.63 / 4.74 GHz) right before
each interactive run, full with the app stopped. The Multi gate ran one backend per launch, one
warm-up + five timed runs per sentence; medians:

| Backend | Sentence (splitter) | Encoded tokens / text slots | Tokenize + embed (ms) | Graph (ms) | Decode (ms) | Max confidence difference |
|---|---|---:|---:|---:|---:|---:|
| GPU FP32 | English (whitespace) | 49 / 15 | 3.223 | 24.778 | 5.691 | 4.4e-5 |
| GPU FP32 | Japanese (char) | 79 / 30 | 7.193 | 23.444 | 8.007 | 2.4e-5 |
| CPU, 4 threads | English (whitespace) | 49 / 15 | 3.591 | 35.488 | 5.485 | 4.4e-5 |
| CPU, 4 threads | Japanese (char) | 79 / 30 | 4.033 | 35.143 | 5.290 | 2.4e-5 |

Both sentences return the official span sets on both backends, with identical encoded IDs and
first-subword / `[E]` positions. In the interactive app on GPU FP32, the Japanese sentence's first
extraction after startup showed tokenize + embed / graph / decode **10.55 / 24.11 / 11.69 ms** and
12.97 / 30.58 / 11.38 ms in a second launch, with the four official spans. Without the s256 graph, a
60-slot English text shows “Missing gliner25_multi_s256_wfp16.tflite. Run scripts/install_to_device.sh
--multi, then retry.” The Small debug gate on the same phone right after: 10/10 texts on GPU and on CPU
(maximum confidence difference 2.4e-4).
