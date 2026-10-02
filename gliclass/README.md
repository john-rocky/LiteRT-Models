# GLiClass Edge — Android zero-shot text classification

Type English text and the labels to choose from, one per line. Pick **Single label** (softmax, the
top label) or **Multi-label** (sigmoid, every label at or above a threshold), GPU or CPU, and tap
**Classify** for every label's score with the chosen labels highlighted, the graph window and
timings. An optional prompt (a task description) goes right before the text. For the prefilled
request (an invented support message with five labels) the app answers `connectivity problem`
(0.856), the same decision as the official gliclass pipeline.

## Model and requirements

- Model: [litert-community/GLiClass-Edge-v3.0-LiteRT](https://huggingface.co/litert-community/GLiClass-Edge-v3.0-LiteRT).
- Upstream: [knowledgator/gliclass-edge-v3.0](https://huggingface.co/knowledgator/gliclass-edge-v3.0)
  revision `df03993a2ed98e5e4a0d2dd7efbbd105abe874cf`, Apache-2.0; encoder
  [jhu-clsp/ettin-encoder-32m](https://huggingface.co/jhu-clsp/ettin-encoder-32m) (ModernBERT, MIT).
- Semantics: `gliclass` 0.1.20 `ZeroShotClassificationPipeline` (uni-encoder): single-label and
  multi-label classification with an optional prompt.
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile/target SDK 35.
- Runtime: LiteRT **2.2.0** `CompiledModel`; Material 1 Compose with MVVM.

**Explicit GPU FP32 precision is mandatory.** On a Galaxy S26 the default GPU precision ran the
128-token graph but changed decisions near the decision boundary: 475 of 482 single-label and 455
of 482 multi-label decisions matched the official pipeline. The app requests
`CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)`. CPU (four
threads) is a second backend you choose in the app; the app never switches to it on its own.

## Download, build and install

Use JDK 17, Android SDK platform 35 / build-tools 35.0.0, Android platform-tools (`adb`) and the
Hugging Face CLI (`hf`). The app needs four files of the model repository:

```bash
hf download litert-community/GLiClass-Edge-v3.0-LiteRT \
  gliclass_edge_v3_s128_fp32.tflite gliclass_edge_v3_s256_fp32.tflite \
  host_assets/tok_embeddings_fp16.bin host_assets/tokenizer.json \
  --local-dir "$HOME/Downloads/GLiClass-Edge-v3.0-LiteRT"
./gradlew :app:assembleDebug
# Optional when multiple devices are connected:
# export ANDROID_SERIAL=your-device-serial
adb install app/build/outputs/apk/debug/app-debug.apk
./scripts/install_to_device.sh "$HOME/Downloads/GLiClass-Edge-v3.0-LiteRT"
adb shell am start -n com.gliclass/.MainActivity
```

The install script's argument defaults to the download directory above. It validates all four
files before contacting the device, pushes through `/data/local/tmp/gliclass/`, copies with
`run-as com.gliclass` into private `files/` and removes the temporary files. Install the debug APK
first; `run-as` requires a debuggable package. Model files are external and are not needed to
build the APK. A launch before the install script ran shows "Model unavailable" with "Missing
<file>. Run scripts/install_to_device.sh, then retry."

## External files

The app loads the graphs by path and memory-maps the embedding table. Four files, 149,936,711
bytes.

| File | Bytes | Purpose |
|---|---:|---|
| `gliclass_edge_v3_s128_fp32.tflite` | 53,703,480 | Up to 128 tokens (labels + prompt + text) |
| `gliclass_edge_v3_s256_fp32.tflite` | 53,965,476 | Up to 256 tokens |
| `host_assets/tok_embeddings_fp16.bin` | 38,684,160 | Float16 `[50370,384]` token-embedding table, upcast to float32 on lookup |
| `host_assets/tokenizer.json` | 3,583,595 | Byte-level BPE vocabulary, merges and added tokens (the upstream file, unchanged) |

Each graph takes `inputs_embeds [1,N,384]`, `attention_mask [1,N]` and `label_routing [1,25,N]`
(float32) and returns `logits [1,1,1,25]`; the first n logits belong to the n labels. The model
repository also holds float16-weight graphs (`*_wfp16.tflite`, half the size); this app uses the
float32 graphs.

## App architecture

```text
Text + labels (+ prompt) → linearized string → Kotlin byte-level BPE tokenizer ([CLS] … [SEP])
  → IDs, attention, one-hot <<LABEL>> routing + memory-mapped float16 table upcast to float32
  → LiteRT graph (GPU FP32 or CPU) → 25 logits
  → Kotlin decoder (softmax argmax, or sigmoid ≥ threshold) → labels and scores
```

`MainActivity` observes immutable `UiState`. `MainViewModel` owns the classifier and runs every
model call on one confined dispatcher (`Dispatchers.Default.limitedParallelism(1)`), because LiteRT
reuses its native input and output buffers; `onCleared()` closes the classifier. One LiteRT
Environment is shared per process. s128 compiles at startup; s256 compiles on first use and stays
resident. Before Ready the app repeats the bundled example through the whole pipeline five times;
each window/backend also gets one untimed pass. Graph timing covers the first input write through
output readback because `run()` is asynchronous.

Labels go one per line, or on a single line separated by commas; up to 25 labels. Labels are
trimmed and empty lines are skipped. The request is linearized as the pipeline does it:

```text
<<LABEL>>label 1<<LABEL>>label 2…<<SEP>>{prompt}{text}
```

There is no separator between the prompt and the text. Single-label returns the top softmax
label; multi-label returns every label whose sigmoid is at least the threshold (0.5 by default),
in input order, or none.

The tokenizer follows `tokenizer.json` the way Hugging Face tokenizers applies it: added tokens
are split out first (`<<LABEL>>`, `<<SEP>>`, `[CLS]`; `[MASK]` takes the whitespace on its left),
each remaining segment is NFC-normalized and split on the normalized added tokens (runs of 2–24
spaces, `[unused0]`–`[unused82]`), and every remaining piece gets a leading space before the GPT-2
regex, the byte mapping and the BPE merges.

The smallest fitting window of 128/256 tokens is chosen; more than 256 tokens or more than 25
labels is rejected, never truncated (the pipeline would truncate at 1,024 tokens). A text, prompt
or label that contains `<<LABEL>>` is rejected. Few-shot examples and hierarchical labels of the
pipeline are not ported. The model is English.

## Files

| Path | Role |
|---|---|
| `app/src/main/java/com/gliclass/MainActivity.kt` | Compose host; reads the debug and benchmark launch extras |
| `app/src/main/java/com/gliclass/MainViewModel.kt` | Owns the classifier on the confined dispatcher: load, warm-up, classify, diagnostics |
| `app/src/main/java/com/gliclass/UiState.kt` | Immutable screen state and the result rows |
| `app/src/main/java/com/gliclass/LabelEditor.kt` | The label editor's format |
| `app/src/main/java/com/gliclass/view/GliclassScreen.kt` | The screen; `view/Theme.kt` and `view/Color.kt` hold the Material 1 theme |
| `app/src/main/java/com/gliclass/GliclassTokenizer.kt` | Byte-level BPE tokenizer read from `tokenizer.json` |
| `app/src/main/java/com/gliclass/GliclassInputs.kt` | Linearization, window choice, padding, label routing, float16 table lookup |
| `app/src/main/java/com/gliclass/GliclassClassifier.kt` | LiteRT `CompiledModel` calls (GPU FP32 or CPU), resident windows, timing |
| `app/src/main/java/com/gliclass/GliclassDecoder.kt` | Softmax and sigmoid decisions of the pipeline |
| `app/src/main/java/com/gliclass/GliclassGateFixtures.kt`, `GliclassGateRunner.kt` | Debug fixture gate |
| `app/src/main/java/com/gliclass/GliclassDiagnostics.kt` | First-request and paced reports (debug and benchmark builds) |
| `app/src/debug/assets/gate_fixtures.json` | 152 captured official requests for the debug gate |
| `app/src/test/java/com/gliclass/` | JVM parity tests against the official pipeline |
| `scripts/install_to_device.sh` | Copies the four external files into the app's `files/` |
| `scripts/TEST_DATA.md` | The gate asset and the optional JVM reference data |
| `LICENSE`, `NOTICE`, `licenses/` | Apache-2.0 text, attribution and the retained upstream licenses |

## Measured

Samsung Galaxy S26 SM-S942Q, Android 16, LiteRT 2.2.0, USB powered, thermal status 1, app in the
foreground (top-app). Medians are the middle value of the sorted list (the upper one of the two
for an even count).

- Fixture gate, debug build, GPU FP32 and CPU (four threads), 552 captured official requests (482
  at s128, 70 at s256): on each backend **552/552** requests gave the official single-label and
  multi-label decisions, all logits finite, and the on-device token IDs, `<<LABEL>>` positions,
  windows and padding equal the captured Python inputs for all 552. Max |Δlogit| against the
  fp32 oracle: 9.08e-3 GPU, 9.02e-3 CPU. Both windows run whole on the GPU delegate, in one
  partition.
- Graph (input write → readback), median over the gate requests: GPU s128 **5.86 ms**, s256
  **8.23 ms**; CPU s128 7.83 ms, s256 15.65 ms.
- Cold start, benchmark build, GPU, the prefilled request (61 tokens, five labels): Ready
  **0.59 s** after process start; the first request took **8.05 / 8.38 / 8.20 ms** end to end in
  three cold starts (graph 6.1 ms, tokenize + embed 1.9–2.1 ms).
- Paced, benchmark build, 20 requests one every 2 s: median end to end **13.15 ms** on GPU (min
  9.55, max 18.78; graph 6.92 ms, tokenize + embed 5.9 ms) and 22.36 ms on CPU.
- Not measured: Pixel phones, other Android devices, sustained runs.

## Tests

JVM tests compare the Kotlin host with the official pipeline and the Python `tokenizers` library.
They skip unless `-Pgliclass.fixtures=/path/to/data` is set; see
[scripts/TEST_DATA.md](scripts/TEST_DATA.md). With the conversion run's data: tokenizer 552/552
oracle strings and 261/261 stress strings, inputs 552/552 requests, decoder 552/552 requests, and
the 152 rows of the committed gate asset.

```bash
./gradlew :app:testDebugUnitTest -Pgliclass.fixtures=/path/to/data
```

The debug APK runs the fixture gate over the rows of `app/src/debug/assets/gate_fixtures.json`.
The committed asset holds 152 requests (SemIf, the model card's two examples and six invented
requests), all at s128. The 552-request gate above used an asset that the conversion run
regenerates with the ag_news and banking77 rows, whose text is not redistributed here.

```bash
adb shell am force-stop com.gliclass
adb shell am start -W -n com.gliclass/.MainActivity --ez gate true --es backend gpu --es report app_gate_gpu.json
adb exec-out run-as com.gliclass cat files/app_gate_gpu.json > app_gate_gpu.json
```

For every request the gate checks that the on-device token IDs, `<<LABEL>>` positions, window and
padding equal the captured Python inputs and that the single-label and multi-label (0.5) decisions
equal the official results, and records max |Δlogit| against the fp32 oracle and timing medians.
The report is `files/<report>.partial` while running. Completion lines use `GLICLASS_GATE`.
`--ez firsttap true` records the first classification after a normal startup, and `--ez paced
true` (optional `--ei count` and `--ei interval_ms`) classifies the bundled example repeatedly;
both work in the debug and the non-debuggable `benchmark` build and also print their reports to
logcat under `GLICLASS_GATE`. With these extras the activity shows above a secure lock screen so
that the measured process is in the foreground; a normal launch is unchanged.
