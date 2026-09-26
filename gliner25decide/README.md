# GLiNER2.5 Decide — Android text classification

Type English text and a list of decisions to make about it, one per line, such as
`intent: refund_request, replacement_request, order_status`. Choose GPU or CPU and tap
**Classify** for the chosen label of every task, its probability, the graph window and timings.
Multi-label tasks return every label above their threshold; a task can also carry a question
(`prompt`) about the text. For the prefilled request (an invented customer message with four tasks)
the app answers intent `refund_request`, urgency `high`, sentiment `negative` and topics
`shipping`, `product_quality`, `battery`, the same decisions as the official gliner2 model.

## Model and requirements

- Model: [litert-community/GLiNER2.5-Decide-LiteRT](https://huggingface.co/litert-community/GLiNER2.5-Decide-LiteRT).
- Upstream: [fastino/GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) revision
  `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6`, Apache-2.0; DeBERTa-v3-large encoder (MIT).
- Semantics: gliner2 2.0.0 `classify_text` (classification heads of the span architecture).
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile/target SDK 35.
- Runtime: LiteRT **2.2.0** `CompiledModel`; Material 1 Compose with MVVM.

**Explicit GPU FP32 precision is mandatory.** On a Galaxy S26 the default GPU precision compiled
the s128 graph but changed 28 of 42 decisions. The app uses
`CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)`. `wfp16` describes
stored weights; host tensors and computation remain float32.

## Download, build and install

Use JDK 17, Android SDK platform 35 / build-tools 35.0.0, Android platform-tools (`adb`) and the
Hugging Face CLI (`hf`). The app needs five files of the model repository:

```bash
export HF_HUB_DISABLE_XET=1
hf download litert-community/GLiNER2.5-Decide-LiteRT \
  gliner25_decide_s128_wfp16.tflite gliner25_decide_s256_wfp16.tflite \
  gliner25_decide_s512_wfp16.tflite host_assets/word_embeddings_fp16.bin host_assets/tokenizer.json \
  --local-dir "$HOME/Downloads/GLiNER2.5-Decide-LiteRT"
./gradlew :app:assembleDebug
# Optional when multiple devices are connected:
# export ANDROID_SERIAL=your-device-serial
adb install app/build/outputs/apk/debug/app-debug.apk
./scripts/install_to_device.sh "$HOME/Downloads/GLiNER2.5-Decide-LiteRT"
adb shell am start -n com.gliner25decide/.MainActivity
```

The install script's argument defaults to the download directory above. It validates all five
files before contacting the device, pushes through `/data/local/tmp/gliner25decide/`, copies with
`run-as com.gliner25decide` into private `files/` and removes the temporary files. Install the debug
APK first; `run-as` requires a debuggable package. Model files are external and are not needed to
build the APK.

## External files

The app loads graphs by path and memory-maps the embedding table. Five files, 2,452,978,688 bytes.

| File | Bytes | Purpose |
|---|---:|---|
| `gliner25_decide_s128_wfp16.tflite` | 660,383,872 | 128 encoded tokens (schema + text) |
| `gliner25_decide_s256_wfp16.tflite` | 710,715,520 | 256 encoded tokens |
| `gliner25_decide_s512_wfp16.tflite` | 811,378,816 | 512 encoded tokens |
| `host_assets/word_embeddings_fp16.bin` | 262,166,528 | Float16 `[128011,1024]` embedding table, upcast to float32 on lookup |
| `host_assets/tokenizer.json` | 8,333,952 | SentencePiece Unigram vocabulary and normalization |

Each graph takes `inputs_embeds [1,N,1024]`, `attention_mask [1,N]` and `label_routing [1,32,N]`
(float32) and returns `logits [1,1,1,32]`, one logit per label slot. The model repository's
`HOST_CONTRACT.md` specifies how the host builds them.

## Tasks

One task per line:

```text
intent: refund_request, replacement_request, order_status, technical_support, other
topics: shipping, product_quality, billing, battery | multi 0.4
answer: yes, no | prompt: Did the parcel arrive late?
```

`| multi` makes a task multi-label (sigmoid, every label at or above the threshold, 0.5 by
default; the top label alone when none is). Without it the task is single-label (softmax, top
label). `| prompt: …` adds a question to the task. Labels cannot contain commas in this editor.
Label descriptions (`{label: description}` in gliner2) are supported by `Task.labelDescriptions`
but not by the editor.

## Architecture and limits

```text
Text + tasks → Kotlin word splitter / Unigram tokenizer / gliner2 schema tokens
  → IDs, attention, one-hot label routing + memory-mapped float16 table upcast to float32
  → LiteRT graph (GPU FP32 or CPU) → 32 logits
  → Kotlin decoder (softmax or sigmoid, argmax or threshold) → decisions
```

The host ports gliner2 2.0.0: `( [P] task[: prompt] ( [L] label … ) )` per task, tasks joined by
`[SEP_STRUCT]`, then `[SEP_TEXT]` and the lower-cased text words, every token tokenized on its own;
the text gets a final "." unless it ends with ".", "!" or "?". `MainActivity` observes immutable
state; `MainViewModel` owns the classifier on a confined dispatcher. One LiteRT Environment is
shared per process. s128 compiles at startup; s256/s512 compile on first use and stay resident.
Before Ready the app repeats the bundled example through the whole pipeline five times; each
window/backend also gets one untimed pass. Graph timing covers the first input write through output
readback because `run()` is asynchronous.

English only. The smallest fitting window of 128/256/512 encoded tokens is chosen; more than
512 tokens or more than 32 labels in total is rejected, never truncated (gliner2's long-text
chunking is not ported). Few-shot examples are not supported. Task lists that gliner2 would
decode under another task's settings (a name that prefixes another task's `name: prompt`), empty
tasks and duplicate names are rejected.

## Check it on a device

The debug APK bundles the device-gate fixtures: 84 captured official requests, 42 per window
(the README examples of the model card in all three, plus the longest public dev rows of
fastino/fast-decisions that fit each window).

```bash
adb shell am force-stop com.gliner25decide
adb shell am start -W -n com.gliner25decide/.MainActivity --ez gate true
# One backend: append --es accel GPU or --es accel CPU.
adb exec-out run-as com.gliner25decide cat files/gate/gate_GPU.json > gate_GPU.json
```

For every (fixture, window) pair the gate checks that the on-device token IDs, `[L]` positions and
padding equal the captured Python inputs, that the decisions equal the official result, and
records max |Δlogit| / |Δprob| against the fp32 oracle, max |Δlogit| against desktop LiteRT CPU on
the same graph and table, and timing medians (one warm-up, three timed runs). Completion lines use
`DECIDE_GATE`. `--ez firsttap true` (debug) records the first classification after a normal
startup in `files/gate/first_tap.json`; `--ez paced true` (debug, optional `--ei count` and
`--ei interval_ms`) classifies the bundled example repeatedly and writes `files/gate/paced_<BACKEND>.json`.
With these debug extras the activity shows above a secure lock screen so that the measured process
is in the foreground; a normal launch is unchanged.

The JVM tests skip unless `-Pgliner.fixtures=/path/to/data` is set; see
[scripts/TEST_DATA.md](scripts/TEST_DATA.md).

## Verified on

Samsung Galaxy S26 SM-S942Q (SM8850), Android 16, LiteRT 2.2.0, debug build, app in the foreground
(above the lock screen, see above), USB powered; every run started after the GPU had cooled to
≤ 50 °C with thermal status 0.

- Gate, GPU FP32 and CPU (four threads): **126/126** (fixture, window) pairs give the official
  decisions on each backend, all logits finite; on-device token IDs, marker positions and padding
  equal the captured Python inputs at all 126 pairs. Max |Δlogit| against the fp32 oracle:
  4.78e-3 GPU, 4.82e-3 CPU. Every window compiles fully onto the GPU delegate (1,780/1,780 nodes,
  one partition).
- First request after a cold start (GPU, s128, the bundled example: 110 encoded tokens, 15 labels):
  five warm-up passes took 0.38–0.39 s before Ready; the request then took **77.9 / 72.4 / 75.5 ms**
  end to end in three cold starts (graph 65.8–68.4 ms, tokenize + embed 5.5–9.7 ms, decode ≤ 0.2 ms).
- Paced, 20 requests one every 2 s at s128: median end to end **93.0 ms** on GPU (graph median
  68.4 ms) and **171.9 ms** on CPU. On GPU the tokenize + embed phase rose from about 13 ms to about
  28 ms after the tenth request; the cause was not established.

Graph latency of the same wfp16 graphs through a native CompiledModel runner (GPU FP32, float16-table
inputs, 42 requests per window, 2 warm-ups, then 8 timed runs at s128 and 5 at s256/s512 per
request, each run input write → run → output readback, cool start, screen on):

| Graph | Backend | Decisions vs official | Median / min ms | Opening → closing request ms |
|---|---|---|---:|---:|
| s128 wfp16 | GPU FP32 | 42/42 | 69.6 / 65.9 | 66.3 → 69.9 |
| s256 wfp16 | GPU FP32 | 42/42 | 198.8 / 174.7 | 174.9 → 214.8 |
| s512 wfp16 | GPU FP32 | 42/42 | 733.1 / 573.5 | 575.2 → 912.1 |
| s256 wfp16 | CPU (XNNPACK) | 42/42 | 510.0 / 311.5 | 314.2 → 518.1 |
| s512 wfp16 | CPU (XNNPACK) | 42/42 | 1,666.0 / 863.1 | 879.9 → 1,785.2 |

The s256/s512 runs slowed down within each job as the GPU clock limit stepped down from 1,300 MHz
(to 902 MHz at s256 and 646 MHz at s512), so the median mixes the cool start and the throttled end;
the opening request is closer to a single request on a cool phone.
