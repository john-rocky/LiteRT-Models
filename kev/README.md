# Kev Decide — typed decisions on Android with Kev-0.8B

Give the app a state (a text, or a JSON object or array) and typed questions: **noul** (yes or no,
answered as p(true)), **choice** (one of named options) and **score** (ordered levels, answered as
the expected level). Tap **Decide** and every question gets its answer with calibrated
probabilities, computed as the author's `to_answers` computes them (four decimals). Three invented
requests are bundled: a support ticket, an incident note and a product review. Everything runs on
the phone: a Kotlin tokenizer, one LiteRT graph call per question and the pointer head on the host.

## Model and requirements

- Model: [litert-community/Kev-0.8B-LiteRT](https://huggingface.co/litert-community/Kev-0.8B-LiteRT)
  (row-prefill graphs, pointer head and tokenizer).
- Upstream: [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) tag `v1.0`
  (commit `bf75a6a8`), Apache-2.0: a rank-16 LoRA and a pointer head on Qwen3.5-0.8B-Base
  (revision `dc7cdfe2`, Apache-2.0). Code: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)
  tag `kev-1.0`.
- Semantics: the author's `kev.api.to_record` → `kev.model.encode` / `rows_of` (one causal row per
  question) → `PointerHead` with the calibration temperature T = 2.3510958125672174 →
  `kev.api.to_answers`. The response JSON holds `model`, `answers` and `usage.input_tokens`.
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile/target SDK 35.
- Runtime: LiteRT **2.2.0** `CompiledModel`; Material 1 Compose with MVVM.

The app asks the GPU for FP32 explicitly:
`CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)`. The default GPU
precision was not measured in this app; in the conversion run it gave non-finite outputs on the
S26 GPU and on desktop Metal. CPU (four threads) is a second backend you can choose in the app.
When the GPU graph cannot be compiled, the app runs on CPU and shows the GPU error.

## Download, build and install

Use JDK 17, Android SDK platform 35 / build-tools 35.0.0, Android platform-tools (`adb`) and the
Hugging Face CLI (`hf`). The app needs three files of the model repository; the 1,024- and
2,048-token graphs are optional:

```bash
hf download litert-community/Kev-0.8B-LiteRT \
  kev-0.8b_rowprefill_L512_fp16fc_i8emb.tflite \
  head/kev_0.8b_pointer_head.safetensors tokenizer/tokenizer.json \
  --local-dir "$HOME/Downloads/Kev-0.8B-LiteRT"
./gradlew :app:assembleDebug
# Optional when multiple devices are connected:
# export ANDROID_SERIAL=your-device-serial
adb install app/build/outputs/apk/debug/app-debug.apk
./scripts/install_to_device.sh "$HOME/Downloads/Kev-0.8B-LiteRT"
adb shell am start -n com.kev/.MainActivity
```

The install script's argument defaults to the download directory above. It checks the size of
every file before it touches the device, and it also installs the L1024 and L2048 graphs when they
are in the directory (`WINDOWS="512"` installs only the 512-token graph). Each file goes through
`/data/local/tmp/kev/` into the app's private `files/` with `run-as com.kev`; the script then
removes the temporary copy and checks the copied size. Install the debug APK before the files:
`run-as` needs a debuggable package. A launch before the install script ran shows "Missing
<file>. Run scripts/install_to_device.sh, then reopen the app."

## External files

| File | Bytes | Purpose |
|---|---:|---|
| `kev-0.8b_rowprefill_L512_fp16fc_i8emb.tflite` | 1,264,068,368 | Rows up to 512 tokens (required) |
| `kev-0.8b_rowprefill_L1024_fp16fc_i8emb.tflite` | 1,269,023,216 | Rows up to 1,024 tokens (optional) |
| `kev-0.8b_rowprefill_L2048_fp16fc_i8emb.tflite` | 1,285,227,888 | Rows up to 2,048 tokens (optional) |
| `head/kev_0.8b_pointer_head.safetensors` | 2,099,632 | Pointer head: q and k projections, float32 |
| `tokenizer/tokenizer.json` | 19,989,325 | The checkpoint's tokenizer, unchanged |

Each graph takes `ids` int32 `[1,L]` (right-padded with `<|endoftext|>` 248044) and `valid`
float32 `[1,L]` (1 on real tokens, 0 on pads) and returns `hidden` float32 `[1,L,1024]` after the
final RMSNorm, signature `serving_default`. The three graphs have the same weights (float16
fully connected layers, int8 embedding).

## App architecture

```text
State + typed questions → to_record (rendered state, option texts) → Kotlin byte-level BPE
  → one causal row per question:
    [state] state [question] instructions ([option] option [/option])… [decide]
  → padded to the smallest window that holds the longest row (512 / 1024 / 2048) + valid mask
  → LiteRT graph (GPU FP32 or CPU 4 threads) → hidden [L,1024]
  → hidden at the decide token and at each option's closing token
  → pointer head on the host (float32, z / T, softmax) → to_answers
```

`MainActivity` observes immutable `UiState`. `MainViewModel` owns the engine (tokenizer, head and
one resident graph) and runs every model call on one worker thread with one LiteRT Environment
per process, because LiteRT reuses its native input and output buffers; `onCleared()` closes the
graph. The L512 graph compiles at startup, and one untimed call on question 1 of the ticket
example warms it up. A request whose longest row needs another window closes the resident graph
and compiles that window when its file is installed; otherwise the app names the missing file.
Rows over 2,048 tokens are rejected, never truncated. A graph output with NaN or infinity on a
question's real positions gives that question no answer.

The state is plain text unless the whole text parses as a JSON object or array; JSON is rendered
the author's way (`key: value` lines, `- item` lines, two spaces per level). Options go one per
line: choice `key: description` or `key`, noul `true: …` and `false: …` (both optional), score one
level per line. Text that contains `<|name|>` is rewritten to `<¦name¦>` before tokenizing, as the
author's `user_tokens` does.

Each card shows the time of its graph call: input writes + `run()` + output read-back, in whole
milliseconds. `run()` alone returns before the GPU work ends. The total under the cards adds the
tokenizer time and every question's graph call, head and answer.

## Files

| Path | Role |
|---|---|
| `app/src/main/java/com/kev/MainActivity.kt` | Compose host (`singleTop`); reads the launch extras |
| `app/src/main/java/com/kev/MainViewModel.kt` | Engine on the worker thread: load, Decide, window switch, autoplay, gate and timing runs |
| `app/src/main/java/com/kev/UiState.kt` | Immutable screen state: status, cards, presentation |
| `app/src/main/java/com/kev/view/KevScreen.kt` | The editable screen; `view/PresentationScreen.kt` the read-only demo layout; `view/Theme.kt`, `view/Color.kt` |
| `app/src/main/java/com/kev/KevDecider.kt` | LiteRT `CompiledModel` (GPU FP32 or CPU), its buffers and the process Environment |
| `app/src/main/java/com/kev/KevEngine.kt` | Tokenizer, head and the resident graph |
| `app/src/main/java/com/kev/KevPipeline.kt` | Request → rows → graph → head → answers, Android-free (`RowRunner`) |
| `app/src/main/java/com/kev/KevTokenizer.kt` | Byte-level BPE tokenizer read from `tokenizer.json` |
| `app/src/main/java/com/kev/KevRequest.kt`, `KevEncoder.kt` | Request validation, `render` / `to_record`, rows, windows, padding |
| `app/src/main/java/com/kev/KevPointerHead.kt`, `KevAnswers.kt` | Pointer head (safetensors) and `to_answers` |
| `app/src/main/java/com/kev/KevJson.kt` | JSON with Python's key order and number semantics |
| `app/src/main/java/com/kev/KevDrafts.kt`, `KevAnswerView.kt` | Editor form of a request; the strings an answer shows |
| `app/src/main/java/com/kev/KevGateRunner.kt`, `KevGateCore.kt`, `KevGateChecks.kt` | Debug fixture gate (device shell, Android-free checks) |
| `app/src/main/java/com/kev/KevTimingRunner.kt`, `KevTimingCore.kt`, `KevTimingRows.kt` | Timing protocol (debug and benchmark builds) |
| `app/src/main/java/com/kev/KevDemo.kt`, `KevDemoRun.kt`, `KevDevice.kt`, `KevLaunch.kt`, `KevFiles.kt` | Demo log lines and run JSON, device facts, launch extras, file names |
| `app/src/main/res/raw/example_*.json` | The three bundled requests |
| `app/src/debug/assets/` | Gate fixtures (SemIf 144 + 12 invented requests) and tokenizer probes |
| `app/src/test/java/com/kev/` | JVM parity tests against the author's oracle |
| `scripts/install_to_device.sh` | Copies the external files into the app's `files/` |
| `scripts/make_test_data.py`, `scripts/TEST_DATA.md` | Bundled test data and how to run the tests |
| `LICENSE`, `NOTICE`, `licenses/` | Apache-2.0 text, attribution and the retained upstream licenses |

## Measured

Samsung Galaxy S26 SM-S942Q, Android 16, LiteRT 2.2.0, debug build, USB powered, screen on, app in
the foreground (top-app). A graph call is timed from the input writes through `run()` to the
output read-back. Each timing leg starts at thermal status 0 with no thermal cap on any CPU policy;
with the screen on, the prime cores read 4.19 of their 4.74 GHz. A leg runs 5 untimed warm-ups,
then 20 timed calls or 20 timed five-question requests. Medians are numpy's. The runs used two
earlier debug builds of this module; they differ from this version in layout, not in the timed
code.

- **Fixture gate, GPU FP32, L512:** tokenizer probes 54/54; rows 181/181 identical to the author's
  fp32 oracle; 172 rows run in the graph (9 need L2048), all finite. The argmax equals the
  oracle's on 166/166 rows outside near-ties (oracle top-2 gap ≤ 0.02); of the 6 near-ties, 5 keep
  the argmax and 1 flips (`own_sensor_08/alert`, oracle gap 8.1e-5). Max |Δp| 0.0078, mean 9.1e-4
  over 524 options. The whole graph runs on the GPU delegate in one partition (21,059 of 21,059
  nodes).
- **Fixture gate, CPU four threads, L512, 40 rows:** probes 54/54, rows 181/181 identical, the 40
  rows all finite, argmax 40/40, max |Δp| 0.0052, mean 8.6e-4.
- **L2048, GPU FP32:** the 9 rows that need the 2,048-token window (three invented long requests,
  rows of 1,369–1,805 tokens) ran through the app with row IDs 9/9 identical, argmax 9/9 and max
  |Δp| 0.0015, the whole graph in one GPU partition (34,883 of 34,883 nodes). So all 181 gate
  questions have run through the device graphs (172 at L512, 9 at L2048). These rows took
  3.7–5.2 s each at thermal status 2 with the GPU clock capped, outside the timing protocol.
- **One question, GPU FP32:** L512, 300-token row: median **615.1 ms** (min 611.2, max 649.9,
  n 20), thermal status 0 → 2 during the leg. L1024, 1,000-token row: median **1,333.4 ms** (min
  1,297.6, max 1,395.5, n 20), status 0 → 2.
- **Five-question request, GPU FP32, L512** (rows of 128–142 tokens): per call median 689.7 ms
  (min 617.2, max 798.9, n 100), per request median 3,455.9 ms (min 3,094.0, n 20). The leg started
  6 minutes after the 172-row gate. The phone went from status 0 to 3 during its 125 calls: the
  earliest took 617–630 ms, the last 772–799 ms.
- **The same request from its text, GPU FP32, L512** (tokenize, five graph calls, head,
  `to_answers`): per request median 3,603.7 ms (min 3,314.4, max 3,982.0, n 20); the five warm-up
  requests took 3,137–3,189 ms. Inside a request: tokenizer 7.0 ms, graph 706.7 ms per call (min
  637.4, max 803.5), head 11.6 ms per question. Status 0 → 2.
- **Five-question request, CPU four threads, L512:** per call median 1,832.1 ms (min 1,410.5, max
  1,874.8, n 100), per request median 9,187.9 ms (min 7,328.0, max 9,284.7, n 20). The warm-up
  calls took 975–1,321 ms and the last timed calls 1,821–1,871 ms as the phone went from status 0
  to 2.
- **Graph compile:** GPU L512 16.9 s at a gate launch, 17.5, 18.0 and 18.5 s with the cache
  cleared, 18.9 s at a normal launch (tokenizer, head and graph loaded in 20.4 s); GPU L1024
  21.3 s with the cache cleared; CPU 2.9 and 3.6 s.
- **Memory:** available memory fell from 7.8 to 5.1 GB with the L512 graph resident (about
  2.7 GB) and from 7.6 to 3.9 GB with the L2048 graph (about 3.7 GB). While a graph compiled,
  Android's low-memory daemon reclaimed cached background apps; the sample was not killed.
- **Not measured:** Pixel phones and other devices; the NPU; GPU default precision in this app;
  runs longer than the legs above; a release (non-debuggable) build.

Example 1 (the ticket) shows what a working install answers. On the S26 with GPU FP32 and the L512
graph, **Decide** gives team `billing` 0.9258 (shipping 0.0203, returns 0.0316, technical 0.0203,
account 0.002) with confidence 0.9073, refund (noul) 0.9456, and mood (score) 1.1842 with Calm
0.0772, Annoyed 0.6613 and Angry 0.2614, confidence 0.492. The same graph on a desktop CPU gives
the same four-decimal answers (max |Δp| 6.0e-7 against the phone). The author's fp32 model on the
same request gives billing 0.9256, refund 0.9459 and mood score 1.1865 (Annoyed 0.6579), within
max |Δp| 0.0035 of the phone. The rows are 131, 101 and 93 tokens (`usage.input_tokens` 181). In
one run on a cooled phone the cards showed 656, 627 and 644 ms, and the request took 1,967 ms from
tokenizing to the last answer.

## Tests

JVM tests compare the Kotlin host with the author's fp32 oracle (request → rows, head, answers,
and the whole pipeline with a stand-in graph). They read the conversion run's reference data and
skip without it; see [scripts/TEST_DATA.md](scripts/TEST_DATA.md). On a desktop JVM 17 the rows,
decide and option indices equal the oracle on 402/402 questions (`usage.input_tokens` 377/377),
the head stays within max |Δp| 2.98e-7 of it, and `to_answers` gives the oracle's answers on
402/402. The bundled examples carry their oracle rows and answers in
`app/src/test/resources/examples_oracle.json`.

```bash
./gradlew :app:testDebugUnitTest -Pkev.work=/path/to/kev_work
```

The debug APK also runs a fixture gate on the device: the tokenizer on 54 probe strings, the rows
of all 181 bundled questions, and the graph and head on the rows that fit the resident window,
against the oracle. Gate, timing and demo launches are described in
[scripts/TEST_DATA.md](scripts/TEST_DATA.md).

```bash
adb shell am force-stop com.kev
adb shell am start -n com.kev/.MainActivity --ez gate true --es backend gpu --es report app_gate_gpu.json
adb exec-out run-as com.kev cat files/app_gate_gpu.json > app_gate_gpu.json
```
