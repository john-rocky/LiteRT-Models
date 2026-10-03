# Kev Decide — typed decisions on Android with Kev-0.8B

Give the app a state (a text, or a JSON object or array) and typed questions: **noul** (yes or no,
answered as p(true)), **choice** (one of named options) and **score** (ordered levels, answered as
the expected level). Tap **Decide** and every question gets its answer with calibrated
probabilities, the same response the author's `/v1/systemone` server returns (`to_answers`, four
decimals). Three invented requests are bundled: a support ticket, an incident report and a product
review. Everything runs on the phone: a Kotlin tokenizer, one LiteRT graph call per question and
the pointer head on the host.

## Model and requirements

- Model: [litert-community/Kev-0.8B-LiteRT](https://huggingface.co/litert-community/Kev-0.8B-LiteRT)
  (row-prefill graphs, pointer head and tokenizer).
- Upstream: [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) tag `v1.0`
  (commit `bf75a6a8`), Apache-2.0: a rank-16 LoRA and a pointer head on Qwen3.5-0.8B-Base
  (revision `dc7cdfe2`, Apache-2.0). Code: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)
  tag `kev-1.0`.
- Semantics: the author's `kev.api.to_record` → `kev.model.encode` / `rows_of` (one causal row per
  question) → `PointerHead` (calibration temperature 2.3511) → `kev.api.to_answers`.
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile/target SDK 35.
- Runtime: LiteRT **2.2.0** `CompiledModel`; Material 1 Compose with MVVM.

**Explicit GPU FP32 precision is mandatory.** With the default GPU precision the graph runs in
float16 and part of the rows come back as NaN. The app requests
`CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)`. CPU (four
threads) is a second backend you choose in the app; when the GPU graph cannot be compiled the app
runs on CPU and shows the GPU error.

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

The install script's argument defaults to the download directory above. It checks every file's
size before contacting the device, installs the L1024 and L2048 graphs when they are in the
directory (`WINDOWS="512"` installs only the 512-token graph), pushes through
`/data/local/tmp/kev/`, copies with `run-as com.kev` into private `files/`, checks the copied size
and removes the temporary file. Install the debug APK before the files; `run-as` requires a
debuggable package. A launch before the install script ran shows "Missing <file>. Run
scripts/install_to_device.sh, then reopen the app."

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
graph. The L512 graph compiles at startup and one untimed pass of the bundled request warms it up.
A request whose longest row needs another window closes the resident graph and compiles that
window when its file is installed; otherwise the app names the missing file. Rows over 2,048
tokens are rejected, never truncated. A graph output with NaN or infinity on a question's real
positions gives that question no answer.

The state is plain text unless the whole text parses as a JSON object or array; JSON is rendered
the author's way (`key: value` lines, `- item` lines, two spaces per level). Options go one per
line: choice `key: description` or `key`, noul `true: …` and `false: …` (both optional), score one
level per line. Text that contains `<|name|>` is rewritten to `<¦name¦>` before tokenizing, as the
author's `user_tokens` does.

Times on the cards are input writes + `run()` + output read-back of that question's graph call,
in whole milliseconds; `run()` alone returns before the GPU work ends. The footer total runs from
the start of tokenizing to the last answer.

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

Not yet measured.

## Tests

JVM tests compare the Kotlin host with the author's fp32 oracle (request → rows, head, answers,
and the whole pipeline with a stand-in graph). They read the conversion run's reference data and
skip without it; see [scripts/TEST_DATA.md](scripts/TEST_DATA.md).

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
