# Kev Decide — typed decisions on Android with Kev-0.8B

Give the app a state (a text, or a JSON object or array) and typed questions: **noul** (yes or no,
answered as p(true)), **choice** (one of named options) and **score** (ordered levels, answered as
the expected level). Tap **Decide** and every question gets its answer with calibrated
probabilities, computed as the author's `to_answers` computes them (four decimals). Three invented
requests are bundled: a support ticket, an incident note and a product review. Everything runs on
the phone: a Kotlin tokenizer, LiteRT graphs on the GPU (the rows of up to 256 tokens can run on a
Snapdragon NPU instead) and the pointer head on the host. A request runs in one of two forms, whichever this app predicts to be quicker: one graph call per question
on the smallest installed window that holds its row (64 to 2,048 tokens), or a shared-state pair
that reads the state once and then each question on its own.

## Model and requirements

- Model: [litert-community/Kev-0.8B-LiteRT](https://huggingface.co/litert-community/Kev-0.8B-LiteRT)
  (row-prefill graphs, shared-state pairs, pointer head and tokenizer).
- Upstream: [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) tag `v1.0`
  (commit `bf75a6a8`), Apache-2.0: a rank-16 LoRA and a pointer head on Qwen3.5-0.8B-Base
  (revision `dc7cdfe2`, Apache-2.0). Code: [github.com/jaredpalmer/kev](https://github.com/jaredpalmer/kev)
  tag `kev-1.0`.
- Semantics: the author's `kev.api.to_record` → `kev.model.encode` / `rows_of` (one causal row per
  question) → `PointerHead` with the calibration temperature T = 2.3510958125672174 →
  `kev.api.to_answers`. The response JSON holds `model`, `answers` and `usage.input_tokens`.
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile/target SDK 35.
- Runtime: LiteRT **2.2.0** `CompiledModel`; Material 1 Compose with MVVM.

The app compiles every graph for the GPU with an explicit precision, `FP16_WITH_FP32_ACCUM`
(float16 storage with float32 accumulation):
`CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM)`.
The model repository's graphs are written so that this precision keeps every hidden state finite,
and each one passes the parity bar at it on the Galaxy S26 (see Measured). The launch extra
`--es precision fp32` compiles every graph at FP32 instead. A graph file whose size is one the
repository published before that rewrite (L512 1,264,068,368 B, L1024 1,269,023,216 B, L2048
1,285,227,888 B) runs at FP32. The default GPU precision (float16 activations) is never used: with
the conversion run's original kernel it gave non-finite outputs on the S26 GPU and on desktop
Metal, and the published graphs stay finite at it but miss the parity bar on desktop Metal (L128:
max |Δp| 0.0332, mean 3.59e-3; conversion run). **Run on** chooses the backend: GPU (the
default), NPU (Qualcomm HTP, for the row graphs of up to 256 tokens; see [NPU](#npu-qualcomm-htp))
or CPU (four threads). The app keeps that choice for the next launch. When a GPU graph cannot be
compiled, the app runs on CPU and shows the GPU error; when a graph cannot be compiled for the NPU,
it runs on the GPU and shows the NPU error.

## Download, build and install

Use JDK 17, Android SDK platform 35 / build-tools 35.0.0, Android platform-tools (`adb`) and the
Hugging Face CLI (`hf`). The default install is five files of the model repository: the 128- and
256-token graphs, the shared-state pair for states of up to 128 tokens, the head and the tokenizer:

```bash
hf download litert-community/Kev-0.8B-LiteRT \
  kev-0.8b_rowprefill_L128_fp16fc_i8emb.tflite \
  kev-0.8b_rowprefill_L256_fp16fc_i8emb.tflite \
  kev-0.8b_sharedstate_Ls128_Lq64_fp16fc_i8emb.tflite \
  head/kev_0.8b_pointer_head.safetensors tokenizer/tokenizer.json \
  --local-dir "$HOME/Downloads/Kev-0.8B-LiteRT"
./gradlew :app:assembleDebug
# Optional when multiple devices are connected:
# export ANDROID_SERIAL=your-device-serial
adb install app/build/outputs/apk/debug/app-debug.apk
./scripts/install_to_device.sh "$HOME/Downloads/Kev-0.8B-LiteRT"
adb shell am start -n com.kev/.MainActivity
```

This APK runs on the GPU and the CPU; for the NPU, copy the Qualcomm libraries into the build
before `assembleDebug` (see [NPU](#npu-qualcomm-htp)).

The install script's argument defaults to the download directory above. `WINDOWS` names the row
graphs to copy (`"128 256"` by default; `WINDOWS="64 128 256 512 1024 2048"` copies all six,
`WINDOWS=""` none) and `PAIR` the shared-state pairs by their state length (the Ls128 pair by
default; `PAIR=256`, `PAIR="128 256"`, or `PAIR=0` for none). Download every file it names
beforehand. It checks the size of every file against the published one before it touches the
device; `CHECK_SIZES=0` skips that for graphs you converted yourself and checks each copy against
its source instead. Graphs already on the device that `WINDOWS` and `PAIR` do not name stay there.
Each file goes through `/data/local/tmp/kev/` into the app's private `files/` with `run-as
com.kev`; the script then removes the temporary copy and checks the copied size. Install the debug
APK before the files: `run-as` needs a debuggable package. A launch with no graph installed shows
"Missing <files>. Run scripts/install_to_device.sh, then reopen the app."; a request that needs a
graph that is not installed names it and the install command.

## External files

| File | Bytes | Purpose |
|---|---:|---|
| `kev-0.8b_rowprefill_L64_fp16fc_i8emb.tflite` | 1,258,031,552 | Rows up to 64 tokens (optional) |
| `kev-0.8b_rowprefill_L128_fp16fc_i8emb.tflite` | 1,258,444,912 | Rows up to 128 tokens (installed by default) |
| `kev-0.8b_rowprefill_L256_fp16fc_i8emb.tflite` | 1,259,246,704 | Rows up to 256 tokens (installed by default) |
| `kev-0.8b_rowprefill_L512_fp16fc_i8emb.tflite` | 1,261,233,328 | Rows up to 512 tokens (optional) |
| `kev-0.8b_rowprefill_L1024_fp16fc_i8emb.tflite` | 1,266,799,568 | Rows up to 1,024 tokens (optional) |
| `kev-0.8b_rowprefill_L2048_fp16fc_i8emb.tflite` | 1,284,223,520 | Rows up to 2,048 tokens (optional) |
| `kev-0.8b_sharedstate_Ls128_Lq64_fp16fc_i8emb.tflite` | 1,261,368,160 | Shared-state pair: a state of up to 128 tokens, questions of up to 64 (installed by default) |
| `kev-0.8b_sharedstate_Ls256_Lq64_fp16fc_i8emb.tflite` | 1,261,918,016 | Shared-state pair: a state of up to 256 tokens, questions of up to 64 (optional) |
| `head/kev_0.8b_pointer_head.safetensors` | 2,099,632 | Pointer head: q and k projections, float32 |
| `tokenizer/tokenizer.json` | 19,989,325 | The checkpoint's tokenizer, unchanged |

Every graph has the same weights (float16 fully connected layers, int8 embedding). A row graph
takes `ids` int32 `[1,L]` (right-padded with `<|endoftext|>` 248044) and `valid` float32 `[1,L]`
(1 on real tokens, 0 on pads) and returns `hidden` float32 `[1,L,1024]` after the final RMSNorm,
signature `serving_default`. It computes all L positions, so a smaller window takes less time for
the same row. A pair file has two signatures. `state_prefill_<Ls>` takes the state part of the rows
(`[state]` and the state tokens) as `ids` / `valid` `[1,Ls]` and returns the model's state after it:
48 tensors, `k_<l>` / `v_<l>` of the 6 attention layers and `gdn_state_<l>` / `conv_tail_<l>` of
the 18 Gated DeltaNet layers. `question_step_<Ls>_64` takes one question's tokens as `ids` /
`valid` `[1,64]`, the state call's `valid` as `state_valid` `[1,Ls]` and the 48 tensors, and
returns `hidden` `[1,64,1024]` for that question's tokens, continuing the positions after the
state. The hidden states are those of the full row, so the head and the answers are the row
form's.

## How a request runs

```text
State + typed questions → to_record (rendered state, option texts) → Kotlin byte-level BPE
  → one causal row per question:
    [state] state [question] instructions ([option] option [/option])… [decide]
  → the plan: rows or the pair, by the predicted time
  rows: each row padded to the smallest installed window that holds it (64 … 2048) + valid
        → row graph → hidden [L,1024]
  pair: [state] + state once (padded to Ls) → state_prefill → 48 state tensors on the GPU
        → per question: its own tokens (padded to 64) → question_step → hidden [64,1024]
  → hidden at the decide token and at each option's closing token
  → pointer head on the host (float32, z / T, softmax) → to_answers
```

The planner predicts both forms from this app's Galaxy S26 measurements (GPU
`FP16_WITH_FP32_ACCUM`, calls made while the GPU clock ceiling stayed at 1,300 MHz) and takes the
smaller, the rows on a tie; a request that only one form can take runs on that one. The pair takes
a request when the state part fits Ls and every question part fits 64 tokens; with both pairs
installed, the one with the smaller prediction runs.

| Graph | Predicted ms |
|---|---|
| L64 / L128 / L256 / L512 / L1024 / L2048, one question | 56.6 / 102.2 / 196.3 / 388.8 / 816.0 / 1,803.5 |
| Ls128 pair, state call + one step per question | 116.0 + 62.0 without weight sharing, 152.3 + 92.9 with it |
| Ls256 pair, state call + one step per question | 216.5 + 62.5 without weight sharing, 255.4 + 95.1 with it |

For the bundled ticket on the default install (rows of 131, 101 and 93 tokens) that is 400.7 ms
for the rows (L256 + L128 + L128) against 302 ms for the pair without sharing, so the pair runs; a
five-question request (rows of 128–142 tokens) is 793.3 ms against 426 ms; a single question whose
row fits L128 stays on L128 (102.2 ms against the pair's 178 ms). The prediction depends on what is
already compiled and on the memory available when the request is planned: a second window and an
unshared pair are only predicted when they could be compiled now, and a compiled pair is predicted
as it was compiled.

The table depends on the backend (`KevCosts.forBackend`). With the NPU chosen, the L64, L128 and
L256 entries are this app's NPU times, 45.1 / 65.8 / 121.9 ms (the median of all calls of a leg),
and every other entry stays the GPU's. The bundled ticket is then 253.5 ms for the rows against
431 ms for the pair with sharing (6,019,043,328 B available with L128 and L256 compiled), so it runs
on the NPU; a five-question request is 497.3 ms for the rows against 426 ms for the pair without
sharing, so it runs on the GPU pair.

**Graphs in memory.** A row plan keeps its two largest windows compiled when both are L256 or
smaller and Android reports at least 4,500,000 kB available (`ActivityManager.MemoryInfo`), or
when both are already compiled; otherwise its largest window takes every question. L512 and larger
windows run alone, and the pair runs alone. Graphs a request does not use are closed before a
missing one compiles, and a request the compiled graphs already cover compiles nothing. At startup
the app compiles the plan of the editor's opening example (the ticket) and warms each graph up with
one untimed call.

**Weight sharing in the pair.** The two signatures can share one copy of the weights on the GPU
(`GpuOptions(constantTensorSharing = true)`) or hold one each. Without sharing the pair is quicker
and takes more memory; the answers are the same (identical probabilities on 132 gate questions).
The app decides when it compiles the pair: without sharing when Android reports at least 6,500,000
kB available right before the compile, with sharing below that; a compiled pair keeps its mode.
`--es share on` or `--es share off` fixes it. On the S26:

| Ls128 pair, five-question request | Request | State call | Step | MemAvailable low point | VmHWM |
|---|---:|---:|---:|---:|---:|
| without sharing | 429.6 ms | 116.0 ms | 62.0 ms | 2.51 GB | 6.81 GB |
| with sharing | 623.3 ms | 152.3 ms | 92.9 ms | 5.44 GB | 3.01 GB |

Other apps' memory counts: in one demo take with background apps left open, Android reported less
than 6,500,000 kB at startup, and the bundled ticket ran on the GPU rows (L256 + L128) in 475 ms
instead of on the unshared pair.

**Launch options.** These extras apply to a normal launch, an autoplay, the gate and the timing
runs; see [scripts/TEST_DATA.md](scripts/TEST_DATA.md).

| Extra | Values | Effect |
|---|---|---|
| `--es backend` | `gpu`, `npu`, `cpu` | The backend for this launch; the choice under **Run on** is kept, this one is not. `npu` needs an APK with the NPU libraries. |
| `--es graph` | `auto` (default), `rows`, `pair` | The form; `pair` fails with the reason when the pair cannot take the request. |
| `--es precision` | `fp16acc`, `fp32` | The GPU precision of every graph (default: each graph's own). |
| `--es share` | `auto` (default), `on`, `off` | Weight sharing in the pair. |
| `--es npu_opt` (debug) | `default`, `inference`, `o3`, `prepare` | The HTP optimization level of the NPU compiles. |
| `--es npu_perf` (debug) | `burst` (default), `none`, or another HTP performance mode in lower case | The HTP performance mode in every graph's options. |

The state is plain text unless the whole text parses as a JSON object or array; JSON is rendered
the author's way (`key: value` lines, `- item` lines, two spaces per level). Options go one per
line: choice `key: description` or `key`, noul `true: …` and `false: …` (both optional), score one
level per line. Text that contains `<|name|>` is rewritten to `<¦name¦>` before tokenizing, as the
author's `user_tokens` does. Rows over 2,048 tokens are rejected, never truncated. A graph output
with NaN or infinity on a question's real positions gives that question no answer.

Each card shows the time of its graph call (input writes + `run()` + output read-back, in whole
milliseconds), the graph and where it ran: `105 ms · L128 · GPU` for a row, `62 ms · Q64 · GPU` for
a pair step, `71 ms · L128 · NPU` on the NPU. `run()` alone returns before the GPU work ends. A pair
request also shows the state call, for example `State · 95 tokens · 115 ms`. The engine line names
the backend (with the GPU precision, `GPU FP16 (FP32 accum)`) and the compiled graphs, then the load
time (tokenizer + head + these graphs' compiles) and their compile time, for example `NPU · L128 +
L256 · loaded in 6.0 s (graph compile 5.0 s)` right after a switch from the GPU; while the switch
compiles, it shows only `NPU · no graph compiled`. The status line gives the request's time from
tokenizing to the last answer, and the footer the backend (`LiteRT 2.2.0 · NPU`).

## NPU (Qualcomm HTP)

**What runs where.** Choose NPU under **Run on**, or launch with `--es backend npu`, and the row
graphs L64, L128 and L256 run on the Qualcomm HTP through LiteRT's NPU dispatch, compiled with
`CompiledModel.Options(Accelerator.NPU, Accelerator.CPU)`. The HTP takes every op of these graphs
but the int8 embedding lookup, which runs on the CPU: logcat shows `LiteRT Op #4 'EmbeddingLookup'
(code=7) is not supported in Qualcomm Compiler`, then `Replacing 2 out of 3 node(s) with delegate
(DispatchDelegate)`. L512 and larger windows and the shared-state pairs stay on the GPU when the NPU
is chosen. The GPU is the default. With the libraries in the APK every graph of the process, the
GPU's too, carries `QualcommOptions(htpPerformanceMode = BURST)`; logcat shows `HtpPerformanceMode
: Burst(2)` for each NPU graph also when GPU graphs compiled before it in the same process.

**Build with the NPU libraries.** The ten Qualcomm runtime files are not in this repository: the
QAIRT libraries may be distributed only inside an app. Collect them in one directory from
`litert_npu_runtime_libraries.zip` and `litert_npu_runtime_libraries_jit.zip` (LiteRT GitHub Release
assets) and the QAIRT SDK (`lib/aarch64-android/`, and `lib/hexagon-v81/unsigned/` for the Galaxy
S26's Hexagon v81), then:

```bash
scripts/fetch_npu_libs.sh /path/to/the/libraries   # copies them into app/src/main/jniLibs/arm64-v8a
./gradlew :app:assembleDebug
```

The root README section [Running on the NPU](../README.md#running-on-the-npu) lists each file and
where it comes from; this module already sets `useLegacyPackaging = true`, which the DSP needs to
open its library. The APK grows from 14.6 MB to 61.6 MB. An APK built without the libraries runs on
the GPU and the CPU: the NPU choice is disabled and the screen says "NPU: this APK has no NPU
libraries (README)."

**Compiled on the phone once, then loaded.** A graph that is not in LiteRT's cache is compiled on
the phone (LiteRT's JIT through the Qualcomm compiler plugin). On the Galaxy S26 that took 81.6 s
for L64, 150.1–179.0 s for L128 and 298.2 s for L256. The app runs such a compile alone, with the
other graphs closed: MemAvailable fell to 2.08 GB (L64), 1.85 GB (L128) and 1.21 GB (L256), and
Android's low-memory killer reclaimed 16–54 background processes of other apps per compile; the app
kept running. The status line reads "Compiling L128 for the NPU (takes minutes once; cached after)…"
and counts the seconds.

**The compile cache.** `Environment.create(context, …)` puts LiteRT's compile cache in the app's
`cacheDir`: one file per graph, `cacheDir/<graph file name without .tflite>/<hash of the file's
content>/<hash of the compiler plugin, the build fingerprint, the accelerators and the API>.tflite`,
1.27–1.28 GB each (3.57 GiB for the three windows). A later launch loads a graph from it in
0.8–2.5 s; a normal launch with L128 and L256 is ready in 3.1–4.3 s. A graph file whose content
changed is compiled again: with one byte changed and the size and modification time kept, the
launch compiled for 63.4 s and LiteRT removed the old entry. The key also holds the Qualcomm options: the same L64 got a new entry for each
performance mode and optimization level tried. The app records the level of each compile in
`files/npu_marks/`, so its status line tells a compile from a load, and deletes the old entry when the
level changes. Android may delete `cacheDir` when storage runs low; the next NPU launch then compiles
again.

**Measured on the NPU.** Galaxy S26, this app's debug build, the graphs loaded from the cache, each
leg from thermal status 0 with no CPU frequency cap. NPU: the median of all calls of a leg (60
calls; the CPU caps that set in during a leg do not slow the NPU call). GPU:
`FP16_WITH_FP32_ACCUM`, the medians of [Measured](#measured).

| | NPU | GPU |
|---|---:|---:|
| One question, L64 (rows of 51–64 tokens) | 45.1 ms | 56.6 ms |
| One question, L128 (rows of 73–97 tokens) | 65.8 ms (68.5 ms in another launch) | 102.2 ms |
| One question, L256 (the same rows) | 121.9 ms | 196.3 ms |
| Bundled ticket, three questions | 298 ms on L256 + L128 (287 ms with a screen recording) | 355–367 ms on the pair without sharing (352 ms with a recording) |
| Five-question request from its text | 522.5 ms on the rows (`--es graph rows`) | 464.3 ms on the pair without sharing, the plan's choice with the NPU chosen |
| Three-question email (state 167 tokens), the graph calls | 368.4 ms, three calls on L256 | 409.1 ms on the Ls256 pair without sharing |
| Ready after a normal launch | 3.1–4.3 s (L128 + L256 from the cache) | 16.9–21.2 s (the pair's compile) |
| VmHWM with the request's graphs | 2.7–4.7 GB (L128 + L256) | 5.5–6.8 GB (the pair without sharing) |

In the ticket's process, with L128 and L256 compiled for the NPU, the five-question request took
547 ms on the rows. The gates are in [Measured](#measured).

**Qualcomm options.** Every graph gets `htpPerformanceMode = BURST` and LiteRT's default optimization
level, which is HTP_OPTIMIZE_FOR_INFERENCE_O3 (logcat `OptimizationLevel :
HtpOptimizeForInferenceO3(2)` without any level set). On L64: with no performance mode a question took
157.4 ms (122.5–240.8 over a leg) against 47.6 ms with BURST; at HTP_OPTIMIZE_FOR_PREPARE the compile
took 19.1 s instead of 64–82 s and a question 122.3 ms; O3 and PREPARE gave the default level's
probabilities on every gate row. The debug extras `--es npu_perf` and `--es npu_opt` set them; the
app's marks record the level, not the mode, so after a mode change the status line says it loads
while LiteRT compiles.

**Limits.** Measured on the Galaxy S26 (SM8850, Hexagon v81) only. A shared-state pair does not go to
the NPU: its JIT compile (two signatures in one file) was stopped twice on the 12 GB S26, Android's
low-memory killer ending the app about 5 minutes in with MemAvailable at 0.60 and 1.22 GB, so the app
compiles pairs for the GPU only. LiteRT reports no error when a graph lands on the CPU instead of the
NPU; check logcat for `Replacing 2 out of 3 node(s) with delegate (DispatchDelegate)` (the app's gate
and timing reports carry these lines for each NPU compile).

## Files

| Path | Role |
|---|---|
| `app/src/main/java/com/kev/MainActivity.kt` | Compose host (`singleTop`); reads the launch extras |
| `app/src/main/java/com/kev/MainViewModel.kt` | Engine on the worker thread: load, Decide, the plan's graphs, autoplay, gate and timing runs |
| `app/src/main/java/com/kev/UiState.kt` | Immutable screen state: status, engine line, cards, presentation |
| `app/src/main/java/com/kev/view/KevScreen.kt` | The editable screen; `view/PresentationScreen.kt` the read-only demo layout; `view/Theme.kt`, `view/Color.kt` |
| `app/src/main/java/com/kev/KevDecider.kt` | One row graph on LiteRT `CompiledModel` (GPU with an explicit precision, NPU + CPU, or CPU), its buffers, `KevPrecision` and the process Environment |
| `app/src/main/java/com/kev/KevNpu.kt` | The NPU backend: the Qualcomm options (BURST, the optimization level), the compile-cache marks in `files/npu_marks/` and the logcat lines that show where a graph ran |
| `app/src/main/java/com/kev/KevPairDecider.kt`, `KevPair.kt` | The shared-state pair: two signatures, the state handed to the question call as its own buffers; the pair's contract |
| `app/src/main/java/com/kev/KevPlanner.kt` | Rows or pair by the predicted time (`KevCosts`), weight sharing (`KevPairShare`), Android-free |
| `app/src/main/java/com/kev/KevEngine.kt` | Tokenizer, head and the compiled graphs of the plan |
| `app/src/main/java/com/kev/KevWindows.kt` | Window choice per question and the compiled graphs, Android-free (`KevResidentGraphs`) |
| `app/src/main/java/com/kev/KevPipeline.kt` | Request → rows or state + branches → graph → head → answers, Android-free (`RowRunner`, `PairRunner`) |
| `app/src/main/java/com/kev/KevTokenizer.kt` | Byte-level BPE tokenizer read from `tokenizer.json` |
| `app/src/main/java/com/kev/KevRequest.kt`, `KevEncoder.kt` | Request validation, `render` / `to_record`, rows, windows, padding |
| `app/src/main/java/com/kev/KevPointerHead.kt`, `KevAnswers.kt` | Pointer head (safetensors) and `to_answers` |
| `app/src/main/java/com/kev/KevJson.kt` | JSON with Python's key order and number semantics |
| `app/src/main/java/com/kev/KevDrafts.kt`, `KevAnswerView.kt` | Editor form of a request; the strings an answer shows |
| `app/src/main/java/com/kev/KevGateRunner.kt`, `KevGateCore.kt`, `KevGateChecks.kt` | Debug fixture gate (device shell, Android-free checks) |
| `app/src/main/java/com/kev/KevTimingRunner.kt`, `KevTimingCore.kt`, `KevTimingRows.kt`, `KevCool.kt` | Timing protocol with the GPU clock ceiling read before each call and a wait for the GPU to cool (debug and benchmark builds) |
| `app/src/main/java/com/kev/KevDemo.kt`, `KevDemoRun.kt`, `KevDevice.kt`, `KevLaunch.kt`, `KevFiles.kt` | Demo log lines and run JSON, device facts, launch extras, file names and sizes |
| `app/src/main/res/raw/example_*.json` | The three bundled requests |
| `app/src/debug/assets/` | Gate fixtures (SemIf 144 + 12 invented requests) and tokenizer probes |
| `app/src/test/java/com/kev/` | JVM parity tests against the author's oracle; planner, window and sharing rules |
| `scripts/install_to_device.sh` | Copies the external files into the app's `files/` |
| `scripts/fetch_npu_libs.sh` | Copies the ten Qualcomm libraries into `app/src/main/jniLibs/arm64-v8a` for an NPU build (ignored by git) |
| `scripts/make_test_data.py`, `scripts/TEST_DATA.md` | Bundled test data and how to run the tests and the device runs |
| `LICENSE`, `NOTICE`, `licenses/` | Apache-2.0 text, attribution and the retained upstream licenses |

## Measured

Samsung Galaxy S26 SM-S942Q (12 GB), Android 16, LiteRT 2.2.0, debug build, USB powered, screen
on, app in the foreground (top-app), the files above, GPU `FP16_WITH_FP32_ACCUM` unless a line says
FP32. A graph call is timed from the input writes through `run()` to the output read-back; a pair's
state call from its input writes to the write of `state_valid`, which waits for the state. Each
timing leg starts at thermal status 0 with no thermal cap on any CPU policy, after at least 180 s
of rest, and waits after the compile until the GPU clock ceiling and the GPU temperature are back
to their values before it. A leg runs 5 untimed warm-ups, then 20 timed calls (or 20 timed
requests). The app reads the GPU clock ceiling (`/sys/class/kgsl/kgsl-3d0/max_clock_mhz`) before
each call: the values below are the medians (numpy's) of the calls made while it stayed at
1,300 MHz, and the calls after it fell are given apart. Available memory is Android's
MemAvailable and VmHWM the process's peak resident memory, in GB of 1,000,000 kB.

- **One question per window** (rows of 73, 80 and 97 tokens; L64: 51, 58 and 64; L512 one
  300-token row; L1024 one 1,000-token row; L2048 the nine long gate rows of 1,369–1,805 tokens,
  three requests of three questions, each after the GPU wait):

  | Window | Median | Min–max | n | Compile |
  |---|---:|---|---:|---:|
  | L64 | 56.6 ms | 55.3–60.1 | 60 | 7.4 s |
  | L128 | 102.2 ms | 101.2–104.6 | 60 | 6.8 s |
  | L256 | 196.3 ms | 195.1–262.6 | 31 | 7.5 s |
  | L512 | 388.8 ms | 384.9–392.9 | 15 | 9.0 s |
  | L1024 | 816.0 ms | 806.7–1,178.2 | 18 | 10.6 s |
  | L2048 | 1,803.5 ms | 1,789–1,821 | 8 | not logged |

  At FP32 the same rows take 137.8 ms on L128 and 272.1 ms on L256 (n 30 and 40), measured on
  the conversion run's previous build of those two graphs; at `FP16_WITH_FP32_ACCUM` that build
  gives the published files' probabilities on every gate row.
- **When the GPU clock ceiling falls:** 3–8 s into back-to-back calls it fell from 1,300 MHz, in
  most legs to 578–902 MHz, and a call then took 1.7–2.1 times as long: L256 330.2 ms instead of
  196.3 ms (n 29), the five-question request on the pair 727.6 ms instead of 429.6 ms, and from its
  text 822.5 ms instead of 465.9 ms. The thermal status stayed at 0 or 1 meanwhile, so it does not
  tell. After each compile the ceiling and the temperature were already back (the wait took 0 s).
- **Requests:**

  | Request | Plan | Request time | Inside |
  |---|---|---:|---|
  | Bundled ticket, default install, normal launch | pair without sharing | 367 ms | state 72 tokens 129 ms, steps 61, 63, 62 ms |
  | The same, `--es graph rows` | L256 + L128 | 463 ms | 214, 105, 105 ms |
  | Bundled incident, in the ticket's process | pair (compiled) | 333 ms | state 95 tokens 115 ms, steps 60–63 ms |
  | Five-question request from its text, default install | pair without sharing | 465.9 ms (452.6–624.7, n 15) | tokenize 8.4 ms, state 115.1 ms, step 61.8 ms, head 6.1 ms per question |
  | Five-question request, rows on L256 | rows | 972.1–984.6 ms | 195.0 ms per call |
  | Three-question email, state 167 tokens: Ls256 pair without sharing | pair | 409.1 ms (402.4–420.4, n 12) | state 216.5 ms, step 62.5 ms |
  | The same with sharing | pair | 544.6 ms | state 255.4 ms, step 95.1 ms |
  | The same, rows on L256 | rows | 585.0–593.8 ms | 195.6 ms per call |

  The ticket and incident times run from tokenizing to the last answer, without the demo's waits
  and without a compile; the row requests' ranges are the requests made before the ceiling fell.
- **Fixture gate:** tokenizer probes 54/54, all 181 rows identical to the author's fp32 oracle
  (IDs, decide and option indices) for every graph, no non-finite row. The argmax counts leave out
  the near-ties (oracle top-2 gap ≤ 0.02):

  | Graph | Rows run | Argmax outside near-ties | Near-ties kept | Max \|Δp\| | Mean \|Δp\| |
  |---|---:|---:|---:|---:|---:|
  | L64 | 34 | 33/33 | 1/1 | 0.00656 | 1.03e-3 |
  | L128 | 147 | 144/144 | 3/3 | 0.00736 | 1.11e-3 |
  | L256 | 172 | 166/166 | 6/6 | 0.00917 | 1.05e-3 |
  | L512 | 172 | 166/166 | 5/6 | 0.00907 | 1.07e-3 |
  | L1024 (the asset's opening 40 rows) | 40 | 40/40 | — | 0.00804 | 1.06e-3 |
  | L2048 (the 9 long rows) | 9 | 9/9 | — | 0.0093 | 1.84e-3 |
  | Ls128 pair, with or without sharing | 132 questions | 130/130 | 2/2 | 0.0101 | 1.14e-3 |
  | Ls256 pair, without sharing | 146 questions | 141/141 | 4/5 | 0.0110 | 1.10e-3 |
  | L256 at FP32 | 172 | 166/166 | 5/6 | 0.00780 | 9.13e-4 |
  | Ls128 pair at FP32 | 132 questions | 130/130 | 2/2 | 0.00780 | 9.76e-4 |
  | L2048 at FP32 (the 9 long rows) | 9 | 9/9 | — | 0.0015 | 4.26e-4 |
  | L64 on the NPU | 34 | 33/33 | 1/1 | 0.0105 | 1.20e-3 |
  | L128 on the NPU | 147 | 144/144 | 3/3 | 0.0134 | 1.44e-3 |
  | L256 on the NPU | 172 | 166/166 | 5/6 | 0.0134 | 1.35e-3 |

  The one near-tie that flips is `own_sensor_08/alert` (oracle gap 8e-5). The L64, L128 and L256
  gates give the same probabilities on every row as the conversion run's previous build of those
  graphs. Every graph runs whole on the GPU delegate in one partition (L64 3,912 nodes, L128 4,759,
  L256 6,019, L512 8,515, L1024 13,555, L2048 23,635; the Ls128 pair 4,975 + 3,965, the Ls256 pair
  6,235 + 3,965). On the NPU the HTP takes 3,911 of L64's 3,912 ops, 4,758 of L128's 4,759 and
  6,018 of L256's 6,019; the remaining op, the embedding lookup, runs on the CPU. On the CPU (four
  threads) the Ls128 pair passes the gate on its opening 40 questions (40/40, max |Δp| 0.00525,
  mean 8.6e-4).
- **Graph compile** (cache cleared): L64 7.4 s, L128 6.8 s, L256 7.5 s, L512 9.0 s, L1024 10.6 s;
  the Ls128 pair 15.1 and 16.3 s without sharing, 13.7 s with it; the Ls256 pair 17.9 s without
  sharing (26.1 s in a gate launch at thermal status 2). A normal launch with the default install is ready in 21.2 s (the pair's compile 20.2 s);
  with `--es graph rows` in 16.9 s (L256 and L128).
- **Memory:** one compiled window (L64 to L1024): MemAvailable low points 3.18–3.96 GB, VmHWM
  4.61–5.56 GB. L256 +
  L128 compiled at startup: low point 2.71 GB, VmHWM 7.20 GB. The pair without sharing: low points
  2.21–3.04 GB, VmHWM 6.15–6.81 GB; with sharing: 5.29–5.88 GB, 3.01–3.05 GB. Android's
  low-memory killer stopped the app in none of these runs.
- **Not measured:** Pixel phones and other devices, the NPU of other Snapdragon chips among them;
  the CPU's time on these files; runs longer than the legs above; a release (non-debuggable) build.

Rows over 1,024 tokens have the smallest margin at `FP16_WITH_FP32_ACCUM`: on the 9 long rows the
largest probability change is 0.0093 and the mean 1.84e-3, against 0.0015 and 4.3e-4 at FP32, still
within the bar (0.02 and 0.002). `--es precision fp32` runs every graph at FP32 for such inputs. The
graphs compile when a request needs them: on the S26 7–11 s per window up to L1024 and 14–26 s for a
pair.

Example 1 (the ticket) shows what a working install answers. On the S26 with the default install,
**Decide** runs the three questions on the pair and gives team `billing` 0.9258 (shipping 0.0203,
returns 0.0316, technical 0.0202, account 0.002) with confidence 0.9073, refund (noul) 0.9457, and
mood (score) 1.1844 with Calm 0.0776, Annoyed 0.6605 and Angry 0.262, confidence 0.4907. With
`--es graph rows` (L256 and L128) it gives billing 0.9261, refund 0.9454 and mood 1.1858. The
author's fp32 model on the same request gives billing 0.9256, refund 0.9459 and mood 1.1865
(Annoyed 0.6579). The rows are 131, 101 and 93 tokens (`usage.input_tokens` 181).

## Tests

JVM tests compare the Kotlin host with the author's fp32 oracle (request → rows, head, answers,
and the whole pipeline with stand-in graphs for the rows and for the pair) and check the planner,
window, sharing and default-install rules. They read the conversion run's reference data and skip
without it; see [scripts/TEST_DATA.md](scripts/TEST_DATA.md). On a desktop JVM 17 the rows, decide
and option indices equal the oracle on 402/402 questions (`usage.input_tokens` 377/377), the head
stays within max |Δp| 2.98e-7 of it, and `to_answers` gives the oracle's answers on 402/402. On the
pair's path the 304 requests whose state and questions fit the Ls128 pair (312 questions) give the
row path's and the oracle's answers on 312/312. The bundled examples carry their oracle rows and
answers in `app/src/test/resources/examples_oracle.json`.

```bash
./gradlew :app:testDebugUnitTest -Pkev.work=/path/to/kev_work
```

The debug APK also runs a fixture gate on the device: the tokenizer on 54 probe strings, the rows
of all 181 bundled questions, and the graph and head on the rows that fit the graph it compiles
(`--ei window`, `--es graph pair` with `--ei ls`; without them, the smallest installed window),
against the oracle. Gate, timing and demo launches are described in
[scripts/TEST_DATA.md](scripts/TEST_DATA.md).

```bash
adb shell am force-stop com.kev
adb shell am start -n com.kev/.MainActivity --ez gate true --es backend gpu --es report app_gate_gpu.json
adb exec-out run-as com.kev cat files/app_gate_gpu.json > app_gate_gpu.json
```
