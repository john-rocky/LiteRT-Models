# d1-omni Decide — typed decisions on Android with d1-omni-600M

Give the model a state and typed questions — **noul** (yes or no, answered as P(yes)), **choice**
(one of named options) and **score** (ordered levels, answered as the expected level) — and each
question gets its answer with the probability of every option, in one pass of the model, on the
phone. The model never generates text: each answer is read from its distribution over the
question's options. Everything runs on the phone: a Kotlin tokenizer and request encoder, LiteRT
decision graphs on the GPU and the read-out on the host.

This round of the sample holds the text path and its debug runs (a fixture gate and a timing
protocol). Images, audio and the inbox screen come next.

## Model and requirements

- Model: [litert-community/d1-omni-600M-LiteRT](https://huggingface.co/litert-community/d1-omni-600M-LiteRT)
  (decision graphs per input length, `contract.json`, `tokenizer.json`).
- Upstream: [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) revision
  `414f8d64`, LFM Open License v1.0.
- Semantics: the provider's `prompt.py` (`encode`: one row per question,
  `<bos> <state> state <q> instructions <opt> <mask> option … <decide>`), the model repository's
  host steps (`contract.json` `host_steps.text`), the read-out of `d1_host.readout_f64` (the scores
  at each option's `<mask>`, divided by the checkpoint's temperature for a text request, softmax, a
  noul reversed to [yes, no]) and `prompt.answer`.
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile/target SDK 35.
- Runtime: LiteRT **2.2.0** `CompiledModel`; Material 1 Compose with MVVM.

The app compiles every decision graph for the GPU at an explicit precision, FP32 by default:
`CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)`. At FP32 the
Galaxy S26 gives the provider's answers within the parity bar on every public text row (see
Measured). The launch extra `--es precision fp16acc` compiles at `FP16_WITH_FP32_ACCUM` (float16
storage with float32 accumulation) instead: about 10 % faster per call, but one of the 253 public
text rows then moves past the bar. The GPU's default precision is never used. When the GPU cannot
compile or run a graph, the app runs it on the CPU (four threads) and logs `GPU_FALLBACK <error>`
under the tag `D1OmniDemo`.

## Download, build and install

Use JDK 17, Android SDK platform 35 / build-tools 35.0.0, Android platform-tools (`adb`),
Python 3 (the install script reads `contract.json`) and the Hugging Face CLI (`hf`). The default
install is four files of the model repository: the contract, the tokenizer and the decision graphs
for rows of up to 128 and 256 positions.

```bash
hf download litert-community/d1-omni-600M-LiteRT contract.json tokenizer.json \
  d1-omni-600M_decide_L128_fp16.tflite d1-omni-600M_decide_L256_fp16.tflite \
  --local-dir "$HOME/Downloads/d1-omni-600M-LiteRT"
./gradlew :app:assembleDebug
# Optional when multiple devices are connected:
# export ANDROID_SERIAL=your-device-serial
adb install app/build/outputs/apk/debug/app-debug.apk
./scripts/install_to_device.sh "$HOME/Downloads/d1-omni-600M-LiteRT"
adb shell am start -n com.d1omni/.MainActivity
```

The install script's argument defaults to the download directory above. `BUCKETS` names the
decision graphs to copy (`"128 256"` by default; `BUCKETS="128 256 512 1024 2048"` copies five).
It checks the size of every file against `contract.json` before it touches the device, copies each
file through `/data/local/tmp/d1omni/` into the app's private `files/` with `run-as com.d1omni`,
removes the temporary copy, and checks the copy's sha256 on the phone against `contract.json`.
Install the debug APK before the files: `run-as` needs a debuggable package.

## External files

| File | Bytes | sha256 | Purpose |
|---|---:|---|---|
| `contract.json` | 66,567 | `2082b4c0…2efa376` | Files, signatures, token IDs, buckets, temperatures, request kinds |
| `tokenizer.json` | 4,733,371 | `1efc3a66…1ed2711` | The provider's tokenizer, unchanged |
| `d1-omni-600M_decide_L128_fp16.tflite` | 896,250,176 | `69720aa4…61273f3` | Rows up to 128 positions (installed by default) |
| `d1-omni-600M_decide_L256_fp16.tflite` | 896,315,712 | `eebf0dcc…725089b` | Rows up to 256 positions (installed by default) |
| `d1-omni-600M_decide_L512_fp16.tflite` | 896,446,784 | `f02e1bd1…1b8787` | Rows up to 512 positions (optional) |
| `d1-omni-600M_decide_L1024_fp16.tflite` | 896,708,944 | `22f367fd…19f4e` | Rows up to 1,024 positions (optional) |
| `d1-omni-600M_decide_L2048_fp16.tflite` | 897,233,232 | `e8e45309…a810dc` | Rows up to 2,048 positions (optional) |

The full sha256 of every file is in `contract.json`, which the app reads at startup: it checks the
tokenizer's sha256 and token IDs against it and finds each bucket's graph by the name it gives.
Each decision graph has one signature, `decide_<L>`: `ids` int32 `[1,L]`, `prefix` float32
`[1,L,1024]`, `media` / `pad` / `keep_right` float32 `[1,L]`, `qtype_onehot` float32 `[1,3]` →
`scores` float32 `[1,L]`.

<!-- vision (round 3) -->
### The picture path

`VISION=1 ./scripts/install_to_device.sh` also copies the three files a picture needs (download
them with `hf download litert-community/d1-omni-600M-LiteRT d1-omni-600M_vision_tower_fp16.tflite
d1-omni-600M_projector_fp16.tflite host/vision_position_table.npy`):

| File | Bytes | sha256 | Purpose |
|---|---:|---|---|
| `d1-omni-600M_vision_tower_fp16.tflite` | 171,563,424 | `6835066c…0942202` | Vision tower, one crop per call: `pixels` / `pos` float32 `[1,1024,768]`, `mask` `[1,1024]` → `features` `[1,1024,768]` |
| `d1-omni-600M_projector_fp16.tflite` | 16,791,504 | `9224b508…df80117` | `soft` float32 `[1,256,3072]` → `prefix` `[1,256,1024]` |
| `host/vision_position_table.npy` | 786,560 | `76d764aa…4fdb073` | The tower's 16 × 16 position table, float32 `[16,16,768]` (on the phone: `files/host/`) |

A picture goes through the provider's preprocessing on the phone, in Kotlin, giving the same values
as the model repository's Python host (`host/d1_vision_host.py`) bit for bit:

```text
picture file → BitmapFactory (a PNG without its colour chunks, as Pillow ignores them) → RGB, EXIF orientation applied
  → layout: factor 32, a thumbnail, and above 524,288 pixels up to 10 tiles of 512 px
  → resize on the float path: PyTorch's bilinear kernel with antialias in float32, round half to even
  → per crop: 16 × 16 patches (x − 127.5) / 127.5, the position table resized to the patch grid, a mask
  → vision tower (GPU) → 2 × 2 pixel unshuffle → projector (GPU) → (rows / 2)(columns / 2) prefix rows
  → each question's row after the P prefix rows → decide_<L> → read-out without temperature → answer
```

Each kind of graph has its own GPU precision: the decision graphs follow `--es precision` (FP32 by
default) and the tower and the projector `--es precision_vision` (`FP16_WITH_FP32_ACCUM` by
default; `fp32` selects FP32). Picture files: `D1Vision.kt` (layout, resize, patches, position
table, unshuffle), `D1Image.kt` (decoding), `D1Npy.kt` (the position table), `D1Graph.kt` (one
single-signature graph on `CompiledModel`), `D1VisionEngine.kt` (the tower, the projector and a
picture's prefix rows) and `D1VisionGate.kt` (the debug picture gate and timing, scripts/TEST_DATA.md).
<!-- end vision (round 3) -->

## How a request runs

```text
State + typed questions → prompt.encode per question (Kotlin byte-level BPE, the provider's escape)
  → one row per question: <bos> <state> state <q> instructions (<opt> <mask> option </opt>)… <decide>
  → each row on the smallest installed graph that holds it (128 or 256 positions by default)
  → build_inputs: ids [ids | 0 …], pad 1 on the real positions, media 0, keep_right 1, type one-hot
  → decide_<L> on the GPU → scores [L]
  → the scores at each option's <mask> → / temperature (text) → softmax (float64) → noul as [yes, no]
  → prompt.answer: noul P(yes); choice, its confidence and probabilities; score, the expected level
```

At most two decision graphs stay compiled. Before a graph compiles next to another one the app reads
the memory Android reports available; under 2,500,000,000 bytes it gives up the second graph and
runs every question of the request on the larger one.

## Measured

Samsung Galaxy S26 SM-S942Q (12 GB), Android 16, LiteRT 2.2.0, this app's debug build, USB powered,
screen on, app in the foreground (top-app), the files above. Each run is the debug fixture gate or
the timing protocol (scripts/TEST_DATA.md) on the public text check set of the model repository
(`fixtures/public_text.json`: 210 rows that fit 128 positions, 43 that need 256), compared with the
provider's float32 probabilities. The bar: the same argmax on every row whose reference top-2 gap is
above 0.02, max |Δp| ≤ 0.02 and mean |Δp| ≤ 0.002 over every option, no non-finite value. A call is
timed from the input writes through `run()` to the read-back of `scores`.

| Graph, precision | Rows | Argmax outside near-ties | Near-ties kept | Max \|Δp\| | Mean \|Δp\| | Bar |
|---|---:|---:|---:|---:|---:|---|
| L128, FP32 | 210 | 203/203 | 7/7 | 0.00192 | 2.50e-4 | pass |
| L256, FP32 | 43 | 40/40 | 3/3 | 0.00120 | 1.64e-4 | pass |
| L128, FP16_WITH_FP32_ACCUM | 210 | 203/203 | 6/7 | 0.0400 | 1.85e-3 | **fail** |
| L256, FP16_WITH_FP32_ACCUM | 43 | 40/40 | 3/3 | 0.0064 | 9.3e-4 | pass |

- At `FP16_WITH_FP32_ACCUM` one row is past the bar: `semif_46b7029b9a704138b77a/answer` gives
  [0.193, 0.229, 0.578] against the provider's [0.177, 0.204, 0.618] (the same argmax). Its scores at
  the three markers are 0.6875, 0.93310546875 and 2.2265625 (float16 values; at FP32 the phone gives
  0.613, 0.807 and 2.361, as the desktop CPU does), the same in all 11 calls over three processes.
  The next rows are `semif_1105577da4c8dad4609d/answer` (0.0131) and `tv4_043/answer` (0.0124); the
  near-tie that flips is `semif_fda0ca94652c6d739ecf/answer`.
- The Kotlin tokenizer and encoder give the Python host's ids, markers and serialized state on all
  253 rows on the phone; the Kotlin read-out of the phone's scores equals Python's `readout_f64` to
  5.6e-17. At FP32 the phone's probabilities are within 1.6e-5 of the desktop CPU's on the same rows.
- Every graph compiles whole for the GPU: `Replacing 1173 out of 1173 node(s) with delegate
  (LITERT_CL) node, yielding 1 partitions`. No graph fell back to the CPU.

| One call, median | FP32 | FP16_WITH_FP32_ACCUM |
|---|---:|---:|
| L128 (rows of the gate, 210 calls) | 46.4 ms | 41.9 ms |
| L256 (rows of the gate, 43 calls) | 82.9 ms | 78.2 ms |
| L128, timing protocol: one question (47 tokens), 20 calls | not measured | 39.8 ms (39.0–40.9) |
| L128, timing protocol: a three-question request (47, 56, 51 tokens), 20 requests | not measured | 122.4 ms (117.0–125.1), 40.8 ms per call |

The gate medians leave out each run's first five calls. They were taken with CPU frequency caps
(scaling_max_freq under cpuinfo_max_freq) in most 2 s samples and, in the L128 runs, the GPU clock
under 1,300 MHz in part of them (5 of 9 samples at `FP16_WITH_FP32_ACCUM`, 6 of 9 at FP32), so they
compare the two precisions only roughly. The timing protocol started from thermal status 0 with no
CPU cap and the GPU at 1,300 MHz, waited after the compile for the GPU clock and temperature, ran
five warm-up calls, and kept the GPU at 1,300 MHz on every timed call. Compiling a graph took 2.6 s
(L128) and 3.4 s (L256 next to a compiled L128) at `FP16_WITH_FP32_ACCUM`, 3.0 s (L128) and 3.5 s
(L256) at FP32. With L128 and L256 both compiled, the app's VmHWM reached 5,089,364 kB and
MemAvailable fell to 3,924,960 kB at its lowest (1 s samples); Android reported 5,541,298,176 bytes
available right before the second compile.

Not measured: the normal launch (startup compile and `ENGINE_READY`), FP32 with the timing protocol,
rows over 256 positions, images and audio, other phones.

## Files

| Path | Role |
|---|---|
| `app/src/main/java/com/d1omni/MainActivity.kt`, `MainViewModel.kt`, `view/StatusScreen.kt` | Compose host (`singleTop`), the engine on the worker thread, the status screen |
| `app/src/main/java/com/d1omni/D1Tokenizer.kt` | Byte-level BPE tokenizer read from `tokenizer.json` |
| `app/src/main/java/com/d1omni/D1Prompt.kt` | The provider's `prompt.py`: questions, `escape`, `serialize`, options, `encode`, `answer` |
| `app/src/main/java/com/d1omni/D1Json.kt` | JSON with Python's key order, number semantics and `json.dumps` output |
| `app/src/main/java/com/d1omni/D1Contract.kt`, `D1Request.kt` | `contract.json`; requests `{state, questions}` |
| `app/src/main/java/com/d1omni/D1Rows.kt`, `D1Readout.kt` | Request kinds, rows, `build_inputs`; the read-out |
| `app/src/main/java/com/d1omni/D1Decider.kt`, `D1Residency.kt`, `D1Engine.kt` | One graph on `CompiledModel`; which graphs stay compiled; tokenizer + contract + graphs |
| `app/src/main/java/com/d1omni/D1GateCore.kt`, `D1GateRunner.kt`, `D1TimingRunner.kt` | Debug fixture gate and timing protocol |
| `app/src/main/java/com/d1omni/D1Device.kt`, `D1Demo.kt`, `D1Launch.kt` | Device facts, demo log lines, launch extras |
| `app/src/test/java/com/d1omni/` | JVM parity tests against the provider's code and the Python host |
| `scripts/install_to_device.sh` | Copies the external files into the app's `files/` |
| `scripts/TEST_DATA.md` | The tests' reference data and the device runs |
| `LICENSE`, `NOTICE`, `licenses/` | LFM Open License v1.0, attribution and changes, retained licenses |

## License

The model and the code ported from the provider's `prompt.py` are under the LFM Open License v1.0
(`LICENSE`, unchanged); `NOTICE` lists what this sample changed. The tokenizer port follows Hugging
Face `tokenizers` and `transformers` (Apache License 2.0, `licenses/`).
