# d1-omni Decide — typed decisions on Android with d1-omni-600M

Give the model a state and typed questions — **noul** (yes or no, answered as P(yes)), **choice**
(one of named options) and **score** (ordered levels, answered as the expected level) — and each
question gets its answer with the probability of every option, in one pass of the model, on the
phone. The model never generates text: each answer is read from its distribution over the
question's options. Everything runs on the phone: a Kotlin tokenizer and request encoder, LiteRT
decision graphs on the GPU and the read-out on the host.

This round of the sample holds the text path and the audio path (a 16 kHz mono clip becomes the
prefix the decision graph reads before the question) with their debug runs (a fixture gate and a
timing protocol), and a speaker player for the clips. Images and the inbox screen come next.

## Model and requirements

- Model: [litert-community/d1-omni-600M-LiteRT](https://huggingface.co/litert-community/d1-omni-600M-LiteRT)
  (decision graphs per input length, `contract.json`, `tokenizer.json`).
- Upstream: [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) revision
  `414f8d64`, LFM Open License v1.0.
- Semantics: the provider's `prompt.py` (`encode`: one row per question,
  `<bos> <state> state <q> instructions <opt> <mask> option … <decide>`), the model repository's
  host steps (`contract.json` `host_steps.text` and `host_steps.audio`), the audio front end of
  `host/d1_audio_host.py` (the provider's `audio.py`: waveform, preemphasis, STFT, Slaney mel 128,
  log, per-bin normalization, bucket and mask inputs), the read-out of `d1_host.readout_f64` (the
  scores at each option's `<mask>`, divided by the checkpoint's temperature for a text request,
  softmax, a noul reversed to [yes, no]) and `prompt.answer`.
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile/target SDK 35.
- Runtime: LiteRT **2.2.0** `CompiledModel`; Material 1 Compose with MVVM.

The app compiles every decision graph for the GPU at an explicit precision, FP32 by default:
`CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)`. At FP32 the
Galaxy S26 gives the provider's answers within the parity bar on every public text row (see
Measured). The launch extra `--es precision fp16acc` compiles at `FP16_WITH_FP32_ACCUM` (float16
storage with float32 accumulation) instead: about 10 % faster per call, but one of the 253 public
text rows then moves past the bar. The audio graph has its own precision, `--es precision_audio
fp16acc|fp32` (`FP16_WITH_FP32_ACCUM` by default, `D1AudioEngine.DEFAULT_PRECISION`). The GPU's
default precision is never used. When the GPU cannot compile or run a graph, the app runs it on the
CPU (four threads) and logs `GPU_FALLBACK <error>` under the tag `D1OmniDemo`.

## Download, build and install

Use JDK 17, Android SDK platform 35 / build-tools 35.0.0, Android platform-tools (`adb`),
Python 3 (the install script reads `contract.json`) and the Hugging Face CLI (`hf`). The default
install is four files of the model repository: the contract, the tokenizer and the decision graphs
for rows of up to 128 and 256 positions; audio requests add the audio graph of their length
(`AUDIO="1001"`: clips up to 10 s).

```bash
hf download litert-community/d1-omni-600M-LiteRT contract.json tokenizer.json \
  d1-omni-600M_decide_L128_fp16.tflite d1-omni-600M_decide_L256_fp16.tflite \
  d1-omni-600M_audio_T1001_fp16.tflite \
  --local-dir "$HOME/Downloads/d1-omni-600M-LiteRT"
./gradlew :app:assembleDebug
# Optional when multiple devices are connected:
# export ANDROID_SERIAL=your-device-serial
adb install app/build/outputs/apk/debug/app-debug.apk
AUDIO="1001" ./scripts/install_to_device.sh "$HOME/Downloads/d1-omni-600M-LiteRT"
adb shell am start -n com.d1omni/.MainActivity
```

The install script's argument defaults to the download directory above. `BUCKETS` names the
decision graphs to copy (`"128 256"` by default; `BUCKETS="128 256 512 1024 2048"` copies five)
and `AUDIO` the audio graphs (none by default; `AUDIO="1001"`, or any of 501 / 1001 / 2001 / 3001
for clips up to 5 / 10 / 20 / 30 s). It checks the size of every file against `contract.json`
before it touches the device; a file already in the app's private `files/` with the contract's size
and sha256 stays as it is, any other is copied through `/data/local/tmp/d1omni/` into `files/` with
`run-as com.d1omni`, the temporary copy removed, and the copy's sha256 on the phone checked against
`contract.json`. Install the debug APK before the files: `run-as` needs a debuggable package.

## External files

| File | Bytes | sha256 | Purpose |
|---|---:|---|---|
| `contract.json` | 69,654 | `93304d36…70ccf79` | Files, signatures, token IDs, buckets, temperatures, request kinds |
| `tokenizer.json` | 4,733,371 | `1efc3a66…1ed2711` | The provider's tokenizer, unchanged |
| `d1-omni-600M_decide_L128_fp16.tflite` | 896,250,176 | `69720aa4…61273f3` | Rows up to 128 positions (installed by default) |
| `d1-omni-600M_decide_L256_fp16.tflite` | 896,315,712 | `eebf0dcc…725089b` | Rows up to 256 positions (installed by default) |
| `d1-omni-600M_decide_L512_fp16.tflite` | 896,446,784 | `f02e1bd1…1b8787` | Rows up to 512 positions (optional) |
| `d1-omni-600M_decide_L1024_fp16.tflite` | 896,708,944 | `22f367fd…19f4e` | Rows up to 1,024 positions (optional) |
| `d1-omni-600M_decide_L2048_fp16.tflite` | 897,233,232 | `e8e45309…a810dc` | Rows up to 2,048 positions (optional) |
| `d1-omni-600M_audio_T1001_fp16.tflite` | 225,525,600 | `dfeb70cf…97a2303` | Audio clips up to 10 s (`AUDIO="1001"`) |
| `d1-omni-600M_audio_T501_fp16.tflite` | 221,138,784 | `754c6d20…93e7d3e` | Audio clips up to 5 s (optional) |
| `d1-omni-600M_audio_T2001_fp16.tflite` | 234,229,600 | `4fdb86fc…e13d9a7` | Audio clips up to 20 s (optional) |
| `d1-omni-600M_audio_T3001_fp16.tflite` | 242,933,600 | `7ad87bca…2aa2669` | Audio clips up to 30 s (optional) |

The full sha256 of every file is in `contract.json`, which the app reads at startup: it checks the
tokenizer's sha256 and token IDs against it and finds each bucket's graph by the name it gives.
Each decision graph has one signature, `decide_<L>`: `ids` int32 `[1,L]`, `prefix` float32
`[1,L,1024]`, `media` / `pad` / `keep_right` float32 `[1,L]`, `qtype_onehot` float32 `[1,3]` →
`scores` float32 `[1,L]`. Each audio graph has one signature, `audio_<T>`: `mel` float32
`[1,128,T]`, `mel_valid` float32 `[1,T]`, `v1` / `v2` / `v3` float32 `[1,T1]` / `[1,T2]` /
`[1,T3]` (T1001: 501 / 251 / 126) → `prefix` float32 `[1,T3,1024]`, whose first P rows are the
clip's prefix.

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

An audio request (`D1AudioEngine.audioPrefix`, then `D1Engine.audioRows`) goes first through the
clip, then through the same steps as text, with the clip's P prefix rows before every question:

```text
16 kHz mono PCM16 wav → int16 samples (D1Wav; another rate or channel count is refused)
  → waveform: / 32768, cut to 30 s, zero-padded to 0.5 s
  → preemphasis (float32) → 512-point STFT, Hann 400, hop 160 → |X|² → Slaney mel 128 → log(x + 2⁻²⁴)
  → each mel bin's mean and std over the valid frames (n // 160) → normalized mel [128, T = n // 160 + 1]
  → the smallest installed bucket T_b ≥ T → mel zero-padded to T_b, the four masks → audio_<T_b>
  → its first P rows (P = the subsampled length of the valid frames: T1001 holds up to 125)
  → per question: prompt.encode after P positions (a null state is {}, options in the audio form)
  → build_inputs with the prefix rows first (media 1, keep_right 0 at P − 1) → decide_<L>
  → the scores at P + each option's <mask> → softmax (no temperature) → answer
```

The mel runs every step in float64 (one rounding to float32 at the end), the model repository's
`mel(x, "float64")`: on the six public clips it equals the Python host's float64 mel bit for bit and
is within 5.7e-5 of its float32 form (the provider's own; JVM tests). The Hann window and the
Slaney filter bank are the Python host's float32 values bit for bit.

At most two decision graphs stay compiled. Before a graph compiles next to another one the app reads
the memory Android reports available; under 2,500,000,000 bytes it gives up the second graph and
runs every question of the request on the larger one. One audio graph stays compiled beside them (a
clip of another bucket closes it first).

`D1AudioPlayer` plays a clip through the speaker (`AudioTrack`, USAGE_MEDIA / CONTENT_TYPE_SPEECH,
16 kHz mono PCM16, at the phone's own media volume) and reports when the sound started: the time
frame 0 left the output by `AudioTimestamp`, else the first moving playback head (the sound starts a
fraction of a second after `play()`). It is not used by a screen yet.

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

### Audio

The same phone and build, the public audio check set of the model repository
(`fixtures/public_audio.json`: six 16 kHz mono clips of 7.65–9.63 s, all on the T1001 audio graph, three
questions each = 18 rows, one on L128 and 17 on L256), each clip's wav read on the phone, the decision
graphs at FP32, the audio graph at each precision below, against the provider's float32 probabilities
with the bar above.

| Audio graph, precision | Rows | Argmax outside near-ties | Max \|Δp\| | Mean \|Δp\| | Bar | Audio graph, one call |
|---|---:|---:|---:|---:|---|---:|
| T1001, FP16_WITH_FP32_ACCUM (default) | 18 | 18/18 | 0.00138 | 2.15e-4 | pass | 31.1 ms (31.0–31.3) |
| T1001, FP32 | 18 | 18/18 | 0.00154 | 1.25e-4 | pass | 40.0 ms (39.4–40.4) |

- The app's mel on the phone equals the model repository's float64 mel bit for bit on all six clips
  (and is within 5.7e-5 of its float32 form); n, frames, T, the bucket and P equal the Python host's,
  and the app's own encoding gives the fixture's ids and markers on all 18 rows. The audio graph's
  prefix rows are within 0.063 of the desktop CPU's at `FP16_WITH_FP32_ACCUM` and 1.7e-5 at FP32 (same
  mel). The audio graph compiles whole for the GPU: `Replacing 2423 out of 2423 node(s) with delegate
  (LITERT_CL) node, yielding 1 partitions`.
- A whole request, timing protocol (`aud_food_03`, the inbox demo's clip: 8.7 s, P = 109, its three
  questions on L256; audio graph at `FP16_WITH_FP32_ACCUM`, decision graphs at FP32; five warm-up and 20
  timed requests from thermal status 0 with no CPU cap in any sample; the app read kgsl's clock ceiling
  1,300 MHz before every request, while the three two-second samples taken during the timed requests read
  1,200 MHz, thermal power level 1–2, GPU 99–104 °C): **302.5 ms** median (294.3–311.1) from the wav file
  to the three answers. Per step (medians): reading the wav 0.6 ms, the mel 13.1 ms, the audio graph
  33.1 ms, the encoding of the three questions 1.1 ms, the three decision calls 253.1 ms (84.2 ms each),
  the waveform, the inputs, the prefix copy and the read-out 1.5 ms together.
- With the L256, L128 and T1001 graphs compiled at once, the app's VmHWM reached 6,245,076 kB and
  MemAvailable fell to 3,227,380 kB at its lowest (1 s samples); Android reported 6,988,881,920 bytes
  available before the first compile and 4,102,299,648 after the third. Compiling took 3.0 s (L256),
  3.4 s (L128) and 1.4 s (audio T1001).

Not measured: the normal launch (startup compile and `ENGINE_READY`), FP32 with the timing protocol,
rows over 256 positions, images, the audio graphs other than T1001, the speaker player, other phones.

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
| `app/src/main/java/com/d1omni/D1Audio.kt`, `D1Wav.kt` | The audio host (waveform, mel, buckets, mask inputs, prefix rows); the 16 kHz mono PCM16 wav reader |
| `app/src/main/java/com/d1omni/D1Graph.kt`, `D1AudioEngine.kt` | One float32 single-signature graph on `CompiledModel`; the audio graph and a clip's prefix |
| `app/src/main/java/com/d1omni/D1AudioPlayer.kt` | Speaker playback with the time the sound started |
| `app/src/main/java/com/d1omni/D1GateCore.kt`, `D1GateRunner.kt`, `D1TimingRunner.kt` | Debug fixture gate and timing protocol |
| `app/src/main/java/com/d1omni/D1Device.kt`, `D1Demo.kt`, `D1Launch.kt` | Device facts, demo log lines, launch extras |
| `app/src/test/java/com/d1omni/` | JVM parity tests against the provider's code and the Python host |
| `scripts/install_to_device.sh` | Copies the external files into the app's `files/` |
| `scripts/TEST_DATA.md` | The tests' reference data and the device runs |
| `LICENSE`, `NOTICE`, `licenses/` | LFM Open License v1.0, attribution and changes, retained licenses |

## License

The model and the code ported from the provider's `prompt.py` and `audio.py` (through the model
repository's `host/d1_audio_host.py`) are under the LFM Open License v1.0 (`LICENSE`, unchanged);
`NOTICE` lists what this sample changed. The tokenizer port follows Hugging Face `tokenizers` and
`transformers` (Apache License 2.0, `licenses/`).
