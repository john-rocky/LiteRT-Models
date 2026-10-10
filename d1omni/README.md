# d1-omni Decide — ask your own questions about a voice note, a photo or a message, on the phone

Record a voice note, pick a photo or type a message, write the questions you want answered, and tap **Decide**: d1-omni-600M answers each question on the phone, with the probability of its answer, in a fraction of a second. Questions are typed — **noul** (yes or no, answered as P(yes)), **choice** (one of named options) and **score** (ordered levels, answered as the expected level) — and the model never generates text: each answer is read from its distribution over the question's options. Everything runs on the phone, offline: a Kotlin tokenizer and request encoder, the provider's audio front end and picture preprocessing in Kotlin, LiteRT graphs on the GPU, the read-out on the host.

**Load sample** fills the three screens with an example — an invented pet-grooming shop's customer who leaves a voice note asking to book a bath and a trim for their dog on Saturday morning, sends a photo of the dog, and writes that they were charged twice — with one default question per input. Replace any of it with your own: your voice, your photo, your text, your questions.

## Try it

- **Voice**: ● Record (16 kHz mono from the microphone; it stops at the length the largest installed audio graph holds: 10 s with the default install, 20 / 30 s with `AUDIO="2001"` / `"3001"`), ■ Stop; or Pick WAV (a 16 kHz mono 16-bit PCM file; another format is refused with the reason); Play listens to it. Default question: *What is the customer asking for?* (booking / cancel / prices / complaint).
- **Photo**: Pick photo opens **Recent photos**, the four pictures added to the phone most recently (asks for the photo permission the first time: READ_MEDIA_IMAGES from Android 13, READ_EXTERNAL_STORAGE before); tap one, or Browse… for the system photo picker (no permission needed). Default question: *What animal is in the photo?* (dog / cat / bird).
- **Message**: type or paste any text. Default questions: *Is the customer asking for a refund?* (yes or no) and *Which team should handle this?* (billing / booking / grooming).
- **Questions**: each screen shows its questions folded; **Edit** opens the editor (the same syntax as this repository's Kev Decide sample): a name, the type, the question, and the options one per line — a choice `name: description` or `name` (at least two), a score one level per line from the lowest (2 to 10), a noul optionally `true: what yes means` / `false: what no means`. Reset brings the default back.
- **Decide** answers the screen's questions: each answer's word and probability in large type as its call returns, then the input's milliseconds and the decision graph it ran on (`312 ms · L256`): from the input's samples, bytes or text to the last answer — the mel and the audio graph, or the decoding and the vision graphs, the encoding and one decision call per question.
- **Summary** adds up the inputs you decided: `3 inputs · 4 answers · N ms · airplane mode on`, one line of answers per input (N adds the milliseconds each screen showed).

The sample's answers on the Mac (the provider's float32 model and this repository's Python host agree within 4.8e-5): *booking* 0.999, *dog* 1.000, *yes* 0.999 and *billing* 0.939.

A picture longer than 384 px on its long side is first shrunk to 384 px (the size of the model repository's check-set pictures) with the provider's own float bilinear antialias resize, so that a phone photo becomes one crop whose rows fit the 128- and 256-position graphs (a 12 MP photo would otherwise be cut into ten tiles and need the 4,096-position graph); a smaller picture goes in unchanged. A JPEG is decoded by Android, whose decoder can differ from Pillow's by a level here and there, so a JPEG's answers match the Python host's closely rather than bit for bit (a PNG's decoded pixels are the Python host's exactly).

A recording is kept in the app's `files/` as `recorded-<epoch ms>.wav` (16 kHz mono 16-bit PCM) and is what Decide sends. The microphone is opened with Android's VOICE_RECOGNITION source (tuned for speech to a recogniser, without the call path's processing); the mel normalises each recording by its own mean and spread, so its level does not matter.

## Model and requirements

- Model: [litert-community/d1-omni-600M-LiteRT](https://huggingface.co/litert-community/d1-omni-600M-LiteRT) (decision graphs per input length, `contract.json`, `tokenizer.json`).
- Upstream: [LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) revision `414f8d64`, LFM Open License v1.0.
- Semantics: the provider's `prompt.py` (`encode`: one row per question, `<bos> <state> state <q> instructions <opt> <mask> option … <decide>`), the model repository's host steps (`contract.json` `host_steps.text` and `host_steps.audio`), the audio front end of `host/d1_audio_host.py` (the provider's `audio.py`: waveform, preemphasis, STFT, Slaney mel 128, log, per-bin normalization, bucket and mask inputs), the read-out of `d1_host.readout_f64` (the scores at each option's `<mask>`, divided by the checkpoint's temperature for a text request, softmax, a noul reversed to [yes, no]) and `prompt.answer`.
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile/target SDK 35.
- Runtime: LiteRT **2.2.0** `CompiledModel`; Material 1 Compose with MVVM.
- Permissions: `RECORD_AUDIO` (asked the first time you tap Record) and `READ_MEDIA_IMAGES` (`READ_EXTERNAL_STORAGE` up to Android 12; asked the first time you tap Pick photo). No network permission: everything runs on the phone, in airplane mode too.

Each kind of graph has its own GPU precision; `--es precision fp32|fp16acc` sets every kind, `--es precision_audio` / `--es precision_vision` one kind (a normal launch and the reproduction run; the gate and timing runs read `precision` for the decision graphs only). The app compiles every decision graph for the GPU at an explicit precision, FP32 by default: `CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)`. At FP32 the Galaxy S26 gives the provider's answers within the parity bar on every public text row (see Measured). The launch extra `--es precision fp16acc` compiles at `FP16_WITH_FP32_ACCUM` (float16 storage with float32 accumulation) instead: a little faster per call, but one public text row then moves past the bar (see Measured). The audio graph has its own precision, `--es precision_audio fp16acc|fp32` (`FP16_WITH_FP32_ACCUM` by default, `D1AudioEngine.DEFAULT_PRECISION`). The GPU's default precision is never used. When the GPU cannot compile or run a graph, the app runs it on the CPU (four threads) and logs `GPU_FALLBACK <error>` under the tag `D1OmniDemo`.

## Download, build and install

Use JDK 17, Android SDK platform 35 / build-tools 35.0.0, Android platform-tools (`adb`), Python 3 (the install script reads `contract.json`) and the Hugging Face CLI (`hf`). The default install is four files of the model repository: the contract, the tokenizer and the decision graphs for rows of up to 128 and 256 positions; audio requests add the audio graph of their length (`AUDIO="1001"`: clips up to 10 s).

```bash
hf download litert-community/d1-omni-600M-LiteRT contract.json tokenizer.json \
  d1-omni-600M_decide_L128_fp16.tflite d1-omni-600M_decide_L256_fp16.tflite \
  d1-omni-600M_audio_T1001_fp16.tflite d1-omni-600M_vision_tower_fp16.tflite \
  d1-omni-600M_projector_fp16.tflite host/vision_position_table.npy \
  --local-dir "$HOME/Downloads/d1-omni-600M-LiteRT"
./gradlew :app:assembleDebug
# Optional when multiple devices are connected:
# export ANDROID_SERIAL=your-device-serial
adb install app/build/outputs/apk/debug/app-debug.apk
AUDIO="1001" VISION=1 ./scripts/install_to_device.sh "$HOME/Downloads/d1-omni-600M-LiteRT"
adb shell am start -n com.d1omni/.MainActivity
```

The app needs all eight files (`AUDIO="1001" VISION=1` with the default `BUCKETS`): it compiles the five graphs at startup (9–10 s on the Galaxy S26).

The install script's argument defaults to the download directory above. `BUCKETS` names the decision graphs to copy (`"128 256"` by default; `BUCKETS="128 256 512 1024 2048"` copies five) and `AUDIO` the audio graphs (none by default; `AUDIO="1001"`, or any of 501 / 1001 / 2001 / 3001 for clips up to 5 / 10 / 20 / 30 s). It checks the size of every file against `contract.json` before it touches the device; a file already in the app's private `files/` with the contract's size and sha256 stays as it is, any other is copied through `/data/local/tmp/d1omni/` into `files/` with `run-as com.d1omni`, the temporary copy removed, and the copy's sha256 on the phone checked against `contract.json`. Install the debug APK before the files: `run-as` needs a debuggable package.

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

The full sha256 of every file is in `contract.json`, which the app reads at startup: it checks the tokenizer's sha256 and token IDs against it and finds each bucket's graph by the name it gives. Each decision graph has one signature, `decide_<L>`: `ids` int32 `[1,L]`, `prefix` float32 `[1,L,1024]`, `media` / `pad` / `keep_right` float32 `[1,L]`, `qtype_onehot` float32 `[1,3]` → `scores` float32 `[1,L]`. Each audio graph has one signature, `audio_<T>`: `mel` float32 `[1,128,T]`, `mel_valid` float32 `[1,T]`, `v1` / `v2` / `v3` float32 `[1,T1]` / `[1,T2]` / `[1,T3]` (T1001: 501 / 251 / 126) → `prefix` float32 `[1,T3,1024]`, whose first P rows are the clip's prefix.

<!-- vision (round 3) -->
### The picture path

`VISION=1 ./scripts/install_to_device.sh` also copies the three files a picture needs (download them with `hf download litert-community/d1-omni-600M-LiteRT d1-omni-600M_vision_tower_fp16.tflite d1-omni-600M_projector_fp16.tflite host/vision_position_table.npy`):

| File | Bytes | sha256 | Purpose |
|---|---:|---|---|
| `d1-omni-600M_vision_tower_fp16.tflite` | 171,563,424 | `6835066c…0942202` | Vision tower, one crop per call: `pixels` / `pos` float32 `[1,1024,768]`, `mask` `[1,1024]` → `features` `[1,1024,768]` |
| `d1-omni-600M_projector_fp16.tflite` | 16,791,504 | `9224b508…df80117` | `soft` float32 `[1,256,3072]` → `prefix` `[1,256,1024]` |
| `host/vision_position_table.npy` | 786,560 | `76d764aa…4fdb073` | The tower's 16 × 16 position table, float32 `[16,16,768]` (on the phone: `files/host/`) |

A picture goes through the provider's preprocessing on the phone, in Kotlin, giving the same values as the model repository's Python host (`host/d1_vision_host.py`) bit for bit:

```text
picture file → BitmapFactory (a PNG without its colour chunks, as Pillow ignores them) → RGB, EXIF orientation applied
  → layout: factor 32, a thumbnail, and above 524,288 pixels up to 10 tiles of 512 px
  → resize on the float path: PyTorch's bilinear kernel with antialias in float32, round half to even
  → per crop: 16 × 16 patches (x − 127.5) / 127.5, the position table resized to the patch grid, a mask
  → vision tower (GPU) → 2 × 2 pixel unshuffle → projector (GPU) → (rows / 2)(columns / 2) prefix rows
  → each question's row after the P prefix rows → decide_<L> → read-out without temperature → answer
```

Each kind of graph has its own GPU precision: the decision graphs follow `--es precision` (FP32 by default) and the tower and the projector `--es precision_vision` (`FP16_WITH_FP32_ACCUM` by default; `fp32` selects FP32). Picture files: `D1Vision.kt` (layout, resize, patches, position table, unshuffle), `D1Image.kt` (decoding), `D1Npy.kt` (the position table), `D1Graph.kt` (one single-signature graph on `CompiledModel`), `D1VisionEngine.kt` (the tower, the projector and a picture's prefix rows) and `D1VisionGate.kt` (the debug picture gate and timing, scripts/TEST_DATA.md).
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

An audio request (`D1AudioEngine.audioPrefix`, then `D1Engine.audioRows`) goes first through the clip, then through the same steps as text, with the clip's P prefix rows before every question:

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

The mel runs every step in float64 (one rounding to float32 at the end), the model repository's `mel(x, "float64")`: on the six public clips it equals the Python host's float64 mel bit for bit and is within 5.7e-5 of its float32 form (the provider's own; JVM tests). The Hann window and the Slaney filter bank are the Python host's float32 values bit for bit.

At most two decision graphs stay compiled. Before a graph compiles next to another one the app reads the memory Android reports available; under 2,500,000,000 bytes it gives up the second graph and runs every question of the request on the larger one. One audio graph stays compiled beside them (a clip of another bucket closes it first).

Play on the voice screen plays the clip through the speaker (`D1AudioPlayer`: `AudioTrack`, USAGE_MEDIA / CONTENT_TYPE_SPEECH, 16 kHz mono PCM16) at the phone's own media volume; the app never changes it.

## Run JSON and logcat (for a recording harness)

Every Decide writes `files/d1omni-run-<epoch ms>.json`: the input (`voice`, `photo`, `message`) and where it came from (`recorded`, `picked`, `typed`, `sample`); the media's sha256 and size (a recording's length, start and stop times, its saved wav and the audio source; a photo's format, EXIF orientation, decoded size and the size the model saw); the questions as asked; per question its row (ids, markers, P, bucket, the sha256 of the int32 ids), its unrounded probabilities, `prompt.answer()` and the strings on screen; the host steps' and media graphs' ms, the work as shown (`item_total_ms`) and unrounded (`item_total_ns`); the device, runtime, resident graphs and their compile times, the memory at ready, airplane mode and the process's cgroup at the start and the end; and where the pill and the answers were drawn (screen px, text size, lines, overflow). Logcat tag `D1OmniDemo`: `ENGINE_READY load_ms= warmup_ms=`, `RECORD_START`, `RECORD_STOP samples= seconds= wav= sha256=`, `PHOTOS_SHOWN n= names=`, `PICKED kind= sha256= name=`, `TYPED chars=`, `SAMPLE_LOADED`, `DECIDE_START item= source=`, `Q_DONE item= qid= ms=`, `DECIDE_DONE item= ms= json=`, `failed …`, `GPU_FALLBACK …` (a graph runs on the CPU).

The reproduction run drives the same code path as the buttons — Load sample, then Decide on the voice, photo and message screens in turn, then the summary — and logs `AUTOPLAY_START` / `AUTOPLAY_DONE json=…`:

```bash
adb shell am start -n com.d1omni/.MainActivity --ez autoplay true [--ei delay_ms 1000] [--ei gap_ms 1500] \
  [--es precision fp32|fp16acc] [--es backend gpu|cpu]
```

## Measured

Galaxy S26 SM-S942Q, Android 16, LiteRT 2.2.0, this app's debug build, connected over USB, the files above; every phone run below ended with the app in the foreground (cgroup `top-app`: 17 of 17 runs). Each check-set run is the debug fixture gate or the timing protocol (scripts/TEST_DATA.md) on the model repository's public check sets (`fixtures/public_text.json`, `public_audio.json`, `public_image.json`), compared with the provider's float32 probabilities. The bar: the same argmax on every row whose reference top-2 gap is above 0.02, max |Δp| ≤ 0.02 and mean |Δp| ≤ 0.002 over every option, no non-finite value. A graph call is timed from the input writes through `run()` to the read-back of its output.

### Text (210 rows that fit 128 positions, 43 that need 256)

| Graph, precision | Rows | Argmax outside near-ties | Near-ties kept | Max \|Δp\| | Mean \|Δp\| | Bar |
|---|---:|---:|---:|---:|---:|---|
| L128, FP32 | 210 | 203/203 | 7/7 | 0.00192 | 2.50e-04 | pass |
| L256, FP32 | 43 | 40/40 | 3/3 | 0.00120 | 1.64e-04 | pass |
| L128, FP16_WITH_FP32_ACCUM | 210 | 203/203 | 6/7 | 0.0400 | 1.85e-03 | **fail** |
| L256, FP16_WITH_FP32_ACCUM | 43 | 40/40 | 3/3 | 0.00644 | 9.27e-04 | pass |

- At `FP16_WITH_FP32_ACCUM` one row is past the bar: `semif_46b7029b9a704138b77a/answer` gives [0.193, 0.229, 0.578] against the provider's [0.177, 0.204, 0.618] (max |Δp| 0.0400, the same argmax). Its scores at the three markers are 0.6875, 0.93310546875 and 2.2265625 (float16 values); at FP32 the phone gives 0.613, 0.807 and 2.361. The same scores came back in all 11 calls of this row over 3 processes. The next rows are `semif_1105577da4c8dad4609d/answer` (0.0131) and `tv4_043/answer` (0.0124); the near-tie that flips is `semif_fda0ca94652c6d739ecf/answer`.
- The Kotlin tokenizer and encoder give the Python host's ids, markers and serialized state on 253/253 rows on the phone; the Kotlin read-out of the phone's scores equals Python's `readout_f64` to 5.6e-17. At FP32 the phone's probabilities are within 1.6e-05 of a run of the same graphs on the Mac's CPU (LiteRT 2.2.0).
- Every row ran on the GPU, and every decision graph compiled whole in every run: `Replacing 1173 out of 1173 node(s) with delegate (LITERT_CL) node, yielding 1 partitions`.

| One call, median | FP32 | FP16_WITH_FP32_ACCUM |
|---|---:|---:|
| L128 (rows of the gate, 205 calls) | 46.4 ms | 41.9 ms |
| L256 (rows of the gate, 38 calls) | 82.9 ms | 78.2 ms |
| L128, timing protocol: one question (47 tokens), 20 calls | not measured | 39.8 ms (39.0–40.9) |
| L128, timing protocol: a three-question request (47, 56, 51 tokens), 20 requests | not measured | 122.4 ms (117.0–125.1), 40.8 ms per call |

The gate medians leave out each run's first five calls and compare the two precisions only roughly: the samples taken during each run (about 1–2 s apart) read a CPU frequency cap in 7 of 9 (L128, FP32), 5 of 6 (L256, FP32), 7 of 9 (L128, FP16_WITH_FP32_ACCUM), 10 of 13 (L256, FP16_WITH_FP32_ACCUM), and a GPU clock ceiling under 1,300 MHz or a raised thermal power level in 6 of 9 (L128, FP32), 2 of 6 (L256, FP32), 5 of 9 (L128, FP16_WITH_FP32_ACCUM), 0 of 13 (L256, FP16_WITH_FP32_ACCUM). Each timing set started after a cool-down (thermal status 0, GPU clock ceiling 1,300 MHz; a CPU frequency cap at the end of the cool-down in 1 of 2 sets), ran five warm-up calls, and the app read the GPU clock ceiling 1,300 MHz before every timed call. Compiling a graph took 2.6 s (L128) and 3.4 s (L256 next to a compiled L128) at `FP16_WITH_FP32_ACCUM`, 3.0 s (L128) and 3.5 s (L256) at FP32. With L128 and L256 both compiled, the app's VmHWM reached 5,089,364 kB and MemAvailable fell to 3,924,960 kB at its lowest (samples about 1 s apart); Android reported 5,541,298,176 bytes available right before the second compile.

### Audio (18 rows: 6 clips of the public audio check set, three questions each, all on the T1001 audio graph)

The same phone and build; each clip's wav read on the phone; the decision graphs at FP32, the audio graph at each precision below.

| Audio graph, precision | Rows | Argmax outside near-ties | Max \|Δp\| | Mean \|Δp\| | Bar | Audio graph, one call |
|---|---:|---:|---:|---:|---|---:|
| T1001, FP16_WITH_FP32_ACCUM (default) | 18 | 18/18 | 0.00138 | 2.15e-04 | pass | 31.1 ms (31.0–31.3) |
| T1001, FP32 | 18 | 18/18 | 0.00154 | 1.25e-04 | pass | 40.0 ms (39.4–40.4) |

- The app's mel on the phone equals the model repository's float64 mel bit for bit on 6/6 clips (within 5.7e-05 of its float32 form); n, frames, T, the bucket and P equal the Python host's on 6/6 clips, and the app's own encoding gives the fixture's ids and markers on 18/18 rows. The audio graph compiled whole for the GPU in every run: `Replacing 2423 out of 2423 node(s) with delegate (LITERT_CL) node, yielding 1 partitions`.
- A whole request, timing protocol (`aud_food_03`, a check-set clip, its three questions on L256; audio graph at `FP16_WITH_FP32_ACCUM`, decision graphs at FP32; five warm-up and 20 timed requests): **302.5 ms** median (294.3–311.1) from the wav file to the three answers. Per step (medians): reading the wav 0.6 ms, the mel 13.1 ms, the audio graph 33.1 ms, the encoding of the three questions 1.1 ms, the three decision calls 253.1 ms, the waveform, the inputs, the prefix copy and the read-out 1.5 ms together. The app read the GPU clock ceiling 1,300 MHz before every request; 3 of the 8 samples taken during the run (about 2 s apart) read a lower ceiling or a raised thermal power level (1,200 MHz at the lowest, power level up to 2, GPU up to 104 °C), and 0 a CPU frequency cap.
- With the L256, L128 and T1001 graphs compiled at once, the app's VmHWM reached 6,245,076 kB and MemAvailable fell to 3,227,380 kB at its lowest (samples about 1 s apart); Android reported 6,988,881,920 bytes available before the first compile and 4,102,299,648 after the third.

### Pictures (12 rows: 5 pictures of the public image check set, 11 crops)

The same phone and build; each picture decoded on the phone; the decision graphs at FP32, the vision tower and the projector at each precision below.

| Tower and projector, precision | Rows | Argmax outside near-ties | Max \|Δp\| | Mean \|Δp\| | Bar | Tower, one crop, median |
|---|---:|---:|---:|---:|---|---:|
| FP16_WITH_FP32_ACCUM (default) | 12 | 12/12 | 0.00154 | 2.51e-04 | pass | 136.5 ms (134.3–147.8, 11 crops) |
| FP32 (the 9 rows that fit 256 positions) | 9 | 9/9 | 0.00330 | 4.23e-04 | pass | 191.7 ms (187.4–195.6, 4 crops) |

- The app's decoded pictures equal the Python host's on 5/5 pictures (sha256 of the RGB), the tower's inputs on 11/11 crops (grid, pixels, positions, mask), P on 5/5 pictures and the ids on 12/12 rows. The tower and the projector compiled whole for the GPU in every run: `Replacing 717 out of 717 node(s) with delegate (LITERT_CL) node, yielding 1 partitions`, `Replacing 5 out of 5 node(s) with delegate (LITERT_CL) node, yielding 1 partitions`.
- A whole request, timing protocol (`img_dogs_01`, the sample's photo, two questions on L128; five warm-up and 20 timed requests): **264.7 ms** median (258.3–272.3) from the picture file to the two answers. Per step (medians): decoding 9.5 ms, resize and patches 21.2 ms, the tower 136.0 ms, the unshuffle 2.1 ms, the projector 4.3 ms, the two decision calls 91.4 ms. The app read the GPU clock ceiling 1,300 MHz before every request; 0 of the 8 samples taken during the run read a lower ceiling and 5 a CPU frequency cap.
- With the vision tower, the projector, L256 and L128 compiled at once, the app's VmHWM reached 6,021,596 kB and MemAvailable fell to 2,334,976 kB at its lowest (samples about 1 s apart).

### The app on the Galaxy S26

Galaxy S26 SM-S942Q, Android 16, LiteRT 2.2.0, this app's debug build, airplane mode on, the phone at its lock screen (the app shows over it); one Decide per input as a person makes it, the taps and typing sent with adb, the screen recorded (take r8t2, the clip of the model's announcement). The numbers are the run JSON's, which the screen shows; each answer was checked against the model repository's Python host on the same input (and, for the recording, the provider's own float32 code): same ids, same argmax, max |Δp| ≤ 0.02.

| Input | Source | Question | Answer on screen | Probability | ms on screen |
|---|---|---|---|---:|---|
| Voice | a recording through the phone's microphone (the Mac's speaker played the sample line) | What is the customer asking for? | booking | 1.000 | 153 ms · L256 |
| Photo | picked from Recent photos | What animal is in the photo? | dog | 1.000 | 211 ms · L128 |
| Message | typed | Is the customer asking for a refund? | yes | 0.999 |  |
|  |  | Which team should handle this? | billing | 0.939 | 96 ms · L128 |

- The recording: 8.30 s of 16 kHz mono from the phone's microphone (VOICE_RECOGNITION) while the Mac's built-in speaker played the sample line in the same room (the distance was not measured); the phone's answer P(booking) 0.999558, the Python host on the same wav 0.999557.
- Recent photos showed 4 pictures (img_dogs_01.png, img_03.png, img_02.png, img_bike_03.png), the four the harness had copied to the phone; the dogs' photo is 384 × 216, so the app used it unchanged.
- Engine load (five graphs compiled on the GPU) 10,421 ms in the take's launch, then one untimed pass over the sample; MemAvailable at ready 4,275,556 kB. The three other takes of the same slot loaded in 9,809, 9,253, 10,459 ms (their run JSONs under demo/device/r8/).
- Each ms on screen is the input's whole work: from its samples, file bytes or text to the last answer (the mel and the audio graph, or the decoding and the vision graphs, then the encoding and one decision call per question).

Not measured: FP32 with the text timing protocol, text rows over 256 positions, picture rows over 256 positions at FP32, the audio graphs other than T1001, recordings and photos other than the ones above, the app's steady state over many Decides, other phones.

## Files

| Path | Role |
|---|---|
| `app/src/main/java/com/d1omni/MainActivity.kt`, `MainViewModel.kt` | Compose host (`singleTop`; the microphone and photo permissions, the system photo and file pickers), the screen's state (the Recent photos sheet from MediaStore), the engine on the worker thread |
| `app/src/main/java/com/d1omni/view/AppScreen.kt`, `QuestionsEditor.kt`, `StatusScreen.kt` | The title and pill, the Voice / Photo / Message / Summary tabs, the Recent photos sheet, the answers in large type; the questions editor; the status screen of the debug runs |
| `app/src/main/java/com/d1omni/D1Sample.kt`, `D1Drafts.kt` | The bundled sample (`res/raw/sample.json`); the editor's questions and their checks |
| `app/src/main/java/com/d1omni/D1Recorder.kt`, `D1Decide.kt`, `D1AppEngine.kt` | The microphone at 16 kHz mono; one Decide per input (and the photo's shrink); the five resident graphs |
| `app/src/main/java/com/d1omni/D1Answers.kt`, `D1Run.kt`, `D1Demo.kt` | The answer's word and probability on screen and the screen's words; the run JSON; the logcat lines |
| `app/src/main/java/com/d1omni/D1Tokenizer.kt` | Byte-level BPE tokenizer read from `tokenizer.json` |
| `app/src/main/java/com/d1omni/D1Prompt.kt` | The provider's `prompt.py`: questions, `escape`, `serialize`, options, `encode`, `answer` |
| `app/src/main/java/com/d1omni/D1Json.kt` | JSON with Python's key order, number semantics and `json.dumps` output |
| `app/src/main/java/com/d1omni/D1Contract.kt`, `D1Request.kt` | `contract.json`; requests `{state, questions}` |
| `app/src/main/java/com/d1omni/D1Rows.kt`, `D1Readout.kt` | Request kinds, rows, `build_inputs`; the read-out |
| `app/src/main/java/com/d1omni/D1Decider.kt`, `D1Residency.kt`, `D1Engine.kt` | One graph on `CompiledModel`; which graphs stay compiled; tokenizer + contract + graphs |
| `app/src/main/java/com/d1omni/D1Audio.kt`, `D1Wav.kt` | The audio host (waveform, mel, buckets, mask inputs, prefix rows); the 16 kHz mono PCM16 wav reader and writer |
| `app/src/main/java/com/d1omni/D1Graph.kt`, `D1AudioEngine.kt`, `D1AudioPlayer.kt` | One float32 single-signature graph on `CompiledModel`; the audio graph and a clip's prefix; speaker playback |
| `app/src/main/java/com/d1omni/D1Vision.kt`, `D1Image.kt`, `D1Npy.kt`, `D1VisionEngine.kt` | The picture host (layout, resize, patches, positions, unshuffle); decoding; the position table; the tower and the projector |
| `app/src/main/java/com/d1omni/D1GateCore.kt`, `D1GateRunner.kt`, `D1TimingRunner.kt`, `D1VisionGate.kt` | Debug fixture gates and timing protocols |
| `app/src/main/java/com/d1omni/D1Device.kt`, `D1Launch.kt` | Device facts; launch extras |
| `app/src/main/res/raw/` | `sample.json`, `sample_voice_note.wav`, `img_dogs_01.png` (the sample and its media) |
| `app/src/test/java/com/d1omni/` | JVM parity tests against the provider's code and the Python host |
| `scripts/install_to_device.sh` | Copies the external files into the app's `files/` |
| `scripts/TEST_DATA.md` | The tests' reference data and the device runs |
| `LICENSE`, `NOTICE`, `licenses/` | LFM Open License v1.0, attribution and changes, retained licenses |

## License

The model and the code ported from the provider's `prompt.py` and `audio.py` (through the model repository's `host/d1_audio_host.py`) are under the LFM Open License v1.0 (`LICENSE`, unchanged); `NOTICE` lists what this sample changed. The tokenizer port follows Hugging Face `tokenizers` and `transformers` (Apache License 2.0, `licenses/`).

The sample's media: `sample_voice_note.wav` is speech synthesised with Kokoro-82M (Apache License 2.0, `licenses/Kokoro-82M-APACHE-2.0.txt`, voice am_michael) from a line written for this sample; `img_dogs_01.png` is a CC0 1.0 photograph from Wikimedia Commons (`File:Two_French_bulldogs_swimming_in_life_jackets.jpg`), its long side resized to 384 px (the model repository's public check-set file). The sample's message, its questions and the grooming shop are invented for this sample.
