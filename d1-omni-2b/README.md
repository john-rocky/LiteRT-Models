# d1-Omni-2B Decide — ask typed questions about a recording, a text or a photo, on the phone

Record a voice message or a sound in your home, and have typed questions about it answered together with a text record in one forward pass: nothing is transcribed, and nothing leaves the phone. Add a recording (**● Record**, up to 30 s from the microphone, or a WAV file), a text or a JSON record and a photo if you have them, write the questions you want answered, and tap **Decide**. A question can also be asked aloud: record it on its card, and the clip takes the place of its text. The answers are typed: yes or no, one or several of your options, a level on your scale, a number in a range, a value read out of the text (an ISO date, a number with its unit, a text) with its span and the probability that it is there, or boxes in the photo. The model generates no tokens: each answer is read from the model's output at the question's slots. On a Galaxy S26 the decision graph runs on the Qualcomm NPU, from files compiled ahead of time for its chip, and the audio encoder and the picture tower on its GPU; on other phones every graph runs on the GPU at FP32. Everything runs on the phone, offline: a Kotlin tokenizer, prompt and audio front end, the model's answer readers ported to Kotlin, LiteRT graphs.

The app comes with four examples:

| Example | Recording | State | Questions |
|---|---|---|---|
| **Three beeps** | The source model card's recording: three short 880 Hz beeps over quiet noise (4 s) | None | How many beeps there are (a number from 0 to 10); whether someone speaks (yes or no) |
| **Asked aloud** | The same recording | None | The same two questions, spoken instead of typed |
| **Voicemail** | A caller asks a bike shop to refund a payment that was charged twice (6.8 s) | The shop's JSON record: its opening days and hours, its closed days and its three teams | Which team should handle the voicemail (workshop, sales or billing); whether the caller asks to be called back; whether the caller wants to come in on a day when the shop is closed |
| **Voicemail, radio on** | A caller asks to bring in a bike whose back brake has stopped working and to be called back, with a song playing as loud as the voice; the whole clip passed through a telephone band (8.6 s) | The same record | Which team should handle the voicemail; whether the caller asks to be called back |

For Three beeps the model card gives 3 beeps (interval 3 to 4) and P(yes) 0.07. In the voicemails the answers follow from what the caller says: billing, no call back and not a closed day for the first; the workshop and a call back for the second.

## Try it

- **Examples**: the four above. Each fills the state, the recordings and the questions; replace any of it with your own.
- **State**: type or paste a message or a note, or switch to **JSON** and paste a record (a read then also gives the path of the value), or leave it empty when the recordings are the whole state. Under **Recordings**, **● Record** records the microphone at 16 kHz mono until **■ Stop** or 30 s; **Pick a WAV file** opens the system file picker; **▶ Play** plays a clip as the app read it (its channels averaged) and **Remove** removes it. A state can hold several clips. **Add a photo** opens the system photo picker; the photo is shrunk to 384 pixels on its long side before the model sees it, so it fits the smallest picture graph.
- **Questions**: each card shows its name, its type and the question. **● Ask aloud** records the question (**WAV file** picks one instead); the card then shows *Asked aloud* with its clip, **▶ Play** and **Type instead**. **Edit** opens the name, the type — Yes / no, Choice, Several choices, Score, Text, Number, Number in a range, Date, Box, Boxes — the options of a choice (a name and what it means), the levels of a score (2 to 10, from the lowest), the range and step of a number in a range, a unit, and the most boxes to read (1 to 16). **Add question** adds a card; **Remove this question** removes one.
- **Decide**: every card gets its answer underneath — the value and P(found) for a read (*Not in the text* when it is absent), the option and its probability for a choice, P(yes) for a yes / no, the level for a score, the median and interval for a number in a range, boxes drawn on the photo in the card's color. The line under the button gives the time of the call on the phone, the tokens it read, the graph length it ran on and the accelerator of the decision graph: `Answered in … ms on this phone · … tokens · L128 graph · NPU`. The first call on a new length adds the graph's compile time.
- **What it refuses**, with the reason on the screen and before any graph runs: a missing model file (it names the file), a clip longer than 30.015 s (3,000 feature frames, the longest audio graph; the app never cuts a clip), an empty question, a WAV file it cannot read (not a RIFF WAVE file, not 16-bit PCM, or cut short), and a sequence longer than the phone's graphs.

Up to 16 questions share one sequence with the state; more questions take another sequence over the same state. Nothing is cached between calls: every Decide sends the state, the clips and the questions as they are on the screen. The app asks for the microphone the first time it records.

## Model and requirements

- Model: [litert-community/d1-Omni-2B-LiteRT](https://huggingface.co/litert-community/d1-Omni-2B-LiteRT) (decision graphs per sequence length, decision graphs compiled for the NPU of the Qualcomm SM8850, picture towers per canvas, audio encoders per clip length, the embedding table, and the model's own `tokenizer.json`, `config.json`, `calibration.json`, `processor_config.json` and `chat_template.jinja` under `provider/`).
- Upstream: [LiquidAI/d1-Omni-2B](https://huggingface.co/LiquidAI/d1-Omni-2B) revision `431f1d66`, LFM Open License v1.0.
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile SDK 36, target SDK 35.
- Runtime: LiteRT 2.2.0 `CompiledModel` (`litert`): the GPU at FP32 with constant tensor sharing (one copy of the weights on the GPU), and on a Galaxy S26 the NPU through Qualcomm's dispatch library. 2.2.0 is the version the NPU files were compiled with (with Qualcomm's QAIRT 2.47) and the version of the model repository's phone checks. Material Compose.
- **Galaxy S26 (12 GB)**: the decision files compiled for its NPU (`npu/sm8850/`, for sequences of up to 128, 256 and 512 tokens), one loaded at a time, with the audio encoder and the picture tower on the GPU at FP32. For a longer sequence the app closes the NPU file and compiles a per-length GPU file (`d1-omni-2b_decide_L1024_fp16fc.tflite` and up) if it is in the model folder; without one it refuses the sequence and says which file it needs. The NPU needs Qualcomm's libraries inside the APK (below). Without them, on a phone without a Qualcomm NPU, or when the NPU files do not load, the decision graph runs on the GPU at FP32 instead and the status line says why.
- **Pixel 11 Pro (16 GB)**: the eight-signature file on the GPU, one 3.0 GB file with the seven length graphs and the graph that runs eight sequences of up to 128 tokens in one call, over one copy of the weights. A phone with 12 GB takes the per-length files instead: on a Galaxy S26 the eight-signature file of the same graphs (the d1-2B file) ran out of memory while it compiled. On a 12 GB phone the app refuses sequences longer than 4,096 tokens.
- Memory: before it compiles anything the app reads MemAvailable. It needs 3.8 GB free for an NPU decision file, 9.4 GB for the eight-signature file and 6.3 GB for a per-length file (the load's need plus 0.8 GB), also when the Galaxy S26 moves from the NPU to a GPU file (after it closes the NPU file it waits up to 3 s for that memory to come back), 0.8 GB before the first compile of the picture tower and 0.5 GB before the first compile of an audio file. When less is free, it first closes the audio files and the picture tower it keeps compiled for later requests and reads again for up to 3 s; if that is still not enough, it shows how much is free and loads nothing. The figures are this app's own where measured: on a Galaxy S26 its load with the 256-token NPU file took 2.9 and 3.0 GB of MemAvailable in two runs, the first compile of an audio file up to 0.47 GB and the 1,024-token GPU file 5.3 GB; on a Pixel 11 Pro its load of the eight-signature file took 8.4 to 8.6 GB in three runs, which left 1.2 to 1.7 GB for the audio files and the tower. The other figures (a per-length file, the tower) are what the loads of the same graphs (the d1-2B files, which differ in their weights only) took on a Galaxy S26 and a Pixel 11 Pro.
- No network permission: the app reads the model from its own folder on the phone. It asks for the microphone (`RECORD_AUDIO`) to record.

## Download, build and install

Use JDK 17, the Android SDK (platform 36), Android platform-tools (`adb`) and the Hugging Face CLI (`hf`).

**Galaxy S26 (NPU).** First copy Qualcomm's runtime libraries into `app/src/main/jniLibs/arm64-v8a/`, as files (not links; git ignores them, and `app/build.gradle.kts` lists them):

| File | From |
|---|---|
| `libLiteRtDispatch_Qualcomm.so` | LiteRT's NPU runtime libraries (`litert_npu_runtime_libraries.zip`, a release asset of [google-ai-edge/LiteRT](https://github.com/google-ai-edge/LiteRT/releases)), its Qualcomm runtime for Hexagon v81 |
| `libQnnHtp.so`, `libQnnSystem.so`, `libQnnHtpV81Stub.so`, `libQnnHtpV81CalculatorStub.so` | Qualcomm AI Runtime (QAIRT) 2.47.0, `lib/aarch64-android/` |
| `libQnnHtpV81Skel.so` | QAIRT 2.47.0, `lib/hexagon-v81/unsigned/` |

The NPU files are compiled ahead of time, so the compiler plugin (`libLiteRtCompilerPlugin_Qualcomm.so`) and `libQnnHtpPrepare.so` are not needed. The libraries come under Qualcomm's and LiteRT's licenses and are not part of this repository. Then:

```bash
pip install -U huggingface_hub
hf download litert-community/d1-Omni-2B-LiteRT \
  provider/tokenizer.json provider/config.json provider/calibration.json \
  provider/processor_config.json provider/chat_template.jinja tables/embed_table.npy \
  npu/sm8850/d1-omni-2b_decide_L128_fp16fc_SM8850.tflite \
  npu/sm8850/d1-omni-2b_decide_L256_fp16fc_SM8850.tflite \
  npu/sm8850/d1-omni-2b_decide_L512_fp16fc_SM8850.tflite \
  d1-omni-2b_tower_G24x24_fp16fc.tflite \
  d1-omni-2b_audio_T504_fp16fc.tflite d1-omni-2b_audio_T1000_fp16fc.tflite \
  d1-omni-2b_audio_T2000_fp16fc.tflite d1-omni-2b_audio_T3000_fp16fc.tflite \
  --local-dir "$HOME/Downloads/d1-Omni-2B-LiteRT"
./gradlew :app:installDebug
# Optional when more than one phone is connected:
# export ANDROID_SERIAL=your-device-serial
adb shell am start -n com.d1_omni_2b/.MainActivity
scripts/push_model.sh "$HOME/Downloads/d1-Omni-2B-LiteRT"
```

That is 14 files, 10,794,311,620 bytes. For sequences of more than 512 tokens, add per-length GPU files, for example `d1-omni-2b_decide_L1024_fp16fc.tflite`.

**Pixel 11 Pro and other phones (GPU).** Without the NPU files and libraries the same app runs every graph on the GPU. Download the eight-signature file instead of the three NPU files:

```bash
hf download litert-community/d1-Omni-2B-LiteRT \
  provider/tokenizer.json provider/config.json provider/calibration.json \
  provider/processor_config.json provider/chat_template.jinja tables/embed_table.npy \
  d1-omni-2b_decide_multi8_fp16fc.tflite d1-omni-2b_tower_G24x24_fp16fc.tflite \
  d1-omni-2b_audio_T504_fp16fc.tflite d1-omni-2b_audio_T1000_fp16fc.tflite \
  d1-omni-2b_audio_T2000_fp16fc.tflite d1-omni-2b_audio_T3000_fp16fc.tflite \
  --local-dir "$HOME/Downloads/d1-Omni-2B-LiteRT"
./gradlew :app:installDebug
adb shell am start -n com.d1_omni_2b/.MainActivity
scripts/push_model.sh "$HOME/Downloads/d1-Omni-2B-LiteRT"
```

That is 12 files, 4,845,895,796 bytes. For a 12 GB phone, download per-length files instead of `d1-omni-2b_decide_multi8_fp16fc.tflite`, for example `d1-omni-2b_decide_L128_fp16fc.tflite`, `d1-omni-2b_decide_L256_fp16fc.tflite` and `d1-omni-2b_decide_L512_fp16fc.tflite` (Three beeps runs on L128, Asked aloud on L256, the two voicemails on L512). `./gradlew -Pd1.noNpuLibs :app:assembleDebug` builds an APK without Qualcomm's libraries even when they are in `jniLibs`.

Open the app once before the push: it creates its folder and says which files are missing. `scripts/push_model.sh` copies the provider files, the table, every `d1-omni-2b_*.tflite` of the folder and the NPU files in `npu/sm8850/` to `/sdcard/Android/data/com.d1_omni_2b/files/model/` and checks each file's size on the phone. Then close the app and open it again: it compiles the decision graph, and the status line says when it is ready, on which accelerator, how long the compile took, and which audio files are missing. A clip goes to the smallest audio file that holds it; a clip that needs a missing audio file is refused with that file's name.

## External files

| File | Bytes | sha256 | Purpose |
|---|---:|---|---|
| `provider/tokenizer.json` | 17,908,634 | `41927bd3…945ec44` | The model's tokenizer, unchanged |
| `provider/config.json` | 5,802 | `cebf53b6…aad7169` | Special tokens, codes, the set size, the audio token |
| `provider/calibration.json` | 4,819 | `182270f6…faf1772` | Temperatures and the calibration tables of the answers |
| `provider/processor_config.json` | 1,928 | `2288d34b…bdcd0e5` | The audio feature extractor's settings |
| `provider/chat_template.jinja` | 13,349 | `f421a240…6c6389c` | The model's chat template (the app checks its audio markup) |
| `tables/embed_table.npy` | 524,288,128 | `0c5cecb5…33c27ef` | The embedding rows, bfloat16 bits `[128000, 2048]` (looked up on the host) |
| `npu/sm8850/d1-omni-2b_decide_L128_fp16fc_SM8850.tflite` | 2,970,355,376 | `f7d29aa9…90f9901` | Up to 128 tokens, compiled for the SM8850's NPU (Galaxy S26) |
| `npu/sm8850/d1-omni-2b_decide_L256_fp16fc_SM8850.tflite` | 2,976,356,016 | `ed4fbb44…e012071` | Up to 256 tokens, NPU |
| `npu/sm8850/d1-omni-2b_decide_L512_fp16fc_SM8850.tflite` | 3,000,534,704 | `5ccb7e3f…101549f` | Up to 512 tokens, NPU |
| `d1-omni-2b_decide_multi8_fp16fc.tflite` | 2,998,830,272 | `47c5a1e4…c95684a` | The seven lengths and the eight-sequence graph as signatures of one file (16 GB phones) |
| `d1-omni-2b_decide_L128_fp16fc.tflite` | 2,934,923,056 | `0c9814d0…f0359c5` | Up to 128 tokens, one length per file (12 GB phones on the GPU) |
| `d1-omni-2b_decide_L256_fp16fc.tflite` | 2,935,382,112 | `ee1e204d…b3b07a1` | Up to 256 tokens |
| `d1-omni-2b_decide_L512_fp16fc.tflite` | 2,938,800,160 | `6464159a…fde9dfd` | Up to 512 tokens |
| `d1-omni-2b_decide_L1024_fp16fc.tflite` | 2,940,897,312 | `16bd0b20…3d475ed` | Up to 1,024 tokens (on a Galaxy S26, the first length above the NPU files) |
| `d1-omni-2b_decide_L2048_fp16fc.tflite` | 2,945,091,616 | `23b043f8…7b2b37d` | Up to 2,048 tokens |
| `d1-omni-2b_decide_L4096_qb1024_fp16fc.tflite` | 2,953,520,352 | `da6d42b4…5496fd4` | Up to 4,096 tokens (the longest the app runs on a 12 GB phone) |
| `d1-omni-2b_decide_L6144_qb1024_fp16fc.tflite` | 2,961,934,864 | `409e949b…99cfc4e` | Up to 6,144 tokens (16 GB phones only) |
| `d1-omni-2b_tower_G24x24_fp16fc.tflite` | 377,915,232 | `b1ea62f7…2b6463b` | The picture tower for pictures up to 24 × 24 patches (384 px) |
| `d1-omni-2b_audio_T504_fp16fc.tflite` | 231,210,512 | `088e8e54…4fe1aa3` | The audio encoder for clips of up to 504 feature frames (5.055 s) |
| `d1-omni-2b_audio_T1000_fp16fc.tflite` | 231,351,376 | `cadef862…8d13b0c` | Up to 1,000 frames (10.015 s) |
| `d1-omni-2b_audio_T2000_fp16fc.tflite` | 231,822,368 | `126a9122…d19aa81` | Up to 2,000 frames (20.015 s) |
| `d1-omni-2b_audio_T3000_fp16fc.tflite` | 232,543,376 | `d8461bbb…76ede99` | Up to 3,000 frames (30.015 s) |

The GPU graphs store the fully connected weights in float16 and compute in float32; the app compiles them at FP32 (the GPU's default float16 activations move the answers outside the checked tolerance). The NPU files run the whole graph on the NPU in float16. The sizes and SHA-256 are the model repository's, at revision `bbbff8de`.

## How a request runs

```text
State (text, or JSON written as the model's prompt writes it) + recordings + photo + typed or spoken questions
  → each clip: a recording (16 kHz mono) or a 16-bit PCM WAV file at any rate → samples / 32768, channels averaged
    → another rate: resampled to 16 kHz as the model's code does (windowed sinc, 6 zeros, roll-off 0.99)
    → padded to whole frames → periodic Hann window of 400 → real FFT of 400, hop 160, no centering → |X|²
    → Slaney mel filters, 128 bands → log10, floor −12 → (x + 4) / 4 → features [128, F]
    → more than 3,000 frames: refused before any graph runs
    → the smallest audio graph that holds F frames (504, 1,000, 2,000 or 3,000; GPU, FP32) → ceil(F / 8) rows
  → prompt: the media's markup (the photo's tokens; each recording as <|vision_start|> <|audio|> × ceil(F / 8)
    <|vision_end|>), the state, then each question's header, instructions and answer slots; a spoken question's
    clip takes the place of its instructions
  → byte-level BPE (Kotlin, read from tokenizer.json); up to 16 questions after one state per sequence
  → photo: shrunk to 384 px → smart_resize to multiples of 32 → PyTorch's uint8 bicubic resize
    → 16 × 16 patches on the smallest tower canvas → tower (GPU) → the picture's token rows
  → embedding rows from tables/embed_table.npy, each clip's rows at its <|audio|> tokens, the picture's rows at
    its tokens
  → decide on the smallest graph length that holds the sequence → hidden states
    (Galaxy S26: the NPU up to 512 tokens, the GPU above; other phones: the GPU at FP32)
  → options: the hidden state at each slot × the options' embedding rows → temperature → softmax
    → calibration → yes / no, choice, several choices, score, number in a range
  → reads: the pointer over the state's positions → the best span → its value parsed as the model's
    code parses it (dates, numbers with units and scales) → text, number or date and P(found)
  → boxes: the four edges' grids → the box and its confidence
```

The audio front end is a line-by-line port of the model repository's `host/d1_audio_features.py`, which reproduces the float32 arithmetic of the provider's feature extractor: the mel filters in float64 as transformers computes them, pocketfft's real FFT as PyTorch runs it, and SLEEF's float32 sine, cosine and logarithm. The WAV reader follows the model's own (`media.read_audio`): 16-bit PCM only, in a plain or an extensible WAV file. A rate whose resampling filter would need more than 1,000,000 values (for example 44,101 Hz) is refused. The rest of the Kotlin host is a port of the model's Python code (`prompt.py`, `api.py`, `box.py`, `calibration.py`, `grid.py`, `json_spans.py`, `parsing.py`, `pointer.py`, the packing of `attention.py`, the slot reading of `runner.py` and the media markup of `media.py`) and of the model repository's host (`host/d1_host.py`, `host/d1_pictures.py`); the photo is read without color management, as the model's image code reads a file.

At most two audio graphs stay compiled; another length closes the one used least recently. On the Galaxy S26 one NPU file is compiled at a time, and a GPU file only for a sequence longer than 512 tokens, after the NPU file is closed.

## Run files (for tests and recordings)

The app runs a list of steps from a JSON file through the same functions the screen uses. Push a file, and the clips it names, to the app's `files/runs/` folder and start the app with it:

```bash
adb push run.json /sdcard/Android/data/com.d1_omni_2b/files/runs/run.json
adb push voicemail.wav /sdcard/Android/data/com.d1_omni_2b/files/runs/voicemail.wav
adb shell am start -n com.d1_omni_2b/.MainActivity --es run run.json
```

```json
{"steps": [
  {"example": "beeps"},
  {"decide": true, "out": "out/beeps.json"},
  {"example": "beeps_spoken"},
  {"decide": true, "out": "out/beeps_spoken.json"},
  {"clips": ["voicemail.wav"]},
  {"edit": {"question": "speech", "instructions": "Does a person leave a message?"}},
  {"decide": true, "out": "out/voicemail.json"}
]}
```

Steps: `{"example": id}`, `{"state": text}`, `{"json": true}`, `{"clips": [file, …]}` (the state's clips, from the runs folder), `{"photo": file}` (read as a picked photo is; `null` removes the photo), `{"questions": {name: question, …}}` (the cards, in the model's question JSON as in `app/src/main/assets/examples.json`; a spoken question's instructions are `{"type": "audio", "path": file}`), `{"spoken": {"question": name, "clip": file}}`, `{"edit": {"question": name, "instructions": text}}`, `{"decide": true, "out": path}`, `{"wait_ms": n}`. A clip, a photo or a question set the app refuses is listed with the reason in the run's log, as the screen shows it. Each `out` file holds the device, the graph form, the request (each clip by its file name, label and SHA-256, the photo by its size and the SHA-256 of its RGB values), every clip's sample rate, channels, samples, frames, tokens and the SHA-256 and shape of its features, the answers in the model's API shape, the token ids of every sequence, and the time of each part of the call (`prep_ms`, `feature_ms`, `tower_ms`, `audio_ms`, `decide_ms`, `readout_ms`, `total_ms`, the compile time, the graph length and the accelerator); a refused Decide writes the reason. `run.json.done` lists the steps when the run ends. A run launch also keeps the screen on and shows the app above the lock screen, so a run can drive a locked test phone; a launch from the launcher does neither.

## Check the Kotlin host against the model's own host

The model repository's `host/` folder is the model's Python host on LiteRT. `scripts/make_fixtures.py` runs it on the app's examples and on test requests of the model repository (a voicemail, a doorbell, a 44.1 kHz stereo file, a text state with a clip and reads, two clips in one state, spoken questions over a text and over a clip, 18 questions over a clip in two sequences, a JSON state with a 24 s clip), a photo with a clip, and two clips at the 30.015 s limit, one sample apart, and records every clip's features and every graph output it read. The JVM tests give the Kotlin host the same WAV files and graph outputs: `AudioTest` checks the WAV reader against the model's reader and the refusals, and `HostParityTest` checks that the Kotlin host computes the same features, builds the same token sequences, audio rows and pixel values and gives the same answers:

```bash
python -B scripts/make_fixtures.py --repo /path/to/d1-Omni-2B-LiteRT --out /tmp/d1-omni-fixtures   # needs the per-length files
./gradlew :app:testDebugUnitTest -Pd1.model=/path/to/d1-Omni-2B-LiteRT -Pd1.fixtures=/tmp/d1-omni-fixtures
```

## Measured

- **The Kotlin host against the model's Python host** (Apple M4 Max; the fixtures from `make_fixtures.py` on the GPU through Metal at FP32; `AudioTest` and `HostParityTest`): on 15 requests with 24 clips — 15 clips in states and 9 spoken questions, among them a 44.1 kHz stereo file and the two clips at the 30.015 s limit — the 15 sequences' token ids, state lengths, question tags and audio rows (their positions, their counts and the order of the audio calls) are identical, and so are the pixel values of the photo. The features differ by at most 2.6e-4 (the 44.1 kHz file, after resampling; at most 2.4e-7 for the 16 kHz clips). The answers agree within 1.4e-6 on the options' probabilities (mean 1.6e-7 over 107 options), within 1.9e-7 on P(found), and exactly on every span, value and box edge; all 131 numbers the screen shows with two decimals are the same. The clip of 3,001 frames is refused, as the Python host refuses it, before any graph call. The WAV reader takes and refuses the same 10 test files as the model's reader, with the same samples.
- **The graphs on the phones** (the model repository's device checks on 2026-11-02, LiteRT 2.2.0, on inputs that the Python host wrote on a Mac; one graph call, from the input writes to the read-back of the output): Galaxy S26, the decision graph on the NPU in a test runner on LiteRT's C API, outside an app, and the audio encoder on the GPU at FP32 in a test app — one question 67.0 ms; a 10 s clip and one question 131.6 ms (audio encoder 25.9 ms + decision graph 105.7 ms); loading the 128-token NPU file 1.0 to 2.1 s. Pixel 11 Pro, the GPU at FP32 in a test app — one question 733.6 ms; a 10 s clip and one question 1,111.9 ms (85.1 + 1,026.8 ms); the eight-signature file compiles all eight signatures in 26.1 s. Against the provider's float32 code, the yes / no, choice and score probabilities of the three NPU files (float16) stayed within 0.0067 on the Galaxy S26, and those of the Pixel 11 Pro's GPU within 4.5e-5.

Not measured in this README: the app's own times (the status line gives them for each call) and other phones.

## Files

| Path | Role |
|---|---|
| `app/src/main/java/com/d1_omni_2b/MainActivity.kt`, `MainViewModel.kt` | Compose host (the system photo and file pickers, the microphone permission, `--es run`); the screen's state, the examples, the recordings, the editor's questions, the decide call on a worker thread, the run files |
| `app/src/main/java/com/d1_omni_2b/Screen.kt` | The examples, the state with its highlights, the recordings, the photo with its boxes, the question cards (typed or asked aloud) and their editor, the answers, the Decide line |
| `app/src/main/java/com/d1_omni_2b/Audio.kt` | The WAV reader and writer, the resampler, the log-mel features (the real FFT, the mel filters, SLEEF's float32 functions) |
| `app/src/main/java/com/d1_omni_2b/Recorder.kt` | Recording from the microphone at 16 kHz mono (`AudioRecord`) and playing a clip back (`AudioTrack`) |
| `app/src/main/java/com/d1_omni_2b/Request.kt` | The editor's request: question drafts, clips, the checks before Decide, the list of model files |
| `app/src/main/java/com/d1_omni_2b/Tokenizer.kt` | Byte-level BPE read from `tokenizer.json`, with offsets |
| `app/src/main/java/com/d1_omni_2b/Prompt.kt` | The model's prompt: questions, codes, slots, the state block, grids, a spoken question's place |
| `app/src/main/java/com/d1_omni_2b/Host.kt` | Sequences of up to 16 questions, the audio markup and rows, embedding rows, the slots' readouts and pointer reads |
| `app/src/main/java/com/d1_omni_2b/Answers.kt`, `Parsing.kt` | The answers in the model's API shape (calibration, boxes, JSON paths, spans); the parser of numbers and dates |
| `app/src/main/java/com/d1_omni_2b/Pictures.kt` | smart_resize, the uint8 bicubic resize, patches, tower canvases |
| `app/src/main/java/com/d1_omni_2b/Engine.kt` | The accelerators (the NPU check), the decision graphs on the NPU or the GPU, the tower and the audio graphs on `CompiledModel`; the MemAvailable checks |
| `app/src/main/java/com/d1_omni_2b/Json.kt` | JSON with Python's key order and number semantics |
| `app/src/main/assets/` | `examples.json` (the examples), `beeps.wav` (the source card's recording), `q_beeps.wav` and `q_speech.wav` (the spoken questions), `shop_voicemail.wav` and `shop_voicemail_radio.wav` (the voicemails) |
| `app/src/test/java/com/d1_omni_2b/AudioTest.kt`, `HostParityTest.kt` | The WAV reader and the refusals; the Kotlin host against the model's Python host on recorded features and graph outputs |
| `scripts/push_model.sh`, `scripts/make_fixtures.py`, `scripts/make_beeps.py` | Copies the model files to the phone; records the Python host's fixtures; writes `beeps.wav` |
| `LICENSE`, `NOTICE`, `licenses/` | LFM Open License v1.0, attribution and changes, retained licenses |

## License

The model and the code ported from the model's Python code are under the LFM Open License v1.0 (`LICENSE`, unchanged), which limits commercial use by organizations with annual revenue of US$10 million or more; this sample is distributed under the same license, and `NOTICE` lists what it ported and changed. The tokenizer port follows Hugging Face `tokenizers` (Apache License 2.0); the audio front end follows transformers' mel filters (Apache License 2.0), pocketfft (BSD-3-Clause), SLEEF (Boost Software License 1.0) and PyTorch (BSD-3-Clause); the picture resize follows PyTorch's kernel (BSD-3-Clause); all in `licenses/`. The LiteRT conversion is not affiliated with Liquid AI.

The sample's media and texts: `beeps.wav` is the source model card's recording, made by its own code (`scripts/make_beeps.py`). `q_beeps.wav` and `q_speech.wav` are speech made with Kokoro-82M (Apache License 2.0, `licenses/Kokoro-82M-APACHE-2.0.txt`), copied from the model repository's test files. `shop_voicemail.wav` and `shop_voicemail_radio.wav` are voicemails made for this sample: speech made with Kokoro-82M from texts written for it, the second mixed at the same loudness with 8.6 s of the song "Frozen Whispers" by szegvari ([Freesound 732895](https://freesound.org/people/szegvari/sounds/732895/), Creative Commons 0 1.0; its page says it was generated with Suno AI) and passed as a whole through a telephone band. The shop, its record and the callers are invented. The examples' states and questions were written for this sample.
