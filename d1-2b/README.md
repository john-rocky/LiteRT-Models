# d1-2B Decide — read values out of a message, a record or a photo with your own questions, on the phone

Paste a message or a JSON record, or add a photo, write the questions you want answered, and tap **Decide**: d1-2B answers every question in one forward pass on the phone's GPU. A read returns the value typed (an ISO date, a number with its unit, a text), the span of the state it came from — highlighted in the state in the question's color — and the probability that the value is there at all; a value the state does not hold comes back as absent instead of a guess. Next to the reads, a question can ask for yes or no, one or several of your options, a level on your scale, a number in a range, or boxes in the photo. The model generates no tokens: each answer is read from the model's output at the question's slots. Everything runs on the phone, offline: a Kotlin tokenizer and prompt, the model's answer readers ported to Kotlin, LiteRT graphs on the GPU.

**Appointment text** (the first example) is a dentist's text that moves a cleaning from one date to another, with five questions: when the appointment is now (a date), at what time (a text), how much is left to pay (a number), the clinic's street address (not in the text) and what to do with the calendar entry (a choice: keep / move / delete). Change the first question to *When was the appointment before?* and tap Decide again: the answer moves to the other date in the same text, and so does its highlight.

## Try it

- **Examples**: Appointment text, Delivery notice, Water bill, Order record (JSON, 18 questions), Photo. Each fills the state and the questions; replace any of it with your own.
- **State**: paste a message or a note, or switch to **JSON** and paste a record (a read then also gives the path of the value). **Add a photo** opens the system photo picker; the photo is shrunk to 384 pixels on its long side before the model sees it, so it fits the smallest picture graph. A state can be text, a photo, or both.
- **Questions**: each card shows its name, its type and the question. **Edit** opens the name, the type — Yes / no, Choice, Several choices, Score, Text, Number, Number in a range, Date, Box, Boxes — the options of a choice (a name and what it means), the levels of a score (2 to 10, from the lowest), the range and step of a number in a range, a unit, and the most boxes to read (1 to 16). **Add question** adds a card; **Remove this question** removes one.
- **Decide**: every card gets its answer underneath — the value and P(found) for a read (*Not in the text* when it is absent), the option and its probability for a choice, P(yes) for a yes / no, the level for a score, the median and interval for a number in a range, boxes drawn on the photo in the card's color. The line under the button gives the time of the call on the phone, the tokens it read and the graph length it ran on (`Answered in … ms on this phone · 253 tokens · L256 graph` for the Appointment text); the first call on a new length adds the graph's compile time.

Up to 16 questions share one sequence with the state; more questions take another sequence. Nothing is cached between calls: every Decide sends the state and the questions as they are on the screen.

## Model and requirements

- Model: [litert-community/d1-2B-LiteRT](https://huggingface.co/litert-community/d1-2B-LiteRT) (decision graphs per sequence length, picture towers per canvas, the embedding table, and the model's own `tokenizer.json`, `config.json` and `calibration.json` under `provider/`).
- Upstream: [LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B) revision `10e7671d`, LFM Open License v1.0.
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile SDK 36, target SDK 35.
- Runtime: LiteRT 2.3.0 `CompiledModel` (`litert` and `litert-gpu`) on the GPU at FP32 with constant tensor sharing (one copy of the weights on the GPU; without it, a test app on a Pixel 11 Pro peaked at 10.7 to 10.9 GiB compiling a per-length file, over its 8 GiB memory limit, and was killed). Material Compose.
- Memory: a phone with 16 GB runs the eight-signature file (one 3.0 GB file with the seven length graphs and the graph that runs eight sequences of up to 128 tokens in one call, over one copy of the weights; tested on a Pixel 11 Pro). On a phone with 12 GB that file does not fit: copy the per-length files you need instead (tested on a Galaxy S26); the app loads one at a time, and switching lengths compiles the next one. On a 12 GB phone the app refuses states longer than 4,096 tokens, with the reason on screen. Before it compiles anything the app reads MemAvailable: it needs 9.3 GB free for the eight-signature file and 6.3 GB for a per-length file (what the load took on those two phones, plus 0.8 GB), and 0.8 GB before the picture tower's first compile; with less it shows how much is free and loads nothing.
- No network permission: the app reads the model from its own folder on the phone.

## Download, build and install

Use JDK 17, the Android SDK (platform 36), Android platform-tools (`adb`) and the Hugging Face CLI (`hf`).

```bash
pip install -U huggingface_hub
hf download litert-community/d1-2B-LiteRT \
  provider/tokenizer.json provider/config.json provider/calibration.json tables/embed_table.npy \
  d1-2b_decide_multi8_fp16fc.tflite d1-2b_tower_G24x24_fp16fc.tflite \
  --local-dir "$HOME/Downloads/d1-2B-LiteRT"
./gradlew :app:installDebug
# Optional when more than one phone is connected:
# export ANDROID_SERIAL=your-device-serial
adb shell am start -n com.d1_2b/.MainActivity
scripts/push_model.sh "$HOME/Downloads/d1-2B-LiteRT"
```

That is the set for a 16 GB phone. For a 12 GB phone, download the per-length files instead of `d1-2b_decide_multi8_fp16fc.tflite`, for example `d1-2b_decide_L256_fp16fc.tflite` and `d1-2b_decide_L512_fp16fc.tflite` (the Appointment text needs L256; the Delivery notice and the Water bill L512; the Order record L512 and L1024; the Photo L2048).

Open the app once before the push: it creates its folder and says which files are missing. `scripts/push_model.sh` copies the provider files, the table and every `d1-2b_*.tflite` of the folder to `/sdcard/Android/data/com.d1_2b/files/model/` and checks each file's size on the phone. Then close the app and open it again: it compiles the decision graph on the GPU (the status line says when it is ready; about 26 s for the eight-signature file on a Pixel 11 Pro).

## External files

| File | Bytes | sha256 | Purpose |
|---|---:|---|---|
| `provider/tokenizer.json` | 17,908,258 | `a39eed58…b75c7e7` | The model's tokenizer, unchanged |
| `provider/config.json` | 2,903 | `ffdc4c99…1d6f332` | Special tokens, codes, the set size |
| `provider/calibration.json` | 4,924 | `e5344644…d884d69` | Temperatures and the calibration tables of the answers |
| `tables/embed_table.npy` | 524,288,128 | `e8d0c746…63fc07a` | The embedding rows, bfloat16 bits `[128000, 2048]` (looked up on the host) |
| `d1-2b_decide_multi8_fp16fc.tflite` | 2,998,830,272 | `afeea7f2…a5845a1` | The seven lengths and the eight-sequence graph as signatures of one file (16 GB phones) |
| `d1-2b_decide_L128_fp16fc.tflite` | 2,934,923,056 | `51cb7b9a…727cd52` | Up to 128 tokens, one length per file (12 GB phones) |
| `d1-2b_decide_L256_fp16fc.tflite` | 2,935,382,112 | `175147ab…20a0a84` | Up to 256 tokens |
| `d1-2b_decide_L512_fp16fc.tflite` | 2,938,800,160 | `62dccd24…66d293c` | Up to 512 tokens |
| `d1-2b_decide_L1024_fp16fc.tflite` | 2,940,897,312 | `1e098e03…1cef1c1` | Up to 1,024 tokens |
| `d1-2b_decide_L2048_fp16fc.tflite` | 2,945,091,616 | `d5a3a5a1…22af04d` | Up to 2,048 tokens |
| `d1-2b_decide_L4096_qb1024_fp16fc.tflite` | 2,953,520,352 | `4827e943…53a9d40` | Up to 4,096 tokens (the longest the app runs on a 12 GB phone) |
| `d1-2b_decide_L6144_qb1024_fp16fc.tflite` | 2,961,934,864 | `57439634…062a10d` | Up to 6,144 tokens (16 GB phones only) |
| `d1-2b_tower_G24x24_fp16fc.tflite` | 377,919,584 | `672bd515…82ecc18` | The picture tower for pictures up to 24 × 24 patches (384 px) |

The decision graphs store the fully connected weights in float16 and compute in float32; the app compiles them at FP32 (the GPU's default float16 activations move the answers outside the checked tolerance).

## How a request runs

```text
State (text, or JSON written as the model's prompt writes it) + photo + typed questions
  → prompt: the state, the picture's tokens, then each question's header, instructions and answer slots
  → byte-level BPE (Kotlin, read from tokenizer.json); up to 16 questions after one state per sequence
  → photo: shrunk to 384 px → smart_resize to multiples of 32 → PyTorch's uint8 bicubic resize
    → 16 × 16 patches on the smallest tower canvas → tower (GPU) → the picture's token rows
  → embedding rows from tables/embed_table.npy, the picture's rows at its tokens
  → decide on the smallest graph length that holds the sequence (GPU, FP32) → hidden states
  → options: the hidden state at each slot × the options' embedding rows → temperature → softmax
    → calibration → yes / no, choice, several choices, score, number in a range
  → reads: the pointer over the state's positions → the best span → its value parsed as the model's
    code parses it (dates, numbers with units and scales) → text, number or date and P(found)
  → boxes: the four edges' grids → the box and its confidence
```

The Kotlin host is a port of the model's Python code (`prompt.py`, `api.py`, `box.py`, `calibration.py`, `grid.py`, `json_spans.py`, `parsing.py`, `pointer.py`, and the packing of `attention.py` and `runner.py`) and of the model repository's host (`host/d1_host.py`, `host/d1_pictures.py`); the photo is read without color management, as the model's image code reads a file.

## Run files (for tests and recordings)

The app runs a list of steps from a JSON file through the same functions the screen uses. Push a file to the app's `files/runs/` folder and start the app with it:

```bash
adb push run.json /sdcard/Android/data/com.d1_2b/files/runs/run.json
adb shell am start -n com.d1_2b/.MainActivity --es run run.json
```

```json
{"steps": [
  {"example": "appointment"},
  {"decide": true, "out": "out/appointment.json"},
  {"edit": {"question": "when", "instructions": "When was the appointment before?"}},
  {"decide": true, "out": "out/appointment_edit.json"}
]}
```

Steps: `{"example": id}`, `{"state": text}`, `{"edit": {"question": name, "instructions": text}}`, `{"decide": true, "out": path}`, `{"wait_ms": n}`. Each `out` file holds the device, the graph form, the request, the answers in the model's API shape, the token ids of every sequence, and the time of each part of the call (`prep_ms`, `tower_ms`, `decide_ms`, `readout_ms`, `total_ms`, the compile time and the graph length); `run.json.done` lists the steps when the run ends. A run launch also keeps the screen on and shows the app above the lock screen, so a run can drive a locked test phone; a launch from the launcher does neither.

## Check the Kotlin host against the model's own host

The model repository's `host/` folder is the model's Python host on LiteRT. `scripts/make_fixtures.py` runs it on the app's examples (plus a state that holds special-token text and a photo that needs resampling) and records every graph output it read; the JVM test `HostParityTest` gives the Kotlin host the same graph outputs and checks that it builds the same token sequences and pixel values and gives the same answers:

```bash
python -B scripts/make_fixtures.py --repo /path/to/d1-2B-LiteRT --out /tmp/d1-fixtures   # needs the per-length files
./gradlew :app:testDebugUnitTest -Pd1.model=/path/to/d1-2B-LiteRT -Pd1.fixtures=/tmp/d1-fixtures
```

## Measured

- **The Kotlin host against the model's Python host** (Apple M4 Max; the fixtures from `make_fixtures.py` on the GPU through Metal at FP32; `HostParityTest`): on 8 requests — the five examples, the appointment with its first question changed, a state with special-token text and a photo resized to 384 × 216 — the 9 sequences' token ids, state lengths, question tags and slot positions are identical, the pixel values of both pictures are identical, and the answers agree within 1.5e-6 on the options' probabilities (mean 9.8e-8 over 28 options), 5.0e-7 on P(found), and exactly on every span, value and box edge.
- **The decision graph on the phones** (the conversion's device runs on the shipped files and a seven-signature file of the same graphs, a harness app on LiteRT 2.2.0, GPU at FP32 with constant tensor sharing; one graph call, from the input writes to the read-back of the output): Galaxy S26 (12 GB), the per-length files — 366.4 ms at L128, 645.7 ms at L256 (the tower at 24 × 24 patches 112.2 ms, timed without sharing), compile 20.9 s at L256. Pixel 11 Pro (16 GB) — 736.8 ms at L128 and 1,047.0 ms at L256 on the seven-signature file, which gives the same outputs as the eight-signature file; the tower 274.4 ms, timed without sharing; the eight-signature file compiled all eight signatures in 25.2 s.

Not measured in this README: the app's own times (the status line gives them for each call) and other phones. On the Galaxy S26 the app ran only the 256-token file.

## Files

| Path | Role |
|---|---|
| `app/src/main/java/com/d1_2b/MainActivity.kt`, `MainViewModel.kt` | Compose host (the system photo picker, `--es run`); the screen's state, the examples, the editor's questions, the decide call on a worker thread, the run files |
| `app/src/main/java/com/d1_2b/Screen.kt` | The examples, the state with its highlights, the photo with its boxes, the question cards and their editor, the answers, the Decide line |
| `app/src/main/java/com/d1_2b/Tokenizer.kt` | Byte-level BPE read from `tokenizer.json`, with offsets |
| `app/src/main/java/com/d1_2b/Prompt.kt` | The model's prompt: questions, codes, slots, the state block, grids |
| `app/src/main/java/com/d1_2b/Host.kt` | Sequences of up to 16 questions, embedding rows, the slots' readouts and pointer reads |
| `app/src/main/java/com/d1_2b/Answers.kt`, `Parsing.kt` | The answers in the model's API shape (calibration, boxes, JSON paths, spans); the parser of numbers and dates |
| `app/src/main/java/com/d1_2b/Pictures.kt` | smart_resize, the uint8 bicubic resize, patches, tower canvases |
| `app/src/main/java/com/d1_2b/Engine.kt` | The decision graphs and the tower on `CompiledModel` (GPU, FP32, constant tensor sharing); the MemAvailable checks before the decision graph's and the tower's first compile |
| `app/src/main/java/com/d1_2b/Json.kt` | JSON with Python's key order and number semantics |
| `app/src/main/assets/` | `examples.json` (the five examples), `bike.png` (the Photo example) |
| `app/src/test/java/com/d1_2b/HostParityTest.kt` | The Kotlin host against the model's Python host on recorded graph outputs |
| `scripts/push_model.sh`, `scripts/make_fixtures.py` | Copies the model files to the phone; records the Python host's fixtures |
| `LICENSE`, `NOTICE`, `licenses/` | LFM Open License v1.0, attribution and changes, retained licenses |

## License

The model and the code ported from the model's Python code are under the LFM Open License v1.0 (`LICENSE`, unchanged), which limits commercial use by organizations with annual revenue of US$10 million or more; this sample is distributed under the same license, and `NOTICE` lists what it ported and changed. The tokenizer port follows Hugging Face `tokenizers` (Apache License 2.0) and the picture resize PyTorch's kernel (BSD-3-Clause), both in `licenses/`. The LiteRT conversion is not affiliated with Liquid AI.

The sample's media and texts: `bike.png` is a CC0 1.0 photograph from Wikimedia Commons (`File:Bike_on_a_snowy_street_in_Quebec_City_at_night.jpg`), resized to 384 × 256. The examples' texts, the order record and the questions were written for this sample; their people, businesses, addresses and numbers are invented.
