# clef-flash home: one wish sets the whole house, on a Mac

Write a wish in a few words, such as "Movie night", and press Decide. The model sets every device of the house at once: the living room light, the curtains, the TV, the air conditioner and the front door, each as one of its options with a probability for each option. The house on screen changes to match. It is one model call that writes no text; on an Apple M4 Max it took 704.9 to 723.4 ms per wish (the median of five presses each, the eight bundled wishes).

The model is Cloudflare's clef-flash, converted to LiteRT ([litert-community/clef-flash-LiteRT](https://huggingface.co/litert-community/clef-flash-LiteRT)), on the Mac's GPU through Metal at float32 precision. The house is a JSON state you can edit: the devices as they are now, the weather outside, the temperature inside, the time, and the wish as its `"request"`. The questions are yours too: one per device, each with its options. This app is the clef-flash sample's editor (`../app/`) with a wish line and a drawing of the house added.

## Run it

```bash
cd clef-flash/home
zsh scripts/fetch_hub.sh                 # 23 files, 18.2 GB, into ../hub/, checked against the repository's SHA256SUMS (resumable; about 16 minutes at 20 MB/s)
python3.14 -m venv venv-demo
venv-demo/bin/pip install -r ../hub/host/requirements-host.txt -r requirements-app.txt
venv-demo/bin/python app/home_demo.py
```

`fetch_hub.sh` downloads only what this app needs (the 512-token row graph, the tables, the joint head, the tokenizer and the host, no picture tower) into `../hub/`, the folder the clef-flash sample's own `scripts/fetch_hub.sh` fills, at the same pinned revision: a file either script has downloaded is kept. A second run only checks the files.

At start the app compiles the 512-token row graph for the GPU and warms it up with one request; the status line counts the seconds. On an Apple M4 Max the compile took 71.3 s in one run and 56.3 s in another, and the second run read READY 58.1 s after the start. Then Decide works.

## The app

- The house: a drawing of the state. The living room light (off, dimmed, full brightness), the curtains (open, closed), the TV (on, off), the air conditioner (off, cooling, heating), the front door's lock, the sky of `"outside"` (night, dusk or day), and `"inside"` and `"time"` as written. After Decide it shows the settings the model picked, and the devices that changed are named in white.
- WISH: the wish line (it writes the state's `"request"`), the eight bundled wishes, and after Decide one row per question: the device, the setting picked and its probability. Under Decide, the request time the app measured, and the request's length in tokens and the graph it ran on.
- HOUSE & QUESTIONS: the state as JSON and one card per question, as in the clef-flash sample's editor: a name, a type (yes / no, choice or score), the question, and the options, one per line (`name: description` for a choice; nothing, or `Yes: …` and `No: …`, for a yes / no). After Decide each card shows the probability of every option. The model reads all of it.
- Any edit puts the house back as the state describes it and clears the answers; press Decide again for the edited request.
- A request longer than the row graph (512 tokens) is refused with a red line; nothing is cut. Each bundled wish is 485 to 490 tokens, so a wish of a few more words fits; with `fetch_hub.sh --length 1024` and `home_demo.py --length 1024` every request runs on the 1,024-token graph instead (about twice the time per request, in the model card).

The picture knows the five devices by their question names (`light`, `curtains`, `tv`, `aircon`, `door`) and reads a setting from the words of the state (`"light": "bright"`) and of the option the model picked (its name, or its description: "Closed" for yes on "The curtains should be closed."). A question with another name gets a row and a card but no place in the drawing; a setting the drawing cannot read shows a red "?". `app/house.py` has the words.

The request time is `time.perf_counter()` right before and after `clef.decide(request)`: the tokenizer, the row graph's inputs, the row graph call with its input writes and output read, the joint head and the response. A compile is never inside it.

## The bundled wishes

Each wish is the same five questions on its own house. Before the model answered any of them, the setting each device was expected to get was written down (`app/wishes.json`, `"expected"`). The model's answers on Metal, the five presses of each wish alike to the last bit:

| Wish | Light | Curtains | TV | Air conditioner | Front door |
|---|---|---|---|---|---|
| Movie night | Dimmed 0.928 | Closed 0.757 | On 0.943 | Off 0.780 | Locked 0.961 |
| I'm leaving for work | Off 0.540 | Open 0.907 (expected: Closed) | Off 0.818 | Off 0.868 | Locked 0.572 |
| Good night | Dimmed 0.710 (expected: Off) | Closed 0.872 | On 0.681 (expected: Off) | Off 0.796 | Locked 0.905 |
| It's freezing in here | Dimmed 0.875 (expected: Full brightness) | Open 0.782 (expected: Closed) | On 0.971 | Heat the room 0.965 | Locked 0.961 |
| Our guests are at the door | Full brightness 0.843 | Open 0.922 (expected: Closed) | Off 0.779 | Off 0.874 | Unlocked 0.903 |
| I want to read a book | Dimmed 0.835 (expected: Full brightness) | Closed 0.844 | Off 0.690 | Off 0.964 | Locked 0.982 |
| It's too hot in here | Off 0.852 | Open 0.872 (expected: Closed) | Off 0.963 | Cool the room 0.987 | Locked 0.952 |
| Let the sunshine in | Off 0.912 | Open 0.970 | Off 0.989 | Off 0.971 | Locked 0.964 |

32 of the 40 settings are the expected ones. The other eight: the curtains opened in four houses where they were closed and expected to stay closed, the light dimmed three times where off or full brightness was expected, and the TV stayed on at Good night. Every device of Movie night got its expected setting. This app does not measure accuracy beyond these eight wishes. The provider reports its own run of a home appliance simulator in the [model card](https://huggingface.co/Cloudflare/clef-flash) (Decision Index: "Home appliance simulator (case exact accuracy)", 97.7 for clef-flash).

Measured in this app on an Apple M4 Max (128 GB, macOS 27.0), ai-edge-litert 2.2.0, Metal at float32 precision, on the 512-token row graph: five presses of Decide per wish through the page (`scripts/press_wishes.py`), the median and the range.

| Wish | Tokens | Request time, median of 5 (range) |
|---|---:|---:|
| Movie night | 485 | 704.9 ms (702.9–1,171.6) |
| I'm leaving for work | 490 | 717.5 ms (709.1–730.5) |
| Good night | 486 | 710.2 ms (706.4–759.6) |
| It's freezing in here | 487 | 710.6 ms (706.9–767.3) |
| Our guests are at the door | 490 | 723.4 ms (707.3–758.4) |
| I want to read a book | 489 | 714.0 ms (708.4–758.4) |
| It's too hot in here | 490 | 714.2 ms (708.5–791.9) |
| Let the sunshine in | 488 | 714.5 ms (704.5–783.3) |
| The recorded scene's request: Movie night, typed into the wish line | 485 | 714.7 ms (706.7–733.3) |

The 1,171.6 ms of the Movie night row is the first Decide after the app read READY; the slowest press of each other wish is its first. In the recorded take below, the Movie night request took 791.9 ms (the 792 ms on its screen): it was the first request after the warm-up, with the window recorder running.

`--accel cpu` runs the same graph on the CPU (XNNPACK, 8 threads): in `scripts/cpu_reference.py`, which runs each wish twice through the app's model code, the second call of each wish took 2,394.7 to 2,411.3 ms (the first call of the first wish, with the compile, 22.0 s).

## Memory

On Metal at float32 the app process's peak memory footprint was 68.2 GB (`/usr/bin/time -l`), and it held 38.0 GB once the row graph was compiled, 37.7 GB after the warm-up. On the CPU, `scripts/cpu_reference.py` peaked at 30.0 GB (`/usr/bin/time -l`). This was measured on a 128 GB Mac; Macs with less memory were not tried.

## Check the answers

The wishes are not records of the model repository's test fixtures, so the provider's own answers do not cover them. The line for this app's answers is the CPU run of the same files and requests:

```bash
venv-demo/bin/python -B scripts/cpu_reference.py                    # the eight wishes on the CPU (run once: it refuses to overwrite out/cpu_ref_wishes.json)
venv-demo/bin/python app/home_demo.py --tag e1 &                    # the app, scripted input accepted (the window appears at READY, top right, and ignores the mouse)
venv-demo/bin/python -B scripts/press_wishes.py e1                  # every wish, five presses each
venv-demo/bin/python -B scripts/check_wishes.py out/e1_manual.run.json
```

`press_wishes.py` taps each wish's button and Decide, as you would, five times each; then the recorded scene's request the way the scene makes it (the Movie night house with the wish typed into the wish line); then it pastes a wish too long for the row graph, which the app refuses. `check_wishes.py` passes when, on every question whose two most likely options are more than 0.02 apart in the CPU run, the most likely option is the CPU run's, every probability is within 0.02 of the CPU run's, and at least 80 % of the wishes' devices got the expected setting. On the Apple M4 Max all nine requests passed (the eight wishes and the typed one): every question gave the CPU run's most likely option (no near tie), the probabilities differed from the CPU run's by at most 6.0e-6, the five presses of each request gave the same probabilities to the last bit, 32 of the 40 settings were the expected ones, and the 541-token wish was refused with `RequestTooLong` and a red line.

`scripts/check_editor.py` checks the editor's rules on the bundled wishes and what the drawing reads from them, without the model.

## Record a take

The app can also play one scene by itself, for a recording. It types into the wish line and presses Decide through the same page and the same code a click reaches.

- Scene M: the house of the Movie night wish with an empty wish line; "Movie night" is typed; Decide; the answers and the house after them.

```bash
venv-demo/bin/python -B scripts/cpu_reference.py   # the line for the take (once; see above)
zsh scripts/take.sh t1                             # the app with --scenes M, the scene recorded, then the checks
```

`take.sh` records the app's window and nothing else (`scripts/record_window.swift`, ScreenCaptureKit's window filter, 30 frames per second, no sound) and needs the Screen Recording permission for your terminal and the Mac's screen unlocked: while it is locked the page paints nothing and the run stops with a `did not paint` / `no mark` error. Each take needs a new tag. A take counts when its checks pass:

- `check_run.py`: scene M's probabilities against `cpu_reference.py` (the same most likely option on each question, every probability within 0.02) and each device's expected setting; every string on screen is the rounding of the run's values, the cards' bars are drawn at their probabilities when the cards are on screen (the scene stays on the Wish tab, where none is drawn), and the house shows the answers' settings; the request time is at most twice the model card's median for a text request on the 512-token row graph (673.3 ms); the page's content equals the request; no graph compiled after READY.
- `check_frames.py`: the recording against the run. Only the app's window is in it, the wish appears before Decide, the house does not change before the answers, and the answers appear after Decide no sooner than the request time allows.

On the Apple M4 Max the recorded take ran 10.2 s (305 frames at 30 frames per second, 1920 × 1080, no sound): the wish was typed in by 2.2 s, Decide was pressed at 3.3 s, and the answers appeared 833 ms later, after a request of 791.9 ms. The settings were Dimmed (0.928), Closed (0.757), On (0.943), Off (0.780) and Locked (0.961), within 1.6e-6 of the CPU run.

On a Mac that runs other GPU work, three environment variables let the scripts wait for a quiet machine and time the scene inside a lock: `HOME_WAIT_CMD` and `HOME_HOLD_CMD` (command prefixes) and `HOME_LOCK_FILE`. Unset, the scripts wait for nothing.

`--tag <name>` runs the app for scripts: the window appears when the model is ready, ignores the mouse, and takes its input from `scripts/drive_editor.py` (the same handlers as your keys and clicks). Every Decide is written to `out/<name>_manual.run.json`.

## Requirements

- A Mac with Apple silicon. Tested on an Apple M4 Max with 128 GB and macOS 27.0.
- Python 3.14 (tested with 3.14.6), with the packages of `../hub/host/requirements-host.txt` and `requirements-app.txt`.
- Disk: 18.2 GB for the download (13.9 GB of it is the row graph).
- Memory: 68.2 GB peak on Metal, 30.0 GB on the CPU (see Memory).
- For recording only: the Screen Recording permission, the Xcode command line tools (`xcrun swiftc`) and ffmpeg.

## Files

- `app/home_demo.py`: the app. The window, the page's Python side, Decide, the startup compile, autoplay and scripted runs.
- `app/editor.py`: how the page's content becomes a request, and back (the clef-flash sample's rules).
- `app/house.py`: what the drawing reads from the state and from the answers.
- `app/wishes.json`: the bundled wishes, their questions, the settings each device was expected to get, and the recorded scene's wish.
- `app/ui/index.html`, `style.css`, `app.js`: the screen and the drawing of the house.
- `scripts/fetch_hub.sh`: the model files this app needs, at a pinned revision, checked with SHA-256.
- `scripts/check_editor.py`, `scripts/drive_editor.py`: the editor's rules and the drawing checked without the model; scripted input.
- `scripts/press_wishes.py`, `scripts/cpu_reference.py`, `scripts/check_wishes.py`: every wish pressed through the page, the CPU run of the same requests, and the answers against both.
- `scripts/take.sh`, `scripts/take_inner.sh`, `scripts/record_window.swift`: a recorded take.
- `scripts/check_run.py`, `scripts/check_frames.py`: a take's checks.

## Licenses

The wishes, the houses and the questions were written for this app. The drawing of the house is made of SVG shapes in `app/ui/index.html`; this directory holds no picture file.

clef-flash, by Cloudflare, is under the Apache License 2.0, and so are its LiteRT files; see the [model card](https://huggingface.co/litert-community/clef-flash-LiteRT). The LiteRT files are a community conversion and are not affiliated with Cloudflare.

The code in this directory is licensed under the Apache License 2.0 (`../LICENSE`), with attribution in `../NOTICE`.
