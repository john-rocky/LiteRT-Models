# clef-flash Decide: questions about a text or a photo, answered on a Mac

Paste a support ticket, a note or a JSON value, add a photo if you like, and write the questions you want answered, each with its options. Press Decide. Every question comes back at once as one of your options, with a probability for each option. The model is Cloudflare's clef-flash, converted to LiteRT, and it runs on the Mac's GPU. It never writes text.

The app opens with a support ticket and its three questions. Four examples are bundled: the ticket, a list of meeting rooms as JSON, a photo of a cup of coffee, and a drawn clock.

A second app, [home/](home/), gives the same model one job: a house written as JSON (its devices, the weather, the time) and a wish in a few words, such as "Movie night". Decide sets the living room light, the curtains, the TV, the air conditioner and the front door at once, and a drawing of the house changes to match. It runs every request on the 512-token row graph (704.9 to 723.4 ms per bundled wish on an Apple M4 Max, the median of five presses each), and its own `scripts/fetch_hub.sh` downloads into this `hub/` what it needs: the 512-token row graph and the tables, head, tokenizer and host that this app's download also holds.

## Run it

```bash
cd clef-flash
zsh scripts/fetch_hub.sh                 # 56 files, 20.0 GB, into hub/, checked against the repository's SHA256SUMS (resumable; about 17 minutes at 20 MB/s)
python3.14 -m venv venv-demo
venv-demo/bin/pip install -r hub/host/requirements-host.txt -r requirements-app.txt
venv-demo/bin/python app/clef_demo.py    # the editor
```

At start the app compiles three graphs for the GPU: the 1,024-token row graph and the two picture towers. On an Apple M4 Max the three compiles took 89.8 s in one run and 70.2 s in another; with one warm-up request per graph, the second run read READY 75.7 s after the start. The status line counts the seconds. Then it reads READY and Decide works.

## The editor

- STATE: plain text, or a JSON object or array (the label next to STATE says which). Leave it empty and add a photo, and the photo is the whole state.
- PHOTO: drop a JPEG or PNG on the window, or press Choose photo…. The app shows the photo as the host opens it, without the EXIF rotation. The model reads it squeezed to a square tile of 448 px (196 picture tokens, the default) or 256 px (64 picture tokens); the two buttons next to the photo's name choose the tile.
- QUESTIONS: one card per question. A card has a name, a type, the question, and the options, one per line:
  - choice: `name: description`, or just `name`; two or more;
  - score: one level per line, from the lowest; two or more;
  - yes / no: nothing, or both `Yes: what yes means` and `No: what no means`.

  The model reads all of it: the name, the question, and each option's name and description.
- Decide: the cards turn into the answers, all at once. Each shows the option the model picked, its probability, and a bar for every option. The footer shows the request time the app measured, and under it the request's length in tokens and the graph it ran on. Any edit clears the answers; press Decide again for the edited request.
- A request longer than the row graph (1,024 tokens) is refused with a red line; nothing is cut.

The request time is `time.perf_counter()` right before and after `clef.decide(request)`: the tokenizer, the photo's tile and patches, the tower call, the row graph's inputs, the row graph call with its input writes and output read, the joint head and the response. A compile is never inside it.

## Which graph runs

The model repository has row graphs for requests of up to 256, 512, 1,024, 2,048 and 4,096 tokens, and a picture tower for each tile. A row graph computes every one of its positions whatever the request's length, so on the 1,024-token graph a short request takes about as long as a long one. The app keeps one row graph compiled, the 1,024-token one, and every request runs on it: it holds any request of up to 1,024 tokens, the state, the photo and the questions together, and each bundled example is 405 to 634 tokens. To run on another graph, download it and name it: `zsh scripts/fetch_hub.sh --length 512` adds the 512-token row graph (13.9 GB), and `venv-demo/bin/python app/clef_demo.py --length 512` runs every request on it and refuses requests over 512 tokens (one call of that graph takes 654.4 ms on Metal, against 1,308.5 ms for the 1,024-token graph, in the model card).

Measured in this app on an Apple M4 Max (128 GB, macOS 27.0), ai-edge-litert 2.2.0, Metal at float32 precision, on the 1,024-token row graph: five presses of Decide per request through the editor (`scripts/press_examples.py`), the median and the range.

| Request | Tile | Tokens | Request time, median of 5 (range) |
|---|---|---:|---:|
| Ticket: a support ticket, 3 questions | — | 436 | 1,356.2 ms (1,347.3–1,420.8) |
| Rooms: a JSON list of rooms, 3 questions | — | 463 | 1,345.7 ms (1,336.9–1,361.4) |
| Coffee: a photo, 4 questions | 448 px | 634 | 1,481.0 ms (1,469.3–1,507.3) |
| Coffee: a photo, 4 questions | 256 px | 502 | 1,425.0 ms (1,376.8–1,441.0) |
| Clock: a drawing, 3 questions | 448 px | 537 | 1,485.7 ms (1,447.4–1,490.4) |
| Clock: a drawing, 3 questions | 256 px | 405 | 1,373.6 ms (1,360.1–1,432.4) |
| The recorded scene's request: the coffee photo, 2 questions | 448 px | 455 | 1,464.4 ms (1,456.1–1,486.3) |

In the recorded take below, the same 455-token request took 1,539.8 ms (the 1,540 ms of its footer): it was the first request after the warm-up, with the window recorder running.

`--accel cpu` runs the same graphs on the CPU (XNNPACK, 8 threads). `scripts/cpu_reference.py` runs the recorded scene's request that way through the app's model code: its second call took 6,212.2 ms (the first, 26.0 s, includes the two compiles).

## Memory

On Metal at float32 the app process's peak memory footprint was 68.5 GB (`/usr/bin/time -l`), and it held 34.1 GB once its three graphs were compiled, 34.8 GB after the warm-up. The model card's own runs, one row graph at a time, peaked at 67.5 to 69.2 GB during the compile. On the CPU, `scripts/cpu_reference.py` peaked at 32.3 GB (`/usr/bin/time -l`). This was measured on a 128 GB Mac; Macs with less memory were not tried.

## Check the answers

The four examples are records of the model repository's public test fixtures, and `hub/fixtures/reference_probs.json` holds the provider's own answers to them: its `systemone()` in float32 on the CPU. The app's answers can be checked against them:

```bash
venv-demo/bin/python app/clef_demo.py --tag e1 &                     # the editor, scripted input accepted (the window appears at READY, top right, and ignores the mouse)
venv-demo/bin/python -B scripts/press_examples.py e1                 # every example, both tiles for a picture
venv-demo/bin/python -B scripts/check_examples.py out/e1_manual.run.json
```

`press_examples.py` taps each example's button, the tile and Decide, as you would, five times each; then the recorded scene's request the same way (the coffee example with two of its four question cards removed); then it pastes a state longer than the row graph, which the app refuses. `check_examples.py` passes when, on every question whose two most likely options are more than 0.02 apart, the most likely option is the provider's, and every probability is within 0.02 of the provider's. On the Apple M4 Max all six requests passed: each of their 20 questions gave the provider's most likely option, and the probabilities differed from the provider's by at most 2.4e-5 (the coffee photo at the 448 px tile; 3.6e-7 for the ticket). The five presses of each request gave the same probabilities to the last bit, and the 2,982-token state was refused with `RequestTooLong` and a red line.

`scripts/check_editor.py` checks the editor's rules and that each example is its fixture record, without the model.

## Record a take

The app can also play one scene by itself, for a recording. It places the photo and presses Decide through the same page and the same code a click reaches.

- Scene P: the editor opens with the coffee example's state, two of its questions (which drink is in the cup, a choice; whether milk was added, yes or no) and no photo; the photo is dropped in; Decide; the two answers.

```bash
venv-demo/bin/python -B scripts/cpu_reference.py   # scene P's request on the CPU, from the same files: the line for the take (run once: it refuses to overwrite out/cpu_ref_take.json, which every later take reads)
zsh scripts/take.sh t1                             # the app with --scenes P, the scene recorded, then the checks
```

`take.sh` records the app's window and nothing else (`scripts/record_window.swift`, ScreenCaptureKit's window filter, 30 frames per second, no sound) and needs the Screen Recording permission for your terminal and the Mac's screen unlocked: while it is locked the page paints nothing and the run stops with a `did not paint` / `no mark` error. Each take needs a new tag. A take counts when its checks pass:

- `check_run.py`: scene P's probabilities against `cpu_reference.py` (the same most likely option on each question, every probability within 0.02; scene P's request is not a fixture record, so the provider's answers do not cover it); every string on screen is the rounding of the run's values, and each bar is drawn at its probability; the request time is at most twice the model card's median for a request with one photo at the 448 px tile (1,493.8 ms); the editor's content equals the request; no graph compiled after READY.
- `check_frames.py`: the recording against the run. Only the app's window is in it, the photo appears after the drop and before Decide, and the answers appear after Decide no sooner than the request time allows.

On the Apple M4 Max the recorded take ran 10.6 s (319 frames at 30 frames per second, 1080 × 1920, no sound): the photo was in place 2.1 s after the first frame, Decide was pressed at 3.0 s, and the answers appeared 1,568 ms later, after a request of 1,539.8 ms. The answers were coffee (0.974) and no (0.994), within 1.8e-7 of the CPU run.

On a Mac that runs other GPU work, three environment variables let the scripts wait for a quiet machine and time the scene inside a lock: `CLEF_WAIT_CMD` and `CLEF_HOLD_CMD` (command prefixes) and `CLEF_LOCK_FILE`. Unset, the scripts wait for nothing.

`--tag <name>` runs the editor for scripts: the window appears when the model is ready, ignores the mouse, and takes its input from `scripts/drive_editor.py` (the same handlers as your keys and clicks). Every Decide is written to `out/<name>_manual.run.json`.

## Requirements

- A Mac with Apple silicon. Tested on an Apple M4 Max with 128 GB and macOS 27.0.
- Python 3.14 (tested with 3.14.6), with the packages of `hub/host/requirements-host.txt` and `requirements-app.txt`.
- Disk: 20.0 GB for the default download, 13.8 to 13.9 GB more for each other row graph.
- Memory: 68.5 GB peak on Metal, 32.3 GB on the CPU (see Memory).
- For recording only: the Screen Recording permission, the Xcode command line tools (`xcrun swiftc`) and ffmpeg.

## Files

- `app/clef_demo.py`: the app. The window, the editor's Python side, Decide, the startup compile, autoplay and scripted runs.
- `app/editor.py`: how the editor's content becomes a request, and back.
- `app/examples.json`: the bundled examples and the recorded scene's questions.
- `app/ui/index.html`, `style.css`, `app.js`: the screen.
- `scripts/fetch_hub.sh`: the model files at a pinned revision, checked with SHA-256.
- `scripts/check_editor.py`, `scripts/drive_editor.py`: the editor's rules checked without the model; scripted input.
- `scripts/press_examples.py`, `scripts/check_examples.py`: every example pressed through the editor, and its answers against the provider's.
- `scripts/take.sh`, `scripts/take_inner.sh`, `scripts/record_window.swift`: a recorded take.
- `scripts/check_run.py`, `scripts/check_frames.py`, `scripts/cpu_reference.py`: a take's checks.

## Pictures and licenses

The examples are records of the public fixtures of [litert-community/clef-flash-LiteRT](https://huggingface.co/litert-community/clef-flash-LiteRT), written for those fixtures and free to use (CC0-1.0); the people, organizations and products in them are invented. The clock is a drawing made with Pillow primitives (CC0-1.0), and its request is the example of the model card. The coffee photo is "Coffee cup and coffee bean" by mcfoodie on [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Coffee_cup_and_coffee_bean.jpg), CC0-1.0. `scripts/fetch_hub.sh` downloads the pictures with the model files; this directory holds no picture.

clef-flash, by Cloudflare, is under the Apache License 2.0, and so are its LiteRT files; see the [model card](https://huggingface.co/litert-community/clef-flash-LiteRT). The LiteRT files are a community conversion and are not affiliated with Cloudflare.

The code in this directory is licensed under the Apache License 2.0 (`LICENSE`), with attribution in `NOTICE`.
