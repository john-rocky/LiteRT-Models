# d1-3B Decide: questions about a text or a photo, answered on a Mac

Paste a support ticket, a note or a JSON object, add a photo if you like, and write the questions you want answered, each with its options. Press Decide. Every question comes back at once as one of your options, with a probability for each option. The model is Liquid AI's d1-3B, converted to LiteRT, and it runs on the Mac's GPU. It never writes text.

The app opens with the model card's support ticket and its three questions. Four examples are bundled: that ticket, an incident note, a product review, and a photo of two kittens.

## Run it

```bash
cd d1-3b
zsh scripts/fetch_hub.sh                 # 27 files, 16.1 GB, into hub/, checked against the repository's SHA256SUMS
python3.14 -m venv venv-demo
venv-demo/bin/pip install -r hub/host/requirements-host.txt -r requirements-app.txt
venv-demo/bin/python app/d1_demo.py      # the editor
```

At start the app compiles four graphs for the GPU: the 64-token shared-state pair, the 512-token row graph, the picture tower and the projector. On an Apple M4 Max that took 77 s; the status line counts the seconds. Then it reads READY and Decide works.

## The editor

- STATE: plain text, or a JSON object or array (the label next to STATE says which). Leave it empty and add a photo, and the photo is the whole state.
- PHOTO: drop a JPEG or PNG on the window, or press Choose photo…. The app shows the pixels as the model reads them, without the EXIF rotation.
- QUESTIONS: one card per question. A card has a name (the answer's key), a type, the question, and the options, one per line:
  - choice: `name: description`, or just `name`; two or more;
  - score: one level per line, from the lowest; 2 to 10 levels;
  - yes / no: nothing, or both `Yes: what yes means` and `No: what no means`.
- Decide: the cards turn into the answers, all at once. Each shows the option the model picked, its probability, and a bar for every option. The footer shows the request time the app measured. Any edit clears the answers; press Decide again for the edited request.
- A row holds the state and one question. A row longer than 4,096 tokens is refused with a red line; nothing is cut.

The request time is `time.perf_counter()` right before and after `host.decide(request)`: the tokenizer, the picture's preprocessing and graphs, every graph call with its inputs and outputs, and the read-out. A compile is never inside it.

## Which graphs run

The model repository has row graphs for rows of 128 to 4,096 tokens and three shared-state pairs (a state of up to 64, 128 or 256 tokens, read once for all questions). For each request, the repository's host picks among the files in `hub/`.

The default download holds the 64-token pair, the 512- and 4,096-token row graphs and the picture graphs. Every request with rows of up to 4,096 tokens runs on it, some on a larger graph than the full repository would use. For those, a grey line under the answers names the graph that ran and the file that `zsh scripts/fetch_hub.sh --all` adds (80 files, 45.3 GB in all). A request whose graph is not compiled yet waits for that compile; at most three text graphs stay compiled.

Measured in this app on an Apple M4 Max (128 GB, macOS 27.0), ai-edge-litert 2.2.0, Metal at float32 precision, with the default download:

| Request | Graph | Request time |
|---|---|---:|
| The card's ticket and its 3 questions, typed into the editor | 64-token pair | 258 ms |
| The kitten photo and 1 question | 512-token row graph, picture tower and projector | 412 ms |
| The incident note and its 3 questions | 512-token row graph, 3 rows | 667 ms |

The ticket and photo rows are from a recorded take, where they ran 6.0 s and 15.2 s after the request before them, while the app typed. The incident note came 0.6 s after another request. Sent back to back, requests take less: the model card gives medians of 153.9 ms for its ticket and 389.3 ms for its own photo with one question. With the full download, the host picks the 128-token pair for the incident note and the 256-token row graph for the photo.

## Record a take

The app can also play two scenes by itself, for a recording. It types into the same fields and presses the same buttons that you do, and Decide runs the same request.

- Scene A: the editor opens with an empty state and the card's three questions; the card's ticket is typed in; Decide; the three answers.
- Scene B: from A's answers, the state is cleared, the three questions are removed, "How many cats are there?" is added with its options (one, two, three or more), the photo is placed in the drop zone; Decide; the answer.

```bash
(cd hub && ../venv-demo/bin/python -B examples/run_example.py --check --accel gpu) 2>&1 | tee out/run_example_check_gpu.log
venv-demo/bin/python -B scripts/cpu_reference.py   # scene B's reference answer, from the same files on the CPU
zsh scripts/take.sh t1                             # the app with --scenes A,B, both scenes recorded, then the checks
zsh scripts/make_media_mac.sh t1                   # the videos: both scenes and an end card, and one per scene
venv-demo/bin/python -B scripts/check_media.py t1
```

`take.sh` records the app's window and nothing else (`scripts/record_window.swift`, ScreenCaptureKit's window filter, 30 frames per second, no sound) and needs the Screen Recording permission for your terminal. Each take needs a new tag. A take counts when its checks pass:

- `check_run.py`: scene A's probabilities against the provider's float32 CPU answers in `hub/examples/run_example.expected.json` (within 1e-5, the same top option, 116 tokens, the 64-token pair); scene B's against `cpu_reference.py` (within 1e-4, the same request, 1 tile, the 512-token row graph); every string on screen is the rounding of the run's values, and each bar is drawn at its probability; each request time is at most twice the card's median for its kind; the editor's content equals the request.
- `check_frames.py`: the recording against the run. Only the app's window is in it, the typed text grows frame by frame before Decide, and the answers appear after Decide no sooner than the request time allows.
- `check_media.py`: every frame of the videos is the frame of the recording it was cut from, and nothing is sped up, edited or captioned.

On a Mac that runs other GPU work, three environment variables let the scripts wait for a quiet machine and time each scene inside a lock: `D1_WAIT_CMD` and `D1_HOLD_CMD` (command prefixes) and `D1_LOCK_FILE`. Unset, the scripts wait for nothing.

`--tag <name>` runs the editor for scripts: the window appears when the model is ready, ignores the mouse, and takes its input from `scripts/drive_editor.py` (the same handlers as your keys and clicks). Every Decide is written to `out/<name>_manual.run.json`. `scripts/check_editor.py` checks the editor's rules without the model.

## Requirements

- A Mac with Apple silicon. Tested on an Apple M4 Max with 128 GB and macOS 27.0.
- Python 3.14 (tested with 3.14.6), with the packages of `hub/host/requirements-host.txt` and `requirements-app.txt`.
- Disk: 16.1 GB for the default download, 45.3 GB with `--all`.
- Memory: the app process held 32.2 GB with the four graphs compiled. A request that needs another graph compiles it in addition.
- For recording only: the Screen Recording permission, the Xcode command line tools (`xcrun swiftc`) and ffmpeg.

## Files

- `app/d1_demo.py`: the app. The window, the editor's Python side, Decide, the startup compile, autoplay and scripted runs.
- `app/editor.py`: how the editor's content becomes a request, and back.
- `app/examples.json`: the bundled examples.
- `app/ui/index.html`, `style.css`, `app.js`: the screen.
- `fixtures/cats_10241192.jpg`, `fixtures/media_sources.json`: the photo and where it comes from.
- `scripts/fetch_hub.sh`: the model files at a pinned revision, checked with SHA-256.
- `scripts/check_editor.py`, `scripts/drive_editor.py`: the editor's rules checked without the model; scripted input.
- `scripts/take.sh`, `scripts/take_inner.sh`, `scripts/record_window.swift`: a recorded take.
- `scripts/check_run.py`, `scripts/check_frames.py`, `scripts/screen_numbers.py`, `scripts/cpu_reference.py`: a take's checks.
- `scripts/media_cuts.py`, `scripts/make_end_card.py`, `scripts/make_media_mac.sh`, `scripts/check_media.py`: the videos and their check.
- `scripts/make_photo.py`: how the photo was made from the Pexels original.

## Photo and licenses

The photo is "Cute Cats Lying Down Together" by Bruno Abdiel on Pexels (https://www.pexels.com/photo/cute-cats-lying-down-together-10241192/), under the Pexels License, resized to 512 × 341 pixels. `fixtures/media_sources.json` records the original's SHA-256 and how it was resized. The model card's own photo (COCO val2017) is used only by `run_example.py --check`, which downloads it into memory; it is not stored here and never shown.

d1-3B and its LiteRT files are under the LFM Open License v1.0; see the [model card](https://huggingface.co/litert-community/d1-3B-LiteRT). The license limits commercial use by organizations with annual revenue of US$10 million or more. The LiteRT files are a community conversion and are not affiliated with Liquid AI.

The code in this directory is licensed under the Apache License 2.0 (`LICENSE`), with attribution in `NOTICE`.
