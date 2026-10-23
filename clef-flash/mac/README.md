# clef-flash mac: one command, seven devices of this Mac

Type a command such as "Quit every app except the TV app, and keep the music playing." and press Decide. The model reads this Mac's devices (the TV app, Calculator, two TextEdit documents that are both named Draft, a music loop, the sound and the screen lock) together with the command, and answers in one call how to handle it and what each device becomes: execute it, ask which device you mean, reject it, keep it for later, block it under the Mac's policy, or change nothing because it is already so. Each answer comes back as one of its options with a probability for each option. With "execute", the app sets the devices to the answered values and reads each one back: Calculator quits, the drafts close, the TV app and the music stay. "Close Draft." gets a question back instead, because two documents are named Draft. One request is 1,845 to 1,904 tokens and took 2,795.2 to 2,899.6 ms on an Apple M4 Max (Metal at float32 precision, the 2,048-token row graph, the median of five presses of each bundled command); the model writes no text. On ten commands written by someone else after the app's questions were fixed, 94.0 % of the questions came back right and 2 of the 10 commands had every question right (see [How well it does](#how-well-it-does)).

The model's answers are never run as commands. `app/actions.py` holds a fixed table from a decided change (a device, a property and a value such as `calculator.running = false`) to a setting; a change the table does not know sets nothing, and the screen is never locked: the app only shows what the answer would do. The model is Cloudflare's clef-flash, converted to LiteRT ([litert-community/clef-flash-LiteRT](https://huggingface.co/litert-community/clef-flash-LiteRT)), on the Mac's GPU through Metal at float32 precision. This app is the clef-flash sample's editor (`../app/`) with a command line and this Mac's devices added.

## The questions: the home appliance benchmark the provider reports

The provider measured clef-flash on a home appliance benchmark: a house's devices as JSON (ids, names, kinds, whether each is available, its settings, named groups, a household policy, the device the last request was about) and an explicit request about them, with choice questions on how to handle the request, which devices it justifiably targets, and what the devices' settings are right after correct handling. Its generator is part of the Decision Index kit ([github.com/apolinario/decision-index](https://github.com/apolinario/decision-index), `decision_index/suite/build/home_appliance.py`, MIT). In the provider's own Decision Index run, clef-flash scored 97.7 on it (case exact accuracy: every question of a request right; [model card](https://huggingface.co/Cloudflare/clef-flash), "Home appliance simulator", Decision Index 0.2.1). That number is the provider's run on the provider's rows, not this app's.

This app asks the same questions, in the benchmark's own words, about this Mac:

- `resolution`: "How should the assistant handle the request?", with the nine ways of the benchmark: execute now, execute available parts now, schedule for later, save an automation rule, ask a clarifying question, make no change because already satisfied, reject as unsupported, block under household policy, answer a state query.
- `target_member:<device>` for each device: "Is <device> in the justified device target set for this request?", yes or no.
- `post:<device>.<property>` for each setting the app can change: "Immediately after correct handling, what is <property> for <device>?", with the value now, the other value of a yes/no setting or the numbers in the command, and "not applicable or not listed".

Where it differs from the benchmark's rows:

- Seven devices instead of fourteen, and no rooms, room aliases or outside temperature: a Mac has none. A request is 1,845 to 1,904 tokens with the bundled commands and runs on the 2,048-token row graph; the benchmark's rows are 2,992 to 3,431 tokens.
- Question and option ids are words (`resolution`, `target_member:tv`, `post:tv.running`; `execute`, `yes`, `true`) instead of hashes, and the questions come in one order (the way of handling, then the targets, then the settings); the benchmark's hashed ids would not fit the 2,048-token graph.
- `post` is asked for every setting the app can change, so that each answer is the value the app sets; the benchmark asks it only for the first two settings the request changes. The benchmark's `outcome` (every change written out as one sentence, which an app cannot list as an option before it knows the answer) and `state_read` questions are not asked.
- The policy says the Mac is presenting and that a lock is blocked while it is (`presenting_lock`, in the place of the benchmark's `away_unlock`).

## What this app changes on your Mac

| Device | Setting | Set with | Read with |
|---|---|---|---|
| `tv` (the TV app) | running | `open -g -a TV`; quit with a TERM signal to its process | its process; `memory_mb` is its resident memory (`ps`) |
| `calculator` (Calculator) | running | `open -g -a Calculator`; the same | the same |
| `draft_1`, `draft_2` (two TextEdit documents named Draft) | open | `out/drafts/1/Draft.txt` and `out/drafts/2/Draft.txt`, written by the app, each opened in a TextEdit of its own (`open -n`), so one is closed by ending its process | whether that process runs |
| `music` | playing | `afplay` playing a chord loop the app writes to `out/music.wav` (32 s, synthesised), started again each time it ends, in a process of its own (`out/music.pid`) that ends with the app (after a crash, at the end of the 32 s) | that process |
| `sound` | volume (0 to 100) | `osascript`: `set volume` (setting a volume unmutes, as on macOS) | `osascript`: `get volume settings` (the mute is shown, not asked) |
| `screen` | locked | never: the app shows the answer and does not lock the screen | the login session |

Nothing else is changed: no TextEdit document of yours (each draft has its own TextEdit process), none of your files, accounts, network or security settings; the Music app and its library are never touched. If Calculator or the TV app was already running when the app started, a command can quit it and the app opens it again when it puts the Mac back. `MAC_TV_APP=<name>` makes the `tv` device another app.

The state also carries the benchmark's named groups (`evening`: the TV app and the music; `work`: the two drafts and Calculator), a policy in the benchmark's shape (`{"multi_action": "best_effort", "allow_partial": true, "security_mode": "presenting", "presenting_lock": "blocked"}`), `schedules` and `rules` (empty: the app keeps no schedule and runs no rule; "schedule for later" and "save an automation rule" are shown, not kept), `device_context` (the first target of the last command that had one) and the time now.

## Putting the Mac back

When the app starts, it reads every device and writes it to `out/mac_before.json`. It puts the Mac back when its window closes, on Ctrl-C and on a TERM signal: the TV app and Calculator running or not as they were, both drafts closed, the music stopped, the volume and its mute as they were. If it was stopped another way (a crash, `kill -9`), it refuses to start again until you run:

```bash
venv-demo/bin/python app/mac_demo.py --restore      # puts the Mac back to out/mac_before.json
venv-demo/bin/python -B scripts/check_restore.py    # the Mac now against the file
```

On the Mac it was tested on, with the README's commands in a new venv: "Start the work group." opened Calculator and one draft (each read back, 0.39 s in all), the app was then killed with `kill -9`, it refused to start again until `--restore`, and `--restore` quit Calculator, closed the draft and left the volume at 44 and muted, as the app had found them (`check_restore.py`: PASS).

## Run it

```bash
cd clef-flash/mac
zsh scripts/fetch_hub.sh                 # 23 files, 18.2 GB, into ../hub/, checked against the repository's SHA256SUMS (resumable)
python3.14 -m venv venv-demo
venv-demo/bin/pip install -r ../hub/host/requirements-host.txt -r requirements-app.txt
venv-demo/bin/python app/mac_demo.py
```

`fetch_hub.sh` downloads only what this app needs (the 2,048-token row graph, the tables, the joint head, the tokenizer and the host, no picture tower) into `../hub/`, the folder the clef-flash sample's and the home app's own `scripts/fetch_hub.sh` fill, at the same pinned revision: a file any of them has downloaded is kept. A second run only checks the files.

At start the app compiles the 2,048-token row graph for the GPU and warms it up with one request; the status line counts the seconds. On the Mac it was tested on, the compile took 79.2 s. Then Decide works. "Open the demo's devices" opens the TV app, Calculator and both drafts, plays the music and sets the volume to 25, so every bundled command has something to act on.

## The app

- THIS MAC: the way of handling in large letters (green changes the Mac, blue asks, red refuses, orange keeps for later, grey changes nothing) with its probability and what it means here (the changes, or "Which Draft? draft_1 or draft_2"), and one row per device: its setting now, the setting it gets, and the probability that it is a target. A row of a target is marked on its left edge. After the Mac is set, each changed row shows ✓ once the device read back as answered, or ✗ with what macOS answered.
- COMMAND: the command line (it writes the state's `"request"`), the bundled commands, "Open the demo's devices", the "Apply to this Mac" switch, Decide, and under it the request time the app measured, the request's length in tokens and the graph it ran on. With the switch off, Decide answers and the Mac stays as it is.
- STATE & QUESTIONS: the state as JSON and one card per question, as in the clef-flash sample's editor: a name, a type (yes / no, choice or score), the question, and the options, one per line. After Decide each card shows the probability of every option. The model reads all of it.
- The state is read from the Mac when the app starts, with "Read this Mac", and again right before each Decide, so the request holds the Mac as it is. Once you edit the state text yourself, your text is used as written until "Read this Mac". The questions follow the state (`app/form.py` builds them from its devices and its command) until you edit a card; then your cards are sent as written until "Rebuild questions". A bundled command's button writes only the command.
- Only "execute now" and "execute available parts now" change the Mac, and only the devices answered as targets whose answered setting differs from the setting now. "Ask a clarifying question" names the devices the command names (two Drafts), or says that none does.
- A request longer than the row graph (2,048 tokens) is refused with a red line; nothing is cut. The bundled commands are 1,845 to 1,904 tokens, so a command of about 140 more tokens fits.

The request time is `time.perf_counter()` right before and after `clef.decide(request)`: the tokenizer, the row graph's inputs, the row graph call with its input writes and output read, the joint head and the response. A compile is never inside it, and neither is setting the Mac, which is timed on its own: it waits on macOS (an app starting, a process ending), not on the model.

## How well it does

Two sets of commands, each with a fixed state and the expected answer to every question written down before the model answered it, by the benchmark's own rules: the target set is the devices the command acts on (none for a clarifying question), and a setting right after correct handling is its value now unless an executed command changes it. One convention was settled for both before the model saw the held-out set: a TextEdit document counts as an app ("Quit every app …" closes the drafts).

| Set | Commands | Questions right | Commands with every question right | Commands where the app does what was expected (defined after the runs) |
|---|---:|---:|---:|---:|
| dev: written for this app; the form of the state and the questions was settled on these (three forms were tried, the first was kept) | 17 | 243 / 255 (95.3 %) | 9 / 17 | 13 / 17 |
| held-out: written by someone else after the form was frozen, run once | 10 | 141 / 150 (94.0 %) | 2 / 10 | 7 / 10 |

Two lines were set for the held-out set before it ran: at least 85 % of its questions right, and at least 6 of its 10 commands with every question right. It passed the first (94.0 %) and missed the second (2 of 10). The last column was defined after both runs had been seen and does not replace those lines: a command counts when the app shows the expected way of handling ("execute now" and "execute available parts now" as one) and, when it changes the Mac, would set exactly the expected changes (`scripts/app_behavior.py`). The held-out commands that do not count: "Start the work group." opens Calculator and one draft of two, "Turn on the printer." is rejected instead of asked about, and "Please lock the screen, I'm stepping out." is not blocked (the app shows the lock and never locks).

These are this app's own numbers, on 27 commands about seven devices of a Mac; the 97.7 above is the provider's own run on the benchmark's rows. Here 15 of the 16 commands that missed missed one or two questions of fifteen ("Close Draft." missed three), and the misses are of a few kinds:

- "Execute available parts now" where every device was available and "execute now" was expected (4 commands). The state's policy says `best_effort` and `allow_partial: true`; in the benchmark's rows that policy comes with a device that is unavailable. On dev, the benchmark's usual policy (`atomic`, no partial) removed these answers but moved others, and did no better (9 of 17), so the first form was kept. For the Mac it changes nothing: both ways set the same devices.
- A device the Mac does not list ("the spreadsheet app", "the printer"): rejected as unsupported where the benchmark asks a clarifying question (2).
- "Lock the screen" while the policy says the Mac is presenting: not blocked, with the screen answered as locked (2); the app never locks the screen.
- "Quit it too.": a clarifying question where the benchmark acts on the device the last request was about (1).
- A clarifying question with the two Drafts answered as targets, where the benchmark answers no device ("Close Draft.", "Open Draft."; 2).
- The music answered as a target when it was the command's condition, the device the last request was about, or neither (3).
- A setting answered as staying when it changes, or as changing when it stays ("Quit the apps using more than 300 MB of memory.", "Start the work group."; 2).

The recorded scene's second command, "Close Draft.", gets 12 of 15: the way (ask which Draft) is right, and both drafts are answered as targets and draft_1 as closing. The app shows the question, names both drafts, and changes nothing.

Every answer below is the most likely option. The held-out set ran on Metal; the CPU run of the same requests (`eval/heldout_cpu.json`) gives the same most likely option on all 150 questions, every probability within 2.6e-5.

dev (`eval/dev.json`, `eval/dev_cpu.json`, CPU):

| | Command | Expected way | Answered way (p) | Questions right | The others |
|---|---|---|---|---:|---|
| d01 | Could you quit Calculator? | execute now | execute now (0.826) | 15 / 15 | — |
| d02 | Quit every app except the TV app, and keep the music playing. | execute now | execute now (0.448) | 15 / 15 | — |
| d03 | Start the evening group, please. | execute now | execute available parts now (0.668) | 14 / 15 | — |
| d04 | Close Draft. | ask a clarifying question | ask a clarifying question (0.831) | 12 / 15 | target draft_1: yes (0.932), expected no; draft_1.open: false (0.656), expected true; target draft_2: yes (0.813), expected no |
| d05 | Leave the music playing. | make no change because already satisfied | make no change because already satisfied (0.906) | 15 / 15 | — |
| d06 | Set the volume to 150%. | reject as unsupported | reject as unsupported (0.857) | 15 / 15 | — |
| d07 | At 10:15 tonight, quit the TV app. | schedule for later | schedule for later (0.905) | 15 / 15 | — |
| d08 | Quit it too. | execute now | ask a clarifying question (0.498) | 14 / 15 | — |
| d09 | Open the spreadsheet app. | ask a clarifying question | reject as unsupported (0.724) | 14 / 15 | — |
| d10 | Lock the screen. | block under household policy | make no change because already satisfied (0.432) | 13 / 15 | screen.locked: true (0.970), expected false |
| d11 | Close the work group, but keep the calculator as it is. | execute now | execute available parts now (0.634) | 14 / 15 | — |
| d12 | Quit the apps using more than 300 MB of memory. | execute now | execute now (0.435) | 13 / 15 | tv.running: true (0.673), expected false; music.playing: false (0.600), expected true |
| d13 | Make the sound 20 points louder. | execute now | execute now (0.773) | 15 / 15 | — |
| d14 | What's the volume right now? | answer a state query | answer a state query (0.890) | 15 / 15 | — |
| d15 | Stop the music. | execute now | execute now (0.775) | 15 / 15 | — |
| d16 | Whenever the music starts playing, set the volume to 30%. | save an automation rule | save an automation rule (0.936) | 14 / 15 | target music: yes (0.661), expected no |
| d17 | Set the volume to 40%. | execute now | execute now (0.825) | 15 / 15 | — |

held-out (`eval/heldout.json`, `eval/heldout_metal.json`, Metal):

| | Command | Expected way | Answered way (p) | Questions right | The others |
|---|---|---|---|---:|---|
| h01 | Make the sound 15 points quieter. | execute now | execute now (0.755) | 15 / 15 | — |
| h02 | Quit every app except Calculator. | execute now | execute now (0.463) | 14 / 15 | target music: yes (0.523), expected no |
| h03 | Start the work group. | execute now | execute now (0.629) | 14 / 15 | draft_2.open: false (0.796), expected true |
| h04 | Open Draft. | ask a clarifying question | ask a clarifying question (0.808) | 14 / 15 | target draft_1: yes (0.900), expected no |
| h05 | Set the volume to 120%. | reject as unsupported | reject as unsupported (0.847) | 14 / 15 | target music: yes (0.616), expected no |
| h06 | At 11:30 tonight, stop the music. | schedule for later | schedule for later (0.847) | 15 / 15 | — |
| h07 | Stop it too. | execute now | execute available parts now (0.367) | 14 / 15 | — |
| h08 | Turn on the printer. | ask a clarifying question | reject as unsupported (0.844) | 14 / 15 | — |
| h09 | Please lock the screen, I'm stepping out. | block under household policy | execute available parts now (0.252) | 13 / 15 | screen.locked: true (0.975), expected false |
| h10 | Shut down the evening group, but keep the TV app as it is. | execute now | execute available parts now (0.594) | 14 / 15 | — |

## Speed

Measured in this app on an Apple M4 Max (128 GB, macOS 27.0), ai-edge-litert 2.2.0, Metal at float32 precision, the 2,048-token row graph fully on the GPU: five presses of Decide per bundled command through the page, with "Apply to this Mac" off (`scripts/press_commands.py`), the median and the range of the request time.

| Command | Tokens | Median of 5 presses | Range |
|---|---:|---:|---:|
| Could you quit Calculator? | 1,849 | 2,801.1 ms | 2,794.2 to 2,920.0 ms |
| Quit every app except the TV app, and keep the music playing. | 1,857 | 2,800.4 ms | 2,793.4 to 2,833.0 ms |
| Start the evening group, please. | 1,848 | 2,806.7 ms | 2,803.5 to 2,845.5 ms |
| Close Draft. | 1,845 | 2,796.2 ms | 2,794.2 to 2,817.0 ms |
| Leave the music playing. | 1,848 | 2,800.7 ms | 2,785.3 to 2,817.8 ms |
| Set the volume to 150%. | 1,873 | 2,808.3 ms | 2,798.1 to 2,856.6 ms |
| At 10:15 tonight, quit the TV app. | 1,857 | 2,803.5 ms | 2,799.7 to 2,807.5 ms |
| Quit it too. | 1,847 | 2,820.5 ms | 2,817.8 to 2,858.6 ms |
| Open the spreadsheet app. | 1,848 | 2,899.6 ms | 2,831.2 to 2,998.7 ms |
| Lock the screen. | 1,847 | 2,799.3 ms | 2,791.1 to 2,811.3 ms |
| Close the work group, but keep the calculator as it is. | 1,856 | 2,800.3 ms | 2,794.2 to 2,871.1 ms |
| Quit the apps using more than 300 MB of memory. | 1,877 | 2,868.6 ms | 2,832.2 to 2,972.1 ms |
| Make the sound 20 points louder. | 1,904 | 2,813.7 ms | 2,788.8 to 2,888.1 ms |
| What's the volume right now? | 1,851 | 2,836.4 ms | 2,797.9 to 3,186.8 ms |
| Stop the music. | 1,847 | 2,795.2 ms | 2,787.7 to 2,827.8 ms |
| Whenever the music starts playing, set the volume to 30%. | 1,894 | 2,831.2 ms | 2,802.2 to 2,905.0 ms |
| Set the volume to 40%. | 1,888 | 2,892.2 ms | 2,812.8 to 2,951.4 ms |

The five answers of each command were the same to the last bit, and the same as the CPU's most likely options on every question (every probability within 3.0e-5 of the CPU run, `eval/dev_cpu.json`); the way of handling and every card on screen showed the call's own values. The same 17 requests decided five times each by `Model.decide` in a process without the app's page (`scripts/run_set.py --repeat 5`, `eval/dev_metal.json`) took 2,820.7 to 2,843.4 ms (medians). On the CPU (XNNPACK, 8 threads) they take 9.5 to 12.2 s each. The row graph alone takes 2,707.0 ms per call on this Mac; the rest is the host's tokenizer, the graph's inputs and the joint head.

Compiling the row graph for Metal took 79.2 s in the app (79.4 to 93.6 s in the other runs). The app's memory footprint peaked at 68.7 GB during the compile and was 38.2 GB after it (39.3 GB after the warm-up); on the CPU path the peak was 32.8 GB.

## Check the answers

The commands are not records of the model repository's test fixtures, so the provider's own answers do not cover them. The line for this app's answers is the CPU run of the same files and requests:

```bash
venv-demo/bin/python -B scripts/check_editor.py eval/dev.json                                       # the requests, the editor's rules and the table, no model
venv-demo/bin/python -B scripts/run_set.py eval/dev.json --accel cpu --out eval/my_dev_cpu.json     # every command on the CPU, held to the expected answers
venv-demo/bin/python -B scripts/run_set.py eval/dev.json --accel gpu --out eval/my_dev_metal.json   # the same on Metal
venv-demo/bin/python -B scripts/check_parity.py eval/my_dev_metal.json eval/my_dev_cpu.json         # Metal against the CPU
venv-demo/bin/python app/mac_demo.py --tag e1 &                                                     # the app, scripted input accepted
venv-demo/bin/python -B scripts/press_commands.py e1 eval/dev.json                                  # every command through the page, five presses each, "Apply to this Mac" off
venv-demo/bin/python -B scripts/check_presses.py out/e1_manual.run.json eval/my_dev_cpu.json
```

`run_set.py` sends each command of a set (its fixed state, the questions built from it) through the app's own `Model.decide` and counts the questions that got the expected answer and the commands with every question right. `check_parity.py` passes when, on every question whose two most likely options are more than 0.02 apart in the CPU run, the most likely option is the CPU run's and every probability is within 0.02 of it. `press_commands.py` turns "Apply to this Mac" off, pastes each command's state into the State tab and taps Decide, as you would, five times each; then it pastes a command too long for the row graph, which the app refuses. `check_presses.py` holds those answers to the CPU run the same way and checks that the screen shows the call's own values. On the Mac it was tested on, Metal and the CPU gave the same most likely option on all 255 dev questions and all 150 held-out questions (no near ties), every probability within 3.0e-5 (dev) and 2.6e-5 (held-out).

## Record a take

The app can also play two scenes by itself, for a recording. It types into the command line and presses Decide through the same page and the same code a click reaches, and the Mac is set for real.

- Scene M: the TV app, Calculator and both drafts open, the music playing; "Quit every app except the TV app, and keep the music playing." is typed; Decide; Calculator and both drafts close, the TV app and the music stay.
- Scene D: both drafts opened again (the take script opens them between the scenes and places their windows); "Close Draft." is typed; Decide; the app asks which Draft and changes nothing.

```bash
zsh scripts/take.sh t1      # saves your Mac, sets the scene's opening, runs the app with --scenes M,D, records, puts everything back, checks
```

`take.sh` records the main display at 1920 x 1080 (`scripts/record_display.swift`, ScreenCaptureKit, 30 frames per second, no sound) with only these apps in the picture: this app (its window, and a plain backdrop it lays over the desktop behind every window), the TV app, Calculator, TextEdit, and the menu bar's system items. Every other app that is running when the recording starts is left out by the recording's filter: its windows, menu bar items and notifications are not in the picture, and nothing of it is closed or changed. For the length of the take it also places the TV app's window and folds its sidebar (the sidebar shows the account's name), through the TV app's preferences, and places Calculator's and the drafts' windows (System Events, which needs the Accessibility permission of your terminal); it puts them back afterwards. It needs the Screen Recording permission for your terminal and the Mac's screen unlocked. Each take needs a new tag. A take counts when its checks pass:

- `check_run.py`: each scene's probabilities against a CPU run of the scene's own request (`cpu_reference.py --from-run`: the same most likely option on each question, every probability within 0.02); each command handled as its expected answers would handle it: the same way of handling ("execute now" and "execute available parts now" count as one, since the app sets the same devices for both), the same changes, and for "Close Draft." the same two drafts to ask between (how many of the fifteen questions got the expected answer is printed as well: "Close Draft." gets 12 of 15, see above); every string on screen is the rounding of the run's values; the request time is at most twice the 2,048-token row graph's own time on this Mac (2,707.0 ms); scene M's changes were set and read back while the TV app and the music went on, scene D changed nothing; the whole command was typed before Decide; the state was the Mac read right before Decide; no graph compiled after READY.
- `check_frames.py`: the recording against the run. The backdrop stays plain below the app's window; each command appears before its Decide; the answers appear after Decide no sooner than the request time allows; in scene M the way of handling turns green and Calculator's and the drafts' windows close only after the answers, while the TV app's window stays; in scene D the drafts are back before the command is typed and the way of handling turns blue. Colours are compared as the recording holds them (the recorder writes BT.709 video, so the backdrop's sRGB #5B6B7C decodes 11 to 13 levels darker per channel).

On the Mac it was tested on, a take was 18.2 s at 1920 x 1080, 30 frames per second, with no sound, and both checks passed. Scene M answered "execute now" (0.515) in 2,981 ms; Calculator quit and both drafts closed, each read back (0.245 s in all), while the TV app and the music went on; all 15 questions got the expected answer. Scene D answered "ask a clarifying question" (0.815) in 2,947 ms, "Which Draft? draft_1 or draft_2", and nothing changed; 12 of its 15 questions got the expected answer, as on dev (both drafts answered as targets, draft_1 as closing). The CPU run of the take's two requests gave the same most likely option on all 30 questions, every probability within 5.8e-6. These two request times are one press each with the screen recorder running, not the medians of the speed table (2,800.4 and 2,796.2 ms for the same commands).

On a Mac that runs other GPU work, three environment variables let the scripts wait for a quiet machine and time the scenes inside a lock: `MAC_WAIT_CMD` and `MAC_HOLD_CMD` (command prefixes) and `MAC_LOCK_FILE`. Unset, the scripts wait for nothing.

`--tag <name>` runs the app for scripts: the window appears when the model is ready, ignores the mouse, and takes its input from `scripts/drive_editor.py` (the same handlers as your keys and clicks). Every Decide is written to `out/<name>_manual.run.json`.

## Requirements

- A Mac with Apple silicon. Tested on an Apple M4 Max with 128 GB and macOS 27.0.
- Python 3.14 (tested with 3.14.6), with the packages of `../hub/host/requirements-host.txt` and `requirements-app.txt`.
- Disk: 18.2 GB for the download (13.9 GB of it is the row graph).
- Memory: a peak memory footprint of 68.7 GB while the row graph compiles for Metal, 38.2 GB after it (32.8 GB on the CPU path).
- The TV app, Calculator and TextEdit (they come with macOS).
- For recording only: the Screen Recording and Accessibility permissions for your terminal, the Xcode command line tools (`xcrun swiftc`) and ffmpeg.

## Files

- `app/mac_demo.py`: the app. The window, the page's Python side, Decide and the settings, the startup compile, putting the Mac back, autoplay and scripted runs.
- `app/form.py`: the request of a command: the state of this Mac's devices in the benchmark's shape, the benchmark's questions on it, and the decision read from the answers.
- `app/actions.py`: the seven devices: how each is read, set and read back, the table from a decided change to a setting, the music, and putting the Mac back.
- `app/editor.py`: how the page's content becomes a request, and back (the clef-flash sample's rules).
- `app/commands.json`: the bundled commands and the recorded scenes' commands.
- `app/ui/index.html`, `style.css`, `app.js`: the screen, light or dark with the Mac.
- `eval/dev.json`, `eval/heldout.json`: the command sets with their fixed states and expected answers; `eval/*_cpu.json`, `eval/*_metal.json`: their runs.
- `scripts/fetch_hub.sh`: the model files this app needs, at a pinned revision, checked with SHA-256.
- `scripts/run_set.py`, `scripts/check_parity.py`: a command set through the app's `Model.decide`, held to its expected answers; one run against another.
- `scripts/check_editor.py`, `scripts/drive_editor.py`: the requests, the editor's rules and the table checked without the model; scripted input.
- `scripts/press_commands.py`, `scripts/check_presses.py`: every command pressed through the page, and the answers against the CPU run.
- `scripts/check_restore.py`: the Mac now against what the app found at its start.
- `scripts/take.sh`, `scripts/take_inner.sh`, `scripts/record_display.swift`, `scripts/cpu_reference.py`: a recorded take and the CPU run of its requests.
- `scripts/check_run.py`, `scripts/check_frames.py`: a take's checks.

## Licenses

The commands, the states, the drafts' text and the music were written for this app (the music is synthesised from sine tones by `app/actions.py`). The questions' wording and the shape of the state follow the Decision Index kit's home appliance generator (MIT, © its authors). The row icons are SVG shapes in `app/ui/app.js`; this directory holds no picture or sound file.

clef-flash, by Cloudflare, is under the Apache License 2.0, and so are its LiteRT files; see the [model card](https://huggingface.co/litert-community/clef-flash-LiteRT). The LiteRT files are a community conversion and are not affiliated with Cloudflare.

The code in this directory is licensed under the Apache License 2.0 (`../LICENSE`), with attribution in `../NOTICE`.
