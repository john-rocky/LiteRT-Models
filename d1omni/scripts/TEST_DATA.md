# Test data

The JVM tests compare the Kotlin host (tokenizer, request → rows, read-out, answers) with the
provider's code and the model repository's Python host. They read two directories kept out of the
source tree; without them, or with a file missing, JUnit reports an assumption skip, and a skip is
not parity evidence.

```bash
./gradlew :app:testDebugUnitTest -Pd1omni.repo=/path/to/d1-omni-600M-LiteRT -Pd1omni.demo=/path/to/demo
# Equivalently: -Dd1omni.repo=… -Dd1omni.demo=…
```

`d1omni.repo` is the model repository directory: `contract.json`, `tokenizer.json` and
`fixtures/public_{text,image,audio}.json` (the public check set: 242 + 5 + 6 requests, 303 questions,
each with the provider's encoded row and its float32 probabilities). `d1omni.demo` holds the
conversion run's test data:

```text
demo/
  fixtures/tokenizer_probes.json   # every string the provider's encode tokenizes for the check set, edge strings,
                                   # serialize / _criterion / str() cases, with the ids of Hugging Face tokenizers
  fixtures/scores_probe.json       # 20 check-set rows through the repository's graphs on a desktop CPU and the
                                   # Python host's read-out, answer() and three-decimal strings
  device/r1/rows_L128.json         # the device gate's rows (210 + 43) and timing sets, made by the Python host
  device/r1/rows_L256.json
  device/r1/timing_rows.json
```

Reports go to `app/build/reports/parity/`, never to the data directories.

| Test | Needs | Checks |
|---|---|---|
| `D1TokenizerTest` | repo, tokenizer cases | the tokenizer.json contract (regex, 509 added tokens, the two `normalized: true` ones, vocabulary size), the Java spelling of the regex, the escape, every case's ids for the raw text and for `escape(text)` |
| `D1PromptTest` | repo, tokenizer cases | all 303 check-set questions → ids and markers bit for bit, `usage_input_tokens` of all 253 requests; `serialize`, `_criterion` and `str()` against Python; options, temperature keys, question validation, answers |
| `D1ReadoutTest` | repo, scores | the read-out from the whole scores vector, `answer()` and the three-decimal strings on 20 rows (text, image, audio; every type; near-ties) |
| `D1GateCoreTest` | repo, device rows, scores | the device rows encode again to their ids (253/253), their inputs, and a stand-in graph with the desktop scores gives the Python probabilities |
| `D1ContractTest` | repo | the file table, buckets and signatures, token IDs, temperatures, request kinds |
| `D1RowsTest` | repo | `build_inputs` (text and media rows), buckets, request-kind rules |
| `D1ResidencyTest` | nothing external | at most two graphs compiled, the memory rule for a second graph |
| `D1JsonTest`, `D1LaunchTest` | nothing external | JSON reading and writing with Python semantics; the launch extras |

## On-device runs (debug APK)

Every run below shows the activity above a secure lock screen and keeps the screen on, so that the
measured process stays in the foreground; each report records the phone's state (thermal status,
GPU clock ceiling and temperature, CPU caps, memory, the cgroup line) at the start and at the end.
A file `files/STOP` ends a gate or timing run after the current graph call; a stale one is removed
at the start. Progress goes to logcat under `D1OmniGate`.

```bash
# Fixture gate: every row of a rows file on its bucket, the probabilities, the scores at the markers,
# the call's times, and whether this app encodes each row's request to the same ids.
adb push rows_L128.json /data/local/tmp/rows_L128.json
adb shell run-as com.d1omni cp /data/local/tmp/rows_L128.json files/rows_L128.json
adb shell am start -n com.d1omni/.MainActivity --ez gate true --es fixture rows_L128.json \
  --es report gate_L128.json [--es precision fp32] [--ei limit 60] [--ei resident 128]
adb exec-out run-as com.d1omni cat files/gate_L128.json > gate_L128.json

# Timing protocol: per set, a wait for the GPU to cool after the compile, warm-up calls, timed rounds.
adb shell am start -n com.d1omni/.MainActivity --ez timing true --es rows timing_rows.json \
  --es report timing.json --ei warmup 5 --ei reps 20 --ei cool_ms 120000 [--es sets card3,one]
```

`--ei resident 128` compiles the L128 graph first and keeps it while the rows file's graph compiles
(two graphs at once); the report holds each compile's memory before and after it.

<!-- vision (round 3) -->
## The picture path

The picture tests read the Python host's dumps under `d1omni.demo` (made by the conversion run's
`demo/scripts/vision_dump_v.py` from the repository's `host/d1_vision_host.py`, the five check-set
PNGs, the vision tower and the projector on a desktop CPU) and the picture rows files:

```text
demo/
  fixtures/vision/<id>/rgb.u8, crop<k>.u8         # load_image() and each crop after the float-path resize (uint8)
  fixtures/vision/<id>/{pixels,pos,mask}<k>.f32  # the tower's inputs per crop (little-endian float32)
  fixtures/vision/<id>/{features,soft,projected}<k>.f32, prefix.f32, meta.json
                                                 # the graphs' outputs on the desktop CPU, the projector input,
                                                 # the prefix rows; layout, sha256, each question's encoded row,
                                                 # its decision inputs' sha256, scores and read-out
  fixtures/vision/layout_cases.json, resize_cases.{json,bin}, positions_sweep.json, orient/, table.json
  device/r3/rows_image{,_small,_L2048}.json, timing_image.json   # the picture gate's and timing's rows
```

| Test | Needs | Checks |
|---|---|---|
| `D1VisionTest` | repo, demo | bit for bit: the decoded RGB of the five PNGs (a plain PNG reader; Android's decoder is checked on the phone), layout() on 6,844 sizes and its grid order, the resample weights (79 size pairs) and the float-path resize (40 arrays), every crop's pixels and tower inputs (pixels / pos / mask, grid), the position table resized to 1,466 grids, the unshuffle and projector input, the prefix rows from the desktop graphs' outputs, each question's six decision inputs and read-out; EXIF orientations 1–8 against Pillow; the rows files parse and encode again |
| `D1NpyTest` | repo, demo | the position table's `.npy` header, sha256 and values; headers it refuses |
| `D1GraphTest` | repo | the declared tensors and feeds of a single-signature graph; the tower and the projector as contract.json declares them |
| `D1VisionLaunchTest` | nothing external | the picture runs' launch extras |

On the phone (debug APK, `VISION=1` install, the pictures and rows files in `files/`):

```bash
# Picture gate: per record the picture decoded and compared with the Python host's RGB (sha256), the
# tower inputs per crop (pixels / pos / mask sha256), the prefix rows (against the desktop CPU's,
# a reference value), then each question on the smallest resident decision graph, its probabilities,
# whether this app encodes the question to the same ids, and the time of every step.
adb shell am start -n com.d1omni/.MainActivity --ez vgate true --es fixture rows_image_small.json \
  --es report vgate.json [--es precision fp32] [--es precision_vision fp16acc] [--ei limit 5]

# Picture timing: per set, a wait for the GPU after the compiles, warm-up requests, then timed
# requests, each = read and decode the picture, the prefix rows, every question's call and read-out.
adb shell am start -n com.d1omni/.MainActivity --ez vtiming true --es rows timing_image.json \
  --es report vtiming.json --ei warmup 5 --ei reps 20 --ei cool_ms 120000 [--es sets dogs2]
```

A picture record's `reference` names `files/` copies of the Python host's RGB and prefix rows
(`vref_<id>_rgb.u8`, `vref_<id>_prefix.f32`); without them the report keeps the sha256 checks only.
<!-- end vision (round 3) -->
