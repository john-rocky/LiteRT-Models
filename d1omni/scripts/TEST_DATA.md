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
