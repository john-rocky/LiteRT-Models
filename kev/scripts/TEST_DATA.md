# Test data

The JVM tests compare the Kotlin host (tokenizer, request → rows, pointer head, answers) with the
author's fp32 oracle: the Kev-0.8B checkpoint run on the CPU with the author's own code (`kev.api`,
`kev.model`, transformers 5.17.0), batch 1, one causal row per question.

## Bundled files (committed)

| File | Size | Content |
|---|---|---|
| `app/src/debug/assets/gate_fixtures.json` | 302,342 B | 156 requests: all 144 items of SemIf `authored144.jsonl` (MIT, `licenses/SemIf-MIT.txt`) and 12 invented requests. Per question: the oracle's row IDs, decide and option indices, float32 probabilities and answer. 181 questions, 172 rows fit the 512-token graph and 9 need the 2048-token graph. |
| `app/src/test/resources/head_fixture.json` + `.f32` | 5,016 + 233,472 B | Question 1 of each invented request: the oracle's hidden states at the decide token and at each option's closing token (float32 little-endian, row after row) with the oracle's logits and probabilities. |
| `app/src/test/resources/tokenizer_probes.json` | 15,613 B | 54 edge strings (Devanagari, the 33 added tokens, emoji sequences, NFD text, whitespace kinds, digits, contractions, other scripts) with the IDs of transformers' `AutoTokenizer` and of the author's `user_tokens`. |
| `app/src/debug/assets/tokenizer_probes.json` | 15,613 B | The same file, for the on-device gate (Android's ICU regex and NFC follow other Unicode versions than the desktop JVM). |
| `app/src/main/res/raw/example_{ticket,incident,review}.json` | 1,110 / 1,218 / 1,043 B | The app's three example requests: the invented demo requests `demo_ticket_01`, `demo_incident_02` and `demo_review_03`, byte for byte, as `{"id", "state", "questions"}`. |
| `app/src/test/resources/examples_oracle.json` | 21,910 B | For each example its sha256 and `usage.input_tokens`, and per question the author's oracle (row IDs, decide and option indices, float32 probabilities, answer) and `graph_l512_cpu`: the shipped L512 graph on a desktop CPU with the head in numpy float32 (probabilities and the 4-decimal answer the app shows). |
| `app/src/test/resources/render_cases.json` | 19,211 B | The author's `render`, `option_text` and `to_record` on edge values, and 13 requests the author's pydantic model rejects. |
| `app/src/test/resources/python_numbers.json` | 90,814 B | CPython 3.12 `sum`, `round(x, 4)` and `repr` on fixed and random inputs, and the author's `to_answers` and confidence functions on 46 distributions. |

All of them are written by `scripts/make_test_data.py`, which checks the sha256 of its sources
before writing and never writes the transfer-v4 rows (below) into the module.

```bash
python3 scripts/make_test_data.py --kev-work /path/to/kev_work            # gate + head fixtures, examples + their answers
/path/to/venv/bin/python scripts/make_test_data.py --kev-work /path/to/kev_work --probes
# --probes also writes the probe, render and number files; it needs Python 3.12,
# transformers 5.17, tokenizers, pydantic and the author's kev package (kev_work/kev).
```

The full fixture set also has 220 rows of the author's transfer-v4 development suite (MMLU,
emotion, tweet_eval, QNLI, PAWS, SciQ and the author's generated holdouts) and one gate arm made
from its MMLU row `tv4_000` by changing one word of the instructions. Their text is not
redistributed (tweet_eval's dataset card gives its licence as unknown, SciQ is CC BY-NC 3.0, and
emotion, GLUE and PAWS declare "other"); the JVM tests read them from the external directory.

## External reference data

Without `kev.work`, or with a missing file, JUnit reports an assumption skip. Skips are not parity
evidence.

```bash
./gradlew :app:testDebugUnitTest -Pkev.work=/path/to/kev_work
# Equivalently: -Dkev.work=/path/to/kev_work
```

Reports go to `app/build/reports/parity/`, never to the data directory.

```text
kev_work/
  fixtures/requests.json                    # 377 requests (sha256 dfe55fb1…)
  oracle/oracle_0.8b.json                   # 402 questions + 377 requests (sha256 d3792f3f…)
  oracle/hidden_0.8b.npz                    # [decide, options…] × 1024 per question (sha256 9f0cafeb…)
  host/kev_0.8b_pointer_head.safetensors    # q / k weight and bias, F32 (sha256 1e3da5cc…)
  host/kev_0.8b_pointer_head.json           # temperature, scale, delimiter IDs (sha256 aad6c553…)
  hf/hub/models--jaredpalmer--kev-0.8b/snapshots/788ddbdd65715bb03a56788c822f6c632c9a551d/tokenizer.json
                                            # 19,989,325 B (sha256 06b95093…)
  demo/fixtures/<id>.json                   # demo requests {"id", "state", "questions"} (the examples' source)
  demo/fixtures/token_lengths.json          # their row lengths from the author's code
  demo/oracle/oracle_demo.json              # their oracle rows (sha256 6b00e472…)
  demo/oracle/expected_v2_L512_cpu.json     # the shipped L512 graph on a desktop CPU (sha256 438044bc…)
  device/timing_rows.json                   # the model card's timing rows
```

The tokenizer is the `tokenizer.json` published with the Kev checkpoint. It is transformers 5.17's
`Qwen2Tokenizer` pipeline as the author's code runs it: the Qwen2 split regex (letters without
`\p{M}`) and 33 added tokens. The `tokenizer.json` of the Qwen3.5-0.8B-Base repo has the same
vocabulary and merges but a different regex (with `\p{M}`) and only 22 added tokens;
transformers rebuilds the pipeline in code and does not use them, so that file gives different
IDs on text with combining marks (Devanagari, for example) or with `<think>`, `<tool_response>`
and the other 9 tokens that only `tokenizer_config.json` lists.

| Test | Needs | Checks |
|---|---|---|
| `KevEncoderTest` | tokenizer, requests, oracle | 377 requests → rows: IDs, decide and option indices of all 402 questions; `usage.input_tokens` of all 377; windows 393 / 0 / 9 over L512 / L1024 / L2048 and 322 / 64 / 7 / 0 / 9 over the five published windows; padding, `valid` mask and the rejection over 2,048 tokens |
| `KevTokenizerTest` | tokenizer, probes | the tokenizer.json contract (regex, 33 added tokens, vocabulary size), the Java spelling of the regex, the `<\|name\|>` rewrite, the 54 probes |
| `KevPointerHeadTest` | head, oracle, hidden states | the 402 oracle questions within 1e-5 (probabilities) and 1e-4 (logits); the 12 bundled questions; the head constants |
| `KevAnswersTest` | oracle, requests, numbers | `to_answers` on the oracle's probabilities equals the oracle's answers (402 questions, 377 requests), CPython's `sum`, `round` and the confidence formulas |
| `KevRecordsTest` | render cases | `render`, `option_text`, `to_record`, request validation |
| `KevJsonTest` | numbers | key order, integer vs float literals, escapes, rejection of malformed JSON, Python's float `repr` |
| `GateFixturesTest` | debug asset, tokenizer, oracle | declared counts (156 requests, 181 questions), rows and indices of every asset question, answers from the asset's probabilities, asset = oracle |
| `KevPipelineTest` | tokenizer, head, requests, oracle, hidden states | the app's decision path with stand-in graphs that check the padded inputs and return the oracle's hidden states at the readout positions, with the graphs `KevResidentGraphs` compiles: L512 / L1024 / L2048 installed (windows 393 / 0 / 9), all five with memory for a second graph (322 / 64 / 7 / 0 / 9, every row on its smallest installed window) and without (319 / 67 / 7 / 0 / 9); answers of all 402 questions and `usage.input_tokens` of all 377 requests in all three; rows over the window and NaN outputs are rejected; the IDs' sha256 |
| `KevWindowsTest` | nothing external | the smallest installed window per row length for the installed sets {256, 512}, {128, 256, 512}, {512} and {512, 2048}; L128 + L256 side by side with enough available memory, L256 alone below the limit, L512 or more alone, the return from L2048 to the small windows, the bundled examples on L256 with the default install, a named window for every row, another backend reopening the primary only; never more than two graphs open |
| `DemoFixtureTest` | tokenizer, demo fixtures | every demo fixture parses and encodes to the row lengths of `token_lengths.json` (`demo_ticket_01`: 131 / 101 / 93) and the demo oracle's row IDs; without the demo files, the bundled ticket example against `examples_oracle.json`; the demo run JSON has every key the recording scripts read |
| `KevDraftsTest` | `examples_oracle.json`, tokenizer | the three examples are the demo requests byte for byte (sha256) and encode to the demo oracle's rows (9/9) and `usage.input_tokens`; `to_answers` on the oracle's and on the shipped graph's probabilities gives their answers (18/18); the editor round trip keeps the record; editor errors; Python's `indent=2` JSON; 4-decimal strings |
| `KevDeviceRunsTest` | debug assets, tokenizer, head, hidden states, timing rows | the on-device gate's checks with a stand-in graph that returns the oracle's hidden states: PASS with probes 54/54, rows 181/181, 172 rows run and 9 skipped as `needs L2048` at L512 and L256, 147 run at L128 (25 skipped as `needs L256`); `limit`, the stop file and a NaN graph cut or fail the run; the timing sets of the launch's window; the timing protocol's numbers of calls (5 + 20, request sets 20 × rows, the request path) |
| `KevGateChecksTest` | debug assets, tokenizer, oracle, timing rows | the gate's assets parse, the device probes equal the test probes, the near-tie gap equals the oracle's float32 gap (15 near-ties), numpy's median, the timing rows (`fiveq` = `own_fiveq_09`'s rows) |

The bundled head fixture still needs the head weights from the external directory; they are
installed on the device, not committed.

## Known JVM difference

Java 17's regex classes follow Unicode 13; the `tokenizers` library (onig) used for the oracle
follows Unicode 16. Code points assigned in Unicode 14–16 (9,787 letters and 130 digits) are not
`\p{L}` / `\p{N}` on a Java 17 JVM, so text containing them can split differently there. Android's
ICU follows newer Unicode versions. A one-off run of 6,000 random strings gave identical IDs for
all of them on the JVM, including the 2,254 strings with code points Java 17 does not define.

## On-device runs (debug APK)

The external files must be installed beforehand (`scripts/install_to_device.sh`). Every run below
shows the activity above a secure lock screen and keeps the screen on, so that the measured
process stays in the foreground; each report records the `/proc/self/cgroup` cpuset line at the
start and at the end (`…:cpuset:/top-app` in the foreground). A file `files/STOP` ends a gate or
timing run after the current graph call (`stopped_early: true`); a stale one is removed at the
start. Progress and results go to logcat under `KevGate`.

Gate and timing runs compile the one window `--ei window` names, or the smallest installed one;
their reports record `avail_mem_bytes_before_compile` and `resident_windows`. The app logs each
request's windows under `KevDemo` (`WINDOWS questions=… compiled=… avail_mem_bytes=… closed=…
resident=…`).

```bash
# Fixture gate: probes, all 181 rows, graph + head on the rows that fit the window.
adb shell am start -n com.kev/.MainActivity --ez gate true --es backend gpu \
  --es report app_gate_gpu.json [--ei window 256] [--ei limit 40]
adb exec-out run-as com.kev cat files/app_gate_gpu.json > app_gate_gpu.json

# Timing protocol (also in the benchmark build): the conversion run's timing_rows.json in files/.
adb push timing_rows.json /data/local/tmp/kev_timing_rows.json
adb shell run-as com.kev cp /data/local/tmp/kev_timing_rows.json files/timing_rows.json
adb shell am start -n com.kev/.MainActivity --ez timing true --es rows timing_rows.json \
  --es backend gpu --es report app_timing_gpu_L512.json [--ei window 512] [--ez clear_cache true] \
  [--es sets T300] [--ez request_path false]
# A rows file may name a set once per window (fiveq at L256 and at L512); a launch times the sets of its window.
```

The gate report (`files/<report>`, `files/<report>.partial` while running) holds the tokenizer
probes (`raw_equal` / `user_equal` of 54), `ids_identical` and `indices_identical` of 181,
`input_tokens_identical` of 156, then for the rows run: `max_abs_dp` and
`mean_abs_dp_all_options` against the oracle, `argmax_equal`, near-ties apart (oracle top-2 gap
≤ 0.02), `nonfinite_rows`, the infer-time median without the 5 rows that come before it, compile
and load times, the cache directory before and after compiling, thermal status and battery
temperature at both ends; per row its key, `row_len`, `ids_sha256`, status (`run`, or `skipped`
with `needs L2048`, `limit` or `stopped`), probabilities and times. PASS needs every probe and row
identical, no NaN, the argmax outside near-ties, max |Δp| ≤ 0.02 and mean |Δp| ≤ 0.002.

The timing report holds, for every set of `timing_rows.json` whose L is the resident window (or
only the sets named by `--es sets`, comma-separated; `none` times no set), 5 warm-up calls and 20 timed calls (a `request` set: 20 requests of its rows back to back) with
median (numpy's), min and max, then (unless `--ez request_path false`) `request_path`: the bundled
`own_fiveq_09` request from its text (tokenize, five graph calls, head, answers), 5 warm-up and 20
timed requests. The sets run back to back in one launch, so the phone is warmer for each set than for
the one before; to start every set at the same temperature, give each set a launch of its own (`--es sets
fiveq --ez request_path false`, then `--es sets T300 --ez request_path false`, then `--es sets none`).
The report records `sets_requested` and `request_path_requested`. A call is input
writes + `run()` + read-back. With `clear_cache` the app's cache directory is emptied before the
graph compiles, so the compile time is a cold one.

The demo autoplay reaches the running app (`launchMode="singleTop"`). Put the request in
`files/` as `{"id", "state", "questions"}`, launch the app normally, wait for `ENGINE_READY`, then:

```bash
adb shell am start -n com.kev/.MainActivity --ez autoplay true \
  --es fixture /data/user/0/com.kev/files/demo_ticket_01.json --ei delay_ms 1500 --ei gap_ms 800
```

`delay_ms` runs from the intent to the request on screen, `gap_ms` before each question. Without
`--ei window` each question runs on the window a Decide would use; `--ei window 2048` runs every
question on that graph (it becomes the second compiled graph). The app
logs under `KevDemo`: `ENGINE_READY load_ms=<tokenizer + head + compile>`, `AUTOPLAY_START
fixture=<path>`, `Q_DONE qid=<id> ms=<the card's ms>`, `AUTOPLAY_DONE json=<path>`, or
`failed <reason>`; `GPU_FALLBACK <error>` when the GPU graph could not be compiled and the app
runs on CPU. The run JSON `files/kev-demo-<epoch ms>.json` records the device, the runtime,
the graphs (`graph`: `file`, `L` and `bytes` of the largest window the questions ran on, `windows`
= every window they ran on, `resident` = the compiled graphs at the end, `compiled` = each compile
for the request with `avail_mem_bytes` read right before it, `closed`, and `second_refused` when
the memory was too low for a second graph), the title and footer
lines as shown, every question's probabilities, the strings on its card (`shown`, `shown_ms` = the
ms only; the card also shows the window), its answer, row IDs, readout indices, `window`,
`ids_sha256` and times (`infer_ms` = the card's ms), `request_total_ms` (tokenize to the last
answer, without the demo's waits and without a graph compile), airplane mode, the cgroup line and
where the screen drew the title, the cards' state indicators and the footer (`layout`).
