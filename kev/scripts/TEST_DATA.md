# Test data

The JVM tests compare the Kotlin host (tokenizer, request → rows, pointer head, answers) with the
author's fp32 oracle: the Kev-0.8B checkpoint run on the CPU with the author's own code (`kev.api`,
`kev.model`, transformers 5.17.0), batch 1, one causal row per question.

## Bundled files (committed)

| File | Size | Content |
|---|---|---|
| `app/src/debug/assets/gate_fixtures.json` | 302,342 B | 156 requests: all 144 items of SemIf `authored144.jsonl` (MIT, `licenses/SemIf-MIT.txt`) and 12 invented requests. Per question: the oracle's row IDs, decide and option indices, float32 probabilities and answer. 181 questions, 172 rows fit the 512-token graph and 9 need the 2048-token graph. |
| `app/src/test/resources/head_fixture.json` + `.f32` | 5,016 + 233,472 B | The first question of each invented request: the oracle's hidden states at the decide token and at each option's closing token (float32 little-endian, row after row) with the oracle's logits and probabilities. |
| `app/src/test/resources/tokenizer_probes.json` | 15,613 B | 54 edge strings (Devanagari, the 33 added tokens, emoji sequences, NFD text, whitespace kinds, digits, contractions, other scripts) with the IDs of transformers' `AutoTokenizer` and of the author's `user_tokens`. |
| `app/src/test/resources/render_cases.json` | 19,107 B | The author's `render`, `option_text` and `to_record` on edge values, and 13 requests the author's pydantic model rejects. |
| `app/src/test/resources/python_numbers.json` | 90,814 B | CPython 3.12 `sum`, `round(x, 4)` and `repr` on fixed and random inputs, and the author's `to_answers` and confidence functions on 46 distributions. |

All of them are written by `scripts/make_test_data.py`, which checks the sha256 of its sources
first and never writes the transfer-v4 rows (below) into the module.

```bash
python3 scripts/make_test_data.py --kev-work /path/to/kev_work            # gate + head fixtures
/path/to/venv/bin/python scripts/make_test_data.py --kev-work /path/to/kev_work --probes
# --probes also writes the probe, render and number files; it needs Python 3.12,
# transformers 5.17, tokenizers, pydantic and the author's kev package (kev_work/kev).
```

The full fixture set also has 220 rows of the author's transfer-v4 development suite (MMLU,
emotion, tweet_eval, QNLI, PAWS, SciQ and the author's generated holdouts) and one gate arm made
from its first MMLU row by changing one word of the instructions. Their text is not
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
| `KevEncoderTest` | tokenizer, requests, oracle | 377 requests → rows: IDs, decide and option indices of all 402 questions; `usage.input_tokens` of all 377; windows 393 / 0 / 9; padding, `valid` mask and the rejection over 2,048 tokens |
| `KevTokenizerTest` | tokenizer, probes | the tokenizer.json contract (regex, 33 added tokens, vocabulary size), the Java spelling of the regex, the `<\|name\|>` rewrite, the 54 probes |
| `KevPointerHeadTest` | head, oracle, hidden states | the 402 oracle questions within 1e-5 (probabilities) and 1e-4 (logits); the 12 bundled questions; the head constants |
| `KevAnswersTest` | oracle, requests, numbers | `to_answers` on the oracle's probabilities equals the oracle's answers (402 questions, 377 requests), CPython's `sum`, `round` and the confidence formulas |
| `KevRecordsTest` | render cases | `render`, `option_text`, `to_record`, request validation |
| `KevJsonTest` | numbers | key order, integer vs float literals, escapes, rejection of malformed JSON, Python's float `repr` |
| `GateFixturesTest` | debug asset, tokenizer, oracle | declared counts (156 requests, 181 questions), rows and indices of every asset question, answers from the asset's probabilities, asset = oracle |

The bundled head fixture still needs the head weights from the external directory; they are
installed on the device, not committed.

## Known JVM difference

Java 17's regex classes follow Unicode 13; the `tokenizers` library (onig) used for the oracle
follows Unicode 16. Code points assigned in Unicode 14–16 (9,787 letters and 130 digits) are not
`\p{L}` / `\p{N}` on a Java 17 JVM, so text containing them can split differently there. Android's
ICU follows newer Unicode versions. A one-off run of 6,000 random strings gave identical IDs for
all of them on the JVM, including the 2,254 strings with code points Java 17 does not define.
