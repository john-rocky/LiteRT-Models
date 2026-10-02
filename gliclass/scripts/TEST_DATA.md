# Test data

## Debug gate asset (committed)

`app/src/debug/assets/gate_fixtures.json` holds 152 captured official requests: all 144 rows of
SemIf `authored144.jsonl` (MIT, GitHub TheoLeeCJ/SemIf at `ca3ba65f`), the model card's two
examples and six invented requests. Each row carries the request (text, prompt, labels), the token
IDs and `<<LABEL>>` positions the official gliclass 0.1.20 pipeline fed the model (CPU fp32,
batch 1), the window, the fp32 logits and the official single-label and multi-label (0.5) results.
All 152 fit the 128-token window, so the committed asset exercises only the s128 graph. The asset
is written by the conversion run's `scripts/app_gate_fixtures.py --subset`.

The full gate also used 200 ag_news test rows and 200 banking77 test rows, 70 of which need the
256-token window. Their text is not redistributed (the ag_news dataset card gives its license as
"unknown"; banking77 is CC BY 4.0). To run that gate, regenerate the asset from the conversion run
without `--subset`, which writes all 552 rows to the same path. Do not commit that file; restore
the subset with `git checkout -- app/src/debug/assets/gate_fixtures.json`.

## Optional JVM reference data

The JVM tests compare the Kotlin host with the official gliclass 0.1.20 pipeline and the Python
`tokenizers` library. The reference data is not distributed with the source. Without
`gliclass.fixtures`, or with a missing directory or file, JUnit reports an assumption skip. Skips
are not parity evidence.

```bash
./gradlew :app:testDebugUnitTest -Pgliclass.fixtures=/path/to/data
# Equivalently: -Dgliclass.fixtures=/path/to/data
```

Reports go to `app/build/reports/parity/`, never to the data directory.

```text
data/
  fixtures/oracle.json             # 552 official pipeline calls (CPU fp32, batch 1): linearized
                                   # string, captured input_ids, label_positions, logits, softmax,
                                   # sigmoid, single-label and multi-label (0.5) results, fits
  fixtures/requests.json           # the 552 requests: text, prompt, labels
  fixtures/tokenizer_stress.json   # edge strings with Python tokenizers ids ([CLS]/[SEP] added)
  tokenizer.json                   # or host_assets/tokenizer.json, or the conversion run's
                                   #   src/gliclass-edge-v3.0/tokenizer.json
  tok_embeddings_fp16.bin          # or host_assets/, or the run's exports/tables/
```

| Test | Needs | Checks |
|---|---|---|
| `GliclassTokenizerTest` | oracle, stress, tokenizer | ids of the 552 linearized oracle strings and of every stress string; special tokens and template |
| `GliclassInputsTest` | oracle, tokenizer, table | linearization, ids, `<<LABEL>>` positions, smallest window, padded ids / attention / routing / embeddings of all 552 requests; rejection rules; float16 upcast of every bit pattern |
| `GliclassDecoderTest` | oracle | oracle logits → the pipeline's single-label and multi-label results (labels exact, scores within 1e-6); tie, threshold and repeated-label rules |
| `GateFixturesTest` | oracle, tokenizer, table, debug asset | every row of the debug asset (152 committed, or 552 regenerated) equals the oracle; host inputs identical for every row; declared row and window counts; the prefilled request is the oracle's `inv_02` |

The oracle, the requests and the stress strings are written by the conversion run's
`scripts/source_oracle.py`, `scripts/make_fixtures.py` and `scripts/tokenizer_stress.py`.
