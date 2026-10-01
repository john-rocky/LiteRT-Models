# Optional JVM reference data

The JVM tests compare the Kotlin host with the official implementation (the source checkpoint's own
`typed_decisions` package). The reference data is not distributed with the source. Without
`opendecision.fixtures`, or with a missing directory or file, JUnit reports an assumption skip.
Skips are not parity evidence.

```bash
./gradlew :app:testDebugUnitTest -Popendecision.fixtures=/path/to/data
```

Reports go to `app/build/reports/parity/`, never to the data directory.

```text
data/
  fixtures/requests.json   # the conversion run's 1,809 requests (author's public test files + examples)
  fixtures/oracle.json     # the official implementation's ids, spans, logits and answers for them
  tokenizer.json           # the source checkpoint's tokenizer.json (also found under hf_staging/ or
                           #   src/open-jev-deberta-v3-large/ of the conversion run)
```

| Test | Needs | Checks |
|---|---|---|
| `InputsParityTest` | requests, oracle, tokenizer | ids, question spans, option spans and the smallest window of all 1,809 requests |
| `TokenizerProbeTest` | tokenizer | edge strings (charsmap, stripping, added tokens, `[UNK]`, emoji, scripts) against the official fast tokenizer |
| `DecoderAndTableTest` | (resources only) | oracle logits → the official answers; float16 widening against NumPy; the question editor |

The resources in `app/src/test/resources/` were written by the conversion run's `scripts/jvm_resources.py`
(the `tokenizers` library on `tokenizer.json`, NumPy's float16 → float32 cast, the oracle).
