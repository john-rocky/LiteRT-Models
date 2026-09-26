# Optional JVM reference data

The JVM tests compare the Kotlin host with the official gliner2 2.0.0 Python path. The reference
data is not distributed with the source. Without `gliner.fixtures`, or with a missing directory or
file, JUnit reports an assumption skip. Skips are not parity evidence.

```bash
./gradlew :app:testDebugUnitTest -Pgliner.fixtures=/path/to/data
# Equivalently: -Dgliner.fixtures=/path/to/data
```

Reports go to `app/build/reports/parity/`, never to the data directory.

```text
data/
  fixtures/oracle_fp32.json            # 361 official classify_text results with captured input_ids,
                                       # schema_special_indices, label_positions, task_results
  host_assets/tokenizer.json           # or cache/hf/hub/models--fastino--GLiNER2.5-Decide/
  host_assets/word_embeddings_fp16.bin #    snapshots/7ee5da4c…/tokenizer.json and
                                       #    exports/host_assets/word_embeddings_fp16.bin
  results/device/inputs_s128/          # optional: device-gate input files + manifest.json
  results/device/inputs_s256/          #   (inputs_embeds / attention_mask / label_routing,
  results/device/inputs_s512/          #   little-endian float32), for DeviceInputsTest
```

| Test | Needs | Checks |
|---|---|---|
| `DecideInputsTest` | oracle, tokenizer | input_ids, attention length, `[P]`/`[L]` positions, smallest window and padded inputs of all 361 fixtures |
| `DecideDecoderTest` | oracle | oracle logits → decisions equal the official result, probabilities within 1e-6; threshold/tie rules |
| `SchemaTest` | oracle, tokenizer | schema token strings and IDs of the `prompt` (readme_18) and `{label: description}` (readme_20) examples |
| `EmbeddingTableTest` | float16 table | rows 0 and 128010 bit-identical to numpy's upcast; every float16 bit pattern |
| `TokenizerProbeTest` | tokenizer | edge strings (normalized `[UNK]`, whitespace, NFC, added tokens inside strings) against Python |
| `DeviceInputsTest` | oracle, tokenizer, table, device inputs | graph inputs byte-identical to the files the S26 gate consumed |
| `GateFixturesTest` | oracle, tokenizer | debug asset equals the oracle; every (fixture, window) input identical |
| `TaskEditorTest` | tokenizer (last case) | editor format, validation, bundled example fits s128 |

The test resources in `app/src/test/resources/` were written from Python by the conversion run's
`scripts/r4_jvm_resources.py` (captured official batches, numpy upcast, gliner2's runtime
tokenizer). The debug asset was written by `scripts/r4_gate_fixtures.py`.
