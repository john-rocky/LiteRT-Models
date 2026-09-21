# Laya validation data

Large model files and captured fixtures are external to the module. Building an APK does not
need them. Host parity tests require a local validation bundle specified by `LAYA_TEST_DATA`;
they must not resolve data from another checkout or download it during a test.

```text
validation-data/
  host_assets/
    tokenizer.json
    laya_ml_calibration.json
    token_embeddings_fp16.bin
    token_embeddings.json
  fixtures/
    ml_fixtures.json
    ml_rows_s256.json
    ml_rows_s512.json
    tokenizer_stress.json
    calibrated_reference.json
    serialization_reference.json
    fp16_conversion_reference.bin
    embedding_lookup_reference.json
    embedding_lookup_reference.bin
    gate_rows_s256.json          # optional: debug device gate
```

Run from the module root after supplying this bundle:

```bash
export LAYA_TEST_DATA="$PWD/validation-data"
./gradlew --no-daemon :app:testDebugUnitTest
```

`LAYA_TEST_RESULTS` optionally selects a separate report directory. A missing or incomplete
bundle must not be counted as a passing parity gate. The model graphs are not used by JVM
host-parity tests.

## Captures and checks

The source checkpoint is `convaiinnovations/laya`, revision
`1c5edc17a7acd8701df6fc341c0d179f1c62c982`, multilingual branch. Captures came from the official
laya 0.3.4 builder and decoder. Each captured row records its fixture/question identity, window,
ordered question schema, unpadded sequence IDs, marker positions, question type, option count,
raw marker logits, raw action logits, and official answer dictionary at T=1.

| Input | Content | Required assertion |
|---|---|---|
| `ml_fixtures.json` | 44 synthetic EN/JA states and schemas | State/schema source for rebuilding captured rows |
| `ml_rows_s256.json`, `ml_rows_s512.json` | 201 question rows per window | 402/402 exact sequence IDs and marker positions |
| Same captured rows | Raw logits and action logits | 402/402 exact official four-decimal dictionaries at T=1 |
| `calibrated_reference.json` | NumPy host decode of the same tensors with shipped temperatures | 402/402 exact calibrated dictionaries |
| `tokenizer_stress.json` | 300 synthetic strings encoded with tokenizers 0.23.2 | 300/300 exact token ID lists |
| `serialization_reference.json` | 20 nested/string/Unicode/boolean/null states and Python `json.dumps` outputs | 20/20 exact strings |
| `embedding_lookup_reference.json` and `.bin` | NumPy FP16-table lookup converted to little-endian float32 for 402 padded rows | 118,554,624 values bit-exact, maximum absolute error 0 |
| `fp16_conversion_reference.bin` | All 65,536 half-float bit patterns converted with NumPy | Float32 bits match, including infinities and NaN payload behavior |

The stress corpus covers Japanese, English, mixed scripts, emoji, URLs, digits/units, repeated
whitespace, boundary whitespace, added tokens, accents, Korean, Chinese, Arabic, and empty text.
The embedding reference binary is **474,218,496 bytes**; its manifest pins the source table and
row IDs. Padded positions must gather token ID 0. Fixture generation uses the pinned tokenizer,
Python serialization, and NumPy host math; generating these references does not execute a model.

## Debug device rows

`gate_rows_s256.json` joins the 201 S256 captures to their source states and question schemas.
It retains IDs, markers, question type, option count, raw captures, and official T=1 answers so
the Android runner can tokenize and build each row independently before model inference.
The current file is **1,018,117 bytes**, SHA-256
`037bd3c83f02f5aade4c559984a55d31cd4c231af493aee5fb880a4af35a7aa8`.

The module does not bundle the gate file or raw calibration corpora. Supply the independently
prepared file with `scripts/install_to_device.sh --gate-rows FILE` alongside the model assets.
The installer places it in the app's private `files/fixtures/` directory. The `gate=true` intent
entry exists only in the debug source set; release builds launch the normal product UI.

Gate acceptance compares all 201 token sequences and marker lists exactly, all 81 choice/score
argmax results under the recorded tie rule, every option/noul/action probability within 0.01 of
the captured official dictionary, and finite outputs. GPU acceptance additionally requires
explicit FP32 precision and the runtime's full-residency, one-partition delegate evidence.
Per-row reports contain creation, tokenizer, embedding lookup, and write/run/read timings;
the first graph call is separate from the remaining 200 warm rows. Matching these captures
checks conversion and host parity, not classification accuracy on a held-out dataset.
