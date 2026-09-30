# Julia-1 validation data

Model files and captured fixtures are external to the module. Building an APK does not need them.
Host parity tests require a local bundle named by `JULIA1_TEST_DATA`; they must not resolve data
from another checkout or download it during a test.

```text
validation-data/
  host_assets/
    tokenizer.json                    # the model repository's tokenizer.json
    julia1_token_table_fp16.bin       # the model repository's float16 table
  fixtures/
    gate_requests.json                # 2,100 oracle requests: request, ids, markers, qtype, logits
    decoder_reference.json            # probabilities and answers from the Python arithmetic
    tokenizer_stress.json             # 300 strings encoded with Hugging Face tokenizers
    fp16_conversion_reference.bin     # all 65,536 binary16 patterns as NumPy float32
    embedding_lookup_reference.json   # three padded rows gathered by NumPy (+ .bin)
    embedding_lookup_reference.bin
    gate_rows_s512.json               # optional: debug device gate, 706 captured rows
```

Run from the module root after supplying this bundle:

```bash
export JULIA1_TEST_DATA="$PWD/validation-data"
./gradlew --no-daemon :app:testDebugUnitTest
```

`JULIA1_TEST_RESULTS` optionally selects a report directory. A missing or incomplete bundle must
not be counted as a passing parity gate. The graphs are not used by the JVM tests.

## Captures and checks

The oracle is the author's runtime (`engine.logits`, CPU FP32, torch 2.14.0, transformers 5.0.0)
on the 2,000 questions of the LocalLLaMA/typed-decisions test (revision
`c76749ec58bd8c3d2ea706b31c333a9059c38f90`) and the 100 requests of `parity-cases.json` in
`SupersonicLabs/Julia-1-ONNX` (revision `82a2fadf`). Each row records the request, the ids,
marker positions and question type of the reference host, and the raw marker logits.

| Input | Content | Required assertion |
|---|---|---|
| `gate_requests.json` | 2,100 requests with reference ids, markers, qtype | 2,065/2,065 identical at S512, 35 rejected; 2,100/2,100 at S1024 |
| `decoder_reference.json` | Python softmax, choice/score/noul answers of the same rows | every probability within 1e-12, same answers |
| `tokenizer_stress.json` | 300 strings from the Laya sample's stress corpus (same tokenizer file) | 300/300 exact token ID lists |
| `fp16_conversion_reference.bin` | NumPy float16 to float32 for all bit patterns | 65,536/65,536 bit-exact |
| `embedding_lookup_reference.*` | NumPy lookup of three requests padded to 512 | 589,824/589,824 values bit-exact |

## Debug device rows

`gate_rows_s512.json` holds the 706 rows of the conversion gate (ids, markers, qtype): all 306
boundary rows, where the reference's top probability is below 0.9, plus 400 others. Install it with
`scripts/install_to_device.sh --fixtures DIR` next to `gate_requests.json`; the installer places
both in the app's private `files/fixtures/`. The `gate=true` intent entry exists only in the debug
source set; release builds launch the product UI. The report's `encoding` block counts the
on-device ids against the reference host, and its `rows` carry the marker logits the host compares
with the author's runtime (`conversion/device_compare.py` in the model repository).
