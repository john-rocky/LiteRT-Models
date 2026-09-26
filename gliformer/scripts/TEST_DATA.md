# JVM and device parity data

The debug source set contains two JSON assets totaling **289,090 bytes**: `corpus.json` and
`unicode.json`. They contain the fixed inputs, official fp32 oracle spans/scores, Python token
captures and character maps. No graph, embedding table, captured float tensor or logit binary is
bundled in any APK. Benchmark and release contain no fixture assets.

The 80 rows comprise ten short examples and a 70-input corpus, with 70 unique texts overall.
There are 70 selected s128 rows (60 unique), five s256 rows and five s512 rows. A separate string
exercises non-BMP characters and Python code-point offsets. Original captures came from
GLiFormer 0.1.2 processing with GLiNER 0.2.29, the published tokenizer and checkpoint
`knowledgator/gliformer-large-v1` at `d0a4e53d09cebe6bc963dd9be319d4279084bb2d`.
Machine-specific source paths are omitted; the original capture/oracle SHA-256 values remain.

## Generate an external bundle

Download the model repository as described in the main README. Use its pinned Python 3.12
runtime dependencies, including `ml_dtypes`, and an empty output directory. The generator uses
CPU CompiledModel, published wfp16 graphs, four threads and one window at a time. It verifies
the captures and oracle before accepting a result. It never replaces the official oracle with
Kotlin output and refuses to overwrite existing references.

```bash
MODEL_DIR="${MODEL_DIR:-models/GLiFormer-Large-NER-LiteRT}"
python3.12 -m venv .python-reference
.python-reference/bin/pip install -r "$MODEL_DIR/requirements-lock.txt"
PYTHONDONTWRITEBYTECODE=1 .python-reference/bin/python scripts/generate_test_data.py \
  --model-dir "$MODEL_DIR" --output test-data
./gradlew :app:testDebugUnitTest -Pgliformer.fixtures=test-data
```

The primary manifest, `references_fp16.json`, uses `HostRuntime(table="fp16")`: fp16 table rows
are upcast to float32 before the wfp16 graphs, matching the app default. Its 140 entries cover
all 80 selected inputs plus the 60 unique short inputs forced to s256. `references_fp32.json`
and `logits/manifest.json` retain the 80 fp32-table diagnostic entries;
`references_fp32_forced_s256.json` adds 60 matching-window diagnostics. Original validation used
Python LiteRT 2.1.6 on Mac CPU; the Android app uses LiteRT 2.2.0. Different runtimes may produce
small numerical differences; do not increase tolerances or relabel a failed comparison.

```text
test-data/
  tokenizer.json, corpus.json, unicode.json
  graph_inputs/manifest.json
  graph_inputs/<id>/{attention_mask,text_routing,parent_routing,label_routing,text_mask}.bin
  references_fp16.json, references_fp32.json, references_fp32_forced_s256.json
  logits/manifest.json, logits/*.bin
  logits_fp16/*.bin, logits_fp32_forced_s256/*.bin
```

Float files are contiguous little-endian float32. Graph-input entries record file, shape, bytes
and SHA-256; reference entries also include input ID, exact window, text capacity, finite status
and Python entities. s128 logits have shape `[1,1,48,15]`; s256/s512 use `[1,1,N,15]`.
`corpus.json` supplies captured input IDs, masks, first subtokens, marker positions, token-to-char
maps and official oracle entities. Tests compare every routing/mask value, span sets, output
ordering and scores; score tolerances are `1e-5` versus matched Python and `1e-3` versus oracle.

With the complete bundle, **15 JVM tests pass with zero skips**: tokenizer 80/80 plus 400 tensor
comparisons, original decoder 80/80, matched-fp16 decoder 140/140, Unicode and decoder branch
checks. Reports go only to `app/build/reports/parity/`. Without `gliformer.fixtures`, five external
data tests skip and ten self-contained tests run. Supplying a missing or incomplete directory
fails the affected tests. Skips never count as parity evidence.

## Install device references

Install the debug APK first, then supply the generated bundle to the same model installer:

```bash
./scripts/install_to_device.sh --serial "$ANDROID_SERIAL" --with-s256 \
  --fixtures test-data "$MODEL_DIR"
```

`package_gate_fixtures.py` verifies sizes and SHA-256 values, includes only the 400 routing/mask
tensors and referenced logits/manifests, and excludes embedding arrays. The installer stages this
archive and extracts it into private `files/gate_fixtures/`; the runner reads it there alongside
the small debug input assets. See the main README for gate commands and report retrieval.
