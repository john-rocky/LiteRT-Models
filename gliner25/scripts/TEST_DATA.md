# Optional JVM reference data

The source distribution includes small debug input/reference assets but does **not** distribute
captured Python batches, packed float32 outputs or decoder traces. All six JVM tests are optional:
without `gliner.fixtures`, or with a missing data directory, JUnit reports an assumption skip with
setup instructions. A Hugging Face download supplies `host_assets/`; it does not supply the full
parity bundle. Tests with unavailable captures also skip. Skips are not parity evidence.

```bash
DATA="$HOME/Downloads/GLiNER2.5-Small-LiteRT"
HF_HUB_DISABLE_XET=1 hf download litert-community/GLiNER2.5-Small-LiteRT --local-dir "$DATA"
./gradlew :app:testDebugUnitTest -Pgliner.fixtures="$DATA"
# Equivalently: -Dgliner.fixtures="$DATA"
```

With only the model download, the bundled-input comparison diagnostic and Unicode-offset test
can run; the other tests require the following additional data. Generated reports always go to
`app/build/reports/parity/`, never to the data directory.

```text
data/
  host_assets/                         # Unmodified HF download
  fixtures/
    captured/00.json ... 09.json       # Ten short Python batches
    f1/captured/00.json ... 69.json    # Seventy unique Python batches
    f1/oracle_fp32.json               # {"fixtures": [{"text": ..., "spans": [...]}]}
    packed/
      reference.json                 # {"entries": [...], "aliases": [...]}
      decoder_traces.json            # {"traces": [...]}
      s128/<input_id>.bin             # Little-endian float32, flattened packed output
      s256/<input_id>.bin
      s512/<input_id>.bin
      s128/short_00.bin ...           # Duplicate short inputs at each fitting window
```

## Regenerate with the published Python runtime

Use Python 3.12 in a separate environment and install the downloaded `requirements-lock.txt`
(including gliner2 2.0.0). The HF repository's `examples/run_example.py` demonstrates the complete
CPU path and provides a first check of your Python installation:

```bash
python3.12 -m venv .python-reference
.python-reference/bin/pip install -r "$DATA/requirements-lock.txt"
.python-reference/bin/python "$DATA/examples/run_example.py" \
  --assets "$DATA/host_assets" --model "$DATA/gliner25_small_s128_wfp16.tflite"
```

To reconstruct the full bundle, use these steps and JSON fields. This is a regeneration recipe,
not an automatically downloaded fixture set:

1. Read `app/src/debug/assets/gate_f1_fixtures.json` and `gate_fixtures.json`. Join a fixture's
   `text_parts` with an empty separator when `text` is absent. Preserve fixture order and the
   supplied official `spans`; do not replace official confidences with newly decoded values.
   Write the 70 text/span rows as `fixtures/f1/oracle_fp32.json`.
2. Import `HostRuntime` from the download's `host_assets/runtime/host_runtime.py`, construct
   `HostRuntime(DATA / "host_assets")`, and call `inputs, captured = host.prepare(text, seq=N)`.
   Execute every fitting N/T window: 128/48, 256/192, 512/384. Do not truncate. The 70 texts give
   195 pairs (60/65/70); the ten duplicate short texts add 30 pairs.
3. Serialize `captured["batch"]` tensors `input_ids`, `attention_mask`, `text_word_indices`,
   `text_word_mask`, `query_marker_indices`, `query_marker_mask` as objects with a `values` key
   containing `tensor.tolist()`. Include `text`, `start_mappings`, `end_mappings`, and
   `token_to_char = list(zip(batch.start_mappings[0], batch.end_mappings[0]))` in each capture.
4. Follow `examples/run_example.py`: compile each published **wfp16** graph with CPU
   `CompiledModel`, order its five inputs by the signature's `args_0` through `args_4` names,
   write float32 input buffers, run, then read the entire packed output. Its size is
   `1108*T + 4574` floats. Save `packed.astype('<f4').tofile(path)` and obtain Python spans with
   `host.decode(captured, packed, inputs)`. Flatten `result['entities']` by adding each label to
   its spans. Assert finiteness and compare label/start/end sets against the bundled reference.
5. `reference.json` needs 195 `entries`, each with `window`, `input_id`, `text`, `captured`
   (relative to the data directory), `packed` (relative to `fixtures/packed/`), `python_spans`,
   and `oracle_spans`. Each span has `label`, `start`, `end`, `confidence`. Add 30 `aliases` for
   the short copies, with `window`, canonical `input_id`, `captured`, and `packed`. Duplicate
   files must equal their canonical `sN/<input_id>.bin` byte-for-byte.
6. Register forward hooks on `host.model.boundary_head.shared_pool_builder` and
   `shared_pool_scorer` while decoding each saved packed output. For valid `pool.mask[0]` rows,
   retain `pool.indices[0]`, `pool.compat_logits[0]`, and the scorer's `output[0][0]` logits in
   their original order. Write `decoder_traces.json` with 195 `traces`, each containing `window`,
   `input_id`, and `candidates`: objects with `start`, `end`, `compatibility`, and `logits`.
7. Run the Gradle command above with the complete data directory and confirm six tests executed,
   zero skips/failures, 80 files / 70 unique inputs / 225 checks and 195 decoder pairs. The
   confidence tolerances are `1e-5` versus Python on the same packed bytes and `5e-3` versus the
   bundled fp32 reference. Different supported CPU runtimes may change the measured small error;
   do not regenerate reference spans from Kotlin or change tolerances to hide a mismatch.
