# Gate rows (device fixture)

The debug gate (`--ez gate true`) reads `files/fixtures/gate_rows_s<window>.json`: one row per public test decision
that fits the window, with the question, the state, the expected token ids and marker positions, and the reference
logits of the publisher's fp32 runtime. The rows are produced by `scripts/make_gate_rows.py` in the conversion
record (`conversion/` of the model repository) from the public typed-decisions test split; they are validation data,
not part of the app download. Install one with `scripts/install_to_device.sh --gate-rows gate_rows_s384.json`.
