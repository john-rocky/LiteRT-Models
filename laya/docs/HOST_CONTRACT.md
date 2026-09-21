# Multilingual host contract

This is the Android integration summary for the pinned multilingual Laya checkpoint
`convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982` and upstream laya 0.3.4.
The complete Python host contract and executable reference are shipment artifacts
`HOST_CONTRACT.md` and `laya_host.py` in [litert-community/Laya-Multilingual-LiteRT](https://huggingface.co/litert-community/Laya-Multilingual-LiteRT). The app uses the embedding-input
graph interface below.

## Tokenizer and prompt

`LayaTokenizer` loads the complete checkpoint BPE JSON with 256,000 vocabulary entries. It
matches added tokens, replaces ASCII spaces with U+2581, applies Metaspace with always-prepend
and splitting, merges by rank, and falls back to UTF-8 byte tokens. Text fragments use
`addSpecialTokens = false`; the builder inserts CLS **2**, SEP **1**, and MASK **4** itself.
PAD is **0** and UNK is **3**. No extra Unicode normalization is applied.

Each call contains one question, with schema type `choice`, `score`, or `noul`. The builder
preserves question and option order, encodes the literal `type question: instructions` prefix,
and renders the upstream criteria without changing them. Options have one MASK marker followed
by at most 48 text tokens. With head budget 256, an option budget below 16 triggers the exact
upstream squeeze rule; the instruction slice keeps `max(8, remaining budget)` tokens.

The sequence is CLS + instruction + SEP + marked options + SEP + state + SEP. String state
passes through; map/list state uses Python-compatible insertion order, Unicode, JSON
booleans/null, comma-space and colon-space separators. Literal `<mask>` in text is replaced
by an ASCII space. State tokens are truncated on the right to fit the selected static window;
the final whole-sequence slice and marker filter follow upstream. A row that loses an option
marker is rejected before inference.

## Graph inputs and outputs

The app uses one `serving_default` signature per graph. Buffers are mapped by signature names
and runtime order. All inputs and outputs below are float32; N is 256 in the UI.

| Graph | Direction | Name | Shape |
|---|---|---|---|
| Main | Input | `inputs_embeds` | `[1,N,768]` |
| Main | Input | `attention_mask` | `[1,N]` |
| Main | Input | `qtype_onehot` | `[1,3]` |
| Main | Output | `token_logits` | `[1,N]` |
| Main | Output | `pooled_cls` | `[1,768]` |
| Action | Input | `pooled_cls` | `[1,768]` |
| Action | Input | `feats` | `[1,4]` |
| Action | Output | `act_logits` | `[1,2]` |

`LayaEmbeddings` maps the little-endian FP16 `[256000,768]` table read-only and converts gathered
rows to float32. Right padding gathers token **0**, including its actual vector; it does not
insert an all-zero vector. Attention is 1 for real sequence positions and 0 for padding.
Question-type order is choice, score, noul. The whole table's FP16 cast round-trip maximum
absolute error is 0. WFP16 storage changes only fully connected weights in the main graph.

## Decoder and calibration

Gather marker logits, then compute stable raw softmax. The action features are raw top-1
probability, top-1 minus top-2, normalized raw entropy using `k = max(K,2)` and a log floor
of `1e-9`, and `k/255`. Action probability is `softmax(act_logits)[0]`; it is uncalibrated.

Option temperature first uses `temperature_by_options` with bucket suffix `2`, `3-5`, `6-10`,
or `11+`, then `temperature[qtype]`. Its floor is `1e-3`. **Calibrated** loads the supplied JSON;
**T=1 (uncalibrated)** uses identity temperatures. The calibrated UI and T=1 conversion gates
are intentionally separate measurements.

Choice returns the first argmax label and ordered probabilities. Score returns the expected
zero-based index, all probabilities, and the original legend. Noul returns continuous true
probability. Choice/score confidence is normalized entropy confidence with a `1e-12` floor;
noul confidence is `max(pTrue, 1-pTrue)`. Values are rounded to four decimal places using
Python's ties-to-even rule on the represented binary double, after intermediate math.

## Executable checks

The host implementation is in `app/src/main/java/com/laya/`: `LayaTokenizer.kt`,
`LayaPromptBuilder.kt`, `LayaDecoder.kt`, `LayaCalibration.kt`, `LayaEmbeddings.kt`, and
`LayaEngine.kt`. JVM parity tests compare all captured IDs, marker positions, official and
calibrated answer dictionaries, state serialization, and embedding values. Fixture layout and
counts are documented in [TEST_DATA.md](../scripts/TEST_DATA.md). No JVM parity result substitutes
for the on-device tokenizer and numerical gate.
