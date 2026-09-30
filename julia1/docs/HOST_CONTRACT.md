# Julia-1 host contract (Android)

This is the Android integration summary for `litert-community/Julia-1-LiteRT`
(revision `b92a0d21eb1aedad6afdc815e027a47896d16eb3`), the LiteRT conversion of
`SupersonicLabs/Julia-1` at revision `a85b127321d580d65176c89ced8273f305745d85`. The Python
reference host `julia_litert.py` in that repository is the executable contract; the Kotlin code in
`app/src/main/java/com/julia1/` ports it.

## Tokenizer and sequence

`JuliaTokenizer` loads the checkpoint's `tokenizer.json` (256,000 entries, Gemma-style BPE with
Metaspace and byte fallback; the same file as the multilingual Laya sample). Text fragments are
encoded without special tokens; the builder inserts `<bos>` **2**, `<eos>` **1** and `<mask>` **4**
itself. PAD is **0**.

One request is one question (`choice`, `score` or `noul`) about one state. Following
`sequence()` in the author's `julia/data.py` with strict encoding:

```text
<bos> "{type} question: {instructions}" <eos> <mask> " option0" <mask> " option1" ... <eos> state <eos>
```

- Options are the criteria descriptions themselves: the mapping values for `choice`, the rubric
  strings for `score`, `[criteria.false, criteria.true]` for `noul`, or the literal `false` and
  `true` when a `noul` question has no criteria. No `level i:` or `false:` prefixes (unlike Laya).
- A request has 2 to 20 options, each at most 48 tokens after its marker. A `noul` request has
  two, ordered [false, true].
- A text state passes through; a map or list state is serialized like Python
  `json.dumps(state, ensure_ascii=False)` (insertion order, `", "` and `": "` separators).
- Strict encoding: a request that would be truncated (option over 48 tokens, question plus
  options over the head budget, whole sequence over the window) raises `EncodingException`.
  Nothing is cut, so the ids do not depend on the window when the request fits. The literal
  `<mask>` in any text is rejected.
- The head budget is `min(512, window - 5)`: 507 for the S512 graph, 512 for S1024.

## Graph inputs and outputs

One `serving_default` signature per graph; buffers are addressed by name. All tensors are float32.
N is 512 (`julia1_s512_fp32.tflite`) or 1,024 (`julia1_s1024_fp32.tflite`).

| Direction | Name | Shape |
|---|---|---|
| Input | `inputs_embeds` | `[1,N,384]` |
| Input | `attention_mask` | `[1,N]` |
| Input | `qtype_onehot` | `[1,3]` |
| Output | `token_logits` | `[1,N]` |

`JuliaEmbeddings` maps the little-endian float16 `[256000,384]` table read-only and converts the
gathered rows to float32. Right padding gathers token **0**, its real row, not a zero vector.
`attention_mask` is 1 for real positions and 0 for padding. `qtype_onehot` order is choice, score,
noul. The GPU path requests `GpuOptions(precision = FP32)`: the answers change under fp16 (see the
model card), so the app offers GPU FP32 and CPU only.

## Decoder

Read `token_logits` at the marker positions and apply softmax at temperature 1 in float64
(`exp(z - max) / sum`, sequential sum, as `julia/typed.py` does). The checkpoint ships no
calibration. `choice` is the key of the highest probability (first maximum on ties), `score` is
the expected zero-based rubric index Σ i·pᵢ without rounding, `noul` is P(true). Choice and score
also report `max_probability`. Probabilities are full float64 values; the screen rounds only for
display.

## Executable checks

JVM parity tests compare the tokenizer with 300 captured encodings, the builder with the
reference host's ids, markers and question types on 2,100 oracle requests (2,065 fit 512 tokens;
35 must be rejected at S512 and accepted at S1024), the decoder with the Python arithmetic on the
same rows, the float16 conversion on all 65,536 bit patterns, and the table lookup on three padded
rows. Fixture layout is in [TEST_DATA.md](../scripts/TEST_DATA.md). No JVM result substitutes for
the on-device gate.
