# SmolVLA (lerobot/smolvla_base) on LiteRT

LiteRT export of the action path of [SmolVLA](https://huggingface.co/lerobot/smolvla_base), a
vision-language-action robot policy: one camera image, a task string and the robot state go in; a
chunk of 50 future actions comes out. The model is split into three fixed-shape `.tflite` graphs
built for the LiteRT `CompiledModel` GPU accelerator. A small numpy host loop glues them together
and reproduces lerobot 0.6.1's `SmolVLAPolicy.sample_actions`.

Status: converted and verified on a Mac (CompiledModel CPU and the Mac's LiteRT GPU accelerator)
and on a Galaxy S26 (CompiledModel GPU, LiteRT 2.2.0). On the S26 every op of all three graphs runs
on the GPU. A whole action chunk chained on the phone, with inputs prepared by the Python host,
matches lerobot's float32 output on the fixture input within 3.4e-2 at the default GPU precision
and within 1.4e-3 with FP32 GPU precision (see [Measured on a Galaxy S26](#measured-on-a-galaxy-s26)). The Android app code (tokenizer, letterbox,
masks and tables, the 10-step loop) is not written; `scripts/smolvla_host.py` is its reference.

```
image ──host letterbox──► [V] smolvla_vision ──► img_emb ─┐
task ──host tokenizer + fp16 embedding lookup──► lang_emb ─┼─► [P] smolvla_prefix (once) ─► k_all, v_all
state ──host normalize + zero-pad──────────────► state ────┘                                 │
noise ─► 10 x [E] smolvla_expert_step (Euler, dt = -0.1) ◄─── masks, RoPE and time tables ◄──┘
      ─► x[..., :6] ─► host un-normalize ─► 50 x 6 actions
```

## Files

| Path | What |
|---|---|
| `scripts/build_smolvla.py` | Builds the three graphs (fp32 and `_f16`), the host assets and `graph_contract.json`. |
| `scripts/smolvla_graphs.py` | The re-authored torch modules that get converted, plus the lerobot loaders. |
| `scripts/smolvla_host.py` | Host side in numpy: letterbox, tokenizer call, embedding lookup, masks, RoPE and time tables, Euler loop, (un)normalization. |
| `scripts/run_smolvla.py` | Command line: image + task + state → 50 x 6 actions, through the `.tflite` graphs. |
| `scripts/verify_reauthored.py` | Torch modules vs lerobot (float32), stage by stage. Writes the fixtures. |
| `scripts/verify_tflite.py` | `.tflite` (CompiledModel CPU, optionally this host's GPU) vs torch modules, then the full chain vs lerobot. |
| `scripts/litert_helpers.py` | CompiledModel runner, flatbuffer op/structure scan, fp16 weight cast, parity metrics, fixture I/O. |
| `out/` | Build outputs: graphs, host assets, fixtures, `build_log.json`, `parity_*.json`, `pip-freeze.txt`. `*.tflite` and `*.bin` are gitignored. |

## Graphs

All tensors are float32 with batch 1, and no tensor has rank above 4. Inputs are listed in signature
order, which is the order of `create_input_buffers(0)`. `S = 64 * num_cameras + 48 + 1` (113 for one
camera).

**V `smolvla_vision.tflite`**: SigLIP-B/16 at 512 px (12 layers, native tanh-GELU), then the
post LayerNorm, the x4 pixel-shuffle connector and a Linear from 12288 to 960. Run it once per camera.

| # | Input | Shape | Meaning |
|---|---|---|---|
| 0 | `image` | [1,3,512,512] | letterboxed RGB in [-1, 1] |
| 1 | `pos_embed` | [1,1024,768] | constant table from `vision_pos_embed.bin` (fed as an input on purpose) |
| out | `img_emb` | [1,64,960] | connector output, not yet scaled by sqrt(960) |

**P `smolvla_prefix.tflite`**: the 16 SmolVLM2 text layers lerobot keeps, run over
`[image tokens | 48 language tokens | state token]`.

| # | Input | Shape | Meaning |
|---|---|---|---|
| 0 | `img_emb` | [1,64,960] | V output (cameras concatenated on axis 1) |
| 1 | `lang_emb` | [1,48,960] | raw rows of `embed_tokens_f16.bin` for the 48 token ids, pads included |
| 2 | `state` | [1,32] | normalized state, zero-padded to 32 |
| 3 | `attn_bias` | [1,1,S,S] | 0 where lerobot's `make_att_2d_masks` allows attention, -30000 elsewhere |
| 4 | `rope_cos` | [1,1,S,32] | cos of `pos / 10000**(2i/64)`, `pos = cumsum(pad) - 1` |
| 5 | `rope_sin` | [1,1,S,32] | sin of the same angles |
| out 0 | `k_all` | [1,80,S,64] | post-RoPE keys of all 16 layers (layer l = channels 5l..5l+4) |
| out 1 | `v_all` | [1,80,S,64] | values of all 16 layers |

**E `smolvla_expert_step.tflite`**: one Euler velocity evaluation. The suffix embedding
(action_in_proj, time MLP) feeds the 16 action-expert layers; even layers self-attend over prefix +
suffix and odd layers cross-attend to the projected prefix K/V. Then the final norm and
`action_out_proj`.

| # | Input | Shape | Meaning |
|---|---|---|---|
| 0 | `x_t` | [1,50,32] | current Euler state (noise at step 0) |
| 1 | `time_emb` | [1,1,720] | `create_sinusoidal_pos_embedding(t, 720, 0.004, 4.0)`, float64 math then float32 |
| 2 | `k_all` | [1,80,S,64] | P output, the same for all 10 steps |
| 3 | `v_all` | [1,80,S,64] | P output, the same for all 10 steps |
| 4 | `bias_self` | [1,1,50,S+50] | prefix pad columns, then a causal 50 x 50 block (0 / -30000) |
| 5 | `bias_cross` | [1,1,50,S] | prefix pad columns (0 / -30000) |
| 6, 7 | `rope_cos_s`, `rope_sin_s` | [1,1,50,32] | positions `prefix_offset + i`, where `prefix_offset` = number of non-pad prefix tokens |
| 8, 9 | `rope_cos_c`, `rope_sin_c` | [1,1,50,32] | positions `0..49` (cross-attention queries) |
| out | `v_t` | [1,50,32] | velocity; the host does `x_t += -0.1 * v_t` |

`out/graph_contract.json` holds the same contract in machine-readable form. It also records the
host recipe step by step, the RoPE time scales, the Euler times and the tokenizer settings. The two
prefix outputs share a shape: `output_0` is `k_all` and `output_1` is `v_all`. In ai-edge-litert
2.2.0 the Python call is `get_output_buffer_requirements(output_index, signature_index)`, with the
output index first.

**Norms.** In V and P the RMSNorm/LayerNorm are computed in exact down-scaled forms (see
[Re-authoring](#re-authoring-all-exact-up-to-float-summation-order)), so the graphs are correct at
the GPU's default fp16 precision. With the plain formulas they are not: the prefix residual stream
reaches about 2,570 and `x * x` leaves the fp16 range. `--norms exact` builds those plain-formula
graphs as a reference (`*_exactnorm*.tflite`); they are correct on the CPU and with FP32 GPU
precision only.

### Host side

- **Image:** RGB `uint8 / 255` → lerobot `resize_with_pad` to 512 x 512 (bilinear,
  `align_corners=False`, no antialias, zero padding on the left and top) → `x * 2 - 1`.
- **Task:** append `"\n"` if missing, then tokenize with the SmolVLM2 tokenizer (GPT-2 byte-level
  BPE, `max_length=48`, `padding="max_length"`, right padding, truncation, no BOS, pad id 2).
  `lang_emb` is the float32 view of the fp16 table rows for all 48 ids.
- **Masks:** `pad = [1]*64N + lang_mask + [1]` and `att = [0]*(64N+48) + [1]`. Then
  `mask[i,j] = cumsum(att)[j] <= cumsum(att)[i] and pad[i] and pad[j]`, and
  `positions = cumsum(pad) - 1`.
- **Loop:** P runs once, then E runs 10 times with `t = float32(1.0 + step * -0.1)`.
  The actions are `x[..., :6]`.
- **(Un)normalization:** `(s - mean) / (std + 1e-8)` on the state and `x * std + mean` on the
  actions, but only when `norm_stats.json` has entries. For `lerobot/smolvla_base` both are `null`
  (identity), because that is what lerobot 0.6.1 applies to this checkpoint (see
  [Known limitations](#known-limitations)).

## Build and verify

```bash
~/.local/bin/uv venv ~/venvs/smolvla --python 3.12
~/.local/bin/uv pip install --python ~/venvs/smolvla/bin/python "torch==2.11.*" "torchvision==0.26.*" \
    "litert-torch==0.9.4" "lerobot==0.6.1" "transformers>=5.4,<5.6" num2words "accelerate>=1.14,<2" \
    safetensors huggingface_hub ai-edge-quantizer pillow
# exact versions used: out/pip-freeze.txt

# from the repo root
export KMP_DUPLICATE_LIB_OK=TRUE
PY=~/venvs/smolvla/bin/python
$PY smolvla/scripts/build_smolvla.py --stage all --subprocess --out smolvla/out   # ~70 s, one process per graph
$PY smolvla/scripts/verify_reauthored.py --out smolvla/out     # torch vs lerobot; writes out/fixtures/
$PY smolvla/scripts/verify_tflite.py --out smolvla/out --gpu   # tflite (CPU, and this host's LiteRT GPU) vs torch vs lerobot
$PY smolvla/scripts/run_smolvla.py --image tipsv2/scripts/test.jpg \
    --task "pick up the red cube and place it in the box" --state 0.25 -0.6 0.9 -0.3 0.45 -1.1 --seed 0
$PY smolvla/scripts/build_smolvla.py --stage scan --out smolvla/out            # flatbuffer checks only

# reference graphs with the plain norm formulas (see "Norms" above)
$PY smolvla/scripts/build_smolvla.py --stage all --subprocess --norms exact --out smolvla/out
$PY smolvla/scripts/verify_tflite.py --out smolvla/out --gpu --norms exact
```

The first run downloads `lerobot/smolvla_base` (907 MB) and `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
(lerobot loads its processor and weights) into the Hugging Face cache. Two builds of the same code
produced byte-identical graphs, assets and fixtures.

`--repo <checkpoint>` builds from another lerobot SmolVLA checkpoint (hub id or local directory),
for example a fine-tune, and writes that checkpoint's normalization stats into `norm_stats.json`.
Only `lerobot/smolvla_base` has been verified.

### Fixtures for a device runner

`out/fixtures/shapes.json` indexes raw little-endian float32 files, one per graph input, in signature
order (`vision/in_00_image.bin`, ...). It also lists the expected outputs of the torch modules on
those exact inputs (`out_00_*.bin`). The `chain/` group holds the noise and, for each of the 10 Euler
steps, the `x_t` and `time_emb` inputs and the torch and lerobot `v_t` outputs. It also holds the
final actions from lerobot and from the torch chain. The fixture uses `tipsv2/scripts/test.jpg`
(1546 x 1213, so the letterbox pads 111 rows at the top), the task above, state
`[0.25, -0.6, 0.9, -0.3, 0.45, -1.1]` and `torch.manual_seed(0)` noise.

## Measured on a Galaxy S26

Galaxy S26 (SM-S942Q, Snapdragon SM8850, Android 16, 10.9 GiB of RAM visible to Android) with LiteRT
`com.google.ai.edge.litert:litert:2.2.0`: `CompiledModel` on `Accelerator.GPU` at the default
precision or with `GpuOptions(precision = FP32)`, and on `Accelerator.CPU`. The runs use the
`parity` and `chain` instrumentation tests of this repo's `npubench/` app, fed the fixture tensors
above. Both tests are in commit
[bdfa02b](https://github.com/john-rocky/LiteRT-Models/commit/bdfa02bbbf6844fef6ebb115d3d89bd4bb1cb810)
on the `klein-sample` branch, which is not merged into `main` yet:

- `parity` runs one graph, 5 warm-up runs, then the median of 10 timed runs, each including the
  readback of every output. Its output dumps are compared with the torch module on the same inputs.
- `chain` loads the three graphs into one process and runs whole chunks the way an app does:
  V → P → 10 × E, each graph fed the previous graph's output through the host. It reports the median
  of 5 chunks after 2 warm-up chunks, and the process memory.

Thermal status was NONE before and after every run (thermal headroom 0.58 to 0.75). The test
process ran in the foreground cpuset (cores 0-5), so the CPU figures do not use the two prime cores.

Every graph file, default and reference, fp32 and fp16 weights, is fully delegated. The native log
reads `Replacing N out of N node(s) with delegate (LITERT_CL) node, yielding 1 partitions` for all
ten files at the default precision, and for the three default fp16-weight files with FP32 precision.

### Whole chunk (image + task + state → 50 x 6 actions)

Default graphs with fp16 weights (692 MB of files), three graphs resident in one process, fed the
fixture inputs that the Python host prepared on the Mac; the 10-step loop runs in the test's Kotlin
code. The difference is against lerobot float32 `sample_actions` on the same inputs, over the
50 x 6 actions (reference values from -0.75 to 1.70). Memory is the process PSS (in GB = 10^9 bytes)
after the timed chunks; the part in parentheses is the GPU (graphics) memory.

| GPU precision | Chunk time (median) | V / P / 10 x E | Memory (PSS, of which GPU) | Max abs diff | corr |
|---|---|---|---|---|---|
| default | 207 ms | 71 / 23 / 112 ms | 2.4 GB (0.83 GB) | 3.4e-2 | 0.99988 |
| FP32 | 352 ms | 137 / 34 / 181 ms | 3.8 GB (1.58 GB) | 1.4e-3 | 0.9999996 |

The 3.4e-2 comes from the GPU's default reduced-precision arithmetic: with FP32 precision the same
files give 1.4e-3, which is the fp16 weight rounding (the phone's CPU and a desktop CPU give 1.4e-3
too). P includes moving `img_emb` in and the K/V out and
into the expert's input buffers. Loading (compiling) the three graphs took 4.3 s (default) and 4.6 s
(FP32). Two runs of the same chunk, one in this process and one as ten separate `parity` runs,
gave bit-identical actions.

The same chunk with the other file sets (per-graph `parity` runs chained through the host):

| Files | GPU precision | Max abs diff | corr |
|---|---|---|---|
| default graphs, fp32 weights | default | 2.1e-2 | 0.99989 |
| reference `*_exactnorm`, fp32 weights | FP32 | 7.9e-6 | 1.000000000 |
| reference `*_exactnorm`, fp32 weights | default | 2.26 (wrong) | 0.59 |

### Per graph (`parity`)

Difference = device output vs the torch module on the same inputs (prefix: non-pad tokens). The
fp16-weight rows include the weight rounding (see [Known limitations](#known-limitations)).

| Graph | Weights | GPU, default precision | GPU, FP32 precision | CPU |
|---|---|---|---|---|
| V vision | fp16 | 71 ms, corr 0.99968, max abs diff 3.5 (values up to 71) | 138 ms, 1.7e-4 | - |
| V vision, reference `_exactnorm` | fp32 | 84 ms, wrong (corr 0.93) | 165 ms, 2.3e-4 | 1028 ms, 1.6e-4 |
| P prefix | fp16 | 19 ms, K corr 0.99998, max abs diff 0.47 (values up to 17); V 0.99997, 0.14 | 31 ms, 1.2e-3 | - |
| P prefix, reference `_exactnorm` | fp32 | 19 ms, wrong (corr 0.40) | 29.5 ms, 1.6e-5 | 122 ms, 2.0e-5 |
| E expert step | fp16 | 10.0 ms, corr 0.99999, max abs diff 3.3e-2 (values up to 4) | 17.2 ms, 5.8e-4 | 44.5 ms, 5.8e-4 |

## Measured on the Mac (ai-edge-litert 2.2.0)

Reference: lerobot 0.6.1 `SmolVLAPolicy` with every module upcast to float32 (`policy.model.float()`),
on CPU, with the same image, task, state and noise.

### Per graph (CompiledModel CPU, default graphs)

| Graph | Ops | Max rank | Banned ops | fp32 / fp16 size | tflite vs torch, fp32 (corr, max abs diff) | tflite vs torch, fp16 weights |
|---|---|---|---|---|---|---|
| V vision | 814 | 4 | none | 390.0 / 195.4 MB | 1.000000000, 1.3e-4 (max abs value 71.0) | 1.000000000, 1.1e-4 |
| P prefix | 1195 | 4 | none | 592.8 / 296.6 MB | k 2.0e-5, v 7.7e-6 (non-pad tokens) | k 1.2e-3, v 6.1e-4 |
| E expert step | 978 | 4 | none | 399.6 / 199.9 MB | 1.000000000, 2.9e-6 (max abs value 4.0) | 0.999999990, 5.8e-4 |

The reference `*_exactnorm` graphs have 566 (V) and 947 (P) ops and the same sizes. The banned-op
list checked is GATHER(_ND), SELECT(_V2), (NOT_)EQUAL, GREATER, LESS, TOPK_V2, CAST, PACK, SPLIT,
BROADCAST_TO, CUMSUM, TRANSPOSE_CONV, EMBEDDING_LOOKUP, CUSTOM and Flex ops. None of them appear in
any graph. The scan also found none of the following: ops whose inputs are all constants, constants
over 4096 elements that feed non-weight operands, or graph outputs that are also consumed inside the
graph. `out/build_log.json` has the full op lists.

### End to end (host + V → P → 10 x E), same inputs as lerobot `sample_actions`

Max abs diff over the 50 x 6 actions:

| Configuration | default graphs | reference `*_exactnorm` graphs |
|---|---|---|
| CPU, fp32 weights | 5.0e-6 | 3.5e-6 |
| CPU, fp16 weights | 1.4e-3 | 1.2e-3 |
| Mac GPU (Metal), fp32 weights, `enforce_f32` | 3.9e-6 | 9.2e-6 |
| Mac GPU (Metal), fp32 weights, default precision | 2.9e-2 (corr 0.99990) | 2.27, wrong (corr 0.59) |

A second input checked with host tables rebuilt from lerobot's own functions (a 1280 x 720 frame, a
9-token task, raw state values, another noise seed) gives 1.9e-5 (fp32 weights) and 1.7e-3 (fp16
weights) on the CPU, for actions up to 8.3.

The differences are in lerobot's postprocessor units, which for smolvla_base equal the model's
normalized units (the postprocessor is identity).

The torch re-authoring on its own (`verify_reauthored.py`, fed lerobot's own tensors) matches as follows:
- prefix K/V: bit-exact on every non-pad token in all 16 layers;
- `img_emb`: max abs diff 1.2e-4. SigLIP runs through SDPA in lerobot and eager attention here;
- expert step 0: max abs diff 2.6e-6;
- whole torch chain: final actions within 3.7e-6.

### fp16 range (real input, float32 torch)

| Graph | max abs value in the residual stream | Norm inputs with an element above 255.9 in magnitude (its square overflows fp16) | Largest mean(x²) per token |
|---|---|---|---|
| V vision | 578 (output of layer 11) | 1 of 25 LayerNorms (`post_layernorm`) | 779 |
| P prefix | 2200 to 2568 (image tokens, from `img_emb * sqrt(960)`); language tokens up to 950 | all 31 RMSNorms | 5.7e4 |
| E expert | 11.2 | 0 of 33 | 4.7 |

### Latency (context only; median of 5, this Mac)

| | V | P | E per step | full chain incl. host |
|---|---|---|---|---|
| CPU, 8 threads, fp32 graphs | 101 ms | 25 ms | 15.5 ms | 0.3 s |
| Mac GPU (Metal), default precision | 22 ms | 7.5 ms | 5.0 ms | 0.1 s |
| Mac GPU (Metal), `enforce_f32` | 25 ms | 8.5 ms | 5.9 ms | 0.1 s |

## Re-authoring (all exact up to float summation order)

- `apply_rope` (`empty_like` plus slice assignment) is rewritten as
  `cat([x1*cos - x2*sin, x2*cos + x1*sin], -1)`. The host tables use lerobot's own formula (base
  10 000, not the HF config's `rope_theta`).
- GQA (15 query heads, 5 KV heads) uses no expand and no 5D tensor. `q [1,15,S,64]` is reshaped to
  `[1,5,3S,64]`. The additive bias is tiled to `[1,5,3S,K]` with CONCATENATION.
- The attention products in V, P and E are `torch.einsum`, which litert-torch lowers to rank-4
  BATCH_MATMULs (`[1,5,3S,64] x [1,5,K,64]` with `adj_y`, then `[1,5,3S,K] x [1,5,K,64]`).
  `torch.matmul` would become rank-3 `[H,N,d]` BATCH_MATMULs, a batch-collapsed attention form that
  has silently miscomputed on ML Drift (Mali) in other graphs, and the bias ADD would mix ranks 3
  and 4.
- `torch.where(mask, w, finfo.min)` becomes `w + bias` with bias -30000. The softmax is identical
  for every row with at least one visible key. Only pad-token rows differ, and nothing downstream
  reads them.
- `cat([action_emb, time_emb.expand]) @ W^T` is split into `W_a @ action_emb + (W_t @ time_emb + b)`.
  The suffix MLP runs as 1x1 convs, so every operand stays 4D: as plain Linears, the converter
  flattens them to `[50,720]` and mixes ranks in broadcast ADD/MUL ops.
- RMSNorm in V and P: `safe_rms` with `m = max(1, max|x|)`,
  `(x/m) * rsqrt(mean((x/m)²) + eps/m²)`. LayerNorm: `d * rsqrt(mean(d²) + eps/S²)` with
  `d = x/S - mean(x/S)` and `S = max(1, max|x|/8)`; the variance is never scaled back up by `S²`,
  which can itself overflow fp16. Against the plain formulas they differ by at most 9.9e-5 (V) and
  4.6e-5 (P) in torch.
- Prefix layer 15 computes only its K/V; its attention output and MLP are never read.
- The vision graph follows the zoo's SmolVLM recipe: `padding=0` patch conv, fixed `arange(1024)`
  positions (here a runtime `pos_embed` input), eager attention and 12 native GELU ops with
  `approximate=True`.

## Known limitations

- **Memory on an 8 GB phone is not measured.** On the S26 (10.9 GiB visible to Android) the process
  holding the three graphs used 2.4 GB (PSS) at the default precision and 3.8 GB with FP32 precision.
- **fp16 GPU precision** costs up to 3.4e-2 on the actions (see above); FP32 precision brings that to
  the fp16 weight rounding (1.4e-3) for about 1.7x the time and 1.6x the memory.
- **lerobot applies no normalization for `lerobot/smolvla_base`.** Its processor files store stats
  under keys like `so100.buffer.action`. lerobot looks up `observation.state` and `action` exactly
  (`rsplit(".", 1)`), so both steps are identity. This was checked by running the pipelines:
  `postprocessor(a) == a`. `norm_stats.json` records exactly that. A fine-tune saved by lerobot
  carries the proper keys, and `build_smolvla.py --repo <checkpoint> --stage assets` picks them up.
  That path was not tested with a real fine-tune.
- **fp16-weight variants are not bf16-exact.** The converter folds the RMSNorm scales of the plain
  formulas into the following FC weights, and the action/state projections are float32 in the
  checkpoint, so casting to fp16 rounds those weights. End-to-end error on the CPU is 1.4e-3.
- **The embedding table is fp16.** 1803 of 47.3M entries are not exactly representable (fp16
  subnormals, max error 3.0e-8); the fixture's 12 tokens are all exact.
- **Host RoPE tables** are computed in float64 and rounded, so they differ from torch's float32
  `sin`/`cos` by at most 6.0e-8. The time embedding and the masks match lerobot bit-exactly.
- **`--num-cameras N > 1`** is implemented, but it was only smoke-tested at torch level (N = 3,
  final actions within 8.5e-7 of lerobot). It was not converted or verified as `.tflite`. Image
  padding masks (`*_padding_mask`, `empty_cameras`) are not supported.
- **Pad-token K/V entries differ from lerobot's** by design (see the masks note above). Compare
  K/V only at non-pad positions; `out/fixtures/prefix/valid_token_index.bin` lists them.
- **The tokenizer runs on the host** with `transformers`. An Android port needs a GPT-2 byte-level
  BPE implementation of `tokenizer.json`.

## Not measured

- Phones other than the Galaxy S26: an 8 GB phone, and GPUs other than Adreno (Mali in particular).
- The Android app side: the tokenizer, letterbox, masks and tables and the loop in Kotlin.
- The LiteRT 2.1.x Android runtimes. All device numbers use LiteRT 2.2.0.
- Task success on a robot, closed-loop behavior, and fine-tuned checkpoints.
