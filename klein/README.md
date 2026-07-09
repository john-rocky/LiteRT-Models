# FLUX.2-klein-4B on LiteRT GPU

Text-to-image with **FLUX.2 [klein] 4B** (Apache-2.0) running end-to-end on a phone
GPU through LiteRT `CompiledModel` (ML Drift). Black Forest Labs ship klein as the
model that "runs on consumer GPUs with as little as 13 GB VRAM" — this module runs
the same weights on a Pixel 8a's Mali-G610, which has no VRAM at all.

Nothing runs on the CPU: the Qwen3-4B text encoder, the 4B rectified-flow DiT and
the VAE decoder are all `Accelerator.GPU` graphs.

![generated on a Pixel 8a](docs/pixel8a_generated.png)

*"a red apple on a wooden table, studio lighting" — 4 steps, 256x256, generated on
a Pixel 8a. PSNR 36.8 dB against the fp32 `diffusers` pipeline.*

| | |
|---|---|
| Model | `black-forest-labs/FLUX.2-klein-4B` (Apache-2.0) |
| Text encoder | Qwen3-4B, hidden states from layers 9 / 18 / 27 |
| DiT | 5 double-stream + 20 single-stream blocks, dim 3072, 24 heads |
| Steps | 4 (step-wise distilled — **no classifier-free guidance**) |
| Output | 256 x 256 |
| Deploy graphs | 12 x int8, 6.2 GB total, largest 912 MB |
| Device | Pixel 8a (Mali-G610), 306 s: encoder 35 s, ~70 s/step, VAE 7 s |
| Quality | PSNR 36.8 dB / corr 0.9987 vs the fp32 `diffusers` pipeline |

## How a 6.2 GB model fits in a phone

`CompiledModel` cannot load a graph over 2 GB (flatbuffer limit) and the GPU budget
is well under that anyway. So the model is **split into chunks that are resident one
at a time** — the same recipe as the Z-Image module next door:

```
  ke_enc0/1/2   Qwen3 layers 1-9 / 10-18 / 19-27          912 MB each
  [host]        interleave the three taps -> [1,512,7680]
  kc_prep       x/context embedders + 3 modulation FCs     166 MB
  kc_double0/1  3 + 2 double-stream blocks                 739 / 492 MB
  [host]        joint = cat([encoder, image], dim=1)
  kc_single0-3  5 single-stream blocks each                615 MB each
  kc_final      adaLN-continuous norm + proj_out            19 MB
  kv_vae        VAE decoder                                 50 MB
```

Chunks of roughly 500 MB - 1 GB are the sweet spot. A 1.8 GB chunk does not merely
load slower, it sends the ML Drift shader compiler into a pathological blowup
(>10 min for one graph).

Everything a GPU graph cannot express is precomputed on the host and pushed as
`.bin`: tokenization, `embed_tokens`, the causal+padding mask, both rotary tables,
the flow-matching sigmas, and the two tail permutations. See
[`scripts/gen_prep_klein.py`](scripts/gen_prep_klein.py).

The tail is worth a note. `unpack by position ids` and the 2x2 `unpatchify` are
**pure permutations** of the flat buffer, so they are recovered exactly by pushing
an `arange` probe through the stock pipeline functions and shipped as int32 gather
maps — no reimplementation, no chance of an index bug. Only the per-channel batch-norm
denormalization between them is real arithmetic.

## GPU-compatibility work

Beyond the chunking, the graph needed these rewrites. All are exact — no
accuracy-changing approximations.

**Found by the desktop op checker:**

- **RoPE without `GATHER_ND`.** FLUX interleaves even/odd channels. Baking that
  permutation into the rows of `to_q` / `to_k` (and the fused `to_qkv_mlp_proj`)
  turns it into a contiguous half-split rotation. Exact, because `q · k` is
  invariant to a permutation applied to both.
- **GQA `repeat_kv` builds a rank-5 tensor.** `x[:, :, None].expand(...)` exceeds
  the 4D limit.
- **`amax` on a 4D tensor** trips litert-torch's NHWC layout pass
  (`NHWC node rewriter not found: amax`) — reshape 4D→3D around the norm.
- **`GroupNorm` in the VAE** → `ManualGroupNormND` (the mid-block attention
  normalizes a 3D tensor).

**Found only on the device** — these compile and run, and the checker is clean:

- **`BROADCAST_TO` is rejected outright** ("Operation is not supported"), and
  `Tensor.expand` lowers to it. So the 4D `repeat_kv` rewrite still fails. It has to
  become a `CONCATENATION`:
  `torch.cat([k[:, i:i+1] for i in range(n_kv) for _ in range(rep)], dim=1)`.
- **A broadcast `ADD` whose left operand is a `BATCH_MATMUL` result is silently
  miscomputed.** That is exactly `softmax(q @ kᵀ * scale + mask[1,1,S,S])`, i.e. every
  masked attention. There is no error and no NaN: the probabilities still sum to 1 and
  still respect the causal and padding masks, but the logits are wrong and the image
  comes out as structured garbage. **Fix: hand the mask in pre-expanded to
  `[1, num_heads, S, S]`.** Zero extra ops, formula unchanged.

  The diagnostic signature is worth remembering: *token 0 is bit-exact and every later
  token is wrong.* Token 0 attends only to key 0, so it is the one row insensitive to
  key mixing.
- **fp16 overflow** in RMSNorm/LayerNorm → max-normalized safe variants, and the
  runtime is asked for `GpuOptions(precision = FP32)`: the modulated (adaLN) blocks
  return 100% NaN in fp16.

Quantization is **INTEGER-compute int8** (`full_dynamic_recipe(INT8, CHANNELWISE)`);
weight-only-FLOAT int4/int8 hangs the GPU compile. Note that the desktop int8 gate is
*pessimistic*: the same graphs score 36.4 dB on the host CPU int8 path and 44.1 dB on
the device, so do not reject a recipe on desktop emulation alone.

## Build and run

```bash
# 1. Convert (needs ~40 GB RAM; writes twelve .tflite next to the scripts)
python scripts/build_klein_enc.py --export        # ke_enc0/1/2
python scripts/chunked_export_klein.py --export   # kc_prep, kc_double*, kc_single*, kc_final
python scripts/vae_deploy_klein.py --export       # kv_vae

# 2. Host prep: fp32 reference image + every .bin the device needs
python scripts/gen_prep_klein.py

# 3. Prove the device loop on the host, driven only by the .tflite graphs
python scripts/gen_verify_klein.py
#    -> [gate] tflite-only loop vs fp32 pipeline: PSNR 33.7 dB  corr 0.99583

# 4. Ship it
./gradlew :app:installDebug
./install_to_device.sh <graphs_dir> <graphs_dir>/klein_bins
```

The app generates on launch and writes `generated.png` and `gen_log.txt` to its
external files dir. Model files are never committed.

`scripts/probe_enc_layer.py` exports one encoder layer as an fp32 graph with taps at
every stage — that is what localized the masked-attention bug above, and it is the
tool to reach for when a device result is wrong but nothing errors.

## Notes

- The prompt is baked into `scripts/gen_prep_klein.py`; changing it means re-running
  step 2 (the tokenizer and Qwen3 embedding table stay on the host).
- Each graph is recompiled every time it is loaded:
  `GpuOptions(serializeProgramCache = true)` aborts the ML Drift OpenCL delegate on
  this runtime, so there is no program cache to reuse. That, not the arithmetic,
  dominates wall-clock.
- The `Environment` is created once and shared. Creating one per `CompiledModel` leaks
  the OpenCL context and aborts the process after roughly twenty compiles.
- `TensorBuffer.close()` after every run; leaked native buffers cumulatively OOM the
  process part-way through a generation.
