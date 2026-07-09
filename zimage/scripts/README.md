# Z-Image conversion + reference pipeline

Desktop-verified. These produce the three int8 LiteRT graphs and a runnable
prompt→image reference (`generate_full.py`) using only those graphs. **Python is
the ground truth for the app's host-prep.**

## Environment

- Python 3.10, `torch 2.12`, `litert-torch 0.10`, `ai-edge-litert 2.1.5`,
  `diffusers 0.38` (has `ZImageTransformer2DModel`), `transformers 5.12`.
- Weights are pulled from `Tongyi-MAI/Z-Image-Turbo` (Apache-2.0) on first run.

## Graphs (INTEGER-compute int8 = the GPU-runnable path)

Quantize via `litert_torch` **generative** recipe
`full_dynamic_recipe(weight_dtype=INT8, granularity=CHANNELWISE)` (INTEGER compute).
Do NOT use weight-only + FLOAT compute — it converts but hangs/overflows in the GPU
(WebGPU/Metal) delegate.

```
python export_int4_integer.py --bits 8      # zdit  (S3-DiT, 30 layers, pad-token fixed)
python vae_deploy.py                         # zvae  (AutoencoderKL decoder)
python qwen_enc.py --real --layers -1        # qwen_enc (Qwen3-4B, hidden_states[-2])
python generate_full.py                      # prompt -> image using ONLY the 3 graphs
```

## Conversion notes (the non-obvious bits, all learned the hard way)

- **S3-DiT is convertible only with a fixed-shape export wrapper** (`deploy_dit.py`):
  host does patchify / position-ids / RoPE / unpatchify; the graph is the block stack.
- **Axial RoPE**: `view_as_complex` + stride-2 slice → GATHER_ND (GPU-banned). Bake the
  even/odd de-interleave into the `to_q`/`to_k` weights → contiguous-slice half-split
  real RoPE, bit-exact (q·k is invariant to a shared channel permutation).
- **fp16-safe RMSNorm** via max-normalization; reshape 4D→3D for the `amax` (litert-torch
  NHWC layout pass rejects `amax` on 4D).
- **pad-token substitution** at padded positions must be in the graph as `x*(1-m)+pad*m`
  (SELECT is GPU-banned). Omitting it corrupts conditioning.
- **VAE**: `ManualGroupNormND` (mid-attn applies group_norm to a 3D tensor).
- **Encoder**: `inputs_embeds` in (host embed_tokens gather), causal mask constant,
  output `hidden_states[-2]`; `litert_torch` generative path converts Qwen3 natively.
- **int4** naive-blockwise wrecks the image (PSNR 18); int8 is visually faithful.

Parities (fp32): DiT full-model corr 0.99999946; end-to-end fp32 vs diffusers PSNR 59 dB.
int8 end-to-end is visually close (DiT-only PSNR 28.3 dB).
