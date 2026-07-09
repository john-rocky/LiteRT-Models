# Z-Image-Turbo — on-device text-to-image (LiteRT)

Alibaba Tongyi-MAI **Z-Image-Turbo** (6B, Apache-2.0), a Single-Stream Diffusion
Transformer (S3-DiT), generating images fully on the phone GPU through **LiteRT
`CompiledModel`**. First diffusion text-to-image model in this zoo, and — via
chunked sequential residency — the first 6B generator verified generating a real
image end-to-end on a commodity 8 GB phone (Pixel 8a, Mali GPU).

![generated on a Pixel 8a](docs/pixel8a_generated.png)

*Generated entirely on a Pixel 8a Mali GPU (LiteRT), prompt "a red apple on a
wooden table, studio lighting", 8 steps.*

- **Pipeline:** Qwen3-4B text encoder → S3-DiT denoiser (8 steps, FlowMatchEuler
  with classifier-free guidance) → VAE decoder.
- **Every heavy stage is a LiteRT INTEGER-int8 graph** (int×int integer compute —
  the path that runs on the GPU delegate; the weight-only-FLOAT path hangs/overflows
  on the GPU delegate, so it is NOT used here).
- **Device-verified end-to-end** on a Pixel 8a: the full 8-step denoising loop plus
  VAE decode run on the Mali GPU and produce the image above (≈ 27 min, FP32-compute,
  recompile-per-step). The `scripts/` reproduce every graph and the exact host loop.

## How it fits an 8 GB phone: chunked sequential residency

The monolithic S3-DiT is a single ~6 GB int8 graph, which exceeds LiteRT's file-load
limit and a phone's GPU budget. It is split into graphs that each compile fully on
the ML Drift OpenCL delegate and load one at a time, so the peak footprint is a
single graph:

| Graph | Role | int8 size |
|-------|------|-----------|
| `z_embx` | image patch embedder | 0.3 MB |
| `z_refx` | noise refiner (2 blocks, adaLN) | 363 MB |
| `z_embc` | caption embedder | 10 MB |
| `z_refc` | context refiner (2 blocks) | 355 MB |
| `zc_main0..5` | 5 S3-DiT layers each (30 total) | 866 MB × 6 |
| `zc_final` | final adaLN + projection | 1.2 MB |
| `zvae` | VAE decoder | 50 MB |

The refined image/context tokens meet as one unified hidden state `[1,288,3840]`
passed between chunks; the composition is bit-exact to the monolithic DiT
(corr 1.000000 desktop, 0.966 on-device int8/FP32-GPU).

## Device constraints solved (all GPU-delegate-only, invisible to the desktop checker)

- **`bc coord for BATCH axis` compile wall (C19 sibling).** A MUL right after the
  embed FC (the learned pad-token substitution) makes ML Drift assign the token dim
  to the batch axis and the OpenCL kernel fails to generate. Fix: move the pad-token
  mask (`x*(1-pad) + pad_token*pad`) and the x/c concat to the **host** — the graphs
  are then a bare FC or a refiner stack, both of which compile.
- **fp16 NaN in the adaLN path.** The noise-refiner / main layers (modulation=True)
  overflow fp16 to all-NaN on the GPU; the context refiner (no modulation) is clean.
  Fix: `GpuOptions(precision = FP32)` — Mali honors it here (this is *not* an
  fp16-locked op).
- **Cumulative OpenCL context leak.** A null `Environment` per `CompiledModel.create`
  aborts the process after ~20 FP32 compiles. Fix: create one `Environment` and pass
  it to every graph; close each input/output `TensorBuffer` per run.

## Host-side glue (ported per platform; Python reference is authoritative)

The graphs do all the matmul-heavy compute; these stay on the host (trivial, and
reproduced exactly in `scripts/gen_prep.py` + `gen_verify.py`):

1. **Tokenize** the prompt (Qwen2 BPE), embed, run the **encoder graph**, slice to
   the valid prompt length → `cap_feats`.
2. Precompute the fixed inputs: 3-axis **RoPE** `cos/sin`, per-step **adaLN**
   timestep embeddings, FlowMatchEuler **sigmas**, the initial latent, and the
   patchify / unpatchify index permutations.
3. Per denoise step (in latent space): patchify → run the chunked DiT for the
   **cond** and **uncond** prompts → combine with the Z-Image CFG
   `noise_pred = -(pos + guidance*(pos - neg))` → Euler update
   `latent += dsigma * noise_pred`. The image branch (`embx → refx`) is shared by
   cond/uncond; the context branch (`embc → refc`) is step-independent (computed once).
4. Decode the final latent with the **VAE graph** → RGB.

⭐ Host-prep details that each cost a garbage image if wrong:
- The DiT conditions on the encoder's **penultimate** hidden state (`hidden_states[-2]`).
- The **pad-token substitution** at padded positions (arithmetic form, not SELECT).
- The Z-Image CFG is `pos + g*(pos - neg)` **negated** (from `pipeline_z_image.py`),
  not the textbook `neg + g*(pos - neg)`; CFG is active for `guidance > 0` (batch 2).

## Quality / precision

int8 (INTEGER-compute) is the GPU-runnable precision and renders a faithful image
(desktop DiT-only PSNR 28.3 dB; on-device sequential-chunk output corr 0.966 vs the
fp32 reference — a subtle grid texture, the apple clearly correct). int4 is garbage
(PSNR 18), so int8 is the floor.

## Build & run

1. **Convert** the graphs (conversion env; see `scripts/README.md`):
   ```
   python scripts/chunked_export.py --export     # z_refx/z_refc + zc_main0..5 + zc_final
   python scripts/prep2.py --export              # z_embx / z_embc / refiners (host-mask split)
   python scripts/vae_deploy.py                  # zvae_int8_256.tflite
   python scripts/gen_prep.py                    # host inputs (gen_bins/) + fp32 reference image
   python scripts/gen_verify.py                  # confirms the exact device loop (corr ~0.99)
   ```
2. **Stage to device:** push the chunk `*.tflite` and `gen_bins/` to the app files dir
   (`install_to_device.sh`).
3. **Build the app** on a connected device (`./gradlew :app:installDebug`). It runs the
   full generation on launch and writes `generated.png` to the app files dir.

`minSdk 26`, `arm64-v8a`, LiteRT `CompiledModel` GPU.

### Kotlin structure

- `ZImageGen.kt` — the full generation loop: chunked DiT (cond + uncond), Z-Image CFG,
  FlowMatchEuler update, unpatchify, VAE decode → bitmap. Device-verified.
- `ChunkRunner.kt` — loads one graph on the GPU (FP32, shared `Environment`, per-run
  buffer free), runs it, returns the output.
- `MainActivity.kt` — Compose UI (generate on launch, show the image).

> `scripts/` is ground truth; `gen_verify.py` reproduces the exact device loop in fp32
> (corr 0.989 vs the diffusers pipeline). Per repo policy, functional validation is on
> the device.
