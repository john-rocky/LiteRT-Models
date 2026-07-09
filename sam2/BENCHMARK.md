# SAM 2.1 (vision): LiteRT vs MLX on the same Apple GPU — the decode-gap thesis, reproduced outside LLMs

Measured on an iPhone 17 Pro (A19 Pro) and a Mac Studio M4 Max. SAM 2.1 Hiera-Tiny, image path
(encoder once + prompt mask decoder per point), fp16 on both sides, warm median.

## TL;DR

- On the same Apple GPU, MLX beats LiteRT for SAM 2.1 — reproduced on a real iPhone 17 Pro and on the
  M4 Max. The Apple-GPU gap we measured for LLM decode is not LLM-specific; it shows up in a pure
  vision model.
- Same-device iPhone 17 Pro, both fp16: MLX encoder **159.6 ms** / decoder **12.0 ms** vs LiteRT (Metal)
  **248.2 ms** / **16.3 ms** → MLX is **1.55×** faster on the encoder and **1.36×** on the decoder.
- The gap is runtime **kernel efficiency** — not the model, the precision, or the conversion. Every graph
  is fully GPU-resident on every target (fullyGPU = YES) and the LiteRT output is numerically exact
  (corr 1.0) against the PyTorch / MLX reference.
- The most useful result is not the ratio but that its **cause is hardware-dependent**: the mask decoder
  is 5.5× slower on LiteRT on the M4 Max, yet only 1.36× slower on the iPhone. On a strong GPU the tiny
  decoder is dispatch-bound (MLX's fused, low-overhead graph wins big); on the mobile GPU it becomes
  compute-bound and the gap collapses.

## 1. What was measured

SAM 2.1 Hiera-Tiny (`facebook/sam2.1-hiera-tiny`), image-segment path only: the heavy Hiera image encoder
runs once per image, the small mask decoder runs once per point. Identical 1024×1024 input, one positive
point, fp16 weights on both runtimes, warm median (5 warm-up + 20–30 timed iterations).

Every row was gated on **correctness**, not just latency: each build reports fully-GPU residency and a
mask foreground count of ≈64.8k px against the 64.9k px reference — so no row is a silently-degraded or
CPU-fallback measurement.

On iOS, LiteRT runs through the **Metal delegate**: it attempts an OpenCL context, fails, and falls back
to Metal — the same path we saw with LiteRT-LM on iPhone.

## 2. Numbers (median, ms)

| Runtime | Silicon / GPU | Encoder (ms) | Decoder (ms) |
|---|---|---:|---:|
| MLX | Mac M4 Max | 41.1 | 2.7 |
| LiteRT | Mac M4 Max (Metal) | 70.7 | 14.8 |
| **MLX (mlx-swift)** | **iPhone 17 Pro (A19 Pro)** | **159.6** | **12.0** |
| **LiteRT** | **iPhone 17 Pro (A19 Pro, Metal)** | **248.2** | **16.3** |
| LiteRT | Pixel 8a (Mali, ML Drift) | 610 | 76 |

The Mac LiteRT row is executed through the iOS simulator, which runs on the same M4 Max GPU as the MLX
row, so the two Mac rows are a same-silicon comparison. The iPhone rows are on the physical device. The
Pixel 8a row is the Android baseline, not part of the runtime comparison (different silicon).

## 3. Why — where the gap lives

- **Kernel layer (primary).** MLX ships native, hand-tuned Metal kernels for Apple GPUs (fused GEMM /
  conv / attention, simdgroup use). LiteRT's Apple-GPU path runs generic, portability-first delegate
  kernels. This is the same conclusion as the Metal-level LLM decode decomposition — kernel efficiency,
  not bit-width — now observed inside a vision graph.
- The **encoder is compute-bound** (~105 GMACs at 1024²): both runtimes are limited by raw GPU
  throughput, so the gap narrows to ~1.5–1.7×.
- The **decoder** is a 2-layer two-way transformer on small tensors. On the M4 Max it is dominated by
  per-op dispatch overhead, where MLX's lazy, fused graph wins 5.5×. On the A19 Pro the same graph
  becomes compute-bound and the gap falls to 1.36×.
- **Not the cause:** identical model, conversion corr 1.0, fp16 on both sides, fully GPU-resident
  everywhere.

## 4. LiteRT-side conversion — and a GPU-delegate correctness bug we found and fixed

The SAM 2 Hiera encoder was made GPU-clean for CompiledModel with three numerically-exact rewrites
(parity held at corr 1.0 after each): bake the windowed positional embedding, which is constant for a
fixed 1024² input (removes the bicubic `GATHER_ND` and the tiled `BROADCAST_TO`); re-express window
partition / unpartition with ≤4-D tensors; replace the fused 5-D `qkv` reshape with a channel-wise q/k/v
slice. Result: encoder `GPU_BAD = 0`, `>4-D = 0`. The SAM 2 mask decoder converted unchanged.

**Worth recording for the team.** The published `litert-community` SAM 2.1 mask decoder was pinned to the
CPU because its GPU output was wrong on a Pixel 8a. We reproduced it headlessly and bisected it on
device. The cause is **not fp16 and not LayerNorm** — the decoder's attention was written with the batch
dim collapsed (q/k/v shaped `[heads, N, d]`, rank 3), and the ML Drift GPU delegate **silently
mis-computes that form**: the graph compiles, delegates 358/358 nodes, passes the op gate and matches
PyTorch on the host, yet returns masks at correlation **0.265** against CPU (**0.473** with fp32 GPU
compute forced, so it is a correctness bug, not a precision limit). Keeping the leading batch dim
(`[1, heads, N, d]`) is numerically identical on the host and restores correlation **0.9998** /
binary-IoU **0.999** on the GPU while running **~20 % faster** (6.8 ms vs 8.5 ms per tap). The fixed
model is a drop-in replacement: it is published as `sam2_tiny_mask_decoder_v2_fp16.tflite` and the sample
now runs **both graphs on the GPU** (encoder 867/867, decoder 425/425 LITERT_CL nodes). Note the image
encoder's rank-3 SDPA *is* GPU-correct on the same device, so a healthy sibling graph proves nothing —
only a numeric GPU-vs-CPU check on device catches this.

## 5. Findings

- The Apple-GPU kernel-efficiency gap is **architecture-independent**: LLM decode and a vision
  encoder/decoder both exhibit it.
- The **magnitude** of the gap is set by whether the graph is dispatch-bound or compute-bound. Small
  graphs on strong GPUs expose LiteRT's per-op dispatch overhead most sharply.
- **Coverage is not the problem here** — LiteRT converted and ran the entire SAM 2 image path fully on
  GPU on both Mali and Metal. Speed is. And **correctness needs a device check**: full delegation plus a
  clean op gate plus desktop parity can all pass while the GPU output is garbage.

## 6. Ask

- **Prioritise the Apple-GPU kernel layer** (native Metal kernels and fusion). It is the same ask as the
  LLM decode work; SAM 2 shows the payoff extends beyond LLMs.
- **Reduce per-op dispatch overhead for small graphs** — that is where the gap is largest (5.5× on the
  decoder on M4 Max).
- **Fix the rank-3 batched-attention miscompute in the ML Drift GPU delegate** (repro and bisect
  available). Until then, a lint that warns when an attention / BMM chain is emitted at rank 3 would have
  caught this at conversion time. The workaround (keep the batch dim) is already shipped in the
  `litert-community` decoder and the interactive-segmentation sample.

## 7. Reproduction

All three harnesses live in this repository. Models are **not** committed — the conversion script
produces them, or fetch them from
[`mlboydaisuke/SAM2-hiera-tiny-LiteRT`](https://huggingface.co/mlboydaisuke/SAM2-hiera-tiny-LiteRT).

**Conversion** (produces the GPU-clean encoder + decoder, and self-checks parity vs PyTorch):

```bash
SAM2_CKPT=facebook/sam2.1-hiera-tiny SAM2_OUT=./out python sam2/scripts/convert_sam2.py
# prints: encoder GPU_BAD=NONE >4D=0 / decoder GPU_BAD=NONE >4D=0 / PARITY corr=1.000
```

**LiteRT on Android** (Pixel 8a / Mali, ML Drift). Put the three model files in
`sam2/app/src/main/assets/`, then:

```bash
cd sam2 && ./gradlew :app:installDebug && adb logcat -s SAM2
# BENCH GPU compile=..ms enc_median=..ms dec_median=..ms iters=20 mask_fg=..
```

**LiteRT on iOS** (Metal, LiteRT `CompiledModel` C API). Put the models in `sam2-ios/Resources/`,
set your own Development Team in Xcode, then:

```bash
cd sam2-ios && xcodegen generate && open SAM2.xcodeproj   # run on device
# console: SAM2BENCH  encoder: fullyGPU=YES  enc_median=..ms  dec_median=..ms  mask_fg=..
```

**MLX on iOS** (a full mlx-swift port of the Python MLX SAM 2 image path, validated at corr 1.0 against
the Python reference before measuring). Put `sam2_tiny.safetensors` (fp16) in `sam2-mlx-ios/Resources/`:

```bash
cd sam2-mlx-ios && xcodegen generate && open SAM2MLX.xcodeproj   # run on device
# console: SAM2MLXBENCH  enc_median=..ms  dec_median=..ms  mask_fg=..
```

**MLX on Mac**: [`avbiswas/sam2-mlx`](https://github.com/avbiswas/sam2-mlx)
(`load_image_segmenter` → `encode_image` / `predict_from_encoded`).

**Device correctness probe** used for the decoder bisect (Pixel 8a). A GPU-vs-CPU numeric check is the
only thing that catches a silently mis-computed graph:

```bash
# on device, /data/local/tmp, needs LD_LIBRARY_PATH=/data/local/tmp
# multi-input: <in>.<i>;  dumps every output to <out>.<i>
./gpu_test_bin dec.tflite 1 dec_in.bin dec_out.bin        # default fp16 GPU
FP32=1 ./gpu_test_bin dec.tflite 1 dec_in.bin dec_out.bin # forces fp32 GPU compute
```

`FP32=1` is the discriminator: if the output is still wrong with fp32 compute, it is a delegate
correctness bug, not an fp16 precision wall.

## 8. Open follow-ups

- The same-silicon Mac row was measured through the iOS simulator (M4 Max GPU); a native macOS LiteRT GPU
  path for vision `.tflite` would tighten it (`litert-mac-verify` is LiteRT-LM / LLM-only).
- Larger sizes (small / base+ / large) if the team wants the full curve.
- The rank-3 attention miscompute deserves its own bug report against the ML Drift delegate.
