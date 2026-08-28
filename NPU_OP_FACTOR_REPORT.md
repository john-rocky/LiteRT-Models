# Why the NPU loses on 9 of 50 — a factor analysis with on-device experiments

Galaxy S26 (SM8850, Hexagon v81), LiteRT `CompiledModel`, JIT NPU path, one
accelerator per process, N=50 medians, **every quoted row taken at thermal
status NONE** unless marked. Raw rows: `bench_all.txt` (the original 50-model
sweep), plus this session's `phase2.log` / `phase3.log` (experiment arms).
Op scans verified **by file content on 28 of the 50** (`opscan_all.json`);
statements about op composition refer to those 28 only.

## TL;DR

The 41-vs-9 split is not "CNN vs attention" and not fp16-vs-fp32. On this
hardware the GPU runs everything inside a narrow effective-throughput band
(~0.9–4.4 GF/ms across all 50), while the NPU spans **0.05–12.5 GF/ms**
depending on *how the graph was written*, not what the model is. Three factors
were isolated by single-variable experiments:

| factor | effect on NPU latency | effect on GPU |
|---|---|---|
| weight storage dtype (fp16+DEQUANTIZE vs fp32) | **none** (≤3%, within noise) | none |
| flatbuffer layout (offset-buffers vs inline) | none | — |
| sequence length "shape" (1024 vs 1025 vs 1500 vs 2048) | 1025: +31%; **1500: −57% throughput**; 2048: fastest | not measurable with these probes (see caveat) |
| GELU flavor (tanh-approx vs sigmoid) | **up to 5.8× slower** with tanh-GELU | none visible on real pairs |
| LayerNorm impl (manual SUM/MUL vs native) | +22% | — |

A synthetic 4-layer ViT-S-style block with **tanh-GELU + manual LayerNorm at
T=1025 reproduces the real DINOv2-S NPU rate exactly (0.71 vs 0.74 GF/ms)**;
the same block with sigmoid-GELU + native LN at T=1024 runs 3.2× faster.
Whisper-encoder's loss is reproduced by the T=1500 probe alone (0.97 vs 0.98
GF/ms) — it is not an attention problem, it is a "1500 is an awkward number"
problem.

## The natural experiment that started this

DINOv2-S standalone loses (86.0 vs 53.6 ms, 0.62×). MoGe-2 — the same ViT-S
backbone at the same 1025 tokens (identical BMM shapes `(6,1025,64)×(6,64,1025)`,
identical attention/FC GFLOPs 19.36/43.53) plus a 233-GF conv head — **wins
1.90× and its NPU total (56.2 ms) is lower than the bare backbone's 86.0 ms.**
So the standalone file must be leaving >2× on the table. Graph diff: the
DINOv2 file is fp16-weights + DEQUANTIZE×49, tanh-GELU (TANH×12), manual
LayerNorm (SUM×50 + MUL chains); MoGe is fp32, sigmoid-GELU (LOGISTIC),
native-LN lowering (MEAN/SQUARED_DIFFERENCE/RSQRT).

## Experiment 1 — weight dtype is innocent (refutes the best static correlate)

Static correlation looked damning: every scanned DEQUANTIZE>0 model sits
≤2.8 GF/ms on the NPU; every DEQ=0 model except whisper sits ≥3. Tested by
folding the fp16 weights to fp32 **in place** (DEQ ops deleted, nothing else
touched; host CPU output corr ≈ 1.0 vs original):

| arm | NPU ms (a / b) | GPU ms |
|---|---|---|
| dinov2 ctrl (fp16+DEQ, offset-buffers) | 85.9 / 86.6 | 54.7 |
| dinov2 **fold** (fp32, DEQ removed) | 83.3 / 89.2 | 55.3 |
| dinov2 repack (fp16+DEQ, inline buffers) | 86.8 | — |
| zipformer ctrl | 81.6 / 84.7 | 36.5 |
| zipformer **fold** | 84.5 / 85.0 | 36.7 |

No effect, both families, replicated. Synthetic twin agrees (probe_t1024
9.36 ms fp32 vs 9.35 ms fp16-ized). QNN folds DEQUANTIZE at compile. The DEQ
column was a proxy for *conversion-recipe era*: the older recipes that
produced fp16 files also used tanh-GELU and manual LN — those are the causes.
File format (offset-buffer vs inline flatbuffer) is likewise innocent.

## Experiment 2 — sequence-length shape, not length

Matched-architecture probes (4 pre-LN blocks, d=384, h=6, sigmoid-GELU,
native LN, fp32; only T varies):

| T | ms | GF | GF/ms | vs T=1024 |
|---|---|---|---|---|
| 512 | 4.10 | 8.9 | 2.16 | 97% |
| 1024 | 9.36 | 20.9 | 2.24 | 100% |
| **1025** | 12.29 | 21.0 | **1.71** | 76% |
| **1500** | 36.30 | 35.1 | **0.97** | 43% |
| 2048 | 23.63 | 54.8 | **2.32** | 104% |

Non-monotonic: T=2048 (2.6× the compute of T=1024) is the *fastest* per-FLOP;
T=1500 halves throughput; one extra token (1024→1025) costs 31%. The JIT
compiler also struggles at the awkward lengths (compile 6.5 s @1024, 63.6 s
@1500, 18.8 s @2048). Whisper-encoder (T=1500, fp32, sigmoid-GELU — none of
the other risk factors) runs at 0.98 GF/ms; the T=1500 probe alone lands at
0.97. DINOv2/TIPS run at T=1025/1026; ViTs that win (SmolVLM 1024, CLIP 50)
sit on friendly sizes.

## Experiment 3 — GELU flavor and LayerNorm implementation

Same probe, T fixed, one substitution at a time (N=50, NONE):

| variant | ms | GF/ms | vs baseline |
|---|---|---|---|
| baseline (sigmoid-GELU, native LN, T=1024) | 9.36 | 2.24 | 1.0× |
| manual-LN (SUM/MUL chains) | 11.41 | 1.84 | 1.22× slower |
| tanh-GELU *formula* with LOGISTIC in place of TANH | 22.11 | 0.95 | 2.36× slower |
| **tanh-GELU** | **53.81** | **0.39** | **5.75× slower** |
| tanh-GELU + manual-LN + T=1025 ("dino style") | 29.53 | 0.71 | 3.2× slower |

The third row splits the 5.75× into its two parts: rewriting `tanh(y)` as
`2σ(2y)−1` keeps the exact polynomial chain but swaps the op — so the
elementwise chain itself (x³ polynomial over the `[1,1024,1536]` MLP tensor)
costs ~2.4×, and the TANH op costs a further ~2.4× on top of the identical
structure. Both are real; TANH is the bigger single lever only in combination.

The combined variant reproduces the real DINOv2-S rate (0.71 vs 0.74 GF/ms).
Composition is **not multiplicative** — tanh+manualLN+1025 is *faster* than
tanh alone — so per-factor numbers say "this substitution can cost this
much", not "add them up". The HTP compiler's fusion choices are pattern-
dependent; treat any recipe change as needing a measurement.

Consistency across the 28 scanned models: the models carrying TANH ops are
zipformer×3 (0.38–0.42×), dinov2 (0.62×), tipsv2 (0.85×) — and w2v2-head /
matcha-decoder, which still win because their GPU side is even slower.
Winners' GELUs are LOGISTIC (sigmoid) or the builtin GELU op (SmolVLM, DA3).

## Experiment 4 — the flip: one op type turns the real DINOv2-S from loser to winner

DINOv2-S rebuilt from the official checkpoint three times with the same
builder (fp32, SafeLayerNorm, baked pos_embed/LayerScale — the tanh build
reproduces the shipped file: corr 1.000000, and matches its latency):

| GELU in the MLP | NPU ms | GPU ms | NPU vs GPU | features vs shipped |
|---|---|---|---|---|
| tanh decomposition (shipped) | 85.9 | 54.2 | 0.63× loses | corr 1.000000 |
| sigmoid `x·σ(1.702x)` | 49.3 | 54.1 | 1.10× wins | corr 0.950 (drifts) |
| **builtin GELU op (exact erf)** | **41.9** | 55.2 | **1.32× wins** | **corr 0.999992** |

The builtin-GELU build is 2.05× faster on the NPU than the shipped math,
*more* faithful to the official model (DINOv2 uses exact GELU; tanh is itself
an approximation), and the GPU runs it at the same speed as the others —
on this device (Adreno, LiteRT 2.2.0) ML Drift accepts the GELU op, so the
"GELU must be rewritten for GPU" rule from the Mali-era conversion guide does
not apply here. Mali remains unverified; re-check before changing the global
recipe. The sigmoid build confirms the memory note that sigmoid-GELU drifts
DINOv2's features (0.950) — it flips the speed but is not shippable for
feature fidelity.

Artifacts: `dinov2_ab_{tanh,sig,erf}_fp32.tflite`, builder
`npubench/factors/build_dinov2_ab.py`, rows in `phase5.log`.

Follow-up (2026-08-28): the promised `GELU(approximate=True)` build measured
38.476 ms NPU (gated 50-run median, `npubench/factors/c0_ab_tanhop.log`) —
same band as erf, 2.23× vs the shipped chain, numerics identical to it
(max_abs 9e-05). Root cause of the missing fusion isolated
(`npubench/factors/gelu_assoc_matrix.py`): among the tanh-GELU spellings,
only the left-associative coefficient-first cube `0.044715*x*x*x`
(= `((0.044715*x)*x)*x`, the exact spelling in our builders) escapes the
converter's six MatchGeluApproximate* patterns; pow, parenthesized cubes,
and both outer orders already fuse. Fix submitted as
tensorflow/tensorflow#126296; follow-up posted on litert-torch#1190
(issuecomment-5446164149).

## Experiment 5 — second factor batch (the rewrite catalog grows)

Same probe family, N=50, thermal NONE; baseline re-measured same-day (9.36 →
9.41 ms, drift negligible):

| substitution | before → after ms | factor | equivalence |
|---|---|---|---|
| decomposed softmax → builtin SOFTMAX | 20.34 → 9.36 | **2.17×** | bit-exact (1.4e-06) |
| T=1500 → T=1536 (pad to 12·128) | 36.30 → 14.29 | **2.54×** | not equivalent as-is (needs mask/pos_embed work per model) |
| SafeLayerNorm → native LN | 11.36 → 9.36 | 1.21× | bit-exact — but SafeLN is the Mali-fp16 overflow guard; per-accelerator trade |
| naked TANH → 2σ(2x)−1 identity | 9.18 → 8.33 | 1.10× | identity — contrast with 2.43× *inside* the gelu chain: TANH's cost is fusion-context, not kernel speed |
| rank-3 → rank-4 attention layout | 9.36 → 9.86 | 0.95× (5% cost) | bit-exact; buys GPU compilability (rank-3 probe refuses on ML Drift) |

The rank-4 probe also gives the first matched-graph GPU-vs-NPU comparison at
probe scale: GPU 16.55 ms vs NPU 9.86 ms — NPU 1.68× faster on an identical
graph. T=1536 runs at 2.84 GF/ms, the best rate in the entire grid — the
projected real-whisper fix (pad 1500→1536) would turn its 0.71× loss into
roughly a 1.7× win, pending a real rebuild with masking.

Practical fallout: zipformer's plain TANHs are *not* a lever (1.10×), so its
deficit stays with graph granularity; conversions that keep softmax decomposed
pay 2×; the machine-readable factor rows live in
`litert-compat/data/perf_factors_staging/`.

## Experiment 6 — the flip generalizes: TIPSv2-B14-DPT

Second real-model flip, same one-variable protocol (builtin GELU replacing the
tanh decomposition, official weights, fp32 rebuild; N=50, thermal NONE):

| TIPSv2-B14-DPT (depth+normals+seg) | NPU ms | GPU ms | NPU vs GPU |
|---|---|---|---|
| tanh-GELU (shipped math) | 326.9 | 282.5 | 0.86× loses |
| **builtin GELU** | **142.0** | 273.3 | **1.92× wins** |

2.30× on the NPU from the GELU swap; GPU flat as always. Host parity vs the
shipped file: depth corr 0.999984, normals 0.999999, seg 0.999993. Two for
two — the 0.85× "loser" class with TANH ops converts to solid NPU wins.

## Experiment 7 — conv-family rewrites, and two inversions of the Mali-era guide

Bit-exact pairs (host max_abs_diff 0.0 / 8.8e-08), N=50, thermal NONE:

| pair | NPU ms | factor | GPU (Adreno 2.2.0) |
|---|---|---|---|
| PixelShuffle lowering (TRANSPOSE×8+RESHAPE) vs **zero-stuff conv** | 16.17 vs **5.45** | **2.97×** | PixelShuffle form refuses to compile; zero-stuff runs 9.8 ms |
| bilinear-as-const-matmuls vs **builtin RESIZE_BILINEAR** (align_corners=True) | 2.86 vs **2.03** | **1.41×** | **INVERTED**: builtin resize compiles and runs (4.8 ms); the const-matmul form refuses (constant-LHS BATCH_MATMUL) |

Two conclusions. First, the zoo's SR conversions (EDSR, Real-ESRGAN) adopted
zero-stuff for GPU compatibility and that turns out to be the NPU-fast form
too — the naive PixelShuffle path is both GPU-incompatible and 3× slower on
the NPU. Second, the align_corners=True resize ban and the const-matmul
workaround were real on the Mali-era stack and are *both inverted* on this
Adreno/LiteRT-2.2.0 rig — compat rewrites must be pinned to runtime+device,
which is exactly what the perf-factor rows in
`litert-compat/data/perf_factors_staging/` (16 rows) encode.

## Shippable bundles produced

- `dinov2_erf_fp16.tflite` (44.9 MB, fp16 weights — dtype measured free):
  builtin-GELU DINOv2-S, corr 0.9999996 vs its fp32 parent, 41.9 ms NPU /
  2.05× faster than the shipped file. In `dinov2-work/`.
- `tipsv2_erf_fp16.tflite` (318.3 MB): builtin-GELU TIPSv2-B14-DPT, all
  outputs corr 0.999997+ vs its fp32 parent, 142.0 ms NPU / 2.30× faster.
  In `tipsv2-work/`.
- **Neither is a drop-in replacement for the module assets yet** — the sample
  apps' GPU lane runs on Mali (Pixel 8a), where the builtin GELU op is
  unverified (Adreno runs it at full speed). Ship as NPU-lane variants; run
  the Mali check when the Pixel is next connected.

## What this does NOT yet explain

- **memorize (0.04×)**: partially localized (experiment-13 follow-ups): the
  stem costs ~50 ms, ~200 ms sits in ops 240–601; finer localization pending
  round-2 cuts. The stem's own ~50 ms for a [1,2M]-input conv/LN front is
  itself unexplained.
- **zipformer's deficit**: dtype ruled out (exp 1), granularity ruled out
  (exp 11 — 3000 serial ops cost ~1.2 ms). Remaining suspects: awkward
  lengths 796/398/199/100 (consistent with exp 8's mod-128 rule) and TANH
  content. A length-padded zipformer rebuild would settle it.
- **GPU sensitivity to the probe factors**: largely closed by the phase-9
  rank≤4 probes — GPU arms on the decoder block track compute, not shape
  (51.2 ms @1500 vs 59.4 @1536), and the whisper pair moves +17% where the
  NPU moves 2×. GPU indifference to *style* is now probe-isolated too.
- Bands quoted in GF/ms are comparable **within** a family/architecture scale;
  the probe's absolute 2.2 GF/ms vs MoGe's 5.3 reflects d_model=384 vs bigger
  matrices, not a contradiction.

## Practical rules (S26 / v81 / LiteRT 2.2.0 JIT, as measured)

1. Ship whichever weight dtype you like — fp16 vs fp32 storage does not move
   NPU (or GPU) latency. Pick by file size.
2. Prefer sigmoid-GELU (`x·σ(1.702x)`) or the builtin GELU over the tanh
   approximation when an NPU target matters. Worth up to ~6× — about half of
   that from the extra elementwise chain, half from the TANH op itself.
   Keep activation epilogues short in general.
3. Prefer native LayerNorm lowering over hand-rolled SUM/MUL chains (~20%).
4. Token/sequence counts: powers of two are safe; 2ᵏ+1 (CLS token!) costs
   ~30%; sizes like 1500 can halve throughput. If the architecture allows,
   pad the sequence to a friendly size.
5. None of these substitutions moves the Adreno GPU measurably — a graph
   tuned for the NPU loses nothing on the GPU (measured on the dinov2 and
   zipformer pairs, and probe-isolated in phase 9).
6. LLM prefill: pad the prompt to a **multiple of 128** — worth 4–5× at
   d=1024 scale, and at that scale the awkward lengths ≥~1537 do not even
   JIT-compile on device (memory blowup). Padding is not just speed, it is
   compilability.
7. Never write `torch.erf` in an NPU-bound graph (no ERF builtin; the
   emitted 96-op approximation is 3.8× — worse than the tanh chain). The
   builtin GELU op is the only fast exact-GELU spelling.
8. Weight-only DEQUANTIZE quantization (fp16/int8/int4) buys zero NPU decode
   latency even in the bandwidth-bound T=1 regime — quantize for file size,
   not for CompiledModel speed.
9. SafeRMS with a constant scale is NPU-free; the max-norm (runtime-scale)
   variant costs ~23%. Guard Mali overflow with the constant-scale form.

## Experiment 8 — LLM prefill: the sequence-length rule at decoder scale (2026-08-28)

Decoder-style probe (RMSNorm + GQA 16q/4kv hd=64 + baked RoPE + causal mask +
SwiGLU, d=1024, d_ff=2816, L=2, fp32, rank≤4 throughout), one variable = T.
N=50 medians, thermal NONE, S26 JIT (`probe_batch4.py`, rows in
`~/Downloads/meeting/npubench-phase9-probes/phase9.log`):

| T | NPU ms | GF/ms | JIT compile |
|---|---|---|---|
| 1151 | 55.5 | 1.13 | 135 s |
| **1152** | **13.7** | **4.58** | 10.6 s |
| 1153 | 73.3 | 0.86 | 207 s |
| 1279 | 75.6 | 0.94 | (cache) |
| **1280** | **16.1** | **4.42** | (cache) |
| 1281 | 86.7 | 0.82 | 12.7 min |
| 1407 | 98.3 | 0.81 | 5.6 min |
| **1408** | **19.3** | **4.13** | 26 s |
| 1409 | 102.5 | 0.78 | 5.8 min |
| 1500 | 121.4 | 0.71 | 8.3 min |
| 1535 | 116.1 | 0.76 | 8.5 min |
| **1536** | **22.9** | **3.86** | 35 s |
| 1537 | **compile OOM** | — | killed ~5.5 min |
| 1663 | **compile OOM** | — | SIGABRT ~7 min |
| **1664** | **23.9** | **4.08** | 38 s |
| 1665 | **compile OOM** | — | — |

Three results. (1) **T ≡ 0 (mod 128) is confirmed and the effect is 4–5×**,
far larger than the encoder probe's +31%: one token past a 128 boundary costs
4.0–5.3×. "Pad the prompt to a multiple of 128" is now a shippable one-liner
worth ~5× NPU prefill at this scale. (2) The JIT compiler tracks the same
rule catastrophically: awkward lengths compile 10–70× slower, and **from
T≈1537 upward they no longer compile at all** — two distinct memory deaths,
lmkd low-watermark kill (1537, replicated twice) and Scudo
`internal map failure (Out of memory)` abort (1663) — while 1536/1664 compile
in <40 s. At d=1024 the awkward-length penalty ends in on-device-JIT
infeasibility, not just latency. (3) GPU reference arms are flat
(pf1500 51.2 ms, pf1536 59.4 ms — tracking compute, not shape), so the same
block flips accelerator by length choice alone: at T=1500 the GPU wins 2.4×,
at T=1536 the NPU wins 2.6×.

## Experiment 9 — RMSNorm decomposition flavors (T=1536 arm, one variable)

No RMS_NORM builtin exists in the flatbuffer schema (checked 2.1.6 and
2.3.0.dev20260823, 210 builtins each) — decomposition choice is the only
lever. Same probe at T=1536:

| RMSNorm impl | NPU ms | vs naive |
|---|---|---|
| naive `x·rsqrt(mean(x²)+ε)` | 22.9 | 1.00 |
| SafeRMS scale-before-square, s=64 | 22.5 | **0.98 — free** |
| max-norm SafeRMS (ABS+REDUCE_MAX+DIV, runtime s) | 28.2 | **1.23** |

The Mali-fp16 overflow guard with a *constant* scale costs nothing on the
NPU; the *runtime max* variant costs 23%. Flip candidate confirmed: the
qwen3 embedder carries max-norm SafeRMS ×113 (opscan 2026-08-25), so a
constant-scale rebuild should recover ~20% NPU there.

## Experiment 10 — which op breaks the elementwise-chain fusion

Same gelu-polynomial chain as phase 3, one op X in the tanh slot
(`probe_batch6.py`; exp/pow arms use a smaller leading constant, sqrt/rsqrt
arms add 2 ops for a positive domain — noted, both ~noise-scale):

| X in slot | NPU ms |
|---|---|
| LOGISTIC | 18.5 |
| EXP | 20.6 |
| POW | 22.3 |
| *(slot empty — RELU fused away)* | 23.7 |
| MAXIMUM | 25.3 |
| ABS | 26.4 |
| SQRT / RSQRT | 31.6 / 31.6 |
| **TANH** | **54.0** |
| **torch.erf (96-op rational chain)** | **89.8** |

LOGISTIC/EXP/POW are fusion-safe — *faster than the chain with nothing in
the slot*, which restates the phase-3 lesson that HTP pattern choice is
non-monotonic in op content: you cannot model these as baseline+op-cost.
TANH is the only single builtin that detonates the chain (2.3× the empty
control). Worse still is writing `torch.erf`: there is no ERF builtin, the
converter emits a 96-op clamp+rational approximation, and it lands at 3.8× —
slower than the tanh chain it would replace. Exact GELU on the NPU has
exactly one fast spelling: the builtin GELU op.

## Experiment 11 — op granularity is not a cost (zipformer axis refuted)

Serial 1×1 Conv+ReLU chains, c=32 on 32×32 (per pair 2.1 MFLOP), N = pair
count; equal-FLOPs monolithic reference:

| N | NPU ms | µs/op |
|---|---|---|
| 10 | 0.52 | 52 |
| 100 | 0.54 | 5.4 |
| 1000 | 1.19 | 1.2 |
| 3000 | 1.69 | 0.6 |
| mono (one 6.6-GF conv) | 2.69 | — |

A 3000-op serial graph costs ~1.2 ms over the 0.5 ms invoke floor —
~0.4 µs/op marginal — and **beats the single fat conv of the same FLOPs by
1.6×**. Op count per se cannot explain zipformer's 0.38–0.42× (3085 ops
would account for ~1.5 ms of its ~82 ms); its deficit stays with its
awkward lengths (796/398/199/100) and TANH content. GPU refs: gr300
2.63 ms vs NPU 0.61 (the NPU is *better* at fine-grained graphs than the
GPU here).

## Experiment 12 — decode-GEMV weight dtype: null even when bandwidth-bound

8× Linear(4096²)+ReLU at T=1 — 0.27 GF against 537 MB of weights, the
opposite regime from experiment 1's compute-bound graphs. Same math, four
storage dtypes (`make_fp16.py` / `make_int8.py`, all host-verified
executable):

| storage | file | NPU ms |
|---|---|---|
| fp32 | 537 MB | 4.07 |
| fp16+DEQ | 268 MB | 4.06 |
| int8+DEQ | 134 MB | 4.07 |
| int4+DEQ | 67 MB | 4.08 |

Flat to 0.6%. The weight-only DEQUANTIZE pattern buys **zero** decode
latency on this path even where bandwidth is everything — consistent with
QNN materializing weights to one internal representation at compile
(inference, not directly observed; the equality across a 8× file-size range
is the measured fact). If int4 is to speed decode here it must reach a
quantized kernel, not a DEQUANTIZE graph. Notably int4+DEQ *compiles and
runs* on the NPU; ML Drift (GPU) refuses the int8/int4 DEQ forms outright
(`Failed to compile model`), while GPU fp32 runs 4.58 ms — both accelerators
sit on the same ~DDR floor.

## Experiment 13 — whisper-encoder pad flip: the third real-model flip

Whisper-tiny encoder rebuilt at T=1536 (mel 3072): checkpoint pos table for
real rows + formula rows for the pad tail, conv1 tail re-zeroing mask, key
mask −1e4, output sliced back to [1,1500,384] (`build_whisper_pad.py`).
Equivalence: **bit-exact vs the same-builder T=1500 control on host**
(max_abs 0.0), and end-to-end transcripts through the ONNX decoder are
identical on both test clips (`wer_check.py`).

| whisper-tiny encoder | NPU ms | GPU ms |
|---|---|---|
| T=1500 control | 35.8 | 26.9 |
| **T=1536 padded** | **17.9** | 31.6 |

2.00× on the NPU from the pad alone; the encoder flips from losing 0.75× to
**winning 1.5× vs GPU** (the GPU pays a mild +17% for the extra tokens).
Prediction from the T-shape probe (~15 ms) landed within 20%. Bundle:
`whisper_enc_pad1536.tflite` in `~/Downloads/meeting/npubench-phase9-probes/`
(as `probe_wh_pad1536.tflite`); the app-side change is zero-padding the mel
to 3072 columns.

## Follow-ups landed with these experiments (2026-08-28)

- **C-0 / issue #1190**: the `GELU(approximate=True)` DINOv2-S build runs
  **38.5 ms** NPU (N=50, NONE; `npubench/factors/c0_ab_tanhop.log`) — same
  band as the erf build (41.9), 2.23× vs the shipped chain. The converter
  fact behind the issue: litert-torch 0.9.3 already fuses four spellings of
  the tanh-GELU chain to the GELU op; the one that escapes every
  `MatchGeluApproximate*` variant is Python's natural
  `0.044715 * x * x * x` (coefficient-first association — no x³ subterm).
  Matrix: `npubench/factors/gelu_assoc_matrix.{py,log}`. The DINOv2/TIPSv2
  builders use exactly that spelling — that is why the slow files exist.
  Comment + pattern PR are owned by the 08-28 handoff session.
- **Mali re-check (C-9)**: on Pixel 8a + LiteRT 2.2.0 (npubench), the
  builtin-GELU DINOv2 compiles and runs (198.8 ms GPU) and
  align_corners=True RESIZE_BILINEAR compiles and runs (25.4 ms) — both
  Mali-era bans are gone *at this runtime*; per-module pins still need their
  own check before unifying zoo assets on erf builds.
- **memorize bisect, rounds 1+2**: working prefix cuts k100/140/240/252/309/
  350 run 49.5 / 54.7 / 56.0 / 59.5 / 59.8 / 56.3 ms vs full 256.1
  (`phase9.log`, `phase10.log`) — the first 350 ops cost only ~60 ms, so
  **~196 ms of the 0.04× disaster sits in ops 350–601**, exactly the
  batch-256 `[1,64]×[64,16]` matmul + `[1,64,64,1024]` GELU region the
  static scan flagged. Every cut ending inside 380–549 compiles but fails
  QNN invoke (all run fine on host CPU), so the region resists prefix
  truncation — finer localization needs op stubbing instead of cutting.

## Corrections to earlier statements

- Previous session's "CNN 30/31 vs attention 11–8" framing: the axis is real
  as a correlation but wrong as an explanation. TwinLiteNet (the "CNN that
  tied") ties because its dilated-ESP convs run ~0.56 GF/ms on *both* devices
  (my earlier 93-GF figure for it was an adjX-flag bug in the FLOP counter;
  its big matmul is actually ~0.1 GF).
- README line "The cause is not established; op count, dtype and architecture
  family are all unmeasured" — now partially established as above; dtype is
  measured-irrelevant.
