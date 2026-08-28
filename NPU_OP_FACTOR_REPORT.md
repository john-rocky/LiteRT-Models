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

- **memorize (0.04×)**: no TANH, friendly-ish shapes, 12.6 GF in 251 ms.
  Suspicious patterns (batch-256 `[1,64]×[64,16]` matmuls, builtin GELU over
  `[1,64,64,1024]`) are unbisected. Open.
- **zipformer's full deficit**: dtype ruled out; TANH present but so are 3085
  ops at lengths 796/398/199/100. The granularity-vs-length split is unmeasured.
- **GPU sensitivity to the probe factors**: the probe graphs contain a rank-5
  tensor at the qkv permute, which ML Drift refuses to compile, so GPU arms
  exist only for the real-model pairs (where GPU was flat: 54.7↔55.3,
  36.5↔36.7). GPU indifference to *style* is inferred from the 50-model band
  plus those pairs, not probe-isolated.
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
   zipformer pairs).

## Corrections to earlier statements

- Previous session's "CNN 30/31 vs attention 11–8" framing: the axis is real
  as a correlation but wrong as an explanation. TwinLiteNet (the "CNN that
  tied") ties because its dilated-ESP convs run ~0.56 GF/ms on *both* devices
  (my earlier 93-GF figure for it was an adjX-flag bug in the FLOP counter;
  its big matmul is actually ~0.1 GF).
- README line "The cause is not established; op count, dtype and architecture
  family are all unmeasured" — now partially established as above; dtype is
  measured-irrelevant.
