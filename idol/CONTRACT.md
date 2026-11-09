# CONTRACT — Idol on LiteRT (pre-registered 2026-11-09 14:0x JST, before any line below was run)

What is being shipped: speridlabs/idol (one image → 3D Gaussian splat) with the speridlabs/idol-turbo LoRA (5
guidance-free steps), as LiteRT graphs run by `CompiledModel` on the phone GPU, plus host glue (Kotlin) for the parts
that are not tensor programs (octree sampling, Gaussian activation, PLY writing).

The lines never change after the first number. A change made before that is dated in "Changes before any number".

## The reference ("the author's pipeline")

- Code: idol-sdk @4702a7f (`idol/`), run by `scripts/ref_pipeline.py`, on the Mac CPU in float32.
- Weights: speridlabs/idol @2731c229 (denoiser, decoder, decoder_gs, birefnet); speridlabs/idol-turbo @45cdd41f
  (`idol_turbo_lora.safetensors`, rank 32, alpha 128, slow expert for t ≥ 0.5, fast below); DINOv3 ViT-H+/16 from
  Comfy-Org/Idol @e907d8dc (bf16 repack of facebook/dinov3-vith16plus-pretrain-lvd1689m, computed in float32); FLUX.2
  VAE from black-forest-labs/FLUX.2-klein-4B @e7b7dc27 `vae/` (bf16, computed in float32).
- Sampler: idol-turbo's (`turbo.py`): Euler, 5 steps, t = 3s / (1 + 2s) over s = linspace(1, 0, 6), velocity
  model(x, 1000 t, cond), no guidance. 262,144 Gaussians = 8,192 anchors × 32.
- Deviations from idol-sdk, none in the math: no encoder-access check; float32 everywhere; the random draws below
  are given instead of drawn inside.

## Shared randomness (both pipelines get the same numbers)

From `g = torch.Generator().manual_seed(seed)`, in this order: VAE posterior noise `randn(1, 32, 128, 128)` (latent =
mean + exp(logvar / 2) · noise), latent noise `randn(1, 8192, 16)`, camera noise `randn(1, 1, 5)`.
From `g1 = torch.Generator().manual_seed(seed + 1)`: the octree pool `rand(8, 8192)` and the leaf pool
`rand(1, 8192, 3)`. The octree sampler draws per row: at level l, parent row i of the current list takes the offset
u0 = pool[l − 1, i] / n_i (n_i = that row's point count) instead of idol-sdk's per-group `torch.rand(r, 1)`; the
leaf positions take the leaf pool instead of `torch.rand_like`. Same distribution as idol-sdk's sampler; the change
only makes the draws replayable on the phone. All five arrays are saved per input (`ref/<input>/noise.npz`) and fed
to both pipelines.

## Fixed inputs

| # | File | sha256 | Source / licence | Mode (crop margin) | Seed |
|---|---|---|---|---|---|
| 1 | `nb-armchair.webp` (768², RGB) | 78e92542fe244d0e174802102446addfab7eef9b14e652d97ec1151deb24ff09 | idol-sdk `demo/examples` (author's gallery, Apache-2.0 repo) | synthetic (1.2) | 7 (author's preset) |
| 2 | `fig-axolotl.webp` (768², RGB) | 298980bf091a6457ea4369368784fa53ece4b301d79db60d236857f3e2c17f09 | idol-sdk `demo/examples` (author's gallery) | synthetic (1.2) | 42 (author's preset) |
| 3 | `pexels_20324601.jpg` (4500², RGB): a succulent in a metal pot on a table | 527d8f87e0eee7c94d652e9f581991c3e860966e1f8eeb0228e19551e52d0255 | https://www.pexels.com/photo/plant-in-silver-flowerpot-20324601/ (Komet Flicker, Pexels License) | real (1.7) | 42 |
| 4 | one photo of a desk object shot with the Galaxy S26 rear camera through the app's camera path, in the first device hold after the go | added here (sha256, time) before any line runs on it | ours | real (1.7) | 42 |

Input 3 stands in for "a photo we shot" until input 4 exists (the phone is not ours before the go); input 4 runs
every line that inputs 1–3 run.

## Lines

### (a) Part agreement — same input, LiteRT output vs the reference

Each LiteRT part gets the reference's own input for that part (no chaining) and is compared with the reference's
output for it. cos = uncentred cosine over the flattened tensor, float64. Also recorded for every row: relative L2,
max |Δ|, max |ref|.

| Part | Compared tensor | fp16 line | int8 line |
|---|---|---|---|
| DINOv3 encoder (all its graphs in a row) | feature1 [1, 4101, 1280] (after the final layer norm) | cos ≥ 0.99 | cos ≥ 0.97 |
| VAE encoder | the 4,096 latent tokens of feature2 [1, 4096, 128] (posterior with the shared noise, 2×2 patchify, bn) | cos ≥ 0.99 | cos ≥ 0.97 |
| Denoiser, one step (refiners, 24 joint blocks, heads) | latent velocity [1, 8192, 16] at each of the 5 steps, from the reference x_t and cond | cos ≥ 0.99 at every step | cos ≥ 0.97 |
| Octree decoder | child probabilities (softmax) [1, L, 8] at each of the 8 levels, from the reference cells and latent | cos ≥ 0.99 at every level | cos ≥ 0.97 |
| Gaussian decoder | each activated channel group: xyz, colour (f_dc), scale, rotation, opacity; from the reference anchors and latent | cos ≥ 0.99 each | cos ≥ 0.97 |

Metered without a line: the camera velocity [1, 1, 5] per step.

Where: Mac = LiteRT 2.3.0 `CompiledModel` GPU (Metal, default precision = fp16), every fixed input. Pixel 11 Pro =
pass / fail, input 1. Galaxy S26 = numbers, input 1, GPU in the shipped precision.

### (a′) Background removal

The app cannot run BiRefNet on the GPU (its decoder's dense deformable convolutions have no LiteRT GPU path), so it
uses IS-Net DIS general-use (xuebinqin/DIS, Apache-2.0; the litert-community/DIS-ISNet-LiteRT form) for the alpha.
Line: on each fixed input, at the preprocessing size (short side 1024), the substitute's uint8 alpha vs BiRefNet's
(the author's code, float32, Mac CPU), both thresholded at ≥ 128: **IoU ≥ 0.95**. Metered: the crop square of each
(author's rule) and the difference of its corners in pixels.

The app's preprocessing is the author's rule (`idol/preprocess.py`): resize the short side to 1024 (Lanczos), alpha
as uint8 (×255, truncated), 3×3 minimum filter, square crop of side margin × the bounding box of the nonzero alpha,
resize to 1024 (Lanczos), composite on black.

### (b) The work — the whole phone pipeline vs the whole author pipeline

Per fixed input: the author's pipeline (raw image → BiRefNet preprocessing → reference) and ours (raw image → the
app's preprocessing with the matting graph → our graphs and glue) with the shared randomness, turbo 5 steps,
262,144 Gaussians → two `.ply` files.

Both are rendered by the same renderer: `scripts/render_views.py` = gsplat 1.5.3 (nerfstudio-project/gsplat,
Apache-2.0) pure-PyTorch projection and tile intersection (`gsplat.cuda._torch_impl`), with gsplat's `accumulate`
compositing (alpha = min(0.999, opacity · exp(−σ)), weight = alpha × the product of (1 − alpha) in front), on the Mac
CPU, float32, black background, colour = clamp(0.5 + 0.28209479 · f_dc, ≥ 0), image clamped to [0, 1].

Cameras, in the PLY frame (y down, object inside [−0.5, 0.5]³): 4 pinhole cameras at azimuth 0°, 90°, 180°, 270°
about the y axis, 20° above the object (camera y = −2.2 · sin 20°), distance 2.2 from the origin, looking at the
origin, image up −y, vertical field of view 40°, 512 × 512.

Line: **PSNR ≥ 25 dB per input**, MSE pooled over the 4 views (RGB, [0, 1]). Per-view PSNR recorded. The side-by-side
image of both renders (4 views × 2 rows) is kept at `contract/b_<input>.png` — the line you can see.

Where: Mac stand-in first (our graphs on Metal + the glue in Python, the same steps as the Kotlin), then the S26
(the app in fixture mode, given the shared randomness; its `.ply` is pulled and rendered on the Mac).

### (c) Speed — S26 GPU

The app on the Galaxy S26, GPU, the shipped precision: wall time from "preprocessed 1024² image ready" to ".ply
written", turbo 5 steps, 8,192 anchors (262,144 Gaussians), graphs already loaded and compiled. Median of 3 runs after
one warm-up run; thermal status and the per-stage times (DINOv3, VAE, denoiser, octree, Gaussian decoder, PLY)
recorded; load and compile time reported apart. Metered; the first number goes on the card.

### (d) Failure paths do not crash

On the Pixel 11 Pro and the S26: (1) model files missing → a message that names the missing file and the install
step; (2) a picture with no object (an all-black image) → a "no object found" message; (3) camera permission denied →
a message, the gallery still works. Pass = the process stays alive (no crash, no ANR) and the message is on screen
(screenshot kept).

### (e) Peak memory — S26

During one generation: the app's VmHWM and the system's GPU pages (`/sys/class/kgsl/kgsl/page_alloc`) maximum.
Metered.

## Forms

GPU precision fp16 (on the phone with fp32 accumulation); fp32 only for a part that fails its line in fp16; int8 is a
speed-ladder step and meets the int8 column. Each GPU call stays ≤ 2 s on the S26 (finding 2 of `F0.md`: a 9 s call
starved the compositor and the process was killed). The turbo LoRA is either merged per phase (two graph sets) or
kept as side matmuls with a phase input (one set), decided by measurement; the lines are the same either way.

## Changes before any number

- 2026-11-09 14:13, before any (b) number: the cameras' up axis. The contract said "y up"; the first render of the
  armchair reference (no comparison, no PSNR) showed the object upside down. idol's `save_ply` writes the y-down frame
  3DGS viewers use (its comment says y-up, the files are not), so the cameras now sit above the object at −y with
  image up −y. Same 4 azimuths, elevation, distance, field of view and size.
