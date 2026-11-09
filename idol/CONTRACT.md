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

## Changes to the app after the first numbers (no line changed)

- 2026-11-09 17:16, before any (c) number: "graphs already loaded and compiled" cannot hold on the S26 (12 GB): one
  denoiser block takes about 0.7 GB of GPU memory and the 46 graphs do not fit at once, so the app keeps as many as fit
  above a memory reserve and compiles the others during each run. (c) is measured as the contract says otherwise
  (median `generate` of runs 1–3 after a warm-up), with the graphs compiled inside each measured run reported next
  to it (count and seconds). The e2e plans keep a 2.5 GB reserve (a 1 GB reserve let lmkd kill the app during a
  preload, LANE.md 17:11).

- 2026-11-09 17:00: input 4 is shot in the first device hold that runs the app, before any app run of inputs 1–3
  (the table's "first device hold after the go" was round g1, and round g2 followed it: both ran the graph runner,
  no app). Its sha256 and time go into the table before any line runs on it.
- 2026-11-09 17:00: the graph set becomes v3 = v2 with two graphs rebuilt after the first S26 sweep (device round g1,
  `device/plan_g1.tsv`); same arithmetic, no line changed.
  - The VAE GroupNorm's scale (`idol_graphs.PooledGroupNorm`): the per-group max of |x − mean| is a max pool by the
    same k as the mean, then a max over the 32 × 32 map (a max of maxes: the same value). g1: `vae_0` took 8.0 s per
    call on the S26 (one REDUCE_MAX over 1,048,576 values per channel), against the ≤ 2 s per call of "Forms". Round g2:
    0.83 s, output equal to g1's (cos 1.0000000).
  - BiRefNet (`bir.tflite`, CPU): g1 ran out of memory at the first call (the Mac CPU peaks at 9.3 GB in one call).
    The memory-lean form `scripts/bir_lean.py`: the deformable-convolution gathers in output-row tiles that run one
    after another (each tile's offsets plus an exact zero computed from the previous tile's output, so the LiteRT CPU
    cannot run every tile's lookup before the work that consumes them), the ASPP's 1×1 convolution applied per
    branch, and the decoder's last 1×1 convolution applied before the 4× resize and folded into the last 3×3
    convolution. PyTorch: map cos 1.000000000 against the author's (max |Δ| 5.5e-6). Round g2 on the S26: one call,
    VmHWM 4.15 GB; uint8 alpha on input 1 differs from the author's in 2,542 of 1,048,576 pixels, each by 1 (IoU
    1.0). Mac CPU peak 3.2 GB.

- 2026-11-09 15:21, after the (b) numbers with the floor below: the app's matting becomes the author's BiRefNet
  itself (`bir.tflite`, run by `CompiledModel` on the CPU); DIS stays in the app as an option, off by default. BiRefNet's
  20 deformable convolutions are rewritten as a GATHER on the token axis with bilinear weights and one matmul
  (`scripts/deform_conv.py`, the arithmetic of torchvision's `deform_conv2d`): the whole BiRefNet in PyTorch is 4.0e-6
  (max |Δ|) from the author's map, and through LiteRT CPU on input 1 its uint8 alpha differs from the author's in
  2,543 of 1,048,576 pixels, each by 1, with the same crop. It runs on the CPU because the LiteRT GPU has no runtime
  gather (GATHER_ND is not supported; EMBEDDING_LOOKUP with a runtime table ended the process with a segfault on Metal).
  Why: splitting input 1's (b) put the whole loss in DIS's alpha edges, which no floor fixes (DIS alpha with the
  author's crop: 22.83 dB; the author's alpha with DIS's crop: 30.35 dB); with the floor DIS passed input 1 by 0.76 dB
  (25.76 dB) and input 2 at 30.48 dB, and input 3's crop stayed 8 % larger than the author's. Line (a′) now compares
  the author's BiRefNet run by LiteRT CPU with the author's BiRefNet in PyTorch; its threshold is unchanged. The DIS
  numbers stay recorded (`contract/standin/`, `contract/a_<input>_mac.json`).
- 2026-11-09 15:07, after the first (b) numbers (Mac stand-in: armchair 24.04, axolotl 19.00, succulent 19.15 dB, all
  FAIL): the app's DIS alpha gets a floor: uint8 values below 8 are set to 0 before the author's preprocessing above.
  Why: DIS leaves faint responses BiRefNet does not (1/255 at the image corners on the axolotl, a faint wisp over the
  succulent's top leaf), and the author's rule crops to the nonzero alpha, so the crop came out 20 % (axolotl) and
  25 % (succulent) larger than BiRefNet's. The value was chosen on 31 held-out images of the author's gallery (every third file, inputs 1–2
  excluded; `scripts/calibrate_matting_floor.py`, `contract/matting_floor.json`) as the floor whose crop box best
  matches BiRefNet's: mean error 0.14 % of the side, worst 0.59 % (no floor: mean 9.0 %, 20 of 31 images over 2 %).
  Line (a′) thresholds at 128 and is unaffected. The first numbers stay recorded (`contract/standin_v1/`,
  `contract/a_<input>_mac_v1.json`, LANE.md).
- 2026-11-09 15:05: the graph set became v2 (DINOv3 and the context refiner with the key-chunked attention from
  4,096 keys); no line changed.

## Changes before any number

- 2026-11-09 14:13, before any (b) number: the cameras' up axis. The contract said "y up"; the first render of the
  armchair reference (no comparison, no PSNR) showed the object upside down. idol's `save_ply` writes the y-down frame
  3DGS viewers use (its comment says y-up, the files are not), so the cameras now sit above the object at −y with
  image up −y. Same 4 azimuths, elevation, distance, field of view and size.
