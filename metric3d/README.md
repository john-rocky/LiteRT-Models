# Metric3D v2 (ViT-S) — Monocular Metric Depth on-device (LiteRT GPU, fully GPU)

[Metric3D v2](https://github.com/YvanYin/Metric3D) (CVPR/TPAMI 2024, BSD-2-Clause) monocular **metric**
(absolute, in-meters) depth running **fully on the LiteRT CompiledModel GPU** (ML Drift). Unlike Depth
Anything (relative) / MoGe (affine-invariant) / DSINE (normals), the output is absolute depth in meters.
The DINOv2 ViT-S encoder **and** the RAFT-DPT decoder both ride the GPU — no CPU/ONNX fallback. The demo
estimates depth on a bundled image and on any image picked from the gallery, rendering a depth colormap and
the near/far metric range.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **2447 / 2447** LITERT_CL (full residency) |
| compile | ~2.2 s (one-time) |
| inference | **~44 ms** |
| fp16 size | 78 MB |
| accuracy | depth corr **0.96** vs original Metric3D (robust 0.96–0.98 across indoor 0.7–4 m / mid 4–17 m / outdoor 11–200 m scenes) |

```
image[1,3,448,448] (ImageNet-normalized) →[GPU: DINOv2 ViT-S → RAFT-DPT (4 iters)]→ depth[1,1,448,448] (meters)
```

The model outputs depth for a **canonical camera** (focal 1000 at the canonical resolution). For a
calibrated camera multiply by `fx / 1000` (the de-canonical transform) using the real focal length; with no
intrinsics the demo shows the raw canonical-metric depth, which is already in meters and qualitatively correct.

## How it converts (and why it's fully GPU)

Input is fixed at **448×448** (= 32×32 patches = 1024 tokens, the device-proven token count). Every GPU
blocker is re-authored to a numerically-equivalent GPU-clean form (tflite-vs-torch corr **1.0**):

**Encoder — DINOv2 ViT-S/14 + register tokens** (the same suite as MoGe-2):
- fused-QKV 5D reshape → 3 separate q/k/v Linear + manual 4D attention
- LayerScale γ baked into `attn.proj` / `mlp` last Linear (eliminates the MUL shape clash)
- bicubic `interpolate_pos_encoding` → baked constant pos-embed for the fixed 32×32 grid

**Decoder — RAFTDepthNormalDPT5** (the new work):
1. **Convex upsample (6/7-D)** → **depth-to-space via `ZeroStuffConvT2d`**: 16 per-subpixel
   softmax-over-9-neighbour combines (kept 4D) → `cat → [N,96,H,W]` → a fixed `ConvTranspose2d(96→6, k4, s4)`
   wrapped in `ZeroStuffConvT2d`. **Device-critical:** the naive "nearest-upsample ×4 + mask at the in-block
   offset" gives correct results on desktop but **corr 0.57 on Mali** — the ML Drift `RESIZE_NEAREST` uses a
   different half-pixel/rounding convention at *non-stride* positions. `ZeroStuffConvT2d` masks **only
   stride-aligned positions** (exact under any nearest convention) and the conv kernel places the in-block
   offset, so it is robust.
2. **`ConvTranspose2d`** (DPT `Token2Feature`, 2× upsample) → **`ZeroStuffConvT2d`** (Mali rejects `TRANSPOSE_CONV`).
3. **GELU → accurate tanh approximation** (`0.5x(1+tanh(0.7978(x+0.044715x³)))`, POW-free). The usual
   `x·sigmoid(1.702x)` approximation is **not** good enough here: at the coarse top of the log-depth bin range
   (up to 200 m) its error is amplified and depth corr collapses to **0.51**; the tanh form restores **0.96**.
4. **`norm_normalize` `F.elu`** (→ `SELECT`) rewritten SELECT-free as `exp(-relu(-k))+relu(k)+min_κ` (exact).
5. The DPT `ConvBlock` uses `nn.ReLU(inplace=True)` on its residual input, so the residual is `relu(x)+convs`,
   **not** `x+convs` — replicated exactly.

The whole model is GPU-clean (banned ops NONE, all tensors ≤4D) and runs in one LITERT_CL partition.

## Files

| file | what |
|---|---|
| `app/src/main/java/com/metric3d/MetricDepth.kt` | model wrapper (ImageNet norm → CompiledModel GPU → depth) |
| `app/src/main/java/com/metric3d/MainActivity.kt` | image picker + depth colormap UI |
| `scripts/build_m3d.py` | conversion: re-author + op-check + fp16 + parity |
| `scripts/load_m3d.py` | mmcv-stub loader for `torch.hub` Metric3D |
| `scripts/device_gate.py` | real-image torch-vs-tflite parity fixtures |
| `scripts/install_to_device.sh` | push `metric3d_fp16.tflite` into the app's `filesDir` |

## Build & run

```bash
# 1) Convert (needs the litert-torch conversion env; produces metric3d_fp16.tflite)
python scripts/build_m3d.py all          # or download from Hugging Face

# 2) Build + install the app
./gradlew :app:installDebug

# 3) Push the model into the app's private storage, then launch
./scripts/install_to_device.sh <dir-with-metric3d_fp16.tflite>
```

The first launch fails with "Model not found" until the install script has been run.

Model: `mlboydaisuke/Metric3D-v2-LiteRT` (Hugging Face). Upstream: Metric3D v2, BSD-2-Clause.
