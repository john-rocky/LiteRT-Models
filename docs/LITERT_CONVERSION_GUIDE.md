# LiteRT Model Conversion Guide

Practical findings from converting various model architectures to TFLite for CompiledModel GPU inference on Android.

## Conversion Tools Comparison

| Tool | Best For | Avoid For | Layout |
|------|----------|-----------|--------|
| **litert-torch** | Vision Transformers, attention models | Models with dynamic control flow | NCHW (preserved) |
| **onnx2tf** | Pure CNN models (YOLO, ESRGAN) | ViT, attention layers (destroys accuracy) | NHWC (converted) |
| **SavedModel → TFLiteConverter** | Models already in TF/Keras | PyTorch-only models | NHWC |
| **Native Keras reimplementation** | Maximum accuracy control | Quick prototyping | NHWC |

## Vision Transformer (ViT) Models

### The Problem

onnx2tf converts NCHW → NHWC during conversion. For CNNs this works fine, but **attention mechanisms break** because onnx2tf incorrectly transposes batch/spatial dimensions in MatMul operations.

Measured accuracy loss with onnx2tf on TinyViT (MobileSAM encoder):
- **onnx2tf**: corr = 0.29 (unusable)
- **litert-torch**: corr = 0.99 (excellent)

### The Solution: litert-torch

```python
import litert_torch

model.eval()
dummy = torch.randn(1, 3, 1024, 1024)
result = litert_torch.convert(model, (dummy,))

# Save — returns TfLiteModel object
result.export("model.tflite")
```

**Key points**:
- Preserves NCHW layout (no transpose errors)
- Output is `TfLiteModel` object — use `.export("path")` to save
- Requires `torch.export` compatibility (no dynamic control flow)
- `F.interpolate` must use `align_corners=False` (GPU rejects `half_pixel_centers=True` + `align_corners=True`)

### GELU Handling

TFLite has no native `Erf` op. The standard GELU `x * 0.5 * (1 + erf(x/√2))` produces FlexErf ops.

**Solution**: Replace with sigmoid approximation before conversion:

**When downstream is latent/logit-sensitive** (continuous latents, flow/diffusion heads), the
classic approximations measurably shift outputs (Pocket TTS: tanh-GELU moved flow latents by up
to 0.1). Use the fitted odd tanh-polynomial instead — max |gelu err| 7.1e-5, still MUL/ADD/TANH/RELU
only: `erf(z) ≈ tanh(z(c1 + z²(c3 + z²(c5 + z²·c7))))`, c1=1.1280827604, c3=0.10434222081,
c5=-1.9996018773e-3, c7=4.5717509263e-5, with z clamped to ±5.5 via RELUs before the Horner so
fp16 never sees large powers. Implementation: `ErfGELU` in `pockettts/scripts/build_pockettts.py`.

```python
class SigmoidGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

# Replace all nn.GELU modules
for name, child in model.named_modules():
    if isinstance(child, nn.GELU):
        setattr(parent, name, SigmoidGELU())

# Also patch functional calls
F.gelu = lambda x, approximate='none': x * torch.sigmoid(1.702 * x)
```

Max error vs real GELU: ~0.01 (negligible).

### ONNX Graph Surgery (Alternative)

If you must use onnx2tf (e.g., for a mixed CNN+attention model), you can replace Erf nodes in the ONNX graph:

```python
# erf(z) ≈ 2 * sigmoid(2.407 * z) - 1
# Coefficient: 1.702 * √2 = 2.407
```

**Warning**: Even with correct Erf replacement, onnx2tf still breaks attention accuracy. This only eliminates FlexErf ops — the underlying NCHW→NHWC issue remains.

## CNN Models (YOLO, ESRGAN)

### onnx2tf Works Well

For pure CNN architectures, onnx2tf is the recommended path:

```bash
onnx2tf -i model.onnx -o output/ -osd
```

Key flags:
- `-osd`: Output SavedModel directory
- `-ois input:1,3,H,W`: Override input shape
- `-dsm`: Disable strict mode (skip accuracy correction if it errors)
- `-ebu`: Enable BatchMatMul unfold (for models with matmul ops)

### GPU-Incompatible Ops

Common ops that prevent CompiledModel GPU:
- `TOPK_V2`, `GATHER`, `GATHER_ND` — Reconvert with these ops removed
- `PACK`, `SPLIT` — Use SavedModel export path instead
- `CAST` (float↔int) — Keep everything as float
- `Erf` (FlexErf) — Replace with sigmoid approximation
- Dynamic `RESHAPE` with -1 dimensions — Use static shapes
- `RESIZE_BILINEAR` with `align_corners=True` — Use `align_corners=False`

### 4D Tensor Limit (Critical)

CompiledModel GPU (ML Drift) only supports **4D tensors (BHWC)**. Any intermediate tensor with 5+ dimensions causes compilation failure. Window partition (Swin, 2D perceivers) is the classic offender — `view(B, Hg, w, Wg, w, C)` is 6D.

Standard ViT (global attention) works because Q/K/V are always 4D: `(B, heads, tokens, dim)`.

**Window partition CAN be made 4D (EdgeTAM 2D Spatial Perceiver).** A non-overlapping window partition `(B,H,W,C) → (B·nWin, w, w, C)` is exactly a **space-to-depth**, which can be done with a single **grouped one-hot `Conv2d`** (stride `w`, `groups=C`, weight `[c·w²+i·w+j, 0, i, j] = 1`) followed by 4D `view`/`permute`/`reshape` (drop `B=1`). This stays ≤4D and is GPU-clean. Naive alternatives fail: a `view(...,−1,2)`/reshape route produces the 6D tensor; `F.pixel_unshuffle` also lowers to 6D; strided slicing `x[:, :, i::w, j::w]` lowers to `GATHER_ND` (banned). The grouped-conv space-to-depth is the one that works — so window attention with a *fixed* window size is **not** fundamentally GPU-incompatible (only dynamic/variable partitions are).

**On-device-only: ops on constant-only inputs are rejected.** Beyond the desktop op-blocklist, ML Drift's compiler rejects `MEAN` / `DIV` / `SELECT` (and similar) when **all their inputs are constants** (the desktop GPU_BAD-name check passes; the on-device *compile* fails with e.g. `MEAN: Expected 1 const input tensor(s), but node has 2 const input(s)`). Seen in EdgeTAM's perceiver: (a) `LayerNorm` applied to a constant `latents` parameter → `MEAN` over a const → **taint the constant to runtime** with `+ 1e-9 * x.mean()` (non-folding, numerically negligible); (b) softmax over a single-element sequence (1 token attending to itself) → `exp(0)/exp(0)` = `DIV` of a tensor by itself → **special-case `seq_len==1`** (the attention weight is identically 1.0, so the output is just the value); (c) a runtime `sine` position-encoding that emits `GATHER_ND`/>4D → **bake it to a constant** for the fixed feature size.

## litert_gpu_toolkit — canonical patch catalog

The patches described throughout this guide are packaged in `litert_gpu_toolkit/` at the
repo root. **Import from the toolkit instead of re-implementing inline in a conversion
script** — every re-authoring below is numerically verified against its PyTorch reference
(`float-noise` level unless noted).

`convert_for_gpu(model, dummy_input, output_path)` applies the always-safe set
automatically. Everything else is opt-in:

| Utility | Fixes | When to use | Proven in |
|---|---|---|---|
| `SigmoidGELU` / `patch_gelu` | `Erf` (FlexErf) ban | Default GELU replacement (fp16-safe, err ~0.01) | ViT backbones everywhere |
| `TanhGELU` / `patch_gelu(m, approximation="tanh")` | same | Regression heads where 0.01 shifts output | Metric3D, D-FINE |
| `ZeroStuffConvT1d/2d` / `patch_conv_transpose(m, dummy)` | `TRANSPOSE_CONV` rejected on device | Any deconv decoder; exact incl. grouped/output_padding | DAC, Matcha, Mimi, EDSR, PP-OCR, DewarpNet, TwinLiteNet |
| `pixelshuffle_to_conv_transpose(r, c)` | PixelShuffle → 6D reshape | Swap manually, then `patch_conv_transpose` | EDSR x4 |
| `ZeroPadMaxPool` / `patch_maxpool_zeropad` | PADV2(-inf) rejected | Padded MaxPool on non-negative input (ResNet stems) | Places365, PlantNet, BiSeNet, SINet-V2 |
| `patch_safe_layernorm(scale=...)` | fp16 sum-of-squares overflow in LayerNorm | Device output wrong at full GPU residency; `adaptive_v2` default | Parakeet, RF-DETR, NAFNet, D-FINE |
| `safe_rms` / `patch_rmsnorm` | fp16 overflow in RMSNorm (deep residual stacks) | Output collapses to 0 on device | Qwen3 embedding/reranking |
| `hierarchical_mean` / `SafeInstanceNorm2d` / `patch_instance_norm` | fp16 overflow in global spatial reductions | Large maps; **pow2 spatial dims only** | MODNet |
| `patch_grid_sample` | `grid_sample` → GATHER_ND | Deformable attention, fixed-size value maps | RF-DETR |
| `ManualGroupNorm` / `patch_groupnorm` | GroupNorm unsupported | always-safe set | DSINE, Matcha |
| `patch_window_attention` / `patch_patch_merging` | Swin GATHER_ND / 6D | always-safe set | Swin variants |
| `patch_weight_standardization` | Conv2d_WS dynamic weight norm | always-safe set | DSINE |
| `patch_interpolate` / `patch_normalize` / `patch_einops` | align_corners / div-broadcast / einops 6D | always-safe set (global monkey-patches) | various |

After any fp16-wall patch (`safe_*`), re-verify on device — desktop CPU/GPU parity does
not exercise the delegate's fp16 accumulation (residency ≠ correctness).

### Latency figures: time run() + readback

`CompiledModel.run()` only enqueues the GPU work; the output readback is what waits for it. Time the
two together (medians, with the thermal status) or the number is the enqueue. Figures in this
repository dated before 2026-08 may be run()-only: ormbg's "~10 ms/frame on a Pixel 8a" (a 1024²
ISNet, ~320 GFLOPs) measured 246 ms with the readback on 2026-09-05 (`ormbg/INTEGRATION.md`); DIS
quotes "~11 ms" for the same shape and has not been re-measured. DINOv2 ViT-S/14 has the same
signature: "~8 ms" in its Pixel 8a conversion note against 53.58 ms on a Galaxy S26 GPU with the
readback (`npubench`, 2026-08) — the S26 GPU is not 6× slower than a Pixel 8a, so the note is the
enqueue. Not re-measured (2026-09-08).

## Three PyTorch → Android routes on one model (measured 2026-09-05)

One model (conv stem + `nn.MultiheadAttention` block + head, 1×3×224×224, random weights shared by every route), three converters, one parity protocol (golden + 8 random inputs, atol 1e-4 / rtol 1e-3, argmax equal, reference = eager PyTorch fp32). Identical results on Python 3.12.13 and 3.14.6 (torch 2.13.0, M4 Max).

| Route | Converter call | Artifact | Worst max abs / rel diff vs PyTorch |
|---|---|---:|---|
| litert-torch 0.9.4 → ai-edge-litert 2.2.0 | `litert_torch.convert(model, (x,)).export("model.tflite")` | 1,073,260 B | 6.9e-7 / 7.5e-4 (Interpreter and `CompiledModel` CPU identical) |
| ExecuTorch 1.4.1, XNNPACK partitioner | `to_edge_transform_and_lower(torch.export.export(model, (x,)), partitioner=[XnnpackPartitioner()]).to_executorch()` | 1,073,320 B | 6.9e-7 / 6.0e-4 |
| torch.onnx.export (opset 18) → onnxruntime 1.29.0 | `torch.onnx.export(model, (x,), "model.onnx", opset_version=18, dynamo=False)` (MHA fast path disabled first) | 1,065,628 B | 7.5e-7 / 2.9e-4 |

`CompiledModel` GPU (macOS Metal backend of the same runtime) on the litert-torch artifact:

| Graph | Result | vs fp32 PyTorch |
|---|---|---|
| unmodified `nn.MultiheadAttention` | refused: `RESHAPE: Tensor dimensions must be less than 5` ×2, `TRANSPOSE: Permutation for transpose is invalid` (the 5-D head split; same rule as the Android GPU delegate) | — |
| attention re-expressed in 4-D `(B, heads, N, head_dim)`, same weights | fully accelerated, default fp16 | 2.8e-3, argmax 9/9 |
| same 4-D graph, `GpuOptions(enforce_f32=True)` | fully accelerated | 3.6e-7 |

Android side: `org.pytorch:executorch-android:1.4.0` ships only `XnnpackBackend`; `onnxruntime-android` 1.29.0's NNAPI provider loads on an API 36 emulator but NNAPI is deprecated from Android 15; `com.google.ai.edge.litert:litert:2.2.0` gives GPU/NPU through `CompiledModel.Options`. Write-up with the emulator numbers: pending publication — until then see the naming table in [README § LiteRT or TensorFlow Lite? The names](../README.md#litert-or-tensorflow-lite-the-names) <!-- TODO(lane B, articles): replace this in-repo anchor with the published article URL -->

## Model-Specific Notes

### MobileSAM

| Component | Format | Converter | Reason |
|-----------|--------|-----------|--------|
| Encoder (TinyViT) | TFLite | litert-torch | ViT attention |
| Decoder (MaskDecoder) | ONNX | torch.onnx.export | Boolean indexing + cross-attention incompatible with all TFLite converters |

Decoder limitations tried:
- onnx2tf: BatchMatMul shape mismatch in cross-attention
- litert-torch: `NonConcreteBooleanIndexError` in mask selection
- onnx_tf: Works but produces FlexErf ops (no GPU)

### RMBG-1.4 (ISNet)

Converter: litert-torch (pure CNN, 247 ops, all GPU-compatible).

Key points:
- ISNet is a U2-Net variant — only Conv2d, BN, ReLU, MaxPool, bilinear upsample, concat, sigmoid
- Model outputs 6 side masks — wrap with `model(x)[0][0]` to get primary mask
- Normalization: `(pixel/255 - 0.5)`, NOT ImageNet mean/std
- Output is sigmoid-activated (0-1), no additional sigmoid needed
- `F.interpolate` must use `align_corners=False` for GPU compatibility

### BiRefNet-lite (Swin Transformer) — NOT GPU-compatible

**Attempted and failed.** Swin Transformer's window attention creates 5D+ tensors (`[B, num_windows, 1, 49, 49]`) that CompiledModel GPU rejects (4D max). This is an architectural limitation, not a conversion issue. Patches attempted:
- Replaced GATHER_ND (relative position bias → pre-computed static tensors)
- Replaced SELECT/NOT_EQUAL (attention masks → pre-computed)
- Replaced DeformableConv2d → regular Conv2d
- Replaced GELU → sigmoid approximation
- Duplicated backbone for dual-resolution pass

All ops became TFLite-native but the 5D tensor constraint blocked GPU compilation. **Swin Transformer ≠ CompiledModel GPU.**

### YOLO11 / YOLO26

Converter: SavedModel → TFLiteConverter (eliminates PACK/SPLIT from Ultralytics export).

### YOLO26 Pose

Converter: **litert-torch** (NOT onnx2tf — see below).

Output: NCHW `[1, 3, 384, 384]` → `[1, 56, 3024]` where `56 = 4 bbox (cx,cy,w,h) + 1 person conf + 17 keypoints * 3 (x,y,vis)`.

**Bypass the end-to-end head**: the default YOLO26 head emits `(N, 300, 6+kp)` after `torch.topk`, which compiles to `TOPK_V2/GATHER` and is rejected by CompiledModel GPU. Drop the topk by flipping three flags on the head module before forward:

```python
yolo = YOLO("yolo26n-pose.pt")
head = yolo.model.model[-1]
head.end2end = False  # bypass NMS-free TopK / Gather
head.export = True    # use the export-mode forward path
head.format = "tflite"
```

This exposes the legacy one-to-many head output `[1, 56, N]`. **Bbox channels are `(cx, cy, w, h)` in input image pixel space — NOT `(x1, y1, x2, y2)`.** The xyxy form is only emitted by the end-to-end head we just disabled. Keypoint xy are also in input pixel space; conf and keypoint visibility are sigmoid-activated.

**Why not onnx2tf**: Ultralytics' default TFLite export pipeline goes ONNX → onnx2tf, but onnx2tf trips a channel-tracking bug at the YOLO26 backbone's `model.2/m.0/Add` (`Dimensions must be equal, but are 32 and 16`). This is the same class of failure that breaks ViT attention through onnx2tf — the tool mis-tracks NCHW channel positions through residual paths in newer YOLO blocks.

**`BATCH_MATMUL` is a false alarm**: `litert_gpu_toolkit`'s checker historically flagged `BATCH_MATMUL` as incompatible. The C2PSA attention block produces 4 BMM ops, and the existing `yolo26n.tflite` in this repo also has 4 BMM ops — both run cleanly on the LiteRT GPU delegate (`DELEGATE: 3` in op distribution). Treat `BATCH_MATMUL` as a warning, not a blocker.

### Real-ESRGAN

Converter: onnx2tf (pure CNN, no issues).

### MoGe-2 (DINOv2 ViT-S)

Converter: litert-torch. Most complex conversion in the repo — 9 patches required.

**Architecture**: DINOv2 ViT-S backbone (12 blocks, 384 dim, 6 heads) + ConvStack multi-scale decoder with 4 heads (points, normal, mask, scale). 35M params, 835 TFLite ops, 136 MB.

**Critical finding — LayerScale breaks GPU delegate**: DINOv2 uses `LayerScale` (per-channel gamma multiply) after each attention and MLP block. The FC output is a 2D tensor `[N, C]` which the GPU delegate interprets as `{N, 1, 1, C}` (batch=N). The subsequent LayerScale MUL with `[1, 1, C]` triggers a shape conflict: `{1, 1, N, C}` vs `{N, 1, 1, C}`. SmolVLM's SigLIP works because it has no LayerScale. **Fix**: bake gamma into the preceding Linear's weight and bias, eliminating the MUL entirely.

**Other patches**:
- Fused qkv `Linear(dim, 3*dim)` → 3 separate `Linear(dim, dim)` to avoid 5D reshape+unbind
- `torch.stack` of multi-layer features → element-wise add
- Position embedding interpolation (bicubic → pre-computed buffer for fixed 32×32 grid)
- `ConvTranspose2d` → `F.interpolate(bilinear, 2x)` + `Conv2d(1x1)` (TRANSPOSE_CONV rejected by Pixel 8a delegate despite desktop checker saying compatible)
- Constant UV buffers need `+ image_slice * 1e-10` to prevent constant folding — GPU delegate rejects Conv2d with constant-only inputs ("input must be a runtime tensor")
- `nn.Upsample(scale_factor=2)` → fixed-size `F.interpolate` (dynamic RESIZE_BILINEAR rejected)
- `padding_mode='replicate'` → `'zeros'` (×40 Conv2d layers)
- `F.interpolate` bicubic → bilinear
- **Global average pool `x.mean((2,3))` → two single-axis means `x.mean(3).mean(2)`** (EdgeTAM RepViT SqueezeExcite). A multi-axis `mean`/`SUM` reducing a large spatial extent (~65k elements) lowers to a single multi-axis `SUM` op that the Pixel 8a ML Drift delegate **mis-computes → silent NaN** (FP32 too, so it is not an FP16 overflow). The graph compiles and runs; only the output is garbage. Splitting into two sequential single-axis reductions is numerically identical and computes correctly. `F.avg_pool2d(x, kernel=spatial)` (→ `AVERAGE_POOL_2D`) also works; `F.adaptive_avg_pool2d(x,1)` does **not** (still a single multi-axis `SUM`).

**Key lesson**: The desktop GPU compatibility checker (checking op names against a blocklist) is necessary but not sufficient. The on-device ML Drift GPU delegate imposes additional constraints: no constant-only Conv2d inputs, no TRANSPOSE_CONV, no dynamic RESIZE sizes, and FC output shape interpretation depends on surrounding ops (LayerScale MUL specifically). **There is also a "compiles + runs but silently mis-computes" class** — e.g. multi-axis reductions over large tensors returning NaN, or a transformer block whose residual **collapses only when fused** into a large graph at high activation magnitude (correct as a standalone graph — see Matcha-TTS) — that neither the desktop checker nor a compile/run smoke test catches. Only an on-device GPU-vs-CPU numeric comparison (CPU is the trusted reference) catches it; bisect with sub-graphs that each output an intermediate to localize the broken op.

### Roboflow Soccer (YOLOv8x detect + YOLOv8x pose)

Sister project: `~/Downloads/SoccerAIDemo`. Ports Roboflow's
[Soccer AI](https://github.com/roboflow/sports/tree/main/examples/soccer) end-to-end
to Android (player detection + 32-keypoint pitch detection + ByteTrack + SigLIP
team classification + radar via DLT homography).

Converter: **litert-torch** for both YOLOs. The Roboflow YOLOv8x weights trip the
**same** onnx2tf channel-tracking bug as YOLO26 — failure at `model.2/m.0/Add`,
`Dimensions must be equal, but are 160 and 80` for an imgsz=640 export. The flag
recipe is identical to the YOLO26 Pose section above:

```python
head = yolo.model.model[-1]
head.end2end = False
head.export = True
head.format = "tflite"
```

Outputs:
- `football-player-detection.tflite` — 260 MB FP32, NCHW `[1, 3, 640, 640]` →
  `[1, 8, 8400]` (4 bbox + 4 class scores: ball, goalkeeper, player, referee).
- `football-pitch-detection.tflite` — 267 MB FP32, NCHW `[1, 3, 640, 640]` →
  `[1, 101, 8400]` (4 bbox + 1 class score + 32 keypoints × 3 (x, y, vis)).

Both pass `litert_gpu_toolkit` GPU compatibility check (`Status: COMPATIBLE`,
ops: CONV_2D / MUL / LOGISTIC / ADD / SLICE / CONCATENATION / TRANSPOSE / RESHAPE /
PAD / DELEGATE — no banned ops, no BATCH_MATMUL false alarm).

**Pitch keypoint order is non-trivial**: the model emits keypoints in the order
defined by `sports/configs/soccer.py` `labels`:
`01..13, 15, 16, 17, 18, 20..32, 14, 19`. Indices 30 and 31 of the model output
are vertices 13 and 18 (1-indexed: 14, 19) — easy to miss; if you skip the
remap, the homography fits but the radar overlay collapses subtly. Fix: store
`keypointOrderToVertex[i] = int(labels[i]) - 1` and apply it before pairing
keypoints with `SoccerPitchConfiguration.vertices` for DLT.

**Homography for the radar view**: pure-Kotlin DLT (8-parameter system, Gaussian
elimination on the normal equations) is sufficient for drone-altitude footage.
SVD / Hartley normalization not needed — the keypoint coord ranges and pitch
coord ranges (cm) are similar in magnitude. RANSAC may help for very oblique
ground-level shots.

### SigLIP-Base (vision-only, for clustering / feature extraction)

Same recipe as the SmolVLM SigLIP wrapper (SigmoidGELU, position embedding
pre-computation, patch_embedding `padding=0`, manual L2 normalization), minus the
pixel-shuffle connector. For Soccer team classification we just need the
mean-pooled L2-normalized feature; UMAP from the Python sample is replaced with
direct KMeans(k=2) on 768-dim embeddings (no dim-reduction needed for binary
clustering of distinct uniforms).

Converter: **litert-torch** (ViT requires it). Output: NCHW `[1, 3, 224, 224]` →
`[1, 768]`, ~327 MB FP32. Too large for APK assets — install to app `filesDir`
via the `install_<model>_to_device.sh` pattern used elsewhere in this repo.

**Surprise that wasn't broken**: `Skipping import of cpp extensions due to
incompatible torch version` (cpp ext requires torch >= 2.11.0; venv has 2.9.1)
prints a warning but the pure-Python fallback path still produces a valid TFLite
file via the SavedModel intermediate. Don't waste time chasing the warning —
verify with a numerical sanity check (`output norm == 1.0`) and move on.

### DeepPhonemizer (English G2P) — sequence model → LiteRT (free-text TTS input)

A non-vision case: an on-device **grapheme-to-phoneme** model that makes free-text TTS input work
on **LiteRT** (`scripts/convert_dp_g2p_litert.py`, used by the Kokoro sample's `NeuralG2p.kt`).
Source: DeepPhonemizer `en_us_cmudict_forward` (**MIT**), a non-autoregressive forward Transformer,
char → stress-less ARPABET. Converted via litert-torch and run on the **CompiledModel CPU**
accelerator (the consumer app already does its TTS on ORT, but the *G2P* is genuinely LiteRT).

Lessons worth keeping:

- **Variable length does NOT convert (the headline blocker).** Exporting with a dynamic sequence
  `Dim` fails: `Shapes must be 1D sequences of concrete values of integer type, got Traced<int32[]>`
  — litert-torch can't carry the symbolic seq length through the transformer's reshapes. This is the
  same class as the already-reported variable-length converter bug. **Workaround**: a single
  **static `[1, 96]`** graph; right-pad every word, decode back to its real length.
- **Compute the padding mask IN-GRAPH and keep ONE input.** With static max length you must mask, or
  attention over pad corrupts the real positions. Build `pad_mask = (ids == 0)` inside `forward` so
  the Kotlin side passes just one tensor. The `eq`/`SELECT_V2`/`CAST` this adds are CPU-fine (only
  the GPU delegate bans them) — and this G2P is CPU-only anyway.
- **FLOAT input, not int.** Feed char ids as **float32** `[1, 96]` and `ids = text.to(int64)` inside
  the graph. Lets Kotlin use the proven `CompiledModel.writeFloat`/`readFloat` path (the int
  TensorBuffer path in litert 2.1.3 is fiddlier). Small ids are exact in fp32.
- **CPU, not GPU.** Op-check shows `EQUAL`, `SELECT_V2`, `CAST`, and **>4D ×12** (MHA head-split 5D,
  the same C12 fused-attention shape as DA3/MoGe). So `CompiledModel.Options(Accelerator.CPU)`. To
  reach GPU you'd decompose attention to 4D + drop the eq/select — not worth it for a rare fallback.
- **The I/O contract lives in two places** — keep the exporter and `NeuralG2p.kt` in sync:
  `char_repeats=3` input expansion (`[<lang>] + id×3 + [<end>]`) and the **CTC greedy decode**
  (argmax per position → collapse consecutive dups → drop pad/blank `0` and lang/end ids). The
  model is CTC, not 1:1-aligned — an every-3rd subsample looks plausible but silently drops phonemes.
- **macOS converter snags**: litert-torch's min-cut layout pass imports
  `scipy.sparse.csgraph.maximum_flow`, whose transitive `_propack` fails to `dlopen` — stub
  `scipy.sparse.linalg._propack` (SVD is unused by maximum_flow). And torch ≥ 2.6 defaults
  `weights_only=True`, but DeepPhonemizer checkpoints pickle classes → monkeypatch `torch.load`.

### DAC / neural audio codec (ConvTranspose1d + RVQ)

Converter: litert-torch. A neural audio codec (DAC, EnCodec, vocoders) splits into a GPU conv graph + a
CPU RVQ. Two walls (device-verified on Pixel 8a):

1. **ConvTranspose1d.** The real DAC decoder (`upsampling_ratios [8,5,4,2]`, kernel = 2·ratio) does NOT
   convert: the odd **stride-5** transposed conv fails legalization (`mhlo.convolution` `lhs_dilation=5`,
   "explicitly marked illegal"); even strides emit `TRANSPOSE_CONV` which Mali rejects. **Fix =
   `ZeroStuffConvT1d`** (the DA3 zero-stuff C20 trick generalized to 1D, kernel = 2·stride): nearest-upsample
   ×S **in 2D** (`x.unsqueeze(2)` → `F.interpolate(size=(1,L·S),"nearest")` → squeeze — the **1D** interpolate
   lowers to `GATHER_ND`, 2D → clean `RESIZE_NEAREST_NEIGHBOR`) × a constant mask buffer (1 at `::S`) → `conv1d`
   with `weight.flip(2).transpose(0,1)`, `padding=K-1` → crop `[P : P+((L-1)·S+K-2P+out_pad)]`. Numerically
   exact (corr 1.0). Per-layer input length captured via a forward-hook dry run. Applies to any vocoder / 1D
   U-Net decoder with transposed-conv upsamplers.

2. **RVQ → CPU.** The residual vector quantizer (codes ↔ latent) uses `EMBEDDING_LOOKUP` + **int64** code
   indices; on Mali the full codes→audio graph fails with `CAST: Tensor type(INT64) is not supported` +
   `EMBEDDING_LOOKUP: Empty quantization params` (only 464/578 nodes delegate). **Split it out**: run the RVQ
   on CPU (in_proj 1×1 → L2-normalize → cosine-argmax → codebook lookup → out_proj, residual loop; ~1 ms in
   Kotlin), feed the GPU decoder a continuous float latent. The float conv encoder/decoder then stay 100% on GPU.

**On-device (Pixel 8a):** DAC 16kHz encoder **367/367** + decoder **398/398** nodes on `LITERT_CL`, warm RTF
~0.82, reconstruction corr 1.0 vs PyTorch. Scripts: `dac/scripts/convert_dac_{encoder,deconly}.py` +
`dac_rvq_validate_export.py` (RVQ codes match torch 100%).

### Matcha-TTS (CFM acoustic model + HiFi-GAN vocoder) — the FFT-free TTS lane

Converter: litert-torch. Matcha-TTS pairs a conditional-flow-matching (CFM) acoustic model with a
**HiFi-GAN time-domain vocoder**, so there is **no FFT/iSTFT anywhere** in the synthesis path — this is what
lets a TTS model ride the GPU at all (spectral vocoders — Kokoro/iSTFTNet/Vocos — need an FFT kernel the ML
Drift delegate does not provide, so their spectral steps are forced host-side). Three graphs: text encoder,
CFM decoder (run per ODE step), HiFi-GAN vocoder; the Euler ODE loop / duration / length-regulator /
embedding / sinusoidal time-embed run host-side.

**Re-authoring (all numerically-equivalent, per-graph tflite-vs-torch corr 1.0, end-to-end waveform corr ≥0.99):**
`GroupNorm` → manual 4D mean/var; `nn.Mish` → SELECT-free fp16-safe softplus `x·tanh(relu(x)+log1p(exp(-|x|)))`;
`ConvTranspose1d` (Upsample1D) → `ZeroStuffConvT1d` (the DAC 1D trick above); diffusers `Attention` → manual
additive-masked attention; the half-res mask `mask[:,:,::2]` → reshape-decimate (a step-2 slice lowers to
`GATHER_ND`); `SinusoidalPosEmb` → host-side (weight-free sin/cos), the learned `time_mlp` stays on GPU.

**Variable length = pad-to-max + a runtime float mask** (256 phonemes, 512 mel frames). The mask is a runtime
graph input, not dropped: the decoder **adds the raw 0/1 mask** to attention scores (replicating diffusers
`AttnProcessor2_0`'s soft bias — NOT `-1e4`), the text encoder adds `(mask-1)·1e4` (replicating `masked_fill`).
Dropping the mask leaks pad frames through global attention (corr 0.936). With the runtime mask, one compiled
graph handles any length and matches torch exactly (corr 1.0).

**The decoder runs on CPU — a NEW on-device "compiles + runs + silently-wrong" failure mode (graph FUSION, not
an op).** On the Pixel 8a, the CFM decoder's diffusers transformer blocks **mis-fuse at large activation
magnitude**: the up-path transformer (input |x|~60) collapses its residual — device output ±0.7 vs CPU ±60,
**corr 0.006** — giving a NaN/garbled mel (the user hears a buzz/tone). The decisive isolation: the **same
transformer block converted as a STANDALONE graph computes correctly on the GPU (corr 0.984)**, so it is a
graph-fusion/scheduling bug, not a bad op (GroupNorm-4D, Mish, SnakeBeta, ZeroStuffConvT1d, the manual masked
attention are each verified correct on Mali via on-device tap dumps). **fp32 and fp16 both fail** (not a
precision/overflow bug) and it is NOT the "global-pool multi-axis mean → NaN" class above (that was a separate
first bug here, fixed with the `mean(3).mean(2)` split) nor the deep-ViT fp16 variance-overflow class — the
`SafeLayerNorm` scale-before-square fix does **not** help (it NaNs: the variance itself exceeds fp16 max and
the scaled eps underflows in the zero-variance pad). **Workaround:** load the decoder with
`CompiledModel.Options(Accelerator.CPU)` — it is exact on CPU, and the pipeline stays realtime (**RTF ~0.8 on
Pixel 8a**) because the GPU HiFi-GAN vocoder dominates wall time. Text encoder + vocoder stay on the GPU.
Minimal repro: `matcha/scripts/probe_tx_standalone.py` (standalone 0.984 vs fused 0.006). Localize fusion bugs
like this by emitting intermediates as extra graph outputs and comparing each stage device-vs-CPU on the same
inputs (`probe_decoder_taps.py`).

**G2P (espeak-free):** Matcha-LJSpeech is trained on espeak en-us IPA (GPL), so the runtime G2P is a 275k-entry
espeak-IPA dictionary (OpenPhonemizer, Clear BSD) primary + a DeepPhonemizer (MIT) `[1,96]` LiteRT CPU graph
for out-of-dictionary words; output IPA maps 1:1 onto the keithito 178-symbol set. The neural model **alone**
mispronounces common/function words ("this"→ðaɪz), so the dictionary must be primary (same hybrid as kokoro).

Scripts: `matcha/scripts/{build_matcha,convert_final,convert_g2p_matcha}.py`. Models:
[`litert-community/Matcha-TTS`](https://huggingface.co/litert-community/Matcha-TTS).

### Pocket TTS (Kyutai 100M) — flow-matching LM over continuous Mimi latents (litert 2.1.6)

Converter: litert-torch. Full recipe + parity numbers: `pockettts/` (fused packed-KV step graph,
LSD time-embeddings folded into the cond bias, 64-frame block decode). Delegate facts worth the
catalog:

- **KV-step `FULLY_CONNECTED` shapes need litert ≥ 2.1.5 on Mali** (2.1.3 rejects them — same
  class VibeVoice hit; 2.1.6 used here).
- **Block-decoding a sliding-window transformer needs overlap ≥ layers×(window−1)**, not one
  window: layer-2 keys are layer-1 outputs whose own windows reach further back (2 layers ×
  window 250 → 498 positions; overlap 256 was silently wrong from block 2 on, corr 0.999).
- **Mali "compiles + runs but degraded" case, transformer-shaped**: the 2-layer Mimi decoder
  transformer delegates fully (`LITERT_CL` 210/210) but its GPU OUTPUT is audibly degraded
  (voicing HNR 0.9 dB vs 2.8 dB on CPU = the fp32 reference), and `GpuOptions(precision=FP32)`
  does NOT recover it — so not fp16 rounding. Same graph class as the mimi/ module's decoder
  transformer and the VibeVoice σ-VAE finding: ships on CPU (7 calls/utterance, ~2%). The
  SEANet conv graph and the 6-layer LM on the same GPU are clean. Diagnose voice quality with
  HNR + high-band noise vs the fp32 reference, then bisect placement per graph.
- Mali per-step cost on packed-KV AR graphs is **dispatch/sync-bound, not FLOP-bound** (78-MMAC
  step ≈ 43 ms: 11 ms cache upload + ~30 ms spread over run + 4 readbacks). Fusing the flow head
  into the step graph and concatenating all outputs into ONE tensor (one invocation, one
  readback per frame) recovered ~8 ms/frame; Adreno runs the same graphs ~10× faster.

### Mimi (Kyutai 2024 codec) — the C33 generalization test (and its negative result)

Converter: litert-torch. Mimi (Kyutai/Moshi streaming codec, 24 kHz/12.5 Hz, hidden 512) is structurally a
codec with **two 8-layer LLM-style Transformers** in the path (`encoder_transformer`, `decoder_transformer`),
so it was the decisive test of whether the Matcha "transformer-collapses-when-fused" delegate bug (above) is a
**general** ML Drift bug or diffusers-`BasicTransformerBlock`-specific.

**Re-authoring (all GPU-clean, parity ~1.0):** GELU(erf)→**tanh-GELU** `0.5x(1+tanh(√(2/π)(x+0.044715x³)))`
(MUL/ADD/TANH, no POW; tanh beats sigmoid — transformer corr 0.991→0.99999); `MimiRotaryEmbedding`→**baked
const cos/sin + rotate_half** (kills the GATHER_ND position-gather); causal/sliding mask→**baked const additive
bias** `(1,1,S,S)` (NOT dropped — decode IS causal; kills CUMSUM/EQUAL/SELECT_V2); attention→manual
matmul+softmax ≤4D; `MimiLayerScale`→**bake γ into the preceding Linear** (o_proj/fc2); `ConvTranspose1d`
(the `upsample` is **depthwise**, groups=512!)→**grouped-aware `ZeroStuffConvT1d`** (generalize the weight
reshape `(Cin,Cout//G,K)→(Cout,Cin//G,K)`+flip, `F.conv1d(groups=g)`); `MimiConv1d` causal pad→**baked
constant `F.pad`** (its int64-buffer `.item()` is a dynamic value → jax `ConcretizationError` at trace time
otherwise); `nn.ELU`→**`relu(x)−relu(1−exp(min(x,0)))`** (SELECT-free, exact, fp16-safe — the SEANet's 13
ELUs were a `SELECT×13` blocker; EXP is GPU-clean); downsample `replicate`-pad→**SLICE+CONCAT edge-replication**
(tflite PAD is constant-only, replicate emits `GATHER_ND`). RVQ (split: 1 semantic + 31 acoustic, Euclidean
argmin)→**CPU** (int64 + EMBEDDING_LOOKUP, Mali-rejected; `MimiRvq.kt`, validated vs torch).

**On-device result (Pixel 8a) — C33 does NOT generalize.** The decoder transformer's residual stream reaches
**|x|=27**. On device it computes to corr **0.70** vs CPU — but **identically standalone and fused**
(standalone 0.6995 ≈ in-fused-graph tap 0.6987, same absmax 17.5), so this is **fp16 precision loss in the
large-magnitude residual** (L7 damps 27→4.4 via near-cancellation the fp16 compute can't hold), **NOT** a
fusion collapse. So the Matcha C33 bug is **diffusers-specific**, not a broad transformer-fusion bug. Key
differences from Matcha's C33: (a) standalone == fused here (Matcha: standalone 0.984, fused 0.006);
(b) **fp32 and fp16 models give identical device output** (the LITERT_CL delegate computes fp16 internally
regardless of stored precision); (c) `SafeLayerNorm`/sigmoid-GELU/safe-bias **hardening does not help** (it is
residual-accumulation cancellation, not a single op). The SEANet **convs are fp16-exact on GPU** (decoder-only
fed the exact transformer output = audio **48 dB**); full-GPU decode is ~12 dB on real speech (a synthetic
tone hides it). **Deployment = hybrid:** transformers→CPU (tiny: 8L×512×seq~50, trivial), SEANet convs→GPU;
4-graph split (enc_conv GPU, enc_tx CPU, dec_tx CPU, deconly GPU) + CPU RVQ. Pixel 8a **RTF ≈ 0.35**, audio at
the codec's quality floor. This mirrors the Matcha landing (transformer→CPU) but for a **different root cause**
(fp16 precision vs fusion bug). Scripts: `mimi/scripts/{build_mimi,build_hybrid_graphs,mimi_rvq_validate_export}.py`.

### wav2vec2 keyword spotting — all-GPU, and the whole-graph compile limit

Converter: litert-torch. `superb/wav2vec2-base-superb-ks` (Apache-2.0): raw 16 kHz waveform → 1D-conv
feature extractor → 12-layer transformer encoder → weighted-layer-sum → classifier. **No FFT anywhere**
(not even host-side mel — the frontend is conv on the raw waveform), and the transformer residual peaks
at only **|x|≈3.2**, so unlike Mimi there is **no fp16-precision issue: the whole model is fp16-exact on
GPU** (no CPU fallback). Device-verified Pixel 8a: 10/10 keywords correct, device-vs-CPU logits corr 0.9995.

**Re-authoring (all numerically-equivalent, parity corr 1.0):** `nn.GELU`/`GELUActivation` ×20 →
tanh-GELU; feature-extractor `nn.GroupNorm` (num_groups=channels) → GN4D (reshape `(B,G,C//G,T)` mean/var
over `(2,3)`; kills GATHER_ND); pos-conv (kernel-128 grouped Conv1d) `weight_norm` → **fold** to a static
weight (`remove_parametrizations(..., leave_parametrized=True)`; the runtime `_weight_norm` recompute is
otherwise live in-graph); `create_bidirectional_mask()` builds an all-valid mask even when
`attention_mask=None` (arange/ge/expand → SELECT_V2 + BROADCAST_TO) → **monkeypatch it to return None**
(fixed length, no padding → SDPA full attention = BATCH_MATMUL + SOFTMAX clean; also makes pooling a plain
`mean(dim=1)`).

**Two new on-device findings (both general):**
1. **Whole-graph Mali shader-compile limit.** A graph can be fully op-clean AND have each half compile,
   yet **fail to compile when fused** (`Failed to compile model`, the delegate reports e.g. "Replacing 923
   out of 1008 node(s) ... 2 partitions"). The full wav2vec2 graph fails; splitting at the conv-frontend /
   transformer-encoder boundary makes both halves compile (frontend 134/134 + head 893/893 LITERT_CL). This
   is a size/complexity ceiling, not a bad op — when a clean graph won't compile, split it.
2. **`use_weighted_layer_sum` heads on GPU.** This checkpoint's logits use a softmax-weighted sum of ALL 13
   hidden states, not just the last (dropping it flips predictions, corr 0.54 — replicate it exactly). On
   the GPU it must be (a) **accumulated incrementally** (`acc += w[i]·hᵢ` after each layer) — `torch.stack`
   of all 13 keeps every layer output live and splits the partition; and (b) the `softmax(layer_weights)`
   must be **baked to Python-float constants** — the runtime softmax + 13 scalar `w[i]` gathers off a
   runtime tensor break delegation into partitions (3 partitions → compile fail). Baked + incremental →
   893/893 LITERT_CL, 1 partition.

Scripts: `wav2vec2-kws/scripts/{build_w2v2,build_w2v2_split}.py`. Models:
[`litert-community/wav2vec2-keyword-spotting`](https://huggingface.co/litert-community/wav2vec2-keyword-spotting).

### PP-OCRv5 (PaddleOCR 2025) — fully-GPU OCR + ZeroStuffConvT2d

Converter: litert-torch via the **PaddleOCR2Pytorch** port (Apache-2.0, pure-torch, no PaddlePaddle dep;
weights from HF `JoyCN/PaddleOCR-Pytorch`). PP-OCRv5 is a classic CNN OCR pipeline — detection (DBNet:
PPLCNetV4 + RepLKFPN + DB head) + recognition (PPLCNetV3 + SVTR + **CTC head**). It was chosen over the
newer VLM-OCRs (Florence-2, GOT-OCR) precisely because it has **no autoregressive decoder** — the CTC head
means both stages ride the GPU with no CPU/ONNX fallback (a VLM-OCR's AR decoder hits the decoder KV-cache
wall and must run on CPU, the SmolVLM split). Apache-2.0, tiny (det 10MB + rec 17MB fp16). Device-verified
Pixel 8a: det 777/777 + rec 827/827 LITERT_CL, ~9ms each, a 3-line image read 3/3 correct.

**Two blockers, both re-authored (per-graph tflite-vs-torch corr 1.0):**
1. **Detector DB head `ConvTranspose2d` (2× k2s2)** → **`ZeroStuffConvT2d`** = the 2D generalization of the
   1D `ZeroStuffConvT1d` (DAC C20/C32): `F.interpolate` nearest ×s × a stride zero-stuff mask + flipped
   `conv2d(padding=k-1)` + crop. `TRANSPOSE_CONV` is Mali-rejected (#1061); this is RESIZE_NEAREST + MUL +
   CONV_2D, numerically exact (corr 1.0). Reusable for any deconv-upsample CNN head (seg/detection). Guard:
   skip the training-only DB `thresh` branch's ConvTranspose2d (not hit at inference).
2. **Recognizer SVTR `Attention` fused-QKV 5D reshape** `(B,N,3,heads,hd)` [the C12 pattern] → split q/k/v
   to 4D `(B,heads,N,hd)` (numerically identical). The port already drops the NRTR autoregressive branch →
   pure CTC. char_num = dict(18383) + blank + space = 18385; CTC layout = ['blank'] + dict + [' '].

Preprocessing: det = ImageNet mean/std, /255, NCHW, 640×640. rec = resize h=48 keep-aspect pad to 320,
(img/255−0.5)/0.5. DB box postprocess (threshold + connected-components + unclip) and CTC greedy decode are
host-side (Kotlin). **Env note:** `import _stub_propack` FIRST — a NARROW stub of only scipy `_propack`
(the macOS-27 zero-fill dlopen bug) that leaves scipy.optimize/signal real (the repo imports them); the
matcha `_stub` over-stubs scipy.optimize and breaks any librosa/scipy.signal user. Scripts:
`ppocr/scripts/{build_det,build_rec}.py`. Models: [`litert-community/PP-OCRv5-LiteRT`](https://huggingface.co/litert-community/PP-OCRv5-LiteRT).

### RF-DETR Nano (Roboflow / LW-DETR 2025) — first transformer/DETR detector fully on GPU (2-graph split + SafeLayerNorm)

Converter: **litert-torch** (`pip install rfdetr`, Apache-2.0). RF-DETR is a transformer detector
(windowed DINOv2-S backbone + deformable-attention DETR decoder, two-stage, 30.5M). The off-the-shelf
Qualcomm/onnx2tf export is GPU-incompatible (deformable `grid_sample`→GATHER_ND, windowed attn 5D/6D,
TOPK/GATHER) — but with litert-torch re-authoring + a 2-graph split it runs **100% on CompiledModel GPU**.
Device-verified Pixel 8a: Graph A 1381/1381 + Graph B 404/404 LITERT_CL, ≈27 ms; on a real image the
device chain reproduces the PyTorch detections at **IoU 0.98–0.99, same class** (the original
"RF-DETR does NOT ride CompiledModel cleanly" verdict is superseded).

**Re-authoring (per-graph tflite-vs-torch corr 1.0):**
1. **Windowed DINOv2 backbone** — 6D window-partition → a 5-step ≤4D reshape/permute (+ exact inverse for
   un-windowing); SDPA→manual 4D attn; `interpolate_pos_encoding` baked; cls `repeat`→`cat`; tanh-GELU.
   Only **3 of 12 layers are global attention** (rest windowed, 144-token) → backbone survives Mali fp16
   (corr 0.9998), unlike full-global DINOv2 (DA-V2 walled at 0.63). The windowing IS the fp16 mitigation.
2. **Deformable `grid_sample` → GATHER/CAST-free tent-matmul**: `wx=relu(1-|ix-px|)` over `arange(W)`,
   `W=outer(wy,wx)`, `out=input_flat @ W_flat.T` BMM — numerically exact incl. zeros-pad OOB, all ≤4D
   (replaces RF-DETR's own `_bilinear_grid_sample` which uses `.long()`+gather = banned).
3. **MSDeformAttn** re-authored ≤4D (n_levels=1, no 6D sampling tensors); **sine pos-embed** `dim_t` baked
   (kills POW/FLOOR_DIV) + strided interleave `[...,0::2]`→`reshape(d//2,2)` (kills GATHER_ND).
4. torch.export friction: `torch._shape_as_tensor`→const, `torch._assert`→no-op, `net.export()`.

**2-graph split (the ship path — standard for two-stage DETR on edge).** The query selection (top-300
proposals = TOPK_V2+GATHER) has no GPU op, but the proposal **grid is image-independent**, so split there:
- **Graph A (GPU)** = backbone(encoder+projector) + flatten + proposal-grid + enc heads → enc_class[1,576,91],
  enc_coord[1,576,4], memory[1,576,256]. Bake the grid as a const buffer (meshgrid→BROADCAST_TO else). The
  24² grid is all-valid so the validity masked_fill is a no-op (skip it; host needs no validity mask).
- **host (Kotlin)** = top-300 by `max(enc_class,-1)` (descending = torch.topk order) → gather enc_coord → ts.
  `memory_ts`/`boxes_ts` (hs_enc/ref_enc) are **dead at inference** (decoder tgt = learned query_feat; topk
  feeds only the reference points) → host does coord-gather only.
- **Graph B (GPU)** = two-stage reparam combine + 2-layer decoder + bbox/class heads → boxes[1,300,4]+logits.
  lite_refpoint_refine=True → decoder.bbox_embed=None → ref_unsigmoid = the input combined refpoint.

**⭐ fp16 hardening = SafeLayerNorm in BOTH the projector AND the decoder (device-only, not desktop):**
- The **MultiScaleProjector** fuses 4 backbone maps; ConvX outputs hit |x|~440 → channels-first LN channel
  sum-of-squares 256·440² OVERFLOWS fp16 (>65504) on Mali → device memory corr 1.0→0.58. Fix = projector LN
  → NAFNet SafeLayerNorm (down-scale by S=128 before reduce, exact) → 0.9999.
- Decoder layer-0 `nn.MultiheadAttention` output |x|~1068 (trained out_proj amplifies ~222×) → residual into
  norm1/norm3 overflows. Fix = nn.LayerNorm → **ADAPTIVE SafeLayerNorm** `S=max(1, amax/8)` per row. A FIXED
  large S squashes the small norms (final norm ~8 → logits 0.88→0.32) — adaptiveness is essential.
- The decoder logits still cap at device corr ~0.88 (transformer fp16 wall: near-one-hot attention scores
  ~300 → fp16 argmax flips per low-conf query; survivable at 2 layers) — but **real detections are perfect**
  (IoU 0.98–0.99). ⇒ ship criterion for detectors = detection IoU/class on a REAL image, NOT raw output corr.

Preprocessing: square resize 384×384, RGB, ImageNet mean/std, NCHW. Host: sigmoid + threshold + cxcywh→xyxy
+ per-class NMS (light, removes fp16 near-duplicate queries). Scripts: `rfdetr/scripts/build_rfdetr_split.py`
(imports build_rfdetr_full → build_rfdetr_bb). Models: [`litert-community/RF-DETR-Nano-LiteRT`](https://huggingface.co/litert-community/RF-DETR-Nano-LiteRT).

### Parakeet (NVIDIA FastConformer-CTC, ASR) — SafeLayerNorm **v2** (never rebuild the variance)

`parakeet-tdt_ctc-110m` (CTC branch, CC-BY-4.0): the 17-layer FastConformer encoder + CTC head run **fully on
the CompiledModel GPU** — the first big global-attention transformer in this zoo to survive the Mali fp16 path
end to end. On-device transcript matches PyTorch exactly (real-frame logits corr 0.99997), 3105/3105 ops on
LITERT_CL.

**The key finding — SafeLayerNorm v2.** The first device run gave corr 0.44 and a *blank* transcript, looking
exactly like the EoMT/DA-V2 "deep global-attention transformers wall on Mali fp16" verdict. A per-layer device
tap proved otherwise: N=0 (subsampling + pos only) = corr 1.0 but **|x| ≈ 7000**; N=1 (one conformer block) =
0.20, and *every* ablation (drop attention / conv / FFN / rel-shift / plain-LN) stayed 0.20 → a single block
already broken ⇒ a structural fault, not precision compounding. The dw-striding subsampling front-end
legitimately emits |x| ≈ 7000, so the first LayerNorm must normalize it — and **even the adaptive SafeLayerNorm
above overflows here**, because it *rebuilds* the variance: `var = mean(d²)·S²` with `S ≈ amax/8 ≈ 918` gives
`S² ≈ 8.4e5` and `var ≈ 2.5e7`, both **> fp16 max 65504** → `var = ∞` → `y = 0` → output = bias → corr 0.20.

Fix — **stay entirely in the down-scaled domain and never reconstruct the large variance** (the scale cancels
in `y = d/√var`):

```python
def safe_layernorm_v2(x, weight, bias, eps):           # x: [..., C]
    amax = x.abs().amax(-1, keepdim=True)
    S    = (amax * 0.125).clamp(min=1.0)               # per-row; native S=1 for small norms
    xs   = x / S                                       # down-scaled, O(1)
    mu   = xs.mean(-1, keepdim=True)
    d    = xs - mu
    var  = (d * d).mean(-1, keepdim=True)              # down-scaled variance — NEVER ·S²
    return d * torch.rsqrt(var + eps / (S * S)) * weight + bias  # eps / S² = eps in x's units; fp16-safe
```

Divide eps by S²: `var` is the variance of x/S, so a plain `+ eps` is eps·S² in the original units. On
Nemotron-3-Diarization (S ≈ 120) that moved the FP32 logits by 3.0e-3; `eps / (S * S)` keeps FP32 equal to
`nn.LayerNorm` (7.1e-5 after export). The Parakeet model itself was not re-measured with the corrected form.

Every intermediate stays `O(1)…O(amax)`, so it is overflow-free for any input magnitude — **use v2 in place of
the `var = mean(d²)·S²` form going forward.** After this, all encoder taps N=1..17 → device corr 1.0 and the
model ships. Diagnostic lesson: a "fp16 wall" that produces an **all-zero / all-blank** collapse, plus a tap
showing even *one* block broken, is a Safe-norm **overflow** (variance reconstruction), not precision
compounding (which starts near 1.0 and decays gradually). DA-V2 (|x| = 21.6) was a genuine precision wall and
stayed parked; Parakeet's was this overflow → fixable and shipped.

Other re-authoring: `RelPositionMultiHeadAttention` → manual ≤4D matmuls (no SDPA/cache); GLU → `a·sigmoid(b)`
(SPLIT banned); BatchNorm folds; CausalConv1d symmetric zero-pad; CTC `ConvASRDecoder` (Conv1d 512→1025) fused
into the graph. Variable length = a fixed 16 s window with the encoder masking folded into a **GPU-clean
additive attention bias** (`scores += (1-mask)·-3e4`) + a conv frame-mask, so audio ≤16 s is zero-padded
without contaminating real frames. NeMo and litert-torch cannot share a process (a jax/torch mutex) → convert
in two processes, each ending `os._exit(0)`. Host log-mel matches NeMo's preprocessor (note: the model uses
**preemphasis 0.97** even though the config says `None`); greedy-CTC + SentencePiece decode on the host.
Scripts: `parakeet/scripts/` (`build_parakeet_ship.py`, `build_parakeet_tap.py` = the per-layer tap/ablation
harness that nailed the SafeLayerNorm v2 fix). Model:
[`litert-community/Parakeet-tdt-ctc-110m-LiteRT`](https://huggingface.co/litert-community/Parakeet-tdt-ctc-110m-LiteRT).

### Nemotron-3-Diarization (Streaming Sortformer, 8 speakers) — SafeLayerNorm v2 eps fix, fixed-T cache packing, fp16 vs FP32 on Adreno

`nvidia/Nemotron-3-Diarization` (100M, OpenMDW-1.1): a 31-layer RoPE transformer encoder (hidden 512, 8 heads) over
80 ms frames, a sub-pixel Conv1d head back to 10 ms, and a streaming state (Arrival-Order Speaker Cache + FIFO) that
decides which past frames the encoder sees next. Split: graph A = 8-frame stacking + projection (2 ops), graph B =
encoder + head at a fixed T (2,915 ops, fully LITERT_CL, 1 partition on the S26), host = log-mel + sigmoid / pooling
+ the cache (Python and Kotlin ports of transformers' `Nemotron3DiarizationSpeakerCache`). With graph B at GPU
precision FP32 the S26 closed loop equals transformers on the 97.6 s example clip (0 flips in 78,072 cells, 37 / 37
segments, 4 / 4 cache compressions identical).

**SafeLayerNorm v2 needs eps / S².** The LayerNorm inputs reach |x| ≈ 956 (final norm; 804 in layer 30), so the
plain `(x − μ)²` (~9·10⁵) overflows fp16: on the S26 at default precision the plain-LN graph ran with no NaN and no
error but returned garbage (max |Δlogit| 38.2, logit correlation down to 0.0006, 87.8 % agreement). The v2 form
fixes the overflow, but as written in the Parakeet section it adds eps in the down-scaled domain, which is eps·S²
in the original units (S = amax/8 ≈ 120 here). That shifted the FP32 logits by 3.0e-3 vs transformers (a 1e-3 gate
failed); dividing eps by S² restores FP32 equality (7.1e-5 after export):

    d * torch.rsqrt(var + eps / (s * s)) * weight + bias      # var = down-scaled variance, never · s²

Open risk (not tested): eps / S² and even eps = 1e-5 are below fp16's normal range (6.1e-5). The S26 (Adreno)
showed no non-finite value, but a GPU that flushes fp16 subnormals would compute rsqrt(0) on an all-zero padding row
(S = 1, var = 0) → 0 · ∞ = NaN, and the padding keys' NaN values would reach real rows through attention
(0 · NaN). Check on Mali before claiming it there.

**The converter folds LayerNorm γ into the next Linear — only for the stock LayerNorm.** The checkpoint is 100 %
bfloat16 values (99.9945 % exact in fp16, the rest within 3.0e-8). With `nn.LayerNorm` the export folded
`layer_norm2.weight` into `fc1` (file weight = W·diag(γ₂) bit-exact), which is no longer bf16-exact, so the fp16
file lost accuracy (max |Δlogit| 8.3e-3, 3 flips). The SafeLayerNorm graph is not folded and its fp16 file equals
the fp32 file within 1e-7 per weight. Check a fp16 file against its fp32 export per constant, not only end to end.

**Fixed-T cache packing.** Every step packs `[cache ≤ 264 | FIFO ≤ 264 | chunk 9 + look-ahead 4]` = L ≤ 541 rows and
zero rows to T = 541 (offline: 264 + 40 + 380 = 684). Three details made the fixed graph equal the variable-length
reference:
1. `attn_bias [1,1,1,T]` added to the scores: 0 valid key, −3e4 pad key. Positions restart at 0 every chunk, so the
   RoPE tables are the same every step; they are graph inputs (no large baked constant), written once per compile.
2. A row mask before the head's k = 3 convolution, derived in-graph from the same bias: `y *= relu(attn_bias + 1)`.
   The reference convolves exactly L rows; without the mask the last real row's logits moved by up to 13.2.
3. Offline only: the pass masks the key of the frame after the last full hop but still runs it through the head.
   A three-level bias (0 / −16384 masked key / −32768 pad) with `relu(y) − relu(y − 1)`, `y = bias·2⁻¹⁴ + 2`, keeps
   the mask exact in fp16 and avoids RELU_0_TO_1; feeding that row as padding moved the logits by 11.4.

**fp16 vs FP32 on Adreno: judge a stateful model on the closed loop.** `CompiledModel.GpuOptions(precision = …)`:
graph B FP32 136 ms, FP16_WITH_FP32_ACCUM 106 ms, default (fp16) 80 ms per step. One step fed with the reference's
inputs looks fine at fp16 (max |Δp| 0.035, 1 flip in 155,072 cells, the chunk's output rows 100 %), but in the
closed loop those differences change the discrete cache selections (4 / 4 compressions keep 3–17 different frames of
264), later steps see different context, and the segments change (37 → 40, 21 flips). FP16 + FP32 accumulation: 2
flips, 2 compressions one frame off. Graph A (whose rows live in the cache for minutes) at default precision was 0.38
off (|x| ≤ 141), at FP32 2.0e-4, for 0.2 ms: run it FP32 regardless of graph B.

**Host log-mel parity needs torch's FFT rounding.** Quiet mel bins (energy near the 2⁻²⁴ log guard) differ by FFT
rounding: an fp32 radix-2 FFT was 2.7e-4 (log domain) from the processor, even a float64 FFT 1.8e-4. Porting
pocketfft's real FFT (factors [2, 4, 4, 4, 4], radf4 × 4 then radf2, twiddle products `c·e + d·f` with one rounding
= FMA) matched `torch.fft.rfft` on all 65,792 values; numpy's fp32 path (`rfft(norm="forward") * 512`) is 1.9e-6.

**Cache selections break exact ties by summation order.** A numpy port summing the 8 per-speaker log terms in its
own order produced one 1-ulp tie at a boost boundary (97.6 s clip, second compression) and kept a different frame
for 32 steps (outputs still identical). Summing as torch's CPU kernel does ((s, s+4) pairs, left to right), with
sigmoid = 1 / (1 + exp(−x)) and sequential 8-row means, made the Python and Kotlin hosts pick exactly the reference's
frames on both test clips.

**Measure streaming at audio rate.** Back to back (as fast as the steps run) the S26 GPU reached ~104 °C (kgsl
`temp`) after about 8 s with the screen on, `thermal_pwrlevel` went 0 → 7 … 10 (clock cap 1300 → 500 MHz) and graph
B slowed 135 → 288 ms (first / last 10 steps, RTF 0.316); with the screen off and locked the same load only
reached level 1 (1200 MHz)
after ~19 s. Pushed at audio rate, graph B stayed at 136 ms for the whole clip. `gpuclk` is not readable by the shell
user; `clock_mhz`, `max_clock_mhz`, `temp`, `thermal_pwrlevel` and `gpu_clock_stats` are.

Scripts: `nemotron3diar/scripts/` (`nemotron3diar_model.py`, `build_nemotron3diar.py`, `nemotron3_diar_litert.py`
= the host as a Python reference, the `gate_*.py` checks). Model:
[`litert-community/Nemotron-3-Diarization-LiteRT`](https://huggingface.co/litert-community/Nemotron-3-Diarization-LiteRT).

### Metric3D v2 (DINOv2 ViT-S + RAFT-DPT) — fully-GPU metric depth, and three device-only gotchas

Metric3D v2 ViT-S (BSD-2) = DINOv2 ViT-S/14+reg encoder + RAFTDepthNormalDPT5 decoder (4 iters) → absolute
metric depth. Fixed 448×448. Encoder reuses the MoGe-2 ViT-S suite (fused-QKV→4D attention, LayerScale baked
into Linear, baked 32×32 pos-embed). It converts GPU-clean and runs **fully on the GPU** (`2447/2447`
LITERT_CL, Pixel 8a ~44 ms, fp16 78 MB), but desktop fp16 (corr 0.9999) hides **three issues that only the
on-device run reveals** — each one is reusable:

1. **Convex upsample → depth-to-space via `ZeroStuffConvT2d`, NOT nearest+in-block-mask.** The RAFT convex
   upsample is `mask.view(N,1,9,r,r,H,W)` softmax + unfold (6/7-D). Re-author as 16 per-subpixel
   softmax-over-9-neighbour combines (each 4D via pad+slice), `cat → [N, D·r², H, W]` (channel = `s·D+d`),
   then a **fixed `ConvTranspose2d(D·r²→D, k=r, s=r)`** with `weight[s·D+d, d, i, j]=1` (`s=i·r+j`) wrapped in
   `ZeroStuffConvT2d`. The intuitive alternative — nearest-upsample ×r then multiply by a mask selecting the
   in-block `(i,j)` position — is exact on desktop but gives **device corr 0.57** (fp32 too): the Mali ML
   Drift `RESIZE_NEAREST_NEIGHBOR` uses a different half-pixel/rounding convention at **non-stride-aligned**
   output positions, so the mask grabs the wrong replicated pixel. `ZeroStuffConvT2d` masks **only
   stride-aligned positions** (`[::s,::s]`, exact under any nearest convention) and the conv kernel supplies
   the in-block offset. **Rule: never rely on `RESIZE_NEAREST` replication at non-stride outputs on Mali;
   route the offset through a conv kernel.** Broadcast vs full-2D mask is irrelevant — it's the position.

2. **tanh-GELU is mandatory for wide-range regression heads (not `x·sigmoid(1.702x)`).** Metric3D regresses
   depth via a softmax-expectation over log-spaced bins to **200 m**. The standard sigmoid GELU approximation
   tanks far-depth fidelity → **orig-vs-reauth corr 0.51** on an outdoor 11–200 m scene (flat indoor scenes
   hide it at 0.98); the accurate tanh GELU `0.5x(1+tanh(0.7978845608(x + 0.044715x³)))` (x³ = x·x·x, POW-free,
   GPU-clean) restores **0.96**. The coarse top-of-range bins amplify the GELU error — use tanh.

3. **`nn.ReLU(inplace=True)` mutates the residual.** The DPT `ConvBlock.forward` does `out = self.act(x)`
   (inplace) then `return x + out`, so the residual is **`relu(x) + convs`, not `x + convs`**. If you replace
   that leading ReLU with a non-inplace op (to dodge its `where(x>0,x,0)` → `SELECT` lowering), you silently
   change the residual → corr 0.22. Replicate exactly: `xr = relu(x); return xr + convs(xr)`.

`norm_normalize`'s `F.elu` (→ `SELECT`) is rewritten SELECT-free as `exp(−relu(−k)) + relu(k) + min_κ` (exact
identity). `Token2Feature`'s `ConvTranspose2d` (2× upsample) → `ZeroStuffConvT2d`. Input = ImageNet norm in
0–255 scale; output is canonical-camera depth (× `fx/1000` for a calibrated camera, host-side). Scripts:
`metric3d/scripts/build_m3d.py`. Models: [`mlboydaisuke/Metric3D-v2-LiteRT`](https://huggingface.co/mlboydaisuke/Metric3D-v2-LiteRT).

### NAFNet (image restoration) — pure CNN, and the SafeLayerNorm fp16-overflow fix

NAFNet (ECCV 2022, MIT) = a U-Net of NAFBlocks, **no activation functions** (SimpleGate = channel-split
multiply). Pure CNN → Bucket-1. GoPro-width32 (deblur, 17M). Converts GPU-clean and runs fully on the GPU
(`2179/2179` LITERT_CL, Pixel 8a ~42 ms, fp16 38 MB), **device-vs-torch corr 1.0** — but only after the
SafeLayerNorm fix. Three numerically-exact re-authorings: `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)`;
`Conv2d(1×1)+PixelShuffle(2)` → Conv2d + depth-to-space `ZeroStuffConvT2d`; and:

**SafeLayerNorm — fp16 channel-sum overflow (the headline; reusable for any deep-residual CNN/ViT).** NAFNet's
residual stream grows large (`|x|≈175` at the bottleneck — the `beta`/`gamma`-scaled residuals accumulate over
the 28-block deep encoder). A channel LayerNorm reduces over C: `Σ_c x` (~90k over 512 channels) and
`Σ_c (x−μ)²` (~15M) both **exceed fp16's max 65504 → overflow** on the **Mali ML Drift delegate, which computes
in fp16 regardless of the model's dtype** (so a "fp32 model" does NOT help — fp32-device == fp16-device ==
garbage; do not use the fp32-device test to rule out precision). Symptom: the output looks ~right (corr 0.98,
because restoration output is input-dominated) but the **learned residual is destroyed (corr 0.016)** → a
periodic **grid artifact** (the decoder upsamples garbage deep features). Diagnosis: tap a **shallow** block
(32 ch, `|x|≈6` → corr 0.9999) vs the **deep middle** (`|x|≈175` → corr 0.109): divergence ∝ activation
magnitude ⇒ fp16 reduction overflow, not op-semantics. **Always check the residual/structural corr, not just
output corr.** Fix — do the reductions in a **down-scaled domain** (numerically EXACT, LayerNorm is
scale-invariant): `xs=x/S; mu=xs.mean(1); d=xs−mu; var=(d*d).mean(1)*S*S; d=d*S; y=d*rsqrt(var+eps)`. `S=128`
keeps both sums < 65504 up to ~3× the observed magnitude; eps stays in the original domain so shallow blocks
are unchanged → corr 1.0 everywhere. (This is *also* why the channel-attention pool must be `mean(3).mean(2)`
and not a single `mean((2,3))`: the two-step split keeps each single-axis sum small; a 65536-element spatial
sum would overflow the same way.) Scripts: `nafnet/scripts/build_nafnet.py`. Weights:
[`nyanko7/nafnet-models`](https://huggingface.co/nyanko7/nafnet-models). Model:
[`litert-community/NAFNet-GoPro-width32-LiteRT`](https://huggingface.co/litert-community/NAFNet-GoPro-width32-LiteRT).

### RTMPose-s (mmpose top-down pose) — SafeRMSNorm, GAU broadcast-reduce, and the mm-stack build

mmpose RTMPose-s (CSPNeXt backbone + RTMCC/SimCC head, 5.5M params, Apache-2.0). Top-down 2D human pose, 17
COCO keypoints. Converts GPU-clean and runs fully on the GPU (`256/256` LITERT_CL, Pixel 8a **~4 ms**, **fp16
11.1 MB**), **device-vs-torch SimCC corr 0.999, keypoints within 0.3 px**. The CSPNeXt backbone (SiLU) and the
diffusers-free RTMCC head are GPU-clean, but two **on-device-only** Mali issues had to be fixed (both passed
the desktop op-check and reported full LITERT_CL residency — the canonical *residency ≠ correctness* trap):

1. **`ScaleNorm` (RMS norm) fp16 overflow → all-zero head (SafeRMSNorm).** The RTMCC `ScaleNorm`
   (`x / (√(Σx²)·dim^-0.5) · g`) input reaches **≈ |274|**, so its channel `Σ x²` ≈ 3.6M **overflows fp16
   (65504)** on the Mali delegate (which reduces in fp16 even for an fp32 graph) → `norm = ∞` → `x/∞ = 0` →
   the **entire head outputs exactly zero** (every keypoint argmax → bin 0). This is the same class as the
   NAFNet SafeLayerNorm fix, here in an RMS norm, with a *total-collapse* symptom (vs NAFNet's grid artifact).
   Fix = scale `x` down by S=64 **before** squaring, then rescale (math-identical):
   `xs=x/64; norm=√((xs·xs).sum(-1))·64·scale; x/norm.clamp(eps)·g`. ⚠ Replacing `torch.norm` with a manual
   sum-of-squares ALONE does **not** fix it (the manual sum still overflows at |274|) — *scale-before-square*
   is the essential ingredient. Diagnosis = bisect-tap: backbone OK (0.9998) → ScaleNorm out 100% zero →
   input ±274 ⇒ overflow.
2. **GAU attention `act@act` BMM → broadcast-reduce.** The Gated Attention Unit's `q@kᵀ` and `kernel@v` are
   activation×activation batch-matmuls the Mali delegate mis-computes; at K=17 tokens the exact replacement is
   `(q[:,:,None,:]·k[:,None,:,:]).sum(-1)`. (Kept as hardening — it alone did not fix the zero; ScaleNorm did.)

**mm-stack build (no compiled mmcv):** `pip install mmengine mmcv-lite mmpose --no-deps munkres json_tricks`,
then **stub** `xtcocotools` (Cython build fails; COCO-eval only) and `mmdet`/`mmdet.utils`/`mmcv.ops` (the
heads `__init__` eagerly imports RTMOHead→mmdet and EDPoseHead→compiled `mmcv.ops`, neither used by RTMPose)
with a robust `_Stub(ModuleType)` (`__file__="<stub>"`, dunder-safe `__getattr__`) plus an
`inspect.getsourcefile` exception guard. Build via `mmpose.apis.init_model(cfg, ckpt_url)`; wrap as
`head(backbone(img))` → `(simcc_x[1,17,384], simcc_y[1,17,512])`; argmax÷split=2 → pixel in the app.
Scripts: `rtmpose/scripts/build_rtmpose.py`. Model:
[`litert-community/RTMPose-s-LiteRT`](https://huggingface.co/litert-community/RTMPose-s-LiteRT).

The **whole-body (RTMW-m, 133 kpts)** and **hand (RTMPose-m, 21 kpts)** variants reuse this exact recipe
(SafeRMSNorm + GAU broadcast-reduce transfer unchanged — the patches are on the shared `ScaleNorm`/`RTMCCBlock`
classes). RTMW adds a CSPNeXtPAFPN **neck** (handle it in the export wrapper: `head(neck(backbone(x)))`) and an
`nn.PixelShuffle` in its head that lowers to a **6D** tensor (>4D, GPU-rejected) → replace with a fixed
**depth-to-space `ConvTranspose2d`** (the PixelShuffle channel→space permutation as the kernel) wrapped in
`ZeroStuffConvT2d` (same fix as NAFNet/Metric3D). Both device-verified Pixel 8a fully-GPU (RTMW 531/531 ~6ms
fp16 66MB corr 0.999; hand 333/333 ~4ms fp16 28MB corr 0.999). Models:
[`litert-community/RTMW-m-WholeBody-LiteRT`](https://huggingface.co/litert-community/RTMW-m-WholeBody-LiteRT),
[`litert-community/RTMPose-Hand-LiteRT`](https://huggingface.co/litert-community/RTMPose-Hand-LiteRT).

### Places365 ResNet18 (scene recognition) — the ResNet-stem `MaxPool` `-inf`-pad fix

ResNet18 trained on Places365 (CSAILVision, MIT, 365 scene categories). Pure CNN, runs fully on the GPU
(`61/61` LITERT_CL, Pixel 8a **~2 ms**, **fp16 22.8 MB**, device-vs-torch corr **1.0**, top-1 match). Two
numerically-exact re-authorings — the second is a **NEW reusable Mali fix for ResNet-style stems**:

1. global `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)` (the usual multi-axis-pool fix).
2. **ResNet stem `MaxPool2d(3, stride=2, padding=1)` → zero-pad + valid max-pool.** PyTorch's max-pool pads
   with **`-inf`**, which litert-torch lowers to a **`PADV2`** op (pad with a non-zero constant). The Mali ML
   Drift delegate **does not delegate `PADV2`** → it splits the graph into CPU partitions (`Replacing 36 out
   of 61 node(s) … 2 partitions`) and then **fails to compile the whole model** (`Failed to compile model`,
   no op-blocklist hit — desktop op-check passes). Because the stem max-pool always follows a ReLU (inputs
   ≥ 0), padding with **0** is numerically identical (a 0-pad never wins the max over a ≥0 cell, and with
   `padding=1`/`kernel=3` every window has a real cell), and `F.pad(x, …, value=0)` emits a delegatable
   **`PAD`** → `61/61` full GPU residency. Replace `nn.MaxPool2d(3,2,1)` with
   `F.max_pool2d(F.pad(x,(1,1,1,1),value=0.), 3, stride=2)`. Reusable for any ResNet/Places/ImageNet stem.

Result: banned ops NONE, all tensors ≤4D, tflite-vs-torch corr 1.0, device-vs-torch corr 1.0. Scripts:
`places365/scripts/build_places.py`. Model:
[`litert-community/Places365-ResNet18-LiteRT`](https://huggingface.co/litert-community/Places365-ResNet18-LiteRT).

### Fast Neural Style (TransformerNet) — conv-weight scaling via norm scale-invariance (large-activation fp16 fix)

PyTorch examples `TransformerNet` style transfer (BSD-3, 4 styles). Pure CNN encoder-decoder (interpolate-
nearest upsample, no transposed conv → no ZeroStuff). Runs fully on the GPU (`350/350` LITERT_CL, Pixel 8a
**~9 ms**, fp16 **3.5 MB**/style, device-vs-torch corr **0.9999**) after three numerically-exact re-authorings:

1. **`ReflectionPad2d` → zero-pad.** Reflection padding lowers to **`GATHER_ND`** (the reflect index gather,
   banned). Fold a `F.pad(value=0)` into each conv → emits `PAD`. Border-only cosmetic difference.
2. **⭐ Large conv activations → conv-weight scaling (exploit normalization scale-invariance).** The conv
   outputs reach **≈ |5000|**, where the **Mali delegate's fp16 conv accumulation loses precision** → garbage
   (device corr **0.34** at `350/350` full residency; desktop fp16 = 1.0 — the canonical *residency ≠
   correctness*). This is NOT a reduction overflow (SafeInstanceNorm alone made it WORSE, 0.16) — it's the conv
   itself accumulating imprecisely at large magnitude. **Fix: scale each conv's weight+bias down so its output
   is ≈ |10|.** Because every such conv is immediately followed by an `InstanceNorm` (which is
   **scale-invariant**: `IN(a·x) = IN(x)`), this is **mathematically exact** (the IN output is unchanged) yet
   keeps the fp16 accumulation in a precise range. Measure each conv's output max once (the scales are
   independent — the IN between convs decouples them), bake `weight /= max/10`. **General rule: when a
   large-activation CNN garbles on Mali fp16 despite full residency, and a normalization follows the big conv,
   rescale the conv via the norm's scale-invariance.** (Reusable for any IN/BN/LN-normalized generator.)
3. **`InstanceNorm` → SafeInstanceNorm.** Spatial mean/var over 256×256 overflows fp16; two single-axis means
   in a down-scaled domain are fp16-safe and exact (SafeLayerNorm class). Needed in addition to (2).

Scripts: `neuralstyle/scripts/build_style.py`. Model:
[`litert-community/Fast-Neural-Style-LiteRT`](https://huggingface.co/litert-community/Fast-Neural-Style-LiteRT).

### L2CS-Net (gaze estimation) — ResNet50 ZeroPadMaxPool reused; new "Gaze Estimation" task

L2CS-Net (Ahmednull, MIT) gaze estimation — ResNet50 + 2 FC heads (yaw/pitch, 90 angle bins each), Gaze360.
Pure CNN, runs fully on the GPU (`139/139` LITERT_CL, Pixel 8a **~3 ms**, fp16 47.9 MB, device-vs-torch corr
**0.9999**). The two fixes are the standard ResNet pair — confirming the Places365 ResNet recipe transfers to
**any torchvision-ResNet-backed regression/classification head** (also relevant to L2CS variants, face-rec,
gaze, age/expression on a ResNet stem):

1. **stem `MaxPool2d(3,s2,p1)` → zero-pad + valid max-pool** (the `-inf`-pad `PADV2` Mali won't delegate; 0-pad
   is exact post-ReLU → `PAD`). 2. **global `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)`**.

Decode: bake the softmax over the 90 bins into the graph; the host does the expectation `deg = Σ p_i·i·4 − 180`
(no `TOPK`/`GATHER`). Weights: `L2CSNet_gaze360.pkl` is on HF (`tianfxc/l2cs`, `py-feat/l2cs`) — avoids the
upstream gdrive-folder download (which `gdown` silently fails on). Load the L2CS `model.py` via importlib (the
`l2cs` package `__init__` pulls `face_detection`). Scripts: `gaze/scripts/build_gaze.py`. Model:
[`litert-community/L2CS-Gaze360-LiteRT`](https://huggingface.co/litert-community/L2CS-Gaze360-LiteRT).

### MI-GAN (mobile image inpainting / object removal) — the norm-free generator lane (zero re-authoring)

MI-GAN (Picsart, ICCV 2023, MIT, 5.97M) — a mobile "magic eraser". Its **inference** generator
(`migan_inference.py`, the re-parametrized deployable model, NOT the StyleGAN training `.pkl`) converts
**GPU-clean in ONE shot, zero re-authoring**: device-vs-torch corr **0.99998**, `509/509` LITERT_CL, Pixel 8a
**~6 ms** at 512×512, fp16 16.3 MB. Why it's free: the mobile generator is **StyleGAN-style with NO
normalization** (no InstanceNorm/GroupNorm → none of the SafeNorm/conv-scaling fixes the style-transfer /
AnimeGAN generators needed), upsampling is `nn.Upsample(nearest)` + a fixed FIR-filter grouped conv (→
`RESIZE_NEAREST_NEIGHBOR` + `CONV_2D`, **no ConvTranspose → no ZeroStuff 0-byte risk**), convs are
depthwise-separable, and the activation is a leaky-ReLU with gain+clamp (→ `LEAKY_RELU` + `MAXIMUM`/`MINIMUM`,
not `SELECT`). So: **a norm-free, FFT-free, transpose-free generator is the cleanest GPU lane** — contrast the
normalized generators (style transfer, AnimeGAN) that hit the large-activation fp16 conv-accumulation wall.

**I/O contract:** input is 4ch `concat(mask−0.5, rgb·mask)` (rgb ∈ [−1,1], mask 1=keep/0=erase); output [−1,1];
composite `rgb·mask + out·(1−mask)`. Weights `migan_512_places2.pt` = the inference-model state_dict, load
directly into `migan_inference.Generator(resolution=512)` (the repo's `export_inference_model.py` is only for
converting a source `.pkl` → inference model; not needed here). gdown-folder worked for the weights. Scripts:
`migan/scripts/build_migan.py`. Model:
[`litert-community/MI-GAN-512-Places2-LiteRT`](https://huggingface.co/litert-community/MI-GAN-512-Places2-LiteRT).

### YuNet (face detection) — the smallest model in the zoo, zero re-authoring

YuNet (ShiqiYu/libfacedetection, BSD-3, 0.076M params) — a tiny anchor-free face detector. Converts
**GPU-clean in ONE shot, zero re-authoring**: `146/146` LITERT_CL, Pixel 8a **~4 ms** at 640×640, **fp16 0.3 MB
(smallest in the zoo)**, device-vs-torch corr **0.9999**. Pure CNN (depthwise-separable `ConvDPUnit`) + a TFPN
neck whose upsample is **`F.interpolate(mode="nearest")` → `RESIZE_NEAREST_NEIGHBOR`** (no transposed conv → no
ZeroStuff) + non-padded `MaxPool2d` (no `-inf` pad → no `PADV2`). Wrap the head's per-stride
`permute(0,2,3,1).reshape(B,-1,C)` (+ `.sigmoid()` on cls/obj) so the model emits 12 decode-ready tensors
(cls/obj/bbox/kps × strides {8,16,32}, output order identity). **Preprocessing = BGR, 0-255, NO normalization**
(`Normalize(mean=0,std=1,to_rgb=False)`). Decode host-side: anchor-free priors `px=col·s, py=row·s` (offset 0),
score=`cls·obj`, box=`(bbox₀₁·s+prior, exp(bbox₂₃)·s)` center+wh, 5 landmarks `kps·s+prior`, then NMS.
Weights `weights/yunet_n.pth` ship in the libfacedetection.train repo. Scripts: `yunet/scripts/build_yunet.py`.
Model: [`litert-community/YuNet-Face-LiteRT`](https://huggingface.co/litert-community/YuNet-Face-LiteRT).

### UniSal (visual saliency) — the strided-slice→avg_pool fix, gaussian-prior bake, and "smoothing isn't cosmetic"

UniSal (rdroste, Apache, 3.71M) — saliency prediction (where humans look). MobileNetV2 + bilinear decoder.
Converts GPU-clean (`158/158` LITERT_CL, Pixel 8a **~3 ms**, fp16 6.5 MB, device-vs-torch corr **0.9998**) with
three exact fixes:

1. **⭐ Strided subsample `x[..., ::2, ::2]` → `F.avg_pool2d(x, kernel_size=1, stride=2)`** (NEW reusable). A
   stride-2 channel-preserving subsample lowers to **`GATHER_ND`** (banned). A kernel-1 stride-2 average-pool
   selects the *exact same* pixels (kernel 1 = no averaging) and emits `AVERAGE_POOL_2D` — numerically identical.
   (Same class as the EdgeTAM `x[:, :, i::w, j::w]`→grouped-conv finding, but for a simple 2× subsample.)
2. **Bake the Gaussian prior maps.** `_get_gaussian_maps` (meshgrid + per-gaussian `exp`) emits `GATHER_ND` +
   `BROADCAST_TO`; the maps depend ONLY on the (fixed) feature size + learned params, so precompute once (run on a
   zero input of the right size) and concatenate the constant buffer.
3. **`F.pad(mode="replicate")` → 0-pad** for the 41×41 Gaussian smoothing (replicate → `GATHER_ND`).

**⚠ Lesson: the smoothing is NOT cosmetic.** Dropping the 41×41 smoothing made the saliency *anti-correlate*
(−0.56) with the real model — the smoothing **suppresses border/corner artifacts** that otherwise become the
spurious global max. Verify a re-authored pipeline against the FULL reference's *argmax/spatial pattern*, not
just an internal device-vs-tflite corr (which was 1.0 even while the output was wrong). Static-image path:
Bypass-RNN + pin one domain (SALICON) so the domain-specific BatchNorm/smoothing fold to constants; final spatial
log-softmax → host. Scripts: `saliency/scripts/build_unisal.py`. Model:
[`litert-community/UniSal-Saliency-LiteRT`](https://huggingface.co/litert-community/UniSal-Saliency-LiteRT).

### CPGA-Net (low-light enhancement) — the POW→exp/log gamma fix; the smallest model in the zoo

CPGA-Net (Shyandram, MIT, IJPRAI, 0.025M params) — low-light image enhancement (Channel Prior + Gamma
Correction). Converts **GPU-clean in ONE cycle**: `135/135` LITERT_CL, Pixel 8a **~2 ms**, **fp16 0.1 MB
(SMALLEST in the zoo)**, device-vs-torch corr **0.99999**. **This finally ships the low-light task** (Bread
parked on the Mali composition-delegation wall; SCI/PairLIE were no-license — CPGA-Net is MIT + tiny). Three
exact fixes:

1. **⭐ Gamma correction `torch.pow(x, γ)` → `exp(γ · log x)`** (NEW reusable). `POW` is banned on Mali; the
   identity `x^γ = exp(γ·log x)` is exact (clamp the base to [1e-9, 1] first) and emits native `EXP` + `LOG`
   (both delegatable — `LOG` confirmed GPU-clean here, `EXP` already proven). Works for a learned scalar γ
   (broadcast). Reusable for any gamma/power op.
2. **CBAM + gamma global pools**: `AdaptiveAvgPool2d(1)` → `mean(3).mean(2)`; `AdaptiveMaxPool2d(1)` →
   `F.max_pool2d(x, kernel_size=(H,W))` (use max-pool, NOT `torch.amax`, which has no NHWC rewriter in
   litert-torch).
3. The dark/bright **channel prior** (`torch.max`/`torch.min` over dim=1) lowers to `REDUCE_MAX`/`REDUCE_MIN` —
   GPU-clean (small 3-channel reduction).

`isdgf=False` (no FastGuidedFilter → no bicubic). Stub `guided_filter_pytorch` (imported at module level, unused).
Scripts: `lowlight/scripts/build_cpga.py`. Model:
[`litert-community/CPGA-Net-LowLight-LiteRT`](https://huggingface.co/litert-community/CPGA-Net-LowLight-LiteRT).

### wav2vec2-CTC (fully-GPU ASR) + the GroupNorm-reduction-extent fp16 rule (Moonshine park)

wav2vec2-base-960h CTC (Facebook, Apache) — on-device **speech recognition, fully GPU, single forward pass**
(no autoregressive decoder; CTC greedy decode on the host). Device-verified Pixel 8a: `997/997` LITERT_CL
(single graph), **~22 ms** / 10 s, device-vs-torch corr **0.99998**, **exact** transcription. **Zero FFT** (raw
16 kHz waveform → 1D-conv frontend). Reuses the shipped wav2vec2-KWS recipe (TanhGELU + GN4D + fold pos_conv
weight-norm + bidirectional-mask→None); only the head changes (classification → CTC `lm_head`), output = logits
`[1, T', 32]`. fp16 190 MB → filesDir push. Scripts: `asr/scripts/build_w2v2_ctc.py`. Model:
[`litert-community/wav2vec2-base-960h-CTC-LiteRT`](https://huggingface.co/litert-community/wav2vec2-base-960h-CTC-LiteRT).

**⭐ NEW reusable Mali rule — manual GroupNorm fp16-precision depends on the REDUCTION EXTENT.** Moonshine-tiny
(the fresh 2024 on-device ASR) was attempted first and **parked**: its conv-stem `GroupNorm(num_groups=1)`
reduces over the **joint C×T** feature map (288 × 1248 ≈ 360k elements). A manual GN over that extent is
**fp16-imprecise on Mali** (device corr 0.55 at the GN tap, even with a down-scaled explicit sum *or* a staged
native mean — fp16 accumulation error over ~360k terms), and the 0.55 compounds through the 6 transformer layers
to a **constant** output. By contrast wav2vec2's GroupNorm is `num_groups=512` = **per-channel over time only**
(a small ~T reduction) → fp16-precise → ships. **Rule: a manual GroupNorm/LayerNorm whose reduction spans a large
joint (channel×spatial/time) extent will lose fp16 precision on the Mali delegate even when down-scaled; group/
instance norms that reduce over a single small axis are safe.** (conv1 itself was corr 1.0 on device — isolate
norm collapses with a per-stage device tap.) Moonshine encoder otherwise converted GPU-clean via the standard
RoPE recipe: interleaved `rotate_half` → fixed `q @ P` matmul + baked cos/sin (kills the `x[...,0::2]` GATHER_ND
+ the `stack` 5D), tanh-GELU, mask→None. `~/Downloads/meeting/asr-work/build_moonshine.py` (reusable RoPE recipe).

### SSDLite320-MobileNetV3 (torchvision detector)

Converter: **litert-torch, patch-free** (no model-internal op rewrite). The fast clean
CompiledModel-GPU detector — 0.59 GMACs, BSD-3, FP16 7.2 MB. Device-verified on Pixel 8a
(Tensor G3): CompiledModel GPU delegates **all 286 nodes to OpenCL** (`Replacing 286 out of
286 node(s) with delegate (LITERT_CL)`, 1 partition, no CPU fallback), **~30 FPS** live camera.

**Two techniques make it convert clean — both are output/IO choices, not model patches:**

1. **4D-head-tap.** SSD's built-in postprocess (`DefaultBoxGenerator` + box decode + NMS)
   lowers to `GATHER_ND`/`TOPK`/`>4D` (GPU-rejected), and the naive head wrapper emits
   transient 5D `view(N,A,K,H,W)` tensors. Instead, return each feature level's **raw head
   conv outputs** (4D, NCHW): `cls[i] = [1, A·91, H, W]`, `box[i] = [1, A·4, H, W]` for the 6
   levels (H = 20,10,5,3,2,1), and move decode + NMS to app code. Same "choose the output
   point" move as YOLOX raw-head / U²-Net `d0`.

   ```python
   feats = list(m.backbone(x).values())
   ch = m.head.classification_head.module_list
   rh = m.head.regression_head.module_list
   return tuple(t for i, f in enumerate(feats) for t in (ch[i](f), rh[i](f)))  # 12 × 4D
   ```

2. **Keep NCHW I/O — do NOT use `to_channel_last_io`.** Its channel-last pass turns
   MobileNetV3's 8 `SqueezeExcitation` global-avg-pools into `GATHER_ND×8 + 5D`. With NCHW
   input the model converts stock-clean (`BANNED NONE, ≤4D, Flex NONE`); the SE pools lower to
   plain `SUM` (×8) which Mali ML Drift accepts. **Lesson: before patching a model, check the
   "needed patch" isn't an artifact of a convenience transform.** (Clean NHWC input would need a
   converter-side fix for channel-last × global-pool, not a model monkeypatch.)

**Preprocessing gotcha (cost an hour):** SSDLite320 normalizes **mean = std = 0.5** →
`pixel/127.5 - 1` ∈ [-1, 1], **NOT ImageNet**. ImageNet-norm silently caps scores (top 0.31 vs
0.74 correct). Verify your preprocessing against `m.transform([t]).tensors`. Resize = bilinear
**stretch** to 320×320 (`fixed_size`, not letterbox).

**Kotlin decode** mirrors `SSD.postprocess_detections` + `BoxCoder(weights=10,10,5,5)`: rebuild
the 3234 default boxes from the `DefaultBoxGenerator` formula (scales 0.2–0.95, ar {1,2,3,½,⅓};
matches the export to 3e-5), softmax over 91 → best non-background → threshold → decode against
the anchor → per-class NMS. The tflite output order is `(cls, box)` per level; NCHW channel =
`a·K + k` (cls) / `a·4 + j` (box). FP16 end-to-end matches stock torchvision **298/300 boxes @
IoU 0.99**. FP16 recipe = `ai_edge_quantizer` `AlgorithmName.FLOAT_CASTING` +
`ComputePrecision.FLOAT` (the inline `op_config` dict throws `KeyError: compute_precision`).

## 2026-09-19 追記 — GLiNER2.5 Small (DeBERTa-v3-xsmall boundary extractor) on S26 GPU

Shipped as `litert-community/GLiNER2.5-Small-LiteRT` (windows 128/256/512, fp32 + fp16-weight files, host contract). Run dir with every gate JSON: `~/code/codex-conversions/2026-09-19/gliner25-small/` (supervised Codex run, 8 rounds).

**Graph cut.** Export the dense prefix only (DeBERTa encoder → routing → boundary encoder → boundary query head → per-token/per-query projections) and stop at the first data-dependent op (top-k / `unique` / `nonzero` in the upstream proposer). The host keeps the upstream sparse pool/scorer/decoder (16 tensors, 466 KB) and the word-embedding lookup (`inputs_embeds` input). Emit ONE packed rank-4 leaf `[1,1,1,1108*T+4574]` holding the 17 logical outputs (no `[1,N,C]` fan-out).

**ML Drift 2.2.0 compile rejections seen on the first export** (S26, CompiledModel GPU): `BROADCAST_TO`, int64 `CAST` / `LESS_EQUAL` / `ONE_HOT` / `RESHAPE` / `SUM`, `SELECT_V2`, `MAXIMUM`; a `BATCH_MATMUL` whose constant is the LEFT operand is also rejected (put constants on the right). Rewrites that compile 1024–1121 ops in one partition: boolean masks as float arithmetic, routing as matmul with host one-hot rows, prefix sums as a constant upper-triangular matmul, attention kept at rank 4.

**DeBERTa-v2 relative attention beyond a 128 window** needs the exact upstream logarithmic bucket lookup (`position_buckets=256`) and the boundary head's 128-word local-window mask baked as constants; a linear-bucket / full-window simplification is exact only at N ≤ 128 (bit-identical parity there, wrong at 256/512).

**Precision.** Default GPU precision → NaN from the encoder output (`text_states`) onward; `GpuOptions(precision = FP32)` → exact spans (F1 1.000 on 70 inputs, confidence drift ≤ 3e-6 fp32 / ≤ 2.7e-3 fp16-weights). Same class as PP-OCRv6 above.

**Weight storage.** Dynamic-range int8 (ai-edge-quantizer 0.8.0 `dynamic_wi8_afp32`) compiles nowhere on S26 2.2.0 GPU: full recipe and FULLY_CONNECTED-only recipe both fail with `Unable to parse bc coord for BATCH axis` (catalog D13), while an empty-recipe control compiles — the int8 FC constants themselves are the trigger, even with rank-4 inputs. `FLOAT_CASTING` float16 on the 96 FC weights (+ DEQUANTIZE) compiles fully, F1 unchanged, 54–84 MB vs 98–128 MB. CPU int8 F1 was 0.993–0.995 (evidence only).

**Android host port (Kotlin, `gliner25/`).** The tokenizer / input construction / sparse decoder were ported to pure Kotlin and checked two ways: JVM unit tests against captured Python inputs and the Python host decoder (195/195 window-input pairs, max confidence diff 4.2e-7), then an in-app gate on the S26 that also compares the ON-DEVICE tokenizer output with the captured Python inputs (70/70 on GPU FP32 and CPU). The second check is not optional: `Pattern.compile(..., Pattern.UNICODE_CHARACTER_CLASS)` passes every JVM test and throws `IllegalArgumentException: UNICODE_CHARACTER_CLASS flag not supported` at tokenizer init on Android — desktop-JVM parity does not prove Android parity for regex-based splitters; spell the Unicode classes out explicitly and use no flags. The upstream default word splitter is `WhitespaceTokenSplitter` (`DEFAULT_WORD_SPLITTER = "whitespace"`), not the `CharLevelSplitter` regex. Cost split on the S26 at s128 (debug build): graph to readback ~12 ms, Kotlin sparse decode ~10 ms after optimization (first port: ~47–58 ms). Two facts from that pass: ~70% of the first port's decode time came from the app being debuggable (non-debuggable build 15.6 ms vs 58 ms, same code) — time host code in a non-debuggable variant before optimizing it; the rest went with flat primitive buffers, cached tensor views, primitive stable sorts and a small persistent worker pool, with reduction order and top-k/tie semantics untouched (confidences bit-identical on device). A fast steady state is not what the user sees: the first request after launch still ran cold (decode 56.6 ms, tokenize+embed 21.6 ms vs ~7 ms / ~3 ms hot) because only the graph was warmed. Repeating the full pipeline 12 times before the UI reports Ready (0.5–0.6 s at startup) brought the first request to 25–38 ms total; measure it screen-on — with the screen off the S26 power-saves and the same first request is 1.5–2x slower. Evidence: `~/code/codex-conversions/2026-09-19/gliner25-android/` (supervised Codex run, 8 rounds).

### Sopro v2 turbo (zero-shot voice-cloning TTS) — domain gates for reduced precision, exact right-padding of causal conv stacks, the fp32-FFT host trap

Converter: litert-torch (0.9.4, torch 2.11, ai-edge-litert 2.2.0). Sopro v2 turbo (121M core + 36M) is a
three-stage TTS: semantic AR LM (12 × 512, FSQ tokens) → flow-matching DiT (8 layers, 2 Euler steps) →
Vocos vocoder (causal ConvNeXt → log-mag/phase → iSTFT). The author's own ONNX split already keeps every
spectral step (three mel front-ends, iSTFT), the RoPE tables, masks, the token→frame gather index, sampling
and post-processing on the host; the LiteRT port keeps the same split, so all 8 offline graphs + the 3
streaming-vocoder graphs converted **on the first attempt with no re-authoring of any op** (exact GELU kept,
alias probe 0 after cloning every parameter contiguous — litert-torch #1061). Static contracts: reference
exactly 10 s (speaker mel [1,80,1001], Whisper mel [1,80,1002] → 235 tokens, reference mel 938 frames);
AR prefill P_MAX 256 + step CAP 1024 with the packed KV on the host (dia2 / Pocket TTS pattern; a 2-signature
merged file shares the weights: 223 MB vs 421); acoustic buckets (T,N) = (2048,512) and (4096,1024) selected
by the actual length; vocoder offline bucket 1024 frames or streaming 64-frame chunks (start 64→37, step
64→64, flush →27, exact vs offline: max err 3.9e-4). Mac CPU gates only so far (device run pending):
fp32 teacher-forced waveform corr ≥ 0.99999 (24 utterances + 4 long), AR replay 2920/2920, semantic tokens
235/235; the free-running LiteRT pipeline with the same NumPy sampler and seed reproduces the PyTorch token
sequences 24/24.

**Findings worth the catalog (all Mac M4 Max CPU, fp32 I/O):**

1. **Judge reduced-precision TTS graphs in the domain of their output — acoustic graphs in mel space, the
   vocoder in waveform space.** With plain wfp16 (FLOAT_CASTING) weights in the flow-matching DiT, one
   utterance dropped to raw-waveform corr 0.977 against the fp32 chain while its log-mel corr stayed 0.9999,
   HNR moved 0.011 dB and WER / speaker cosine did not move: the 2-step ODE turns fp16 weight perturbations
   into phase differences, not spectral ones. Conversely an absolute tensor rule on the vocoder's phase output
   (absmax ≈ 125) is meaningless — a 0.6 rad error at a near-silent bin leaves the waveform at corr 0.99994.
   Keeping the vocoder's final FC in fp32 did NOT reduce that phase error (0.5998 → 0.5989): the error is the
   phase channel's scale, not the head weights. Gate set that worked: encoders/AR = tensor rule
   `max|diff| ≤ max(1e-2, 2e-3·absmax) ∧ corr ≥ 0.999` + token/greedy replay with near-tie documentation;
   acoustic = solved-mel corr ≥ 0.99; vocoder = single-swap waveform corr ≥ 0.99; chain = log-mel corr ≥ 0.99
   + free-running WER / speaker cosine + HNR and 4–12 kHz band-energy within 1.0 / 1.5 dB of the fp32 chain.
   The quality gates alone (WER, speaker cosine) are blind to phase: int8 vocoder variants with ≈ 15 phase
   wraps of tensor error passed both — the waveform gate is what rejected them.
2. **Exact zero right-padding for a causal conv stack with lookahead** (Vocos: 8 ConvNeXt blocks, lookahead 3
   each + 3 at the embed conv = 27 frames): multiply the residual stream by the runtime frame mask after the
   embed **LayerNorm** and after every block. Masking only the convolution is insufficient — the LayerNorm
   bias populates padded frames and leaks into the next block's lookahead window. fp64 padded-vs-unpadded
   difference 1.2e-13; the fp32 remainder (8.8e-5 on absmax 125) is summation-order roundoff.
3. **Host mel mirrors must use an fp32 FFT.** NumPy's default `np.fft.rfft` dispatches to the fp64 pocketfft
   loop; against torch's fp32 STFT the power spectrum differs by ~1e-7, which the log amplifies to 1e-2 in
   quiet high bands (normalized-mel error 3.8e-3, end-to-end waveform max error 6.3e-3 on one utterance).
   `np.fft.rfft(x, norm="forward") * np.float32(n_fft)` selects the fp32 path bit-exactly (error 4e-7). A
   Kotlin FFT needs its own fp32 parity check; matching the math is not matching the rounding.
4. **Toolchain facts (corrected 2026-09-23):** the first run read a `fold_quantize=True` failure as a rank-3 Conv1d wall.
   A 24-case matrix (Conv1d / Conv2d / Linear × small and real shapes × static / dynamic × fold True / False) shows
   the documented path — `convert_pt2e(..., fold_quantize=False)` — converts every module with per-channel int8
   weights (12/12), and `fold_quantize=True` (torchao's default) fails every module of every rank with
   `'stablehlo.uniform_dequantize' op operand #0 must be ranked tensor of per-tensor integer quantized or per-axis
   integer quantized values` (12/12). The int8 vocoder is absent because it failed the waveform gate, not
   because of the converter. On macOS the ai-edge-litert 2.2.0 wheel's GPU-only `CompiledModel` (Metal accelerator
   registered) SIGSEGVs after `Flatbuffer model initialized` on a trivial Linear+ReLU graph in GPU-only, GPU|CPU and
   `enforce_f32` form; the 2.1.6 wheel runs the same file and script on Metal (same Python 3.12.13), so there are no
   Mac GPU numbers from 2.2.0. The int8 that ships is the AR only (native PT2E per-channel dynamic, 55 MB vs 223 MB
   fp32, greedy replay 2834/2920, free-running WER 1.31 %, speaker cosine 0.925).

Scripts: `sopro/scripts/` (portable copy of the HF repo's `conversion/`), contract in `sopro/contract.json`;
full REPRODUCE and card on [litert-community/sopro-v2-turbo](https://huggingface.co/litert-community/sopro-v2-turbo).

**Android (Galaxy S26, LiteRT 2.2.0, 2026-09-23).** GPU rules learned on Adreno: the speaker and semantic encoders need
`GpuOptions(precision = FP32)` (default precision → non-finite speaker output, 93–100/235 semantic token flips); the acoustic DiT
(condition + velocity) passes the mel-domain gates at default precision (velocity 177 ms on GPU vs 718 ms on CPU); the AR step is
numerically exact on GPU FP32 (2,920/2,920) but bus-bound — 22 ms/step vs 9.5 ms on CPU because the 50 MB packed KV is re-uploaded
every step — so the AR stays on CPU (int8); the Vocos ConvNeXt vocoder miscomputes on the GPU in both precisions (raw corr ≈ 0, HNR
Δ 5.7 dB; offline and streaming graphs; a rank-4 promotion of every rank-3 tensor did not change it); the style prefix is rejected
(`BATCH_MATMUL: non-constant tensor`). Exact rewrites that made the rest GPU-resident: one-hot float selections instead of
GATHER_ND (AR last row, acoustic token/frame gathers), no `BROADCAST_TO` in the DiT, FSQ argmax moved to the host (ARG_MAX/CAST on
INT64 are rejected), shape metadata promoted to rank 4 for the AR step's 24 rank-3 BMMs. Shipped hybrid: TTFA 2.07 s, RTF 0.41,
first tap 1.8–2.9 s after a 3.5 s Ready (GPU compiles); all-CPU 3.34 s / 0.64. Device transfer rule: an app cannot copy into
/data/local/tmp under SELinux — pull with `adb exec-out run-as <pkg> tar -cf - files/<dir>`.

## 2026-09-22 追記 — Laya Multilingual (mmBERT-base typed-decision scorer) on S26 GPU

Shipped as `litert-community/Laya-Multilingual-LiteRT` (windows 256/512, fp32 + fp16-weight files, act head, fp16 token table, calibration JSON, Python host, Android sample `laya/`). The facts in this section were written by Codex (gpt-6-astra) in two supervised runs and re-checked by the supervisor from the raw evidence (gate rows recounted from the pulled logits): `~/code/codex-conversions/2026-09-21/laya/` (conversion, 8 rounds) and `~/code/codex-conversions/2026-09-21/laya-android/` (Kotlin host + S26 gates, 8 rounds).

**Graph cut for a "schema in the prompt, one `[MASK]` per option" scorer.** One question row per call. Inputs `inputs_embeds [1,N,768]`, `attention_mask [1,N]` (float), `qtype_onehot [1,3]` (the type embedding becomes a matmul); outputs `token_logits [1,N]` (the scorer applied at every position) and `pooled_cls [1,768]`. No marker gather, no option count and no top-k inside the graph: the host gathers the marker positions it built, so the option count is unbounded by the graph. The act head (`cat(pooled, 4 features of the raw softmax)` → 2 logits) is a separate 1 MB graph because its features depend on the host-side softmax. `nn.TransformerEncoderLayer` takes PyTorch's fused fast path in eval mode and has to be written out explicitly (pre-norm, ReLU FFN); exact GELU, rank ≤ 4, rotary tables baked per layer type as contiguous clones, padding and sliding-window masks as float constants.

**ML Drift rejects an EMBEDDING_LOOKUP on an fp16-stored table; an fp32 in-graph table and fp16 FC weights are fine.** The first wfp16 graph took `input_ids` and kept the 256000×768 table inside as fp16 + DEQUANTIZE: the S26 GPU delegate reported `EMBEDDING_LOOKUP: Empty quantization params` (and the table's DEQUANTIZE), placed 1677 of 1783 ops, and CompiledModel failed to compile. The fp32 ids graph with the table inside does compile on the same phone (1.29 GB, compile ≈ 7 s, 54.8 ms per question — measured by the typed-decisions-android lane, `~/code/hfmodels-android/tested-runtime-matrix.json`, shipped as `litert-community/laya-LiteRT`); only the wfp16 form was tried in this run before the cut was moved. With the lookup on the host the same network compiles as one partition — 1779/1779 ops with 99 `FLOAT_CASTING` fp16 FULLY_CONNECTED weights + DEQUANTIZE, 1680/1680 in fp32 — under `GpuOptions(precision = FP32)`. The host table can be float16: for this checkpoint fp32→fp16→fp32 is exact (max difference 0 over 196.6M values), and padded positions must gather the real PAD row, not zeros. On the desktop, `ai-edge-quantizer` 0.8.0 also accepts `FLOAT_CASTING` on EMBEDDING_LOOKUP (useful for CPU-only files: 1.29 GB → 644 MB).

**Gate the act logits relatively.** The act head's logits have magnitude ~2e3–5e3, where the fp32 ULP is 2.4e-4, and the unchanged upstream model moves them by 0.24 between batch paddings. An absolute 1e-4 tensor gate is below the rounding floor; use max |Δ| / max |reference| ≤ 1e-4 for such tensors and keep the decision gate at the answer level (same argmax, |Δp| ≤ 1e-3 fp32 / 1e-2 fp16 weights). The act probability was 1.0 on every fixture row upstream and in LiteRT.

**transformers version.** The checkpoint's encoder `config.json` is in the transformers-5 format (`layer_types` + `rope_parameters`). Load it with transformers 5.x (5.17.0 verified: full-attention theta 160000, sliding 10000 for the English ModernBERT-large checkpoint; both 160000 for mmBERT). The multilingual encoder config says `cls_token_id = 1` while the tokenizer and the upstream builder use `<bos> = 2`.

**Kotlin host.** The Gemma-style tokenizer (BPE, Metaspace `prepend_scheme=always`, byte fallback, 256k vocabulary, 580k merges) ports to pure Kotlin without JNI or regex flags: 300/300 stress strings and 402/402 fixture rows match HF `tokenizers` on the JVM, and 201/201 rows match on the device. Loading the 34 MB `tokenizer.json` takes 0.76 s in a non-debuggable build on the S26 (1.5 s debuggable). Python's `json.dumps` separators and `round(x, 4)` (ties-to-even on the binary double) have to be reproduced exactly for the answer dictionaries to match.

**Timing (Galaxy S26, LiteRT 2.2.0, GPU FP32, 256 tokens).** Main + act graph, write → run → readback: warm median 50.9 ms per question (CPU 163 ms); host embedding lookup median 14.6 ms (debug build); GPU compile 1.25 s; launch → Ready 2.3 s and a five-question preset 301 ms in a non-debuggable build. Debug and release UI totals differed by ~10 % — the graph, not the Kotlin host, is the cost.

**Calibration with cleared data only.** The multilingual checkpoint ships T = 1. Temperatures per (question type, option-count bucket) were fitted by NLL on 4,415 licensed EN/JA examples with language-balanced fit splits; a bucket keeps T = 1 when a language slice (n ≥ 40) gets ECE worse by more than 0.05. Judge a dataset by the original terms of the subset, not the hub tag: an Apache-tagged sentiment aggregate wrapped SemEval tweets and Amazon reviews and was excluded.

**English and typed-decisions checkpoints (ModernBERT-large) — same cut, 2026-09-22.** Shipped as `litert-community/Laya-English-LiteRT` (root = English, `typed-decisions/` = the fine-tune; run `~/code/codex-conversions/2026-09-22/laya-en-gpu/`, 5 rounds, supervisor recounts from the raw logits). Facts a port needs: (1) hidden 1024, 28 layers, vocab 50368, PAD 50283; the fp16 token table is exact for both checkpoints (max fp32→fp16→fp32 difference 0 over 51.6M values), 103 MB each. (2) The two attention layer types use **different RoPE tables** (full attention theta 160000, sliding 10000): export them as separate contiguous clones and audit the exported constants against the instantiated transformers 5.x rotary module — equal constants may be deduplicated by value, different ones must stay separate (litert-torch #1061 merges equal buffers). (3) Builder budgets differ from the graph window: English builds with max_len 512 / head 192, typed-decisions with 1024 / 256; a row is run at a window only if its built sequence fits (140/209 English rows fit 256/512, 100/175 typed-decisions rows); the graph window never replaces the builder budget. (4) Device gate without a tokenizer port: a debug copy of the sample with a separate `applicationId` (`com.laya.gate`) reads the graphs, the table and a rows file of captured ids + markers from `/data/local/tmp`, runs them with `GpuOptions(precision = FP32)` and dumps the raw marker/act logits; the Mac decodes them with the same arithmetic as the CPU parity. The shipped `com.laya` release and its files stay untouched; the gate package and the tmp files are removed afterwards. (5) S26 (LiteRT 2.2.0, GPU FP32): one partition for every graph (2223/2223 fp16-FC, 2100/2100 fp32, act 4/4); warm main+act medians 122.9 ms (English s256 wfp16), 125.9 ms (typed-decisions s256 wfp16), 616–623 ms for the s512 graphs measured after the phone had warmed to 40–44 °C (the cold first rows were 290–632 ms) — label such runs contended; compile 2.5–5.8 s; process PSS 3.9–4.7 GB with the 1.48 GB fp32 file loaded. CPU (XNNPACK, 4 threads) 583 ms warm for English s256 wfp16. (6) The typed-decisions oracle: `laya` 0.3.4 defines the four workflow question-id sets (`laya.router._TYPED_DECISION_WORKFLOWS`), not full question presets; fixtures must author wording and criteria in the upstream format and freeze `agent.predict` outputs before conversion.

## 2026-09-26 追記 — GLiFormer Large v1 (layout-DeBERTa-large, 575M) NER path on S26 GPU — and the unrolled-BiLSTM compile wall

Shipped as `litert-community/GLiFormer-Large-NER-LiteRT` (s128 single graph; s256/s512 as encoder + head graphs; fp32 references; fp32 + fp16 token tables; Python host runtime). The facts in this section were written by Codex (gpt-6-astra) in one supervised run of 11 rounds and re-checked by the supervisor from the raw evidence (span sets and score deltas recounted from the gate JSON, the offline example re-run): `~/code/codex-conversions/2026-09-26/gliformer-large-v1/` (ROUNDS.md, results/).

**Graph cut for the NER path of a multi-head GLiFormer checkpoint.** `gliner_config.json`: `backbone_type deberta_2d`, `model_type gliformer-layout`, `use_layout true` — but `LayoutDebertaModel.forward(bbox=None, layout_input_mask=None, page_token_ids=None)` skips the spatial and page embeddings and the layout attention bias, and `predict_entities` on plain text passes none of them; export the text path only and assert those arguments are `None` in the source trace. NER = `NERHead(AnchoredSpanExtractionHead)`: parent anchor (`[SCHEMA]`, A = 1), `anchor_modeling linear` over the C `[ENTITY]` label embeddings, `AnchoredSpanScorer` → `(BN, L, C, 3)` start/end/inside logits; `represent_spans false`, so the decoder is `NERDecoder.decode → SpanDecoder.decode_bio_spans_batch → map_results` (pairing + overlap removal on the host, no top-k in the graph). **Do not drop the BiLSTM:** `num_rnn_layers = 1` is an active one-layer bidirectional LSTM over the pooled words (its input projection is already hoisted upstream); the graph unrolls it for the fixed word capacity T with float validity masks (h/c carried through padding, outputs zeroed, the reverse direction starts at the last real word). Inputs: `inputs_embeds [1,N,1024]` (host lookup, embedding LayerNorm stays in the graph, `position_biased_input false`), float `attention_mask [1,N]`, one-hot `text_routing [1,T,N]`, `parent_routing [1,1,N]`, `label_routing [1,5,N]`, `text_mask [1,T]`; one packed leaf `[1,1,T,15]`. DeBERTa relative positions: exact upstream `make_log_bucket_position` (`position_buckets 256`, linear only within ±128) baked as two projected signed-distance tables per layer, `[1,16,2N,64]` each, selected by a rank-4 matmul + skew/reshape/slice — O(N) constants, so s512 fp32 stays at 1.52 GB (< 2 GB flatbuffer wall). CPU parity vs the official fp32 forward: logits max |Δ| 6.3e-5 (s128), 1.3e-4 (s256/s512); 70/70 identical span sets, micro-F1 1.000, fp32 and fp16-weight.

**The unrolled-BiLSTM head is a compile wall on ML Drift (S26, LiteRT 2.2.0), independent of memory.** Full s128 (T = 48, 4,149 ops) compiles: 1 partition, 82 ms, 10/10. Full s256 (14,341 ops) → SIGSEGV during `Initializing OpenCL-based API from graph` (exit 139, `fault addr 0x100000000000000f`). Isolation: encoder-only s256 (1,773 ops) compiles (peak 3.6 GB); head-only s256 (12,558 ops) crashes; the same head with the input projection hoisted (already the case upstream) crashes identically; head with the two recurrent matrices supplied as runtime inputs (14,086 ops) crashes at **570 MB** RSS — so it is not an out-of-memory; head at T = 128 (6,286 ops) crashes too. A synthetic chain of 4,000 or 8,000 elementwise ops fails differently (`status 3`, `Failed to build program executable - Out of host memory`, 53 s / 444 s of compile), so no universal op-count threshold is established; the trigger is the topology of a long unrolled recurrence. Issue candidate with two reproducer graphs (`exports/probes_round5/P2_s256_wfp16.tflite`, `probes_round6`). Escape used: **encoder graph on GPU + head graph on CPU** (routing matmuls inside the head, the host passes hidden states): s256 encoder 176 ms + head 167 ms (one process 410 ms, 4.57 GB resident), s512 encoder 931 ms (battery 41 °C) + head 349 ms. A chunked-LSTM all-GPU route (48-word chunk graph with h/c state I/O, ≈ 14 ms per chunk per direction from the s128 residual) was costed, not built.

**Memory facts a port must plan for.** A 707 MB fp16-weight s128 graph loads to **2.59 GB resident** on the S26 (fp32 file: 3.24 GB) and peaks at **4.54 GB during compile** (fp32: 5.17 GB); the phone has 11.4 GB total / ~7 GB available. Process cold start to first result 5.0 s (compile 4.7 s), 80 ms afterwards. The CPU head is worse per byte: the 53–56 MB fp16-weight heads load to 2.4 GB (s256) / 4.7 GB (s512), and **fp32 heads do not help** (4.73 GB vs 4.68 GB at s512, 8 % faster) — the per-node materialization of the unrolled recurrence, not fp16 DEQUANTIZE copies, is the cost on both backends. Consequence: s256 is the practical top window in an app; s512 ships for completeness with that caveat.

**Precision.** GPU default precision compiles and returns finite logits but **zero entities** (packed-logit |Δ| 26.8, norm ratio 0.003–0.006); `GpuOptions(precision = FP32)` is exact (score |Δ| ≤ 1.3e-6 fp32, ≤ 2.4e-4 fp16-weight). Same class as GLiNER2.5 (NaN there). No int8 attempted (catalog D13).

**Toolchain.** `pytorch_model.bin` (2.30 GB, 506 fp32 tensors) loads under torch 2.12's default `weights_only=True`; `gliformer` 0.1.2 (PyPI, GitHub `b5c0a0fd`) + `gliner` 0.2.29 (pins transformers < 5.17; 5.16.1 used) + litert-torch 0.9.3 / ai-edge-litert 2.1.6 / ai-edge-quantizer 0.8.0 `FLOAT_CASTING` (FC weights only; BATCH_MATMUL relative tables stay fp32). The base encoder card `knowledgator/DeBERTa-large-joint-3000` is not readable (HTTP 401); only the model's Apache-2.0 is asserted.

## 2026-09-26 追記 — GLiNER2.5-Decide (DeBERTa-v3-large classification) on S26 GPU

Shipped as `litert-community/GLiNER2.5-Decide-LiteRT` (windows 128/256/512, fp16-weight defaults + fp32 references, float16 embedding table, host contract, Python runtime, Android sample `gliner25decide/`). Run dir with every gate JSON: `~/code/codex-conversions/2026-09-26/gliner25-decide-litert/` (supervised run, 6 rounds; `SUPERVISOR.md` holds the recomputed numbers).

**Graph cut.** gliner2's classification path is `classifier(hidden[[L] positions])`, so the graph is the DeBERTa-v3-large encoder (24 layers, hidden 1024) plus `rank4_matmul(label_routing, hidden)` and the two classifier FULLY_CONNECTED layers (1024→2048→ReLU→1): inputs `inputs_embeds [1,N,1024]`, `attention_mask [1,N]`, `label_routing [1,32,N]` (host one-hot rows at the `[L]` markers), output `logits [1,1,1,32]`. The host keeps the embedding lookup, the gliner2 schema/tokenization and softmax / sigmoid / threshold. The span and counting heads are not exported. There is no data-dependent op, so nothing else stays on the host.

**The Small graph type scales unchanged.** GLiNER2.5 Small's `ShapedDebertaLayer` + `rank4_matmul` (above) was reused verbatim for 24 layers at hidden 1024 and exported on the first attempt: 1,634 ops at every window, rank ≤ 4, none of the rejected ops, 1,634/1,634 on the S26 GPU in one partition. Rewritten vs the native transformers encoder: max |Δlogit| 0.0 on 42 / 328 / 361 fixtures. Only the baked relative-position constants grow with N (s256 − s128 = 24 × 2 × 16 × 128 × 64 × 4 B).

**Precision.** Default GPU precision compiled and gave finite logits but only 14/42 decisions (Small gave NaN instead); `GpuOptions(precision = FP32)` gave 42/42. Finite output is not a pass — gate decisions.

**Storage.** `FLOAT_CASTING` fp16 on the 146 FC weights (+146 DEQUANTIZE, 1,780 ops) halves the graphs (0.52–0.57×) with no S26 GPU latency cost (s256 median 198.8 vs 198.5 ms fp32, s512 733.1 vs 742.9 ms) and 100 % decisions. The `[128011,1024]` embedding table ships as float16 (262 MB instead of 524 MB) with an exact host upcast to float32; every graph × table pair kept 100 % decisions on desktop LiteRT CPU (the fp16 table alone moves logits by ≤ 6.6e-4).

**S26 thermals: publish cool and sustained separately.** Even from a cool start (kgsl ≤ 50 °C, thermal status 0), back-to-back s256 / s512 jobs step the GPU clock limit down from 1,300 to 902 / 646 MHz within the job, and the per-request median climbs 175 → 215 ms and 575 → 912 ms. The overall median mixes both; report the opening request (cool) next to the median (sustained), with the conditions.

**App timing behind a secure keyguard.** `am start` on a phone whose lock screen is a credential starts the app behind the keyguard: it is never top-resumed (`TOP_SLEEPING`) and its process sits in the `background` cpuset (cores 0–1 and 4–5 on the S26, without the two fastest cores 6–7), so host-side timings and the CPU backend are measured on the wrong cores. For debug measurement runs only, the sample calls `setShowWhenLocked(true)` + `setTurnScreenOn(true)` when a diagnostic extra is present; verify `topResumedActivity` and `/proc/<pid>/cgroup` (`top-app`) before trusting a number. With that, a request right after a cold start took 72–78 ms end to end at s128 (0.38–0.39 s of warm-up before Ready) and paced requests a median 93.0 ms on GPU FP32.

**Tokenizer.** Decide's `tokenizer.json` marks `[UNK]` as `normalized: true` (Small's has `false`), and the Small Kotlin tokenizer refused to load it. Normalized added tokens must be matched the way Hugging Face tokenizers' `AddedVocabulary::extract_and_normalize` does: split out the non-normalized added tokens on the raw text, normalize each remaining segment once, then split it on the normalized tokens (`"a [UNK] b"` → `▁a ▁ [UNK] ▁b`). A tokenizer port is per checkpoint; re-check `added_tokens` flags when reusing it.

## 2026-09-29 追記 — Laya Multilingual on fp16: the NPU passes after three fp32-exact rewrites, GPU fp16 does not

Evidence: `~/code/codex-conversions/2026-09-29/laya-fp16-gate/` (`STATUS.md`, `device/`, `device_zoo/`, `emulation/`).
Galaxy S26 SM-S942Q, LiteRT 2.2.0, S256 WFP16 graph, 201 fixture rows, thermal status none before each run.

**The shipped graph returns no finite row in fp16.** On the S26 GPU at `Precision.FP16`, on the Hexagon NPU (which
computes float graphs in fp16) and on the Mac through LiteRT's own Metal GPU at default precision, 0/201 rows are
finite. Two causes: the attention masks multiply `(1 - mask)` by −1e9, which is −inf in fp16, so every real token gets
0 × −inf = NaN; and LayerNorm (MEAN / SQUARED_DIFFERENCE / RSQRT) squares activations that fp16 cannot hold. From
layer 12 the `<bos>` row carries 1.3e4–1.4e4, separator rows 1.4e3–3.8e3 and some punctuation / common-word rows
4e3–7.6e3, all in channels 418, 424, 449, 468, 488, 530, 580, 614; `(x − mean)²` reaches 2e8. One layer-11 MLP neuron
(924, GeGLU product up to 36,341) writes the `<bos>` value; layer-12 neurons 957 and 878 write most of the rest.

**Three rewrites, each exact in fp32** (the rewritten graph is bit-identical to the shipped one in PyTorch on 5 probe
rows; Mac CPU 201/201, argmax 81/81, max Δp 1.45e-3 WFP16 as before; +30 MUL, 1809 ops):
1. LayerNorm on `x · 2^-k` with `eps · 2^-2k`, k per LayerNorm from max |input| over the fixture rows with one power of
   two of margin (k = 2 in layers 9–11, 8 in layers 12–21 and the final norm, 4 in the second head layer and the
   scorer, 0 elsewhere). One global 2^-8 for every LayerNorm NaNs instead: small rows underflow to a zero variance.
2. Masks −1e4 (key + band = −2e4, finite in fp16; exp(−1e4) is already 0 in fp32).
3. Layers 11–12: the gate half of `Wi` × 2^-4 and `Wo` × 2^4 (the GeGLU product used 55 % of the fp16 range).

| Graph / accelerator | Finite | Choice/score argmax | Max Δp | Warm median ms |
|---|---:|---:|---:|---:|
| rewritten, GPU FP32 | 201/201 | 81/81 | 0.0014 | 51.8 |
| rewritten, GPU FP16_WITH_FP32_ACCUM | 201/201 | 81/81 | 0.0124 (one row over 0.01) | 40.8 |
| rewritten, GPU FP16 | 201/201 | 80/81 | 0.0331 | 28.2 |
| rewritten, NPU (JIT; one `DispatchDelegate` node) | 201/201 | 81/81 | 0.0069 | 36.5 |
| shipped, GPU FP16 / NPU | 0/201 | — | — | 27.2 / 35.1 |

The NPU result repeated exactly on a second process (JIT compile 40.3 s first, 0.22 s from the cache). The sample app
ships GPU FP32 and the NPU; GPU fp16 stays off.

**Predict fp16 on the Mac before the device.** A PyTorch emulation that rounds every op to fp16 but accumulates matmuls
in fp32 predicted 81/81 / 5.1e-3 — close to the NPU and to FP16_WITH_FP32_ACCUM. LiteRT's Metal GPU at default precision
(`ai_edge_litert` GpuOptions without `enforce_f32`) gave 78/81 / 0.038 — close to S26 GPU FP16. Use the emulation to find
which op overflows and the Metal run as the plain-FP16 pre-gate; the Python API cannot select FP16_WITH_FP32_ACCUM.
Not needed here: scaling q before QKᵀ, keeping softmax output in fp32 or scaling it by 2^10 (attention mass below the
fp16 normal range was ≤ 0.14 % per row; the 2^10 scale made the error worse).

**English, typed-decisions and the token-id form (same day).** ModernBERT-large carries 1.7e3–1.9e3 from layer 6,
3.4e3–3.7e3 from layer 8, 1.2e4 from layer 15 and 3.2e4–3.3e4 from layer 20 (both checkpoints, no GeGLU product above 4096), so
the per-LayerNorm k reaches 9–10; the same rule (k = ceil(log2(max / 64))) and −1e4 masks made every fp32 graph
bit-identical and every NPU run pass. S26 NPU, rewritten graphs, warm median per question:

| Graph | Rows | Choice/score argmax | Max Δp | NPU ms | GPU FP32 ms, same session |
|---|---:|---:|---:|---:|---:|
| English S256 wfp16 (embeds) | 140 | 59/59 | 0.0072 | 65.7 | 192.6 [120.4, 252.8], phone warming |
| English S512 wfp16 (embeds) | 209 | 87/87 | 0.0072 | 167.2 | — |
| typed-decisions S256 wfp16 | 100 | 65/65 | 0.0034 | 72.8 | 124.8 |
| typed-decisions S512 wfp16 | 175 | 112/112 | 0.0034 | 173.7 | — |
| English S256 fp32 (token ids, table in graph) | 140 | 59/59 | 0.0054 | 83.8 | — |
| multilingual S256 fp32 (token ids) | 201 | 81/81 | 0.0115 | 37.1 | — |
| multilingual S256 wfp16 (token ids, fp16 table + DEQUANTIZE) | 201 | 81/81 | 0.0068 | 36.8 | — |

Two NPU facts from this table: the fp16-weight file (DEQUANTIZE-fed FULLY_CONNECTED and EMBEDDING_LOOKUP) compiles whole
on the HTP although ML Drift rejects it on the GPU, and on the same rows it was *more* accurate than the fp32-weight
file with identical weight values (0.0068 against 0.0115) — take the wfp16 file for the NPU. First-launch JIT compile was
31–59 s for these 0.64–1.7 GB graphs; later launches load the cache.

**No integer path keeps the answers.** Post-training int8 weights change answers (per-channel 36/38, max Δp 0.089; GPTQ
38/38 but 0.040; every projection type alone already exceeds 0.01); int4 flips 6–11 of 38. int16 activations pass
(38/38, 5.3e-3) only with 16-bit weights and the few input columns above 16× the median column max (the neurons above)
kept in float. Weight-only int8 / int4 does not change NPU speed anyway (NPU factor lane).

## 2026-09-30 追記 — Julia-1 (mmBERT-small decision model) on the S26 GPU: an F32 checkpoint pays for fp16 storage, and fp16 arithmetic changes its answers

Shipped as `litert-community/Julia-1-LiteRT` (S512 and S1024 fp32 graphs, float16 token table, the source
`tokenizer.json`, Python host, conversion scripts). Evidence: `~/code/codex-conversions/2026-09-30/julia1-litert/`
(`scripts/julia_graph.py`, `scripts/safe_v*.json`, `results/`, `device/`; `card/README.facts.md` names the source file
of every number). Galaxy S26 SM-S942Q, LiteRT 2.2.0, a debug gate app in the Laya English pattern (captured ids +
markers from `/data/local/tmp`, raw marker logits back to the Mac). Reference = the author's runtime raw logits on CPU
FP32; the S512 device rows are all 306 boundary rows (reference top-1 < 0.9) + 400 others = 706.

**The Laya cut carries over unchanged to a 384-wide encoder.** `SupersonicLabs/Julia-1` is the same marker-based
decision model as Laya (type embedding + two pre-norm ReLU `nn.TransformerEncoderLayer` blocks + LayerNorm–Linear–GELU–
Linear scorer) on `jhu-clsp/mmBERT-small`: 22 layers, hidden 384, 6 heads, global attention every third layer, local
band ±64, RoPE theta 160000 for both layer types, vocabulary 256,000. The 09-22 graph — host token lookup,
`inputs_embeds [1,S,384]` + float `attention_mask [1,S]` + `qtype_onehot [1,3]` → `token_logits [1,S]`, RoPE tables
baked per layer type as separate contiguous clones, the band as a constant, the head's fused fast path written out —
exported at the first attempt: 1,704 ops at S512 and at S1024 (only the baked constants grow, 185.1 → 188.5 MB), one
LITERT_CL partition on the S26 under `GpuOptions(precision = FP32)`. Two outputs fewer than Laya: no `pooled_cls` and
no act-head graph, because Julia's public API (`predict`, `logits`, the named questions) runs with
`return_actions=False` — the `act_head` weights in the checkpoint are dead for a port. Oracle: the author's
`scripts/reproduce_typed.py` run unchanged reproduces the published CPU FP32 numbers on `LocalLLaMA/typed-decisions`
(choice 426/600, score 542/800, noul 483/600); the Python host over the S1024 graph returns the same answer on all
2,000 questions with either table (35 of them need more than 512 tokens, max 607 — read the window off the dataset's
token lengths, p50 309 / p99 566). Trap in the author's `Julia-1-ONNX/parity-cases.json`: its 100 requests were
encoded with `head_length` 256, not the runtime default 512; reproduce a parity file at its own setting before
comparing (argmax 100/100, max |Δlogit| 1.0e-4 once matched).

**An F32 checkpoint pays for fp16 storage.** Laya's token tables round-tripped fp32 → fp16 → fp32 exactly (09-22),
so its float16 host table cost nothing. Julia-1 stores float32 weights (170 tensors, all F32), and every fp16 storage
step is a measurable rounding — Mac CPU, S512 graph, 2,065 rows, fp32 arithmetic throughout:

| Storage | Same argmax | Max Δp | Rows > 0.01 |
|---|---:|---:|---:|
| fp32 graph + float32 table | 2,065/2,065 | 0.00007 | 0 |
| fp32 graph + float16 table (shipped) | 2,065/2,065 | 0.0077 | 0 |
| wfp16 graph + float32 table | 2,065/2,065 | 0.031 | 31 |
| wfp16 graph + float16 table | 2,065/2,065 | 0.027 | 33 |

The float16 table passes the fp16-storage bar (same argmax, |Δp| ≤ 1e-2 on every row) and ships as the default; the
wfp16 graph does not and is not shipped. "The table is exact" was a checkpoint fact for Laya, not a rule: measure the
table on every checkpoint and state its cost on the card.

**fp16 arithmetic changes the answers; the fp32-exact rewrites only make it finite.** Both graphs carry two of the
09-29 rewrites (per-LayerNorm 2^-k with eps·2^-2k: k = 8 in layers 12–21 and the final norm, k = 2 in the head and
scorer norms; masks −1e4) plus q × 1/8 before QKᵀ (a power of two, exact; its effect was not isolated) and no GeGLU
gate rescale; the rewritten graph is bit-identical to the plain graph in PyTorch fp32 on 5 probe rows. The
residual stream needs them: ≤ 36 through layer 11, then one layer-11 GeGLU product (14,345) writes 3,147 into layers
12–18 and layer 18's MLP raises it to 5,315 through the final norm, where (x − mean)² reaches 2.8e7. Every fp16 path
then returned finite rows and different answers:

| Path | Graph + table | Rows | Same argmax | Max Δp | Rows > 0.01 | Warm median ms |
|---|---|---:|---:|---:|---:|---:|
| S26 GPU FP32 (session 1) | s512 fp32 + f32 table | 706 | 706 | 0.00005 | 0 | 80.7 |
| S26 NPU, JIT 34 s, one DispatchDelegate node (1803/1803) | s512 wfp16 + f32 table | 706 | 684 | 0.425 | 296 | 27.7 |
| S26 NPU, JIT 41 s (1704/1704) | s512 fp32 + f32 table | 706 | 685 | 0.418 | 286 | 29.3 |
| Mac Metal, `enforce_f32` | s512 fp32 + f32 table | 400 | 400 | 0.00007 | 0 | 11.4 (informational) |
| Mac Metal, default precision | s512 fp32 + f32 table | 400 | 375 | 0.57 | 285 | — |

Three facts a port can reuse. (1) The wfp16-on-CPU number is the NPU's floor: the HTP computes with fp16 weights, so
0.031 on the Mac had already failed the bar before the phone was touched. The two NPU files compiled to identical QNN
runlists (the same `finalize_runlist` line, 14,954 vector + 3,414 matrix ops): on the HTP the storage dtype changes
nothing but the rounding of the weights, so the fp32 file is not a way around that floor. (2) Mac Metal at default
precision predicts plain GPU fp16 (375/400 here; 78/81 on Laya, where S26 GPU FP16 gave 80/81), and Metal
`enforce_f32` predicts S26 GPU FP32 (0.00007 vs 0.00005). Run both before any device fp16 gate. (3) The encoder, not
the head, is the sensitive part: PyTorch in float16 on the Apple GPU (MPS) over the 306 boundary rows gave max Δp 0.087
with only the encoder in fp16 (303/306) and 0.0014 with only the head + scorer in fp16 (306/306). An "encoder on the
NPU, head in FP32" split would not rescue this model, so it was not built; S26 GPU FP16 and FP16_WITH_FP32_ACCUM were
not measured.

**A static 2^-k sized for the massive-activation rows pushes other rows toward fp16 underflow; a per-row scale fixes
the emulation, not the kernels.** In a flush-to-zero emulation (every op rounded to fp16, |x| < 2^-14 → 0, LayerNorm
decomposed as the LiteRT graph computes it) the k = 8 graph returned 0/150 finite boundary rows and k = 6 returned
150/150 (with max Δp 0.71 — finite, not right). The split matches rsqrt(eps·2^-2k) on a row whose scaled variance
flushes to zero: 8.1e4 at k = 8 (above the fp16 maximum 65,504), 2.0e4 at k = 6. A row-scaled LayerNorm — SafeLayerNorm
v2 with the floor lowered from 1 to 2^-6 so small rows are scaled *up* as well (`s = max(max|x|/8, 2^-6)`, the epsilon
term formed as `(√eps / s)²` so `s²` never exists) — kept 149/150 rows finite in that emulation (max Δp 0.079). ML
Drift's Metal kernels never needed it: the static k = 8 graph was already finite on 400/400 rows there, and the
row-scaled graph gave the same wrong answers (374/400, max Δp 0.45, against 375/400, 0.57). Use the flush-to-zero
emulation to locate an underflow, not to predict a kernel; when the fp16-safe graph is finite on Metal and still wrong,
the wall is sensitivity and no LayerNorm rewrite moves it.

**S26 GPU timing moves by session on the same graph; report the minimum with the median.** Three S512 GPU FP32 runs
of the same file (compile 0.88–0.92 s, 1704/1704 LITERT_CL, median over 701 warm rows):

| Run | Start thermal status / battery | Warm median ms | Min | Max | First call |
|---|---|---:|---:|---:|---:|
| session 1, f32 table | 0 / 34.2 °C | 80.7 | 56.3 | 104.1 | 62.9 |
| session 2 run 1, fp16 table | 0 / 35.7 °C | 111.8 | 56.0 | 134.9 | 57.5 |
| session 2 run 2, fp16 table, 60 s later | 1 / 38.2 °C | 112.1 | 56.7 | 134.2 | 78.4 |

The start thermal status and battery temperature did not separate the sessions (both started at status 0, 1.5 °C
apart); the phone had run other work before session 2. The minimum agreed within 0.7 ms across the three runs and the
first call was fast in all three, so the medians reflect the phone's state during the run (GPU clock and kgsl
temperature were not captured), not warm-up and not the table file (the host lookup is outside the timed interval: Kotlin loop median 19.0 ms over the f32 table, 23.6 ms with
`Half.toFloat` over the fp16 table, debug build). S1024 in the same warm session: 100 rows (the 35 typed-decisions
questions over 512 tokens + 65 others, 18 boundary), 100/100, max Δp 0.0077, median 298.5 ms [155.6, 305.2] at thermal
status 2. Publish the minimum and the start state next to the median, as the GLiNER2.5-Decide section says; a median
alone does not reproduce.

## 2026-10-01 追記 — Open Decision (DeBERTa-v3-large typed decisions, com-kotobalabs/open-jev-deberta-v3-large) on the S26 GPU: the GLiNER2.5 shaped-DeBERTa layer reuses as is, the mask constant is the fp16 NaN but not the fp16 wall, and the NPU prepare runs out of memory at S512

Shipped as `litert-community/Open-Decision-DeBERTa-v3-Large-LiteRT` (S256 and S512 fp16-weight graphs, float16
word table, the source `tokenizer.json`, Python host, Kotlin snippet, conversion scripts). Evidence:
`~/code/codex-conversions/2026-10-01/openjev-deberta-litert/` (`scripts/graph.py`, `results/`, `device/`;
`card/README.facts.md` names the source file of every number). Reference = the checkpoint's own
`typed_decisions` package (CPU fp32, transformers 4.57.6) on the author's public test files
(`kotoba-lang/typed-decisions`: 1,809 requests / 4,327 questions, 1,830 of them boundary = top-1 < 0.9).

**The GLiNER2.5-Decide graph type carries over to a span-pool decision head unchanged.** The model is a
`DebertaV2Model` (24 layers, hidden 1,024, 16 heads, relative attention with 256 log buckets) plus a head that scores
each option from `[mean(question text tokens); mean(option text tokens); product]` and softmaxes within each
question at temperature 1.05. The 09-26 `ShapedDebertaLayer` (log-bucket relative positions pre-expanded as
constants, rank-4 batch matmuls, float mask, native GELU) was vendored verbatim; the author's `scatter_add` span
means became two host-built routing inputs `q_routing [1,128,S]` / `o_routing [1,128,S]` (row j = 1/len over the
text tokens of option j's question and of option j), so the pooling is two batch matmuls against the hidden states
and the head stays in the graph: `inputs_embeds [1,S,1024]` + `attention_mask [1,S]` + the two routings →
`logits [1,1,1,128]`. In PyTorch fp32 the rewritten graph is bit-identical to the stock `DebertaV2Model` path.
Export: 1,639 ops (1,785 with fp16 FC weights), no GATHER / CAST / int64, rank ≤ 4, 25 s per window; the 48
relative-position constants stay fp32 under FLOAT_CASTING. Option slots (128) are the only contract knob; the
fixtures need at most 89 (banking77: 77 intents + 10 areas + 2).

| Where | Graph + table | Requests | Questions (boundary) | Same argmax | Max Δp | Median ms |
|---|---|---:|---:|---:|---:|---:|
| Mac CPU | s512 fp32 + f32 table | 1,809 | 4,327 (1,830) | 4,327 | 1.1e-5 | 291 (loaded) |
| Mac CPU | s512 wfp16 + fp16 table (shipped) | 1,809 | 4,327 (1,830) | 4,327 | 0.0016 | 532 (loaded) |
| Mac Metal, `enforce_f32` | s512 wfp16 + fp16 table | 100 | 222 (92) | 222 | 0.0016 | 68 |
| Mac Metal, default precision | any graph | 100 | 222 | 0 — every output non-finite | | |
| S26 GPU FP32 explicit (LiteRT 2.2.0) | s512 wfp16 + fp16 table | 500 | 1,302 (754) | 1,302 | 0.00083 | 697 [576, 1202] |
| S26 GPU FP32 explicit | s256 wfp16 + fp16 table | 419 | 1,074 (712) | 1,074 | 0.00074 | 408 [376, 485] |
| S26 NPU, JIT (experiment: −1e4 mask + SafeLayerNorm k=3) | s256 wfp16 + fp16 table | 100 | 276 (192) | 274 | 0.025 | 110 [107, 117] |

**fp16 weights + fp16 table keep every answer on this F32 checkpoint** (unlike Julia-1, 09-30): the mean pooling
averages the rounding out, and the shipped form is the fp16-weight graph.

**Two fp16 walls, both fixed by fp32-exact rewrites; the NPU then lands two questions short of the bar.**
`torch.finfo(float32).min` (transformers' own DeBERTa mask value, kept by the shaped layer) is −inf in fp16, so
`(1 − mask)·min` is NaN at every real position: Mac Metal at default precision returns all non-finite logits for
the fp32 and the wfp16 graphs alike, and the S26 NPU returns finite but wrong answers (95/276) with the same
constant. A −1e4 mask (fp32 bit-identical; `ShapedDebertaLayerFiniteMask` in `graph.py`) is not enough: Metal fp16
becomes finite but wrong (185/222) and the NPU stays wrong (102/270, 6 non-finite questions), because the
LayerNorm sums of squares overflow fp16 — the residual stream only reaches 28.5, but 1024 × 28.5² ≈ 8e5. With the
09-29 SafeLayerNorm (every LayerNorm on x·2⁻³ with eps·2⁻⁶, bit-identical in fp32; `--ln-shift 3`) on top of the
mask, Metal fp16 keeps 200/200 answers (max Δp 0.015) and the S26 NPU keeps 274/276 (max Δp 0.025, 7 questions
over 0.01) at **110 ms per 256-token request** (JIT 172 s, one DispatchDelegate node) against 408 ms on the GPU
with explicit FP32; k = 2 still overflows (8 non-finite), k = 4 is no better (273/276). The two NPU flips are
near-ties (reference gaps 0.0065 and 0.040), so the fp16-sensitivity remainder is real but small; the strict bar
(every answer, Δp ≤ 0.01) is not met and the NPU form is not shipped. Lesson: "finite but wrong after the mask
fix" means an overflow is still hiding — check the LayerNorm inputs before calling it sensitivity. GPU default
precision on the S26 behaves like Metal (216/274 with the mask fix alone). The S26 GPU warmed from thermal status
0 to 3 (31 → 45 °C battery) over 500 back-to-back 512-token requests; the median above covers that whole run
(min 576 ms); at S256 the GPU FP32 median is 408 ms [376, 485] on 419 requests, 1,074/1,074 answers.

**The HTP JIT prepare of the S512 graph runs out of memory on the S26.** `CompiledModel` with `Accelerator.NPU`
(QualcommOptions BURST, JIT) aborted 94 s into the compile: `Scudo ERROR: internal map failure (Out of memory)` from
`malloc` inside `libQnnHtpPrepare.so GraphPrepare::sequencing_stage` (SIGABRT, no row ran). This is the
compiler's host-side memory, not the graph's op set (the same ops compile on the GPU as one partition and the S256
graph compiles for the HTP in 170–200 s).

**Tokenizer: this checkpoint's `tokenizer.json` keeps SentencePiece's `Precompiled` charsmap** (`Strip` →
`Precompiled` → `Replace " {2,}"`), unlike the GLiNER2.5 files (`Replace` + `NFC` + `Strip`). The sample's
`DecisionTokenizer.kt` ports the charsmap as Hugging Face tokenizers reads it (darts-clone double array over UTF-8
bytes, applied per extended grapheme cluster shorter than 6 bytes, else per code point, first common-prefix match)
and reproduces the official ids and spans on all 1,809 requests on the JVM and 29/29 edge probes (ligatures,
full-width letters, emoji, IPA, Japanese). The official path is the fast tokenizer; `spm.model` (slow) differs on
exotic characters (an IPA "ꜜ" → `[UNK]` vs byte pieces), so a port must follow `tokenizer.json`.

## 2026-10-02 追記 — ModernBERT-Ja-310M Decision (argos1111/modernbert-ja-310m-jev, Japanese cross-encoder) on the S26 GPU and NPU: the Laya ModernBERT cut as a one-logit pair scorer, the NPU keeps every answer but one probability by 0.0113, and fp16 computation still moves answers after the exact rewrites

Shipped as `mlboydaisuke/ModernBERT-Ja-310M-Decision-LiteRT` (CC BY-SA 4.0, the source weights' license, so not litert-community;
S256 and S512 fp16-weight graphs, float16 token table, the source `tokenizer.json`, Python host, Kotlin block, the debug gate app,
conversion scripts). Evidence: `~/code/codex-conversions/2026-10-02/modernbert-ja-decision-litert/` (`scripts/graph.py`,
`results/`, `device/`; `card/README.facts.md` names the source file of every number). Galaxy S26 SM-S942Q, LiteRT 2.2.0, a debug
gate app in the Laya English pattern. Reference = the card's own transformers snippet with transformers 5.17.0 (the author's
version) on CPU FP32: 547 requests / 621 questions / 2,357 pairs (JGLUE v1.3 test 250 JNLI + 250 JCommonsenseQA, the author's
16 hand-written items and 12-question example, 30 invented Japanese requests; 123 boundary questions with top-1 < 0.9).

**A cross-encoder decision model is the Laya cut with one logit out.** `ModernBertForSequenceClassification` (25 layers,
hidden 768, 12 heads, GeGLU, global attention every third layer with RoPE theta 160000, sliding ±64 with theta 10000, CLS
pooling) scores one `<s> context </s><s> candidate </s>` pair per call; the softmax over a question's candidates is the
answer. The 09-21 ModernBERT graph form carries over: host token lookup (`inputs_embeds [1,S,768]` + float
`attention_mask [1,S]`), RoPE baked per layer type as two separate contiguous clones, the band as a constant, the final
LayerNorm applied to position 0 only, the head (dense → GELU → LayerNorm → classifier) inside the graph → `logit [1,1,1,1]`.
1,809 ops at fp32 (1,911 with fp16 FULLY_CONNECTED weights), rank ≤ 4, no GATHER / CAST / int64. In PyTorch fp32 the graph is
within 3.8e-6 of the stock forward. One request costs one call per candidate (the 547 fixture requests took 2,357 calls); a
batched `[K,S,768]` form was not built.

**transformers 4.57.6 cannot load this repository's tokenizer; the `tokenizers` library can.** The checkpoint was saved by
transformers 5.17.0 (`tokenizer_class: TokenizersBackend`, config with `layer_types` and `rope_parameters`). 4.57.6 loads the
model (the v5 rope keys are ignored and the 4.x defaults 160000 / 10000 equal them; logits identical to 5.17.0 on the probe) but
raises on `AutoTokenizer`. The host reads `tokenizer.json` with the `tokenizers` library (`enable_truncation(512,
strategy="only_first")`): same ids as the 5.17.0 `AutoTokenizer` on all 2,357 pairs, and the base model's `tokenizer.json`
(which the author's serving code loads) gives the same ids. The pair template is `<s> A </s><s> B </s>` (ids 1 / 2); `<cls>` 6
and `<sep>` 4 exist but are unused, and the card's "CLS pooling" means position 0 = `<s>`. Take the oracle in a transformers
5.x venv when a checkpoint is saved by 5.x, and compare the ids before trusting the 4.x conversion venv.

**fp32-exact rewrites, calibrated on the fixture pairs.** LayerNorm inputs stay ≤ 64 through layer 13, then 1,200 (layers
14–15), ~2,000 (16–18) and 2,629–2,694 (19–24); the final norm sees 207 at position 0; the GeGLU product peaks at 1,236
(< 4,096, no gate rescale). SafeLayerNorm k = ceil(log2(max/64)) → k 5 in layers 14–18, 6 in 19–24, 2 on the final norm; masks
−1e4. Bit-identical in fp32 on 7 probe pairs at S512 and 5 at S256. The rewrites make every fp16 path finite; they do not
make it correct (next point).

| Where | Graph + table | Requests | Questions (boundary) | Same argmax | Max Δp | Questions > 0.01 | Warm median per pair |
|---|---|---:|---:|---:|---:|---:|---|
| Mac CPU | s512 fp32 + f32 table | 547 | 621 (123) | 621 | 9.8e-6 | 0 | 251 ms (loaded) |
| Mac CPU | s512 fp32 + fp16 table | 547 | 621 (123) | 621 | 0.00039 | 0 | 142 ms |
| Mac CPU | s512 wfp16 + fp16 table (shipped) | 547 | 621 (123) | 621 | 0.00079 | 0 | 198 ms |
| Mac Metal explicit FP32 | s512 wfp16 + fp16 table | 547 | 621 (123) | 621 | 0.00078 | 0 | 33 ms |
| Mac Metal default precision (fp16) | s512 wfp16 + fp16 table | 547 | 621 (123) | 619 (flip gaps 0.027, 0.0018) | 0.042 | 32 | 30 ms |
| S26 GPU explicit FP32, thermal 0 → 2, 34.9 → 44.8 °C | s512 wfp16 + fp16 table | 547 | 621 (123) | 621 | 0.00079 | 0 | 416.4 ms [183.6, 669.0] |
| S26 GPU explicit FP32, thermal 2, 44.8 °C | s256 wfp16 + fp16 table | 546 | 618 (120) | 618 | 0.00079 | 0 | 229.2 ms [199.7, 296.9] |
| S26 GPU default precision / FP16, thermal 3 | s512 wfp16 + fp16 table | 60 | 60 (3) | 60 | 0.0202 | 2 | 231 / 253 ms |
| S26 GPU FP16_WITH_FP32_ACCUM, thermal 3 | s512 wfp16 + fp16 table | 60 | 60 (3) | 60 | 0.0045 | 0 | 403 ms |
| S26 NPU (JIT 29.5 s, one DispatchDelegate node), thermal 2 → 3 | s256 wfp16 + fp16 table | 546 | 618 (120) | 618 | 0.0113 | 1 | 41.7 ms [30.6, 53.4] |
| S26 NPU (JIT 83.0 s, one node), thermal 3 | s512 wfp16 + fp16 table | 547 | 621 (123) | 621 | 0.0113 | 1 | 142.6 ms [110.6, 151.7] |
| S26 NPU (JIT cache 0.5 s) | s256 wfp16 + **f32 table** | 546 | 618 (120) | 618 | 0.0113 (same question) | 1 | 44.6 ms |

**The NPU keeps every winner and misses the Δp bar by one question.** Both windows compile whole for the HTP (unlike the
24-layer / 1024-wide DeBERTa graph, whose S512 prepare ran out of host memory): JIT 29.5 s at S256, 83.0 s at S512. Same
argmax on all 618 / 621 questions including every boundary question, max Δp 0.0113 on one invented internal-document `noul`
question (reference p(true) 0.234 → 0.245, reference gap 0.53, not a near-tie), every other question ≤ 0.0062. The float32
table reproduces the 0.0113 exactly, so the one miss is the HTP's fp16 arithmetic, not the table rounding. 3.4–5.5× faster
than the GPU with explicit FP32 on the same hot phone (42 vs 229 ms at S256, 143 vs 416 ms at S512). The card reports the NPU
with that number; the verified path is the GPU with explicit FP32.

**fp16 computation moves answers after the rewrites — the Laya / Julia-1 pattern on ModernBERT-base.** Mac Metal default
precision flips 2 of 621 (and 2 of 618 at S256) with 32 questions over 0.01; the S26 GPU at default precision and at FP16
(identical numbers) keeps the 60 winners of a 60-request probe but moves 2 over 0.01 (0.0202) at 0.55× the explicit-FP32 time;
FP16_WITH_FP32_ACCUM stays within 0.0045 on the same 60 but runs as slowly as explicit FP32 (403 vs 416 ms). Metal default
precision predicted the S26 GPU fp16 behaviour again (both: finite, a few answers move).

**The float16 table costs 0.00039 on this F32 checkpoint** (104 of 78.6M values flush to zero, max rounding 6.1e-5): fp32
graph + fp16 table 0.00039, wfp16 graph + f32 table 0.00071, both 0.00079 — all under the 0.01 bar with every winner kept, so
the fp16 pair ships (the Julia-1 lesson: measure the table on every F32 checkpoint).

**Rank-3 versus rank-4 attention matmuls.** litert-torch lowers `q @ kᵀ` on `[1,12,S,64]` to rank-3 BATCH_MATMUL ([12,S,64]),
as in the Laya graphs. A `--rank4` export through a marker op (`torch.library.custom_op` + a `dot_general` lowering, the
GLiNER2.5 recipe) keeps rank 4 (1,761 ops in wfp16): identical answers on the Mac and on the S26 GPU at S512 (621/621,
0.00079; 489 ms median at thermal status 3, measured hotter than the rank-3 run, so not a speed comparison). The shipped
files are the rank-3 form (measured on every path); the rank-4 twin is one flag away for a Mali phone, where the rank-3
chain is the known-bad one (Pixel not measured here).

**S26 timing moves with heat within one run.** The S512 GPU run started at thermal status 0 / 34.9 °C and ended at 2 /
44.8 °C after 17 minutes; per-pair time went from 184 ms (minimum, first rows) to 500–670 ms. Every later row started hot
(status 2–3, 44.8–47 °C). Report median with [min, max] and the thermal state; the S256 GPU median (229 ms) is a hot-phone
number and the cool-phone S256 time was not measured.

**Accuracy on our subset (reference, fp32):** JNLI 238/250 (95.2%), JCommonsenseQA 233/250 (93.2%), the author's 16 items
14/16 (one miss is a 0.450 / 0.448 tie on a 3-level score question). The author reports 92.62% / 92.40% / 93.8% on the full
files through their API; the card's printed example (0.9996 for `billing — 請求・返金`) is 0.9981 in fp32 on CPU with either
transformers version (the author trained and ran with bf16 autocast).

## 2026-10-02 追記 — GLiClass-Edge v3.0 (ModernBERT ettin-encoder-32m zero-shot classifier) on the S26 GPU and NPU: the label read-out is one routing matmul, `torch.matmul` lowers below rank 4, and a LayerNorm pre-scale makes fp16 close but not exact

Shipped as `litert-community/GLiClass-Edge-v3.0-LiteRT` (S128 / S256 graphs, fp32 and fp16-weight, float16 token
table, Python host, Kotlin snippet, conversion scripts, the Android sample). Evidence:
`~/code/codex-conversions/2026-10-02/gliclass-edge-litert/` (`card/README.facts.md` names the source of every number).
Reference = pip `gliclass` 0.1.20 on CPU fp32 (it needs transformers ≥ 5): 552 requests (SemIf authored144, ag_news
test 200, banking77 test 200 as 25-label own subsets, the card's 2, invented 6), 482 fit 128 tokens.

**Graph cut: the classification path only.** The pipeline scores the hidden state at each `<<LABEL>>` of
`<<LABEL>>l1…<<LABEL>>ln<<SEP>>` + prompt + text (no separator between prompt and text) against `[CLS]`. The graph
takes `inputs_embeds [1,S,384]` (host lookup) + float `attention_mask [1,S]` + a one-hot `label_routing [1,25,S]` and
returns `logits [1,1,1,25]`: the label states are one batch matmul of the routing against the hidden states, `[CLS]`
is a slice, and both projectors and the MLP scorer stay inside. The scorer's first Linear on `[text; label]` is split
into a text half and a label half that are added, so the text row broadcasts by shape (no BROADCAST_TO). The
ModernBERT layers follow the Laya / Julia-1 cut (Wqkv / Wi split by rows, one RoPE table, ±64 band constant, −1e4
mask, exact builtin GELU): 670 ops (748 with fp16 FC weights), no GATHER / CAST / int64, rank ≤ 4.

**litert-torch 0.9.3 lowered `torch.matmul` below rank 4**: rank-3 attention (`[6,128,64]`) and a rank-2 routing
matmul (`[25,128]@[128,384]`). The vendored `rank4_matmul` marker (Open Decision run) keeps all 21 BATCH_MATMUL at
rank 4 with unchanged logits; use it from the start on ModernBERT graphs.

**The converter folds each LayerNorm scale into the next FULLY_CONNECTED weights** (50 of 78), so FLOAT_CASTING
rounds W·γ, not W. Rounding the folded weights in PyTorch reproduces the 2 near-tie flips of the wfp16 graphs
(`banking77_015`, top-2 gap 0.0012; `banking77_064`, sigmoid 0.4998); rounding W alone flips 4 other rows. As with
Julia-1 (09-30), fp16 storage of an F32 checkpoint costs near ties, so the default download is the fp32 graphs + the
fp16 table (482/482 and 552/552 on desktop CPU; table rounding ≤ 1.2e-4 moves logits by ≤ 0.009).

**fp16 arithmetic: the audit found LayerNorm sums of squares up to 6.4e6** (rows up to 2,064 from their mean in
layers 7–9). SafeLayerNorm at the audit's k (`ceil(log2(max |x − mean| / 64))`: 0 for layers 0–2, 5 for 3–6, 6 for
7–9 and the final norm; 15 norms) is bit-identical in fp32 (PyTorch 552/552, LiteRT CPU and S26 GPU FP32 482/482) and
turns collapse into near ties (top label / label set of 482): Metal default 258/133 → 479/469; S26 GPU default
261/138 → 475/455 at 3.86 ms; S26 NPU (JIT, one partition, 748/748 ops, compile 0.98 s) 480/462 at 2.51 ms, against
5.83 ms on the GPU with explicit FP32. Metal predicted the direction, not the size (16 flipped rows vs 32 on the S26
GPU, 8 shared). NPU flips are all near ties (gaps ≤ 0.0043, sigmoid margins ≤ 0.026), but the bar is every answer:
the shipped mode is GPU explicit FP32 (and CPU), and the sln graphs ship because they are exact in fp32.

**Tokenizer port (byte-level BPE, no JNI).** `add_prefix_space` applies to every split between added tokens, so the
prompt after `<<SEP>>` starts with `Ġ` and text glued to the prompt does not (`fitted.The`). Java's `\s` is
ASCII-only and Android rejects `UNICODE_CHARACTER_CLASS`, so the GPT-2 regex spells out onig's Unicode White_Space.
The Kotlin host matches Python `tokenizers` on 552/552 oracle strings and 261/261 edge strings. On the S26 (LiteRT
2.2.0, GPU explicit FP32) the sample gives the official answers on 552/552 requests from text to labels: graph median
5.86 / 8.23 ms (s128 / s256), a request right after a cold start 8.05–8.38 ms end to end.

## 2026-10-02 追記 — GLiNER2.5 Multi (mDeBERTa-v3-base boundary extractor) on S26 GPU + NPU

Shipped as `litert-community/GLiNER2.5-Multi-LiteRT` (windows 128/256/512, fp16-weight + fp32 graphs, fp16 + fp32 host tables, host contract, Python host). Run dir with every gate JSON: `~/code/codex-conversions/2026-09-19/gliner25-multi/` (supervised Codex run, 8 rounds over 2026-09-19 and 2026-10-02).

**Recipe reuse.** The GLiNER2.5 Small graph type (dense prefix, rank-4 attention, baked log-bucket relative positions, one packed output) exported mDeBERTa-v3-base (hidden 768, 12 layers, vocab 250,112) on the first attempt at all three windows; packed length `1492·T+6494`, 1,148–1,149 ops after fp16 weight casting, no ML Drift rejection. Only the hard-coded dimensions and the sparse-decoder parameter list (traced, 16 tensors, 860,676 B) changed.

**Host table in fp16 (Decide precedent).** `word_embeddings_fp16.bin` = `fp32.astype(float16)` (384 MB vs 768 MB; max cast error 4.9e-4). Every graph-storage × table-storage pair kept identical spans on 70/75/80 inputs; confidence drift ≤ 1.4e-3 for wfp16 × fp16. ML Drift's rejection of an fp16 in-graph table does not matter when the lookup is on the host.

**Two fp32-exact rewrites in the shipped graph, not a separate NPU file.** Attention-mask fill −1e4 (two sites: encoder and boundary attention) and SafeLayerNorm k=3 (`layer_norm(x·2⁻³, …, eps·2⁻⁶)` on all 29 active LayerNorms) are bit-identical to the stock graph in fp32 (60/60 comparisons, max 0.0), so one graph set serves GPU FP32 and the NPU. Mac Metal at default precision as the fp16 predictor: the stock graph was all-NaN on 70/70, the rewritten graph finite with 68/70 and 72/75 identical span sets — within one input of what the S26 NPU then measured (69/70, 72/75, 77/80).

**NPU (Hexagon HTP, JIT).** All three windows compile on the S26 for this 12-layer / 768-wide encoder (first JIT 5.4 / 45.7 / 167 s; the 24-layer / 1024 DeBERTa of the Open Decision run did not at S512), one DispatchDelegate node, 8.3 / 34.4 / 186.5 ms = about 3× the GPU FP32 at s128. Every span-set difference is a 0.5-threshold near-tie (official confidences 0.502 / 0.503 dropped, one extra span at NPU confidence 0.519); median confidence drift 0.004, max 0.048, no non-finite output → fp16 accumulation noise, not overflow, so a k search is pointless. Rule stays: gate decisions, not finiteness — the NPU is reported, GPU explicit FP32 ships. GPU default precision is the same class (finite, 69/70, drift 0.033).

**Tensor-gate tolerance for prefix sums.** `content_prefix` (cumulative sum, |ref| up to 418) tripped an absolute 1e-3 gate with a 2.5e-6 relative difference (one of 1,360 input × output pairs at s512). Rule adopted: pass when max|diff| ≤ max(1e-3, 1e-4 · max|ref| of that output); spans and confidences remain the primary gate.

**Process traps measured this run.** (1) The Codex workspace-write sandbox blocks Metal device creation (`SIGSEGV` in `ml_drift::metal::CreateGpuInfoFromMetalDevice` before any inference); run Metal predictor gates outside the sandbox. (2) A device driver that releases the shared S26 hold between cells is interrupted by any lane whose keeper holds continuously (we lost about 80 minutes at 1/11 cells); hold across the whole cell list or plan for the gap. (3) Under the S26 secure keyguard an `am start` gate activity needs `setShowWhenLocked(true)` + `setTurnScreenOn(true)` to be top-resumed (Decide finding, verified here). (4) s256 / s512 GPU medians rise 1.5–2.1× within one 75–80-input job (63.5 → 93.0 ms, 200.9 → 422.5 ms, thermal status 0 → 1–2); publish the cool opening value and the sustained median with their conditions.

**Japanese.** The official `word_splitter="char"` is the caller's choice (no detector); the processor appends `.` after `。`, so a 48-slot window holds 47 Japanese characters. Upstream organization recall on ten early Japanese sentences was 4/10 — upstream behaviour, preserved by the conversion (the gate is agreement with the official implementation, not accuracy).

## 2026-10-04 追記 — Kev-0.8B (Qwen3.5 hybrid + pointer head, typed decisions) on the S26 GPU: one graph call per question under explicit FP32, the checkpoint's own tokenizer.json, and a Kotlin host that reproduces CPython's float sum and round

Model repo: `litert-community/Kev-0.8B-LiteRT` (row-prefill graphs L64 / L128 / L256 / L512 / L1024 / L2048, shared-state
pairs Ls128 / Ls256, pointer head, tokenizer); Android sample `kev/`. Evidence:
`~/code/standup/handoffs/assets/2026-10-03-kev-sample-app/` (`ROUND1.md`–`ROUND7.md`; `readme.facts.md`, `r5.facts.md`
and `r7.facts.md` name the source of every number; device reports under `device/`). Reference = the author's code
(`kev.api`, `kev.model`, transformers 5.17.0) on CPU fp32: 402 questions of 377 requests. On a desktop JVM the Kotlin
host equals it on 402/402 rows and readout indices (`usage.input_tokens` 377/377), the head stays within max |Δp|
2.98e-7, and `to_answers` gives the oracle's answers on 402/402. On the S26 (LiteRT 2.2.0, debug build, GPU explicit
FP32) all 181 bundled gate questions get identical row IDs. The 172 L512 rows keep the oracle's argmax on 166/166
rows outside near-ties (max |Δp| 0.0078), L256 gives the same on the same rows, L128 matches 147/147 on the rows of
up to 128 tokens, the 9 L2048 rows match 9/9, and each graph runs whole on the GPU delegate in one partition.

**Graph cut: a state-free row prefill.** The conversion run exports one graph per window: `ids` int32 `[1,L]` +
`valid` float32 `[1,L]` → `hidden` float32 `[1,L,d]` (every position after the final RMSNorm, d = 1024 for 0.8B
and 2560 for 4B, no lm_head). No KV or Gated DeltaNet state goes in or out: each call computes one row, the state
plus one question, from scratch. Positions are the constants 0..L−1, the mask is a constant causal mask plus
(1 − valid) × (−1e4), rows are right-padded with 248044, and the L512 / L1024 / L2048 files share their weights.
Kev never decodes, so the graph needs no cache-update ops, and forms that failed on the GPU before stay out of it:
masked_fill, in-place index_put row updates, the rank-5 repeat_interleave of the 4B and a rank-4 PAD. The price is
one call per question, each recomputing the state. As a check, the fp32 graph (not shipped) matches the author's
fp32 PyTorch on 392/392 argmax with max |Δp| 4.4e-6 at L512, L1024 and L2048 (4B at L1024: 9.8e-6).

**Three patched classes, not the stock exporters.** The conversion run swaps three classes of the transformers
5.14.1 Qwen3.5 text model: `PatchedQwen3_5GatedDeltaNet` (chunk kernel `_rank4_chunk_gated_delta_rule`: the tail PAD
becomes a concat, the diagonal and triangular masks are constants, rank ≤ 4), `PatchedQwen3_5TextRotaryEmbedding`
(1-D RoPE) and `PatchedQwen3_5TextModel` (the mask built from `valid`); attention is `kev_eager` (GQA copies K/V by
concat, additive mask). For the 4B, Gated DeltaNet's 16 key heads and 32 value heads (ratio 2) are copied by concat,
which avoids a rank-5 tensor. An anchor check pins the patch to `modeling_qwen3_5.py` (sha256 0e2cd8dc…). The
rewrite moves real-position hidden states at L1024 by 3.2e-5 (0.8B) and 2.6e-4 (4B) over 393 questions; unpatched
transformers 5.14.1 itself differs from the author's oracle (5.17.0) by 5.1e-5 / 1.2e-4. Not used: litert-torch
0.9.4's stock `qwen3_5` static model, which contains the GPU-hostile forms above, and upstream main, whose chunked
delta rule becomes `tfl.custom` ops (`gated_delta_update` and others) that the delegate does not take. Export is
`litert_torch.convert(graph, sample_kwargs={'ids', 'valid'}).export()` with litert-torch 0.9.4, transformers 5.14.1,
torch 2.13.0, ai-edge-litert 2.2.0 and ai-edge-quantizer 0.9.0; the 4B fp32 export peaks at a physical footprint of
50.4 GB (L1024) and 52.7 GB (L2048).

**Quantization ladder: fp16 fully connected weights + an int8 table.** 0.8B, L1024, 392 questions against the
author's fp32; the 15 near-ties (reference top-2 gap ≤ 0.02) are counted apart. The red arm is a request with one
word changed; a runtime detects it above 0.02.

| Variant | Contents | Bytes | Mac GPU (Metal, FP32 precision) | Mac CPU | Red arm | Verdict |
|---|---|---:|---|---|---|---|
| fp32 | not quantized | 3,024,236,656 | not measured | max 4.4e-6, 392/392 | 0.0476 | check only, not shipped |
| V1 `wi8fc` | fully connected weights + embedding table int8 (channelwise, INTEGER compute = dynamic int8) | 776,130,240 | max 0.0701, mean 5.2e-3, 376/377 outside near-ties, near-ties 11/15 | max 0.1465, mean 1.5e-2, 373/377, 9/15 | GPU 0.0447, CPU 0.0159: not detectable | not shipped (breaks the calibration) |
| V2 `fp16fc_i8emb` | fully connected weights fp16 (float casting, through DEQUANTIZE) + embedding table int8 (per-row scale); activations, conv and the delta rule fp32 | 1,269,023,216 | max 0.0104, mean 9.1e-4, 377/377, 13/15 | the same numbers | 0.0477 | shipped (L512 1,264,068,368, L2048 1,285,227,888) |
| V3 `i8emb` | only the embedding table int8, fully connected weights fp32 | 2,264,171,344 | max 0.0103, 377/377, 13/15 | the same numbers | 0.0477 | control: the 0.0104 comes from the int8 table (not shipped) |
| V4 `fp16fc_fp16emb` | fully connected weights and table fp16 | 1,520,323,136 | rejected by the GPU: `EMBEDDING_LOOKUP: Empty quantization params` (CompiledModel creation fails) | max 0.0010, 377/377, 15/15 | 0.0477 | not shipped (does not run on the GPU) |

V2 ships because it is the smallest file that runs whole on the GPU, keeps the probabilities within the bar (argmax
100% outside near-ties, max |Δp| ≤ 0.02, mean ≤ 0.002) and still detects the red arm. Its two near-tie flips
(`tv4_023`, gap 0.0020; `own_sensor_08`, gap 8.1e-05) are the same in V3, so they come from the int8 table. Only V2
was built for the 4B: 383/383 outside near-ties and 9/9 near-ties on 392 questions at L512 and L1024, 392/392 + 9/9
on 401 at L2048, max 0.0152, mean 4.5e-4, red arm 0.0528 (its most likely option changes from c to a), and 2.8e-5
between the GPU and the CPU on the same file.

**fp16 activations give non-finite rows; the 4B does not fit the S26.** At default GPU precision (fp16 activations)
V2 L1024 gives non-finite read-out rows on 18 of 392 questions on desktop Metal (V1 and V3: the same 18), and the
other 374 still reach max |Δp| 0.0399 with one argmax flip outside near-ties. On the Galaxy S26 GPU at default
precision (V2 L512, a 58-question tap run) 18 questions are non-finite and the other 40 reach max 0.0193. The fix is
FP32 precision requested explicitly (Python `GpuOptions(enforce_f32=True)`, Kotlin
`CompiledModel.GpuOptions(precision = Precision.FP32)`): the S26 GPU then gives the Mac's numbers (max 0.0104, the
same 2 flips) with every node in one LITERT_CL partition, and the host answers no question whose row is non-finite.
The 4B V2 L1024 graph (7,799,218,560 B) did not fit the 12 GB Galaxy S26 (one try): the GPU delegate took all 34,313
nodes in one partition, then lmkd reclaimed memory ("min2x watermark is breached even after kill") and killed the
process before the compile finished. The process's VmHWM reached 5,579,828 kB; the lmkd line shows 1,742,792 kB
resident and 6,996,624 kB swapped, with MemAvailable 6.6 GB just before. lmkd reclaimed 68 other processes in the
same window. On the Mac the 4B needs desktop-class memory: 20.1–21.1 GB after compiling on the GPU at FP32
precision (peak 37.1–38.8 GB) and 14.9–16.7 GB on the CPU. Evidence (conversion run): `results/litert_{gpu_f32,cpu}_rows_L1024_v*.json`,
`results/quant_recipes.json`, `results/litert_gpu_f16_rows_L1024_v*.json`, `results/timing_mac_r7_4b.json`,
`device/r6/kev_s26_r6_E_4b.kill_lines.txt`.

**The tokenizer is the checkpoint's own `tokenizer.json`, not the base model's.** The author's oracle tokenizes with
transformers 5.17.0 `Qwen2Tokenizer`, which builds its pre-tokenizer in code: the Qwen2 split regex without `\p{M}`,
ByteLevel without a prefix space, and 33 added tokens (the 22 of the base `tokenizer.json` plus 11 that only
`tokenizer_config.json` lists, `<think>` and `<tool_response>` among them). The `tokenizer.json` published with the
Kev checkpoint (19,989,325 B) is that pipeline saved. The Qwen3.5-0.8B-Base `tokenizer.json` has the same vocabulary
and merges but a regex with `\p{M}` and 22 added tokens; raw `tokenizers` on it differs from the oracle on 4 of 12
probe strings (Devanagari, `<think>`, `<tool_response>`, `<tts_pad>`). The fixture's 2,038 user strings give
identical IDs with either file, so a 402/402 row parity cannot tell them apart; only the probes do. The Kotlin host
reads the Kev file as it is (regex, added tokens, vocabulary, merges) and matches `AutoTokenizer` on 54/54 probes on
the JVM and on the S26. Evidence: `ROUND1.md` §1, `r1_evidence/probe_three.py`, `r1_evidence/probe_added.py`.

**The split regex follows different Unicode rules in onig, java.util.regex and ICU.** onig's case-insensitive
contractions fold `ſ` (U+017F): `'ſtuff` splits as `'ſ` + `tuff`. Java's `(?i)` folds ASCII only, so the port spells
the class out (`'[sSſ]`, written `ſ` in the source). `\s` becomes onig's White_Space class (25 code points),
because Java's `\s` is ASCII-only and Android rejects `UNICODE_CHARACTER_CLASS`. Java 17's regex classes follow
Unicode 13 and onig (tokenizers 0.23.2) Unicode 16: 9,787 letters and 130 digits exist only in onig's `\p{L}` /
`\p{N}`. On the JVM 6,000 random strings still gave identical IDs, 2,254 of them with code points Java 17 does not
define; on the S26 the 54 probes pass under Android's ICU. Evidence: `ROUND1.md` §2, `r1_evidence/probe_fold.py`,
`r1_evidence/Fold.java`, `r1_evidence/onig_classes.json`, `r1_evidence/ClassDiff.java`.

**The response is bit-identical only with CPython's float semantics.** `to_answers` normalizes and takes
expectations with Python 3.12's built-in `sum`, which adds floats with Neumaier compensation, and rounds with
`round(x, 4)`, which rounds the exact binary value half to even: `round(0.12345, 4)` and `round(0.12355, 4)` are
both 0.1235, and `round(0.03125, 4)` is 0.0312. The port sums in the same order with the same compensation
(`KevAnswers.pythonSum`) and rounds with `BigDecimal(x).setScale(4, HALF_EVEN)`; with a plain loop the score and the
confidences do not match bit for bit. A JSON state needs Python's semantics as well: keys keep the file's order (the
JVM `org.json` artifact uses a HashMap), and `1` and `1.0` render differently, so the reader keeps number literals.
Evidence: `KevAnswersTest`, `kev/app/src/test/resources/python_numbers.json`, `ROUND1.md` §1.

**Time to the read-back, and pick the smallest window.** `CompiledModel.run()` returns before the GPU finishes, so
the app times each question from the input writes through `run()` to `readFloat()` of `hidden`; every ms on the
cards and in the README is that span. The graph computes every position of its window, so the time follows the
window, not the row. From thermal status 0, three rows of 73–97 tokens take a median 175.8 ms at L128 and 323.3 ms
at L256, a 300-token row 615.1 ms at L512, and a 1,000-token row 1,333.4 ms at L1024; the five-question rows of
128–142 tokens took 617–630 ms at L512 and 317–325 ms at L256 at the start of their legs. The sample therefore asks,
per question, for the smallest installed window that holds its row. The bundled ticket (rows of 131, 101 and 93
tokens) runs on L256 in 1,046 ms with the default install, against 1,967 ms on L512 before. Evidence:
`device/r3_timing_gpu_T300.json`, `device/r3_timing_gpu_L512.json`, `device/r3b_timing_gpu_T1000.json`,
`device/r5_timing_gpu_L128_S80.json`, `device/r5_timing_gpu_L256_S80.json`, `device/r5_timing_gpu_L256_fiveq.json`,
`device/r5_autoplay_ticket_kev-demo-1791058996833.json`.

**Two graphs in memory only for the small windows.** Running short questions on L128 and longer ones on L256 in the
same request means two compiled graphs, and three runs on the 12 GB S26 measured what that costs. Compiling L256
next to the resident L128 left the app running twice: MemAvailable fell from 4,762,332 to 883,380 kB in one run
(the ticket took 756 ms) and from 5,025,996 to 638,280 kB in the other. Compiling L2048 next to L128 took it from
4,770,840 kB to 623,824 kB, and Android's low-memory killer stopped the app. The sample therefore compiles a second
graph only for L128 + L256 and only with at least 4,500,000 kB available (`ActivityManager.MemoryInfo.availMem`),
and runs L512 and larger windows alone, after closing the others. On the phone, a long request after the ticket
closed L256 and compiled L2048 alone (low point 2,217,164 kB, no kill), and the next short request compiled L256
again; its answers started about 12 s after the request. Evidence: `device/r5_autoplay_ticket_l128.mem.txt`,
`device/r5_autoplay_long.logcat_all_grep.txt`, `device/r5_autoplay_long.mem.txt`, `device/s30_mem.txt`,
`device/s30_smoke.log`.

**FP16_WITH_FP32_ACCUM on the final graphs (2026-10-05).** The published graphs are the conversion run's final
kernel: the Gated DeltaNet chunk goes through an inverse instead of a step-by-step loop, and softplus and exp are
written so that float16 storage stays finite; L64, L128 and L256 also carry a size-1 SUM after two projections for
the NPU, which leaves the GPU outputs bit-identical (the app's gates give the same probabilities on every row).
At the default GPU precision (float16 activations) these graphs stay finite but miss the bar on desktop Metal
(L128: max |Δp| 0.0332, mean 3.59e-3; conversion run), so the precision stays explicit. At
`CompiledModel.GpuOptions(precision = Precision.FP16_WITH_FP32_ACCUM)` (float16 storage, float32 accumulation)
every graph passes the app's gate on the S26: L64 34 rows max |Δp| 0.00656, L128 147 rows 0.00736, L256 172 rows
0.00917, L512 172 rows 0.00907, L1024 40 rows 0.00804, the Ls128 pair 132 questions 0.0101, the Ls256 pair 146
questions 0.0110, and the 9 L2048 rows 0.0093 with mean 1.84e-3 (0.0015 and 4.3e-4 at FP32: the smallest margin,
so the README gives it and the FP32 switch). On the same file it takes about a quarter less time than FP32: L128
102.0 against 137.8 ms, L256 196.0 against 272.1 ms. The precision is a per-graph default in the app: a file of a
size the repository published before the rewrite runs at FP32, and `--es precision fp32` forces FP32 everywhere.
Evidence: `ROUND7.md`, `r7.facts.md`, `device/r7_G1_gate_L128_C7_fp16acc.json`,
`device/r7_G4_gate_pair256_fp16acc_noshare.json`, `device/r7_T1_timing_L128_fp16acc.json`,
`device/r7_F3_timing_L128_fp32.json`, `device/r7_T2_timing_L256_fp16acc.json`, `device/r7_F4_timing_L256_fp32.json`.

**A shared-state pair: two signatures in one file, the state handed over as buffers (2026-10-05).** Every row of a
request starts with the same `[state]` + state tokens, so the conversion run also exports a pair per state length.
`state_prefill_<Ls>` takes `ids` / `valid` `[1,Ls]` and returns the 48 state tensors: `k_<l>` / `v_<l>`
`[1,2,Ls,256]` of the 6 attention layers, `gdn_state_<l>` `[1,16,128,128]` and `conv_tail_<l>` `[1,3,6144]` of
the 18 Gated DeltaNet layers. `question_step_<Ls>_64` takes one question's `ids` / `valid` `[1,64]`, the state
call's `valid` as `state_valid` `[1,Ls]` and the 48 tensors, and returns `hidden` `[1,64,1024]` with positions
continuing after the state. One `CompiledModel` holds both signatures. The app creates the state call's output
buffers once and puts the same `TensorBuffer` objects into the question call's input map, so the state stays on
the GPU and nothing is copied through the host. LiteRT has no call that lists a signature's tensors, so the names
come from the conversion run's contract and the compile checks each one's type and shape. On the JVM the pair's
path gives the row path's answers on all 312 questions (304 requests) that fit the Ls128 pair. On the S26 the
five-question request takes 429.6 ms on the pair against 972.1–984.6 ms as five rows on L256, and a
three-question email with a 167-token state 409.1 ms on the Ls256 pair against 585.0–593.8 ms. Evidence:
`kev/app/src/main/java/com/kev/KevPairDecider.kt`, `KevPipelineTest`, `device/r7_F1_timing_pair128_fiveq_noshare.json`,
`device/r7_T6_timing_L256_fiveq_fp16acc.json`, `device/r7_F2_timing_pair256_email3_noshare.json`,
`device/r7_T8_timing_L256_email3_fp16acc.json`.

**`constantTensorSharing` trades memory for time; the answers stay the same.** With
`GpuOptions(constantTensorSharing = true)` the two signatures share one copy of the weights on the GPU; without it
each holds its own. On the S26 the unshared pair is quicker: the five-question request 429.6 against 623.3 ms
(state 116.0 against 152.3 ms, step 62.0 against 92.9 ms), the Ls256 email 409.1 against 544.6 ms. It also needs
more memory: VmHWM 6.15–6.81 GB against 3.01–3.05 GB, MemAvailable low points 2.21–3.04 GB against 5.29–5.88 GB,
kgsl page allocation 3.1–3.2 GB against 1.7–1.8 GB. The probabilities are identical: the 132-question gate gives
the same numbers both ways. The app decides when it compiles the pair: unshared when Android reports at least
6,500,000 kB available right before the compile, shared below that, and a compiled pair keeps its mode. Evidence:
`device/r7_T5_timing_pair128_fiveq_fp16acc.json`, `device/r7_F1_timing_pair128_fiveq_noshare.json`,
`device/r7_G3_gate_pair128_fp16acc.json`, `device/r7_G6_gate_pair128_fp16acc_noshare.json`.

**The plan predicts both forms from measured times.** `KevCosts` holds this app's S26 medians: one question on each
window, and the state call and one step of each pair, shared and unshared. `KevPlanner` sums them over the request
(rows: the windows `KevResidentGraphs.assign` would use; pair: the state call plus one step per question) and runs
the smaller, the rows on a tie. The prediction follows what the phone holds at that moment: two windows only when
they could stay compiled (at least 4,500,000 kB available, or both already compiled), an unshared pair only when
its compile would see at least 6,500,000 kB, and a compiled pair as it was compiled. The bundled ticket was
predicted at 302 ms on the pair against about 400 ms on L256 + L128 and took 367 ms on the pair; with `--es graph
rows` it took 463 ms. A single question whose row fits L128 stays on L128 (102.2 ms against 178 ms for the pair),
while one whose row needs L256 goes to an unshared pair (178 ms against 196.3 ms). Evidence: `KevPlannerTest`,
`device/r7_A1_autoplay_ticket_auto_kev-demo-1791138907893.json`,
`device/r7_A2_autoplay_ticket_rows_kev-demo-1791139223945.json`.

**Split the timings by the GPU clock ceiling, call by call.** On the S26 the GPU clock ceiling
(`/sys/class/kgsl/kgsl-3d0/max_clock_mhz`) fell from 1,300 MHz 3–8 s into back-to-back calls, in most legs to
578–902 MHz, while the thermal status stayed at 0 or 1, and calls then took 1.7–2.1 times as long (L256 196.3 →
330.2 ms; the five-question request on the pair 429.6 → 727.6 ms). A median over a whole leg mixes the two speeds,
and its value depends on the leg's length. The timing runner therefore reads the ceiling before each call, outside
the timed span, and records it next to the call's time; the representative value is the median of the calls made
at 1,300 MHz, and the others are given apart. In the conversion run a compile alone warmed the GPU (41.1 → 59.7 °C
during a 7.6 s compile, the ceiling at 1,050–1,100 MHz at the opening call), so the runner also waits after the
compile and before each set until the ceiling and the temperature are back to their values before the compile
(`--ei cool_ms`); in this app's legs that wait took 0 s. Evidence: `kev/app/src/main/java/com/kev/KevTimingCore.kt`,
`KevCool.kt`, `device/r7_chain1_table.md`, `device/r7_chain3_table.md`, `kev_work/ROUND14.md`.

**A 360 dp screen sets the demo layout.** The S26 reports density 3.0, a 360 × 780 dp screen; a layout planned at
density 2.625 cut the ticket's last line even at its 16 sp floor. Compose Material 1 `Text` inherits body1's 24 sp
line height, which cut a three-line footer until each line height was explicit. The presentation now measures every
fitted text with `rememberTextMeasurer` inside the layout pass and draws it once, at the size that fits, instead of
shrinking it frame by frame. The editing screen's footer wrapped inside "GPU FP32" at 360 dp; it now puts one group
per line and joins the words of an item with non-breaking spaces. Evidence: `ROUND3.md` §1–2,
`device/s4_autoplay_done.png`, `device/u10_answers.png`.

**Heat decides which timing legs count.** About 2.5 minutes of GPU FP32 at L512 (the 172-row gate) took the S26
from thermal status 0 to 3 (skin 37.9 → 45.0 °C, GPU clock ceiling down to 726 MHz). A launch that timed every set
back to back met the protocol only for its opening set (five-question calls drifted from 618 to 775 ms), so each set
now gets a launch of its own (`--es sets`, `--ez request_path`). Two cooling traps. When the app is not in the
foreground the screen dozes, and a watcher that sends `KEYCODE_WAKEUP` every 17 s starts face unlock; that held the
phone at status 1 for 11 minutes. Cool with the screen off and the watcher idle, then wake the screen 5 s before the
leg. And `am force-stop` right after a leg briefly caps the prime cluster during the keyguard transition, so read
the caps when the next leg starts. Thermal status 0 does not hold the GPU clock either: in a five-question leg at
L256 that started and ended at status 0, the GPU clock ceiling fell from 1,300 to 646 MHz, and the requests went
from 1,596–1,626 ms to 2,390–2,412 ms. Record the ceiling (`/sys/class/kgsl/kgsl-3d0/max_clock_mhz`) before and
after each leg. Evidence: `ROUND3.md` §1–3, `device/wake_keeper.log`, `device/legs_status.txt`,
`device/r3b_chain.sh`, `device/r5_timing_gpu_L256_fiveq.state_after.txt`.

**The NPU backend: NPU + CPU, BURST on every graph, a compile cache keyed by the file's content (2026-10-05).**
The row graphs L64, L128 and L256 run on the S26's Qualcomm HTP (V81) through LiteRT 2.2.0's JIT with
`CompiledModel.Options(Accelerator.NPU, Accelerator.CPU)`. The Qualcomm compiler takes every op but the int8
embedding lookup (`LiteRT Op #4 'EmbeddingLookup' (code=7) is not supported in Qualcomm Compiler`; L64 3,911 of
3,912 ops, L128 4,758 of 4,759, L256 6,018 of 6,019), the DispatchDelegate replaces 2 of the 3 nodes, and the lookup
runs on the CPU; with the NPU alone the compiled model is not created (`LiteRtCreateCompiledModel failed: 504`, the
conversion run's round 11). The app sets `QualcommOptions(htpPerformanceMode = BURST)` on every graph of a process
whose APK carries the libraries, the GPU and CPU graphs too; logcat shows `HtpPerformanceMode : Burst(2)` for each
NPU graph also after GPU graphs compiled in the same process. `Environment.create(context, …)` makes
`context.cacheDir` LiteRT's compile cache. LiteRT 2.2.0 keys an entry by the graph file's name, a hash of the whole
file's content and a hash of the compiler plugin, the build fingerprint, the accelerator set and the API version
(`litert/core/cache/compilation_cache.cc`), and the Qualcomm options set through the Kotlin API change that last
hash too: on the S26 the same L64 content got four entries (BURST; BURST and O3; BURST and PREPARE; no Qualcomm
options). One entry per file content is kept (other content directories of the same name are removed after a save),
and a load touches the file. On the S26 each entry is 1.27–1.28 GB and loads in 0.8–2.5 s (`Flatbuffer model
initialized from cached model.`); a file with one byte changed under the same name, size and modification time was
compiled again (63.4 s) and the old entry removed. The app keeps its own marks (file name, size, modification
time, LiteRT version, build fingerprint, optimization level) to tell its status line whether a compile takes
minutes, and deletes the old entry when the optimization level changes. A graph that is not cached is compiled
alone, with the other graphs closed, because of its memory: 81.6 s at L64, 179.0 s at L128 and 298.2 s at L256,
MemAvailable down to 2.08 / 1.85 / 1.21 GB, VmHWM 4.86–5.66 GB, and Android's low-memory killer reclaimed 16–54
background processes of other apps per compile. The two-signature pair (Ls128 + Lq64) does not compile for the NPU
on the 12 GB S26: lmkd stopped the app in its JIT compile twice, about 5 minutes in (2,374,380 kB resident + 4,914,876
kB swapped; 1,220,640 + 5,249,876 kB), after MemAvailable had held 2.69–2.79 GB for two minutes and fell to 0.60 /
1.22 GB; no cache file was written, so the app keeps the pairs on the GPU. From the cache, one question takes 45.1
ms at L64, 65.8 ms at L128 and 121.9 ms at L256 (the median of all 60 calls; the CPU frequency caps that set in
during a leg do not slow the NPU call), next to 56.6 / 102.2 / 196.3 ms on the GPU at `FP16_WITH_FP32_ACCUM`, and the
bundled ticket 298 ms (367 ms on the GPU pair). The gates pass at max |Δp| 0.0105 / 0.0134 / 0.0134. LiteRT's default
optimization level for the plugin is already HTP_OPTIMIZE_FOR_INFERENCE_O3 (`OptimizationLevel :
HtpOptimizeForInferenceO3(2)` with no level set, QNN `optimization_level=3`); HTP_OPTIMIZE_FOR_PREPARE compiles L64
in 19.1 s instead of 64–82 s but takes 122.3 ms per question against 45.9 ms, with the same probabilities; with no
performance mode L64 takes 157.4 ms (122.5–240.8) against 47.6 ms with BURST. Check every
NPU number against logcat: `Partitioned subgraph<0>, selected N ops, from a total of M ops. resulted in 2
partitions.` for a JIT compile, `Flatbuffer model initialized from cached model.` for a cache load, and `Replacing 2
out of 3 node(s) with delegate (DispatchDelegate)` for the NPU; `with delegate (TfLiteXNNPackDelegate)` or `Failed to
apply compiler plugins` mean the CPU, without an error. Evidence: `ROUND8.md`, `r8.facts.md`,
`device/r8_N1_gate_L64_npu.json`, `device/r8_N2_gate_L128_npu.json`, `device/r8_N3_gate_L256_npu.json`,
`device/r8_T1_timing_L64_npu.json`, `device/r8_T2_timing_L128_npu.json`, `device/r8_T3_timing_L256_npu.json`,
`device/r8_M1.l64_modify.txt`, `device/r8_P1_gate_pair128_npu.logcat_all_grep.txt`,
`device/r8_P1_gate_pair128_npu_again.logcat_all_grep.txt`, `device/r8_U1b_select_npu_first.npu_evidence.txt`.

**Process traps.** (1) The repository's root `.gitignore` has `*.bin`, which silently dropped `head_fixture.bin`
from `git add kev/`; the fixture is `.f32` now. Check new data with `git status --short --ignored <dir>`, and give
`git check-ignore -v` the path from the repository root. (2) Lint `NewApi` caught `BigInteger.intValueExact()` (API
31) in the JSON reader, which would throw `NoSuchMethodError` on API 26–30; the JVM tests pass because the desktop
JVM has the method, so run `:app:lintDebug` after host changes. (3) Gradle reports the test task of a docs-only
change as UP-TO-DATE and prints no results; evidence logs need `--rerun` or `--rerun-tasks`. (4) Samsung's lmkd
writes a kill as `Reclaim '<package>' (<pid>) … oom_score_adj …`, followed by ActivityManager's `Process <package>
(pid N) has died`; a check that looks only for `lowmemorykiller` or `Kill '<package>'` misses it and reports a
timeout instead. Evidence: `ROUND1.md` §2, `ROUND2.md` §1–2, `ROUND5.md`,
`device/r5_autoplay_long.logcat_all_grep.txt`.
