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
