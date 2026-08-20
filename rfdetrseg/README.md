# RF-DETR-Seg Nano — Instance Segmentation on-device (LiteRT GPU, fully GPU)

[RF-DETR-Seg](https://github.com/roboflow/rf-detr) (Roboflow, rf-detr 1.9.3, Apache-2.0) instance
segmentation running **fully on the LiteRT CompiledModel GPU** (ML Drift). RF-DETR-Seg Nano is a
two-stage DETR — DINOv2-S/12 backbone (312×312, all-global attention), deformable-attention decoder,
and a ConvNeXt-style mask head producing a full-image 78×78 mask per query. The demo runs **live
camera** segmentation (CameraX) and overlays per-instance masks + boxes + COCO labels on each frame.

## On-device (Pixel 8a, Tensor G3 — verified)

| graph | nodes on GPU | time |
|---|---|---|
| **Graph A** — backbone + encoder + proposal heads | `1293/1293` LITERT_CL (1 partition) | 17.5 ms |
| **Graph B** — decoder + box/class heads + mask head | `884/884` LITERT_CL (1 partition) | 9.1 ms |

Real-image end-to-end (device chain vs the official PyTorch `RFDETRSegNano.predict`, threshold 0.5):
all detections match with **box IoU ≥ 0.99, mask IoU ≥ 0.995, classes identical** on the test images.

## How it splits (and why it's fully GPU)

The two-stage query selection (top-100 proposals by class score) is `TOPK_V2` + `GATHER`, which have
no GPU op — so the model splits at exactly that point into two GPU graphs with a tiny host step:

```
image[1,3,312,312] + clspos[1,1,384] + pospatch[1,676,384]
  →[GPU Graph A]→ enc_class[1,676,91], enc_delta[1,676,4], memory×2[1,676,256]
  →[CPU: ÷2 → proposal-grid combine → top-100 → gather → reparam(refpoint_embed)]→ refpoint[1,100,4]
  →[GPU Graph B  (memory, refpoint, query_feat[1,100,256])]→ boxes[1,100,4], logits[1,100,91], masks[1,100,78,78]
  →[CPU: sigmoid + threshold + per-class NMS]→ instances (mask inside = logit > 0)
```

The proposal grid is image-independent (26×26, wh = 0.05), so the host step is pure elementwise math
plus a topk — exact off-GPU.

### ⭐ The baked-constant execution bug (why three tensors are host-fed inputs)

**ML Drift silently mis-executes compute chains that consume large baked-constant tensors — the same
graph with the constant fed as a runtime input is exact.** This is not a precision issue: fp32 and
fp16 flatbuffers return identical wrong numbers, and it is compilation-context dependent (a subgraph
can probe clean in isolation and still break in the full build). Minimal pair: `tgt + FFN(tgt)` with
`tgt` a baked `[1,100,256]` weight → device corr 0.966; identical ops with `tgt` as a graph input →
0.99999. Three instances in this one model, all fixed exactly:

1. **DINOv2 LayerScale** `h + λ·f(h)` → corr 0.62; fix = fold λ into the preceding Linear (exact).
2. **Decoder query embedding** (baked `query_feat.weight`) → hs corr 0.97; fix = feed `query_feat`
   as a graph input and move the reparam combine (baked `refpoint_embed`) to the host.
3. **ViT pos-embed add** `patches + POS[1,677,384]` → backbone corr 0.55–0.99 varying per build;
   fix = host-feed `cls_token+pos` (`clspos`) and the patch pos-embed (`pospatch`) as inputs.

The host-fed constants ship as raw-float32 assets (`clspos.bin`, `pospatch.bin`, `query_feat.bin`,
`refpoint_embed.bin`, ~1.1 MB total), written into the input buffers once at load.

### Other GPU re-authoring (per-graph tflite-vs-torch corr ≥ 0.998)

Converted with **litert-torch** (NCHW preserved — `onnx2tf` destroys ViT attention):

- **SDPA / nn.MultiheadAttention → manual rank-4 attention** (ML Drift mis-executes rank-3 batched
  matmuls), backbone and decoder both.
- **Deformable `grid_sample` → GATHER/CAST-free tent-matmul** (bilinear weights `relu(1-|i-p|)` ×
  rank-4 BMM — numerically exact incl. zeros padding).
- **SafeLayerNorm v2** — adaptive per-row down-scale that never reconstructs the large variance
  (`y = d/√var` after `x/S`), fp16-safe at any magnitude; 4D (channels-first) sites drop to a
  `[B,HW,C]` 3D detour because litert-torch's NHWC layout pass cannot rewrite `amax` on 4D tensors.
- **tanh-GELU** (ERF has no GPU lowering); **sine pos-embed** `dim_t` baked, interleave via reshape.
- **memory ×2 output trick** — a `[1,N,C]` tensor that is both consumed and a graph output comes
  back zeroed on Mali; ×2 forces a separate buffer (exact in fp16), the host halves it.
- **seg einsum → rank-4 matmul**; mask-head DepthwiseConvBlock LN via the same 3D detour.

## Files

| File | Description |
|------|-------------|
| `RfDetrSeg.kt` | Both GPU graphs on one shared Environment + host topk/reparam + decode + NMS |
| `MainActivity.kt` | Live camera, per-instance mask overlay + boxes + COCO labels |
| `app/src/main/assets/*.bin` | Host-fed constants (clspos / pospatch / query_feat / refpoint_embed) |
| `app/src/main/assets/coco_labels.txt` | 91-line COCO label table (index = COCO category id) |

## Setup

1. Build the two tflites with `scripts/build_rfdetrseg_split.py` (needs `pip install rfdetr==1.9.3`
   + litert-torch), or download from Hugging Face —
   [litert-community/RF-DETR-Seg-Nano-LiteRT](https://huggingface.co/litert-community/RF-DETR-Seg-Nano-LiteRT).
2. Build/install the app, then push the models into its private storage:
   ```bash
   ./scripts/install_to_device.sh <dir-with-the-tflites>
   ```
3. Launch **RF-DETR-Seg** — it compiles the GPU shaders (~1 s/graph first launch), then segments live.

## Conversion

`scripts/build_rfdetrseg_split.py` applies all patches, builds **Graph A** and **Graph B**, op-checks
both (no banned ops, no >4D tensors), validates split-vs-torch and E2E corr, and writes the fp16
tflites + the host-constant `.npy`s + device-probe fixtures. `scripts/verify_real.py` runs the
real-image ship criterion (device chain vs official PyTorch predict: box/mask IoU + class match).
`scripts/tfm_compat.py` shims transformers 4.57↔5.x (no-op on ≥5.1).

**Original project**: [roboflow/rf-detr](https://github.com/roboflow/rf-detr) (RF-DETR-Seg Nano,
tag 1.9.3) | [Apache-2.0](https://github.com/roboflow/rf-detr/blob/main/LICENSE)
