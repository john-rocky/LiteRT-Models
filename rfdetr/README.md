# RF-DETR Nano — Object Detection on-device (LiteRT GPU, fully GPU)

[RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow 2025, an LW-DETR derivative) object detection
running **fully on the LiteRT CompiledModel GPU** (ML Drift). RF-DETR is a transformer detector
(DINOv2 backbone + deformable-attention DETR decoder) — a model family that previously did **not** ride
the GPU API cleanly (deformable `grid_sample` → `GATHER_ND`, windowed attention → 5D/6D tensors,
two-stage query selection → `TOPK`/`GATHER`). Here every one of those is re-authored or split out, so
the whole detector runs on the GPU with no CPU/ONNX fallback. The demo runs **live camera** detection
(CameraX) and overlays the boxes + COCO labels on each frame.

## On-device (Pixel 8a, Tensor G3 — verified)

| graph | nodes on GPU |
|---|---|
| **Graph A** — backbone + encoder + proposal heads | `1381/1381` LITERT_CL |
| **Graph B** — two-stage combine + decoder + heads | `404/404` LITERT_CL |

Both graphs fully GPU-resident; the **live camera runs at ~9 fps (~110 ms/frame end-to-end)** — a
transformer detector entirely on the GPU. On a real image the device chain reproduces the PyTorch
detections at **IoU 0.98–0.99, same class, matching scores**.

## How it splits (and why it's fully GPU)

RF-DETR is a two-stage DETR. The query selection (top-300 proposals by class score) is `TOPK_V2` +
`GATHER`, which have no GPU op — but the proposal **grid is image-independent**, so the model splits at
exactly that point into two GPU graphs with a tiny host step between them (the standard two-stage-DETR
edge split):

```
image[1,3,384,384]
  →[GPU Graph A]→ enc_class[1,576,91], enc_coord[1,576,4], memory[1,576,256]
  →[CPU: top-300 by max class score → gather coords]→ refpoint_ts[1,300,4]
  →[GPU Graph B  (memory, refpoint_ts)]→ boxes[1,300,4] (cxcywh), logits[1,300,91]
  →[CPU: sigmoid + threshold + cxcywh→xyxy + per-class NMS]→ detections
```

`memory_ts`/`boxes_ts` (the enc auxiliary outputs) are **dead at inference** — the decoder query is a
learned embedding and the top-k only feeds the reference points — so the host step is just a topk +
coord-gather (no feature gather).

### GPU re-authoring (per-graph tflite-vs-torch corr **1.0**)

The model is converted with **litert-torch** (NCHW preserved — `onnx2tf` destroys ViT attention). Fixes:

1. **Windowed DINOv2 backbone** — 6D window-partition → a 5-step ≤4D reshape/permute (+ exact inverse);
   SDPA → manual 4D attention; `interpolate_pos_encoding` baked; cls-token `repeat`→`cat`; tanh-GELU.
   Only **3 of 12 layers use global attention** (the rest are windowed), which is what lets the
   backbone survive the Mali fp16 path (corr 0.9998).
2. **Deformable `grid_sample`** → a **GATHER/CAST-free tent-matmul**: bilinear weights `relu(1-|i-p|)` ×
   BMM against the value map — numerically exact incl. zeros-padding OOB, all ≤4D.
3. **MSDeformAttn** re-authored ≤4D (no 6D sampling tensors); **sine pos-embed** `dim_t` baked + strided
   interleave → reshape (kills `POW`/`GATHER_ND`).
4. **Two-stage topk/gather** → host (the split above).

### ⭐ fp16 hardening (the on-device gate — not visible on desktop)

The Mali delegate computes in fp16 regardless of model dtype, and two LayerNorm sites overflow it:

- **Projector** (fuses 4 backbone maps, `ConvX` outputs reach |x|≈440) → channel sum-of-squares
  `256·440²` ≫ 65504 → device collapsed (memory corr 1.0 → **0.58**).
- **Decoder** layer-0 self-attention output reaches |x|≈1068 → the residual into `norm1`/`norm3`
  overflows likewise.

Both are fixed with a **down-scaled, scale-invariant SafeLayerNorm** (reduce in an `x/S` domain, rescale
back — numerically exact). The decoder version is **adaptive** (`S = max(1, amax/8)` per row): a *fixed*
large `S` squashes the small-magnitude norms (logits corr 0.88 → 0.32), so the scale must follow the
input magnitude. → memory corr **0.9999**, and real detections match PyTorch.

## Files

| File | Description |
|------|-------------|
| `RfDetr.kt` | Both GPU graphs on CompiledModel + host topk/gather + decode + per-class NMS |
| `MainActivity.kt` | Runs detection on a bundled image / gallery pick, overlays boxes + COCO labels |
| `app/src/main/assets/coco_labels.txt` | 91-line COCO label table (index = COCO category id) |

## Setup

1. Build the two tflites with `scripts/build_rfdetr_split.py` (needs `pip install rfdetr` + litert-torch),
   or download from Hugging Face — [litert-community/RF-DETR-Nano-LiteRT](https://huggingface.co/litert-community/RF-DETR-Nano-LiteRT).
2. Build/install the app, then push the models into its private storage:
   ```bash
   ./scripts/install_to_device.sh scripts/output
   ```
   (`test_image.jpg` + `coco_labels.txt` are bundled.)
3. Launch **RF-DETR** — it compiles the GPU shaders (~1 s/graph first launch), then detects objects.

## Conversion

`scripts/build_rfdetr_split.py` (imports `build_rfdetr_full.py` → `build_rfdetr_bb.py`, which apply all
the backbone/decoder patches) builds **Graph A** and **Graph B**, op-checks both (GPU-clean: no banned
ops, no >4D tensors), validates per-graph corr 1.0 vs PyTorch, and writes the fp16 tflites + device-probe
fixtures. `scripts/run_probe.sh` drives an on-device GPU residency/accuracy probe.

**Original project**: [roboflow/rf-detr](https://github.com/roboflow/rf-detr) (RF-DETR Nano) |
[Apache-2.0](https://github.com/roboflow/rf-detr/blob/main/LICENSE)
