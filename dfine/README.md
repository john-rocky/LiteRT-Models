# D-FINE-S — Object Detection on-device (LiteRT GPU, fully GPU)

[D-FINE](https://github.com/Peterande/D-FINE) (USTC, 2024 — `ustc-community/dfine-small-coco`), the SOTA
real-time DETR, running **fully on the LiteRT CompiledModel GPU** (ML Drift). D-FINE is a transformer
detector — HGNetV2 backbone + a hybrid AIFI/CCFM encoder + an **FDR** (Fine-grained Distribution
Refinement) decoder. Both transformer graphs run on the GPU with no CPU/ONNX fallback; only the topk
selection and a tiny per-token tail run on the CPU. This is a **still-image** demo (bundled image +
gallery pick).

## On-device (Pixel 8a, Tensor G3 — verified)

| graph | nodes on GPU |
|---|---|
| **Graph A** — HGNetV2 backbone + hybrid encoder + score head | fully `LITERT_CL` |
| **Graph B** — two-stage combine + FDR decoder + heads | fully `LITERT_CL` |

The device chain reproduces the PyTorch detections **exactly** — on a COCO val image (giraffe + cows)
**5/5** detections, every box at **IoU 0.99–1.00** with matching class and score.

Still-image (not real-time): D-FINE's deformable decoder over the full 8400 tokens / 80×80 feature levels
is GPU-compute-bound on this device (the GATHER-free tent-matmul `grid_sample` turns an O(points) gather
into an O(H·W) matmul). For a real-time camera DETR see [RF-DETR](../rfdetr) — its single 24×24 deformable
level runs at ~9 fps.

## How it splits (and why it's fully GPU)

D-FINE is a two-stage DETR. The query selection (top-300 proposals by class score) is `TOPK_V2` +
`GATHER`, which have no GPU op — but the proposal **grid is image-independent**, so the model splits there:

```
image[1,3,640,640]
  →[GPU Graph A]→ enc_class[1,8400,80], memory_raw[1,8400,256]
  →[CPU: top-300 by max class score; per-token tail on the 300 selected:
         target = enc_output(valid·memory_raw)   (Linear + LayerNorm)
         ref    = enc_bbox_head(target) + anchors (3-layer MLP)]
  →[GPU Graph B  (memory_raw, target, ref)]→ boxes[1,300,4] (cxcywh), logits[1,300,80]
  →[CPU: sigmoid + threshold + cxcywh→xyxy + light NMS]→ detections
```

### GPU re-authoring (per-graph tflite-vs-torch corr **1.0**)

Converted with **litert-torch** (NCHW preserved — `onnx2tf` destroys ViT attention). Fixes:

1. **Deformable `grid_sample`** → a **GATHER/CAST-free tent-matmul** (bilinear weights `relu(1-|i-p|)` ×
   BMM against the value map; exact incl. zeros-padding OOB, all ≤4D); multi-level **MSDeformAttn** ≤4D.
2. **D-FINE FDR head**: the LQE `prob.topk` → an iterative max-and-mask (exact, GPU-clean; done in 3D
   since a 4D last-dim `amax` breaks the NHWC lowering); `distance2bbox`'s `stack` → `cat` (avoids `PACK`).
3. **AIFI** 2-D sine pos-embed baked (`temperature**ω` → illegal `POW`); **SafeLayerNorm**; tanh-GELU;
   `inverse_sigmoid`'s redundant `clamp(0,1)` dropped (→ Mali-rejected `RELU_0_TO_1`); the baked anchors'
   `finfo.max` invalid-border values clamped to ±1e4 (else fp16 `Inf`).

### ⭐ The on-device gate — a Mali 3D-sequence fan-out bug (NOT the FDR decoder)

Both graphs convert GPU-clean and are fully `LITERT_CL`-resident, but a naïve Graph A (emitting
`enc_class`/`enc_coord`/`output_memory`/`memory_raw` all at once) produced **0 detections** on device, and
it first looked like the FDR decoder collapsing in fp16. It was a **red herring.** The real cause is a Mali
delegate bug where a **3-D *token* tensor `[1,N,256]`** (from `conv.flatten(2).transpose(1,2)`) that is both
a graph output and consumed by another node — or that fans out to several consumers — gets clobbered on the
longer branch. (4-D conv-map outputs are fine.) Here the raw `memory` output (Graph B's cross-attention
input) was silently garbage (device corr −0.02), so the decoder cross-attended to noise → no detections.

**Fix:** Graph A emits only the two fp16-clean leaves — `enc_class` and `memory_raw × 2` (the ×2 forces a
separate, clean output buffer; exact in fp16, undone on the host). The per-token tail (`enc_output` +
`enc_bbox_head`) moves to the host, in fp32, over only the 300 selected tokens — exact, because per-token
ops commute with the gather (`gather(f(x),i) = f(gather(x,i))`). With clean memory the **FDR decoder is
fine** (correlation is not the ship criterion — real-image detection IoU is), and every object is found.

## Files

| File | Description |
|------|-------------|
| `DFine.kt` | Both GPU graphs on CompiledModel + host topk + per-token tail + decode + light NMS |
| `MainActivity.kt` | Runs detection on a bundled image / gallery pick, overlays boxes + COCO labels |
| `app/src/main/assets/coco_labels.txt` | 80-line COCO label table (contiguous class id 0–79) |
| `app/src/main/assets/host_params.bin` | `enc_output` + `enc_bbox_head` weights, valid mask, anchors (fp32) |

## Setup

1. Build the two tflites + host params with `scripts/build_dfine_split.py` + `scripts/build_dfine_fix3.py`
   then `scripts/pack_assets.py` (needs `pip install transformers` + litert-torch), or download from
   Hugging Face — [litert-community/D-FINE-S-LiteRT](https://huggingface.co/litert-community/D-FINE-S-LiteRT).
2. Build/install the app, then push the models into its private storage:
   ```bash
   ./scripts/install_to_device.sh <dir-with-the-tflites>
   ```
   (`test_image.jpg`, `coco_labels.txt`, `host_params.bin` are bundled.)
3. Launch **D-FINE** — it compiles the GPU shaders (~1 s/graph first launch), then detects objects.

## Conversion

`scripts/build_dfine_split.py` holds the GPU re-authoring (monkeypatches + Graph B); `build_dfine_fix3.py`
is the **device-correct** Graph A (the 3-D fan-out fix above) and saves the host-tail weights;
`pack_assets.py` renames the tflites and writes `host_params.bin` + `coco_labels.txt`. Both graphs op-check
GPU-clean (no banned ops, no >4D tensors) and validate per-graph corr 1.0 vs PyTorch.

**Original project**: [Peterande/D-FINE](https://github.com/Peterande/D-FINE) ·
`ustc-community/dfine-small-coco` · [Apache-2.0](https://github.com/Peterande/D-FINE/blob/main/LICENSE)
