# RT-DETRv2-S — Object Detection on-device (LiteRT GPU, fully GPU)

[RT-DETRv2](https://github.com/lyuwenyu/RT-DETR) (Baidu, 2024 — `PekingU/rtdetr_v2_r18vd`) object
detection running **fully on the LiteRT CompiledModel GPU** (ML Drift). RT-DETRv2 is a real-time
transformer detector (ResNet18-vd backbone + a hybrid AIFI/CCFM encoder + a plain deformable-attention
DETR decoder). Both transformer graphs run on the GPU with no CPU/ONNX fallback; only the topk selection
and a tiny per-token tail run on the CPU. The demo detects objects in a bundled image and overlays the
boxes + COCO labels.

## On-device (Pixel 8a, Tensor G3 — verified)

| graph | nodes on GPU | time |
|---|---|---|
| **Graph A** — ResNet18-vd backbone + hybrid encoder + score head | fully `LITERT_CL` | ~6 ms |
| **Graph B** — two-stage combine + plain decoder + heads | `704/704` LITERT_CL | ~11 ms |

The device chain reproduces the PyTorch detections **exactly** — COCO val *giraffe* image **7/7**, *cats*
image (`000000039769`) **6/6**, every box at **IoU 0.98–1.00** with matching class and score.

## How it splits (and why it's fully GPU)

RT-DETRv2 is a two-stage DETR. The query selection (top-300 proposals by class score) is `TOPK_V2` +
`GATHER`, which have no GPU op — but the proposal **grid is image-independent**, so the model splits at
exactly that point:

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
   BMM against the value map; exact incl. zeros-padding OOB, all ≤4D); **MSDeformAttn** re-authored ≤4D.
2. **ResNet18-vd stem** `MaxPool(k3,s2,p1)` pads with `-inf` → `PADV2`, which the Mali delegate won't
   delegate → split graph. The pool follows ReLU (inputs ≥ 0) so **zero-pad + valid maxpool is exact**.
3. **AIFI** 2-D sine pos-embed baked (`temperature**ω` → illegal `POW`); **SafeLayerNorm**; tanh-GELU;
   `inverse_sigmoid`'s redundant `clamp(0,1)` dropped (lowers to Mali-rejected `RELU_0_TO_1`); the baked
   anchors' `finfo.max` invalid-border values clamped to ±1e4 (else fp16 `Inf`).

### ⭐ fp16 hardening — the on-device gate (a Mali 3D-sequence fan-out bug)

Both graphs convert GPU-clean and are fully `LITERT_CL`-resident, but the naïve Graph A (emitting
`enc_class`, `enc_coord`, `output_memory`, `memory_raw` all at once) **silently produced wrong boxes on
device** — large objects vanished while small ones stayed perfect. The cause is **not** an fp16-precision
wall: it is a Mali delegate bug where a **3-D *token* tensor `[1,N,256]`** (produced by
`conv.flatten(2).transpose(1,2)`) that is **both a graph output and consumed by another node — or that
fans out to several consumers — gets clobbered on the longer branch.** (4-D conv-map outputs with the
same fan-out are fine.) Here `output_memory` fed both the score head and the box head; the box head (a
3-layer MLP) lost, so its reference-box deltas collapsed to ~0 → boxes shrank to the default anchor →
large objects dropped. The corruption was masked in correlation by the ±1e4 baked anchors ("corr 1.0"
while the real valid-row corr was 0.88).

**Fix:** Graph A emits only the two fp16-clean leaves — `enc_class` (the 1-layer score head survives) and
`memory_raw × 2` (the ×2 forces a separate, clean output buffer; exact in fp16, undone on the host). The
per-token tail (`enc_output` + `enc_bbox_head`) is moved to the host, in fp32, over only the 300 selected
tokens — exact, because per-token ops commute with the gather (`gather(f(x),i) = f(gather(x,i))`) — so the
reference boxes are perfect and every object survives.

## Files

| File | Description |
|------|-------------|
| `RtDetr.kt` | Both GPU graphs on CompiledModel + host topk + per-token tail + decode + light NMS |
| `MainActivity.kt` | Runs detection on a bundled image / gallery pick, overlays boxes + COCO labels |
| `app/src/main/assets/coco_labels.txt` | 80-line COCO label table (index = contiguous class id 0–79) |
| `app/src/main/assets/host_params.bin` | `enc_output` + `enc_bbox_head` weights, valid mask, anchors (fp32) |

## Setup

1. Build the two tflites + host params with `scripts/build_rtdetr_split.py` + `scripts/build_rtdetr_fix3.py`
   then `scripts/pack_assets.py` (needs `pip install transformers` + litert-torch), or download from
   Hugging Face — [litert-community/RT-DETRv2-S-LiteRT](https://huggingface.co/litert-community/RT-DETRv2-S-LiteRT).
2. Build/install the app, then push the models into its private storage:
   ```bash
   ./scripts/install_to_device.sh <dir-with-the-tflites>
   ```
   (`test_image.jpg`, `coco_labels.txt`, `host_params.bin` are bundled.)
3. Launch **RT-DETRv2** — it compiles the GPU shaders (~1 s/graph first launch), then detects objects.

## Conversion

`scripts/build_rtdetr_split.py` holds the GPU re-authoring (monkeypatches + Graph B); `build_rtdetr_fix3.py`
is the **device-correct** Graph A (the 3-D fan-out fix above) and saves the host-tail weights;
`pack_assets.py` renames the tflites and writes `host_params.bin` + `coco_labels.txt`. Both graphs op-check
GPU-clean (no banned ops, no >4D tensors) and validate per-graph corr 1.0 vs PyTorch.

**Original project**: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) (RT-DETRv2) ·
`PekingU/rtdetr_v2_r18vd` · [Apache-2.0](https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE)
