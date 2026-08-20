# SAM 3.1 (image side) on LiteRT CompiledModel GPU

Text-prompted open-vocabulary detection + instance segmentation from the
`facebook/sam3.1` Object-Multiplex checkpoint (detector part: ViT-L/14 @1008 +
CLIP-L text encoder + DETR-style head, 854M params), converted to three LiteRT
graphs that run on the CompiledModel GPU API. The Object-Multiplex video tracker
(19.6M extra params, shares the same ViT trunk) is phase 2.

## Graphs (fp16 weights)

| file | I/O | size |
|---|---|---|
| `sam3_vision.tflite` | image `[1,3,1008,1008]` ((x/255−0.5)/0.5, RGB) → `[fpn288 \| fpn144 \| fpn72]` | 930 MB |
| `sam3_text.tflite` | token embeddings `[1,32,1024]` (host lookup) → text memory `[1, 32·256]` | 607 MB |
| `sam3_head.tflite` | `[fpn×3 \| text_mem \| pad(32)]` → `[logits(200) \| boxes(200×4 cxcywh) \| presence(1) \| mask logits(200×288²)]` | 68 MB |

Host side: CLIP BPE tokenizer (ctx 32, **zero padding**, pad mask = `id==0`),
fp16 token-embedding table (`sam3_token_embed.bin`, 101 MB), score =
`sigmoid(logit)·sigmoid(presence)`, threshold 0.5.

## Build / verify (Mac)

```bash
cd sam3
uv venv --python 3.12 .venv   # torch 2.12 + torchvision 0.27 + litert-torch + ai-edge-litert
.venv/bin/python scripts/build_sam3.py --gpu-mac     # export + parity (torch/CPU/Metal)
.venv/bin/python scripts/run_pipeline_mac.py         # full 3-graph GPU pipeline + overlays
```

`build_sam3.py` asserts exact torch parity of every re-authored module before
exporting (vision/text/head corr 1.0000000 vs the stock model). The conversion
recipes live in `scripts/vit4d.py` (exact 4-D ViT re-authoring: RoPE de-interleave
baked into qkv weights, ≤4-D window partition, SafeLayerNorm) and
`scripts/gpu_patches.py` (4-D manual attention, masked/biased softmax forms the
Metal delegate executes correctly, batch-first DETR decoder, GATHER_ND-free
sine embedding / log-RPB / GroupNorm).

Mac results (M4 Max, `run_pipeline_mac.py`, 5 image/prompt pairs vs all-PyTorch):
- vision fp16 + text f32 + head f32: **~750 ms/frame**, kept-set matches except
  borderline queries (|Δprob| within ~0.1 of the 0.5 threshold), mask IoU ≥ 0.97.
- all f32 (`--vision-f32`): **~950 ms/frame**, kept-set + masks exact (IoU 1.000).

## Android app (`app/`)

Pick an image, type a prompt, get boxes + masks. Vision + head on
`Accelerator.GPU`, text on `Accelerator.CPU` (fp16 GPU execution of the CLIP-L
|x|≈1200 residual stream corrupts some prompts; CPU is exact and fast). Models
are pushed to the app's private storage (adb pattern, ~1.7 GB total):

```bash
cd sam3
./gradlew :app:installDebug
./scripts/install_to_device.sh          # pushes models/out/* into com.sam3 files/
```

First `Detect` compiles the GPU graphs (expect ~a minute on device); later runs
reuse the compiled cache. Vision features are cached per image, so re-prompting
the same photo only runs text + head.

Measured on a Pixel 8a (LiteRT **2.2.0**): vision 9.2 s (GPU fp16, 3104/3104 ops
LITERT_CL), text 0.5 s (CPU), head 1.4 s (GPU) — first prompt ~11 s, re-prompting
the same image ~1.9 s. Detections match the PyTorch reference (wheel 4/4 scores
within 0.02, window 6/6, mask IoU ≥ 0.98).

**LiteRT version matters:** the DETR head triggers two Mali ML Drift bugs. The
decoder output must stay rank-4 — a rank-3 `[1,200,256]` tensor fanning out to the
score/box/mask heads is silently corrupted (all 200 logits identical) — and even
the rank-4 graph is mis-executed by LiteRT **2.1.5**; **2.2.0 runs it correctly**.
The app pins 2.2.0.
