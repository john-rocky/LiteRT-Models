---
license: apache-2.0
library_name: litert
pipeline_tag: depth-estimation
base_model: google/tipsv2-b14-dpt
tags:
  - litert
  - tflite
  - android
  - on-device
  - gpu
  - tipsv2
  - vit
  - dpt
  - depth-estimation
  - surface-normals
  - semantic-segmentation
  - ade20k
---

# TIPSv2-B/14 + DPT heads — depth · normals · segmentation on LiteRT GPU

[TIPSv2](https://huggingface.co/google/tipsv2-b14) (Google DeepMind, CVPR 2026) is a
DINOv2-style ViT-B/14 vision-language backbone; [`google/tipsv2-b14-dpt`](https://huggingface.co/google/tipsv2-b14-dpt)
adds three DPT heads trained on the frozen backbone. This is that model as **one LiteRT
graph** that runs fully on the `CompiledModel` GPU delegate (no CPU fallback) and returns
all three dense outputs from a single 448×448 image:

- **metric depth** (NYU Depth V2, 0.001–10 m)
- **surface normals** (NYU Depth V2, unit vectors)
- **ADE20K semantic segmentation** (150 classes)

![TIPSv2 — input | depth | normals | ADE20K segmentation, on-device fp16 GPU output](hero.png)

*Left to right: input, metric depth (inverse-depth Spectral colormap), surface normals
(xyz → RGB), ADE20K segmentation — all three from the on-device fp16 GPU run (Pixel 8a).*

- **Architecture:** ViT-B/14 (12 blocks, dim 768, 12 heads, 1 register token) → 4 taps
  (blocks 3/6/9/12) → 3 × DPT (reassemble + 4 fusion blocks + head). 86M backbone + 72M heads.
- **Weights:** [google/tipsv2-b14-dpt](https://huggingface.co/google/tipsv2-b14-dpt) · Apache-2.0.
- **Size:** 318 MB (fp16).

## I/O

- **Input:** `[1, 3, 448, 448]` NCHW, RGB in **[0, 1]** — no ImageNet mean/std (TIPSv2 convention).
- **Outputs (in order):**
  1. `[1, 1, 448, 448]` depth in metres
  2. `[1, 3, 448, 448]` unit surface normals
  3. `[1, 150, 256, 256]` segmentation logits at the DPT head's native grid —
     `argmax` over the 150 channels host-side, then upscale (the official pipeline bilinearly
     upsamples the logits to 448 first; argmax agreement between the two is 99.96 %).

## GPU conversion

Fully GPU-resident on a Pixel 8a (**1434/1434 nodes, 1 partition**, ~0.9 s for all three
heads). Re-authored from the HF weights with exact rewrites:

- **4D attention:** fused-qkv split into q/k/v and reshaped to `[1, heads, N, d]` with a manual
  `softmax(qkᵀ/√d)·v` — the delegate rejects the native 5D head-split.
- **LayerScale** baked into `attn.proj` / `mlp.fc2`; **SafeLayerNorm** (deviation pre-scaled by
  1/64 before squaring so the fp16 variance cannot overflow); **tanh-GELU** (no `ERF` kernel —
  the only non-exact rewrite, corr 0.999998 vs the official model).
- **DPT:** readout `cat(patch, cls.expand) @ W` → `patch @ W_a + cls @ W_b` (exact, no
  `BROADCAST_TO`); `ConvTranspose2d(k=s)` → zero-stuff + `Conv2d` (exact; `TRANSPOSE_CONV` is
  rejected on Pixel 8a); the fusion blocks' bilinear ×2 `align_corners=True` (banned on the GPU)
  → two constant-RHS matmuls `U·X·Uᵀ` (exact).
- **Depth-head fp16 range fold:** the depth decoder's activations reach ~1e8 at the logits
  (fp16 max 65504) — the GPU returned a constant depth while normals/seg were fine. The decoder is
  a ReLU/affine chain ending in a scale-invariant normalisation (`relu(l)/Σrelu(l)`), so
  power-of-2 scales are folded into its weights/biases (per-level convs 1/4096…1/64, each fusion
  `out_conv` ×1/4, `project` 1/32, head 1/16 — matching scales at every residual add) keeping
  every stage ≲100. Bit-exact in fp32.

Parity (desktop fp32 re-authored vs official PyTorch): depth corr 0.999998 · normals 0.999999 ·
seg argmax 99.96 %. Device fp16 (Pixel 8a GPU) vs desktop fp32: depth corr 0.99986 · normals
0.99990 · seg argmax 99.3 %. Conversion script: `build_tipsv2.py` (litert-torch).

## Minimal usage

### Kotlin (Android, LiteRT CompiledModel GPU)

```kotlin
// 318 MB — stage into filesDir (adb push + run-as cp) rather than APK assets
val model = CompiledModel.create(File(context.filesDir, "tipsv2_b14_dpt_fp16.tflite").absolutePath,
    CompiledModel.Options(Accelerator.GPU), null)
val inputs = model.createInputBuffers()
val outputs = model.createOutputBuffers()

inputs[0].writeFloat(imageNchw01)        // [1,3,448,448] RGB / 255, no mean/std
model.run(inputs, outputs)
val depth = outputs[0].readFloat()       // [448*448] metres
val normals = outputs[1].readFloat()     // [3*448*448] unit vectors
val seg = outputs[2].readFloat()         // [150*256*256] logits -> argmax per pixel host-side
```

### Python (LiteRT CompiledModel API)

```python
import numpy as np
from ai_edge_litert.compiled_model import CompiledModel

model = CompiledModel.from_file("tipsv2_b14_dpt_fp16.tflite")
inputs = model.create_input_buffers(0)
outputs = model.create_output_buffers(0)
inputs[0].write(np.ascontiguousarray(image, np.float32))    # [1,3,448,448] in [0,1]
model.run_by_index(0, inputs, outputs)
depth = outputs[0].read(448 * 448, np.float32).reshape(448, 448)          # metres
normals = outputs[1].read(3 * 448 * 448, np.float32).reshape(3, 448, 448)
seg = outputs[2].read(150 * 256 * 256, np.float32).reshape(150, 256, 256)
labels = seg.argmax(0)                                                     # ADE20K ids (256x256)
```

## Performance

Measured on a **Pixel 8a** (Tensor G3 / Mali-G715) through LiteRT's own `CompiledModel`
accelerator (`LITERT_CL`), the path the Kotlin sample and the LiteRT API use.

| Runtime | Backend | Graph on GPU | Latency |
|---|---|---|---|
| LiteRT `CompiledModel` (`LITERT_CL`) | GPU | 1434 / 1434 | ~0.92 s / image (all three heads; wall-clock difference between 1 and 6 back-to-back runs) |

GPU compile + load ≈ 5 s on first use. The three heads are ~3/4 of the compute (256-channel 3×3
convs up to 256×256); a single-head build would be correspondingly faster.

## Sample app

The **TIPSv2** Android sample (Kotlin, CompiledModel GPU): photo picker → input | depth,
normals | segmentation with an ADE20K legend. The conversion script `build_tipsv2.py`
(re-authoring + parity + litert-torch convert + fp16 + device harness) is included in this repo.

## License

Apache-2.0 (TIPSv2 / Google DeepMind). Converted with litert-torch. Hero photo: Pexels (free license).
