---
license: apache-2.0
library_name: litert
pipeline_tag: image-feature-extraction
tags:
  - litert
  - tflite
  - android
  - on-device
  - gpu
  - dinov2
  - vit
  - self-supervised
  - feature-extraction
---

# DINOv2 ViT-S/14 — dense features on LiteRT GPU

The self-supervised [DINOv2](https://github.com/facebookresearch/dinov2) ViT-S/14
backbone running its full forward pass on the LiteRT `CompiledModel` GPU delegate
(no CPU fallback). It emits the **dense patch tokens**; a top-3 PCA of those tokens
mapped to RGB gives the classic "what the backbone sees" overlay — semantically
similar patches (object parts vs background) share a color, so the object pops out
with no labels or segmentation.

- **Architecture:** DINOv2 ViT-S/14 (`vit_small_patch14_dinov2` in timm) — 12 blocks,
  dim 384, 6 heads, patch 14. Fixed 448×448 input → 32×32 = 1024 patch tokens.
- **Weights:** [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) · Apache-2.0.
- **Size:** 45 MB (fp16).

![DINOv2 dense feature PCA](hero.png)

*Left: input. Right: top-3 PCA of the DINOv2 patch tokens (on-device fp16 output).*

## I/O

- **Input:** `[1, 3, 448, 448]` NCHW, RGB, ImageNet-normalized.
- **Output:** `[1, 1024, 384]` patch tokens (32×32 grid, cls token dropped).

## GPU conversion

Fully GPU-resident on a Pixel 8a (**864/864 nodes, 1 partition**, ~8 ms) via the
proven ViT re-authorings:

- **4D attention:** the fused-qkv attention is split into q/k/v and reshaped to
  `[1, heads, N, d]` (≤4D) with a manual `softmax(qkᵀ/√d)·v`; the delegate rejects
  the native 5D head-split reshape.
- **SafeLayerNorm:** the deviation is scaled by 1/64 before squaring so the
  per-token sum of squares stays in fp16 range on DINOv2's massive activations,
  then rescaled — algebraically identical to the plain variance.
- **LayerScale** (`ls1`/`ls2`) baked into the following projection weights.
- **tanh-GELU** (`0.5x(1+tanh(0.79788(x+0.044715x³)))`) — near-exact and
  delegate-friendly; the sigmoid-GELU approximation drifts to feature corr 0.968
  over 12 blocks, tanh → 0.99999.
- The **pos_embed is baked at a fixed 448 grid** at model creation, so there is no
  runtime interpolation (no `GATHER_ND`).

Device fp16 patch features vs desktop fp32: corr 0.996. Re-authored torch vs stock
timm: corr 0.999992.

## Minimal usage

### Kotlin (Android, LiteRT CompiledModel GPU)

```kotlin
val model = CompiledModel.create(context.assets, "dinov2_s_fp16.tflite",
    CompiledModel.Options(Accelerator.GPU), null)
val inputs = model.createInputBuffers()
val outputs = model.createOutputBuffers()

inputs[0].writeFloat(imageNchw)          // [1,3,448,448] ImageNet-normalized
model.run(inputs, outputs)
val tokens = outputs[0].readFloat()      // [1024*384] patch tokens -> PCA host-side
```

### Python (LiteRT CompiledModel API)

```python
import numpy as np
from ai_edge_litert.compiled_model import CompiledModel

model = CompiledModel.from_file("dinov2_s_fp16.tflite")
inputs = model.create_input_buffers(0)
outputs = model.create_output_buffers(0)
inputs[0].write(np.ascontiguousarray(image, np.float32))   # [1,3,448,448]
model.run_by_index(0, inputs, outputs)
tokens = outputs[0].read(1024 * 384, np.float32).reshape(1024, 384)

x = tokens - tokens.mean(0)
_, _, vt = np.linalg.svd(x, full_matrices=False)
rgb = x @ vt[:3].T                        # [1024,3] -> normalize -> 32x32 RGB
```

## Performance

Measured on a **Pixel 8a** (Tensor G3, Android 16) with the standard TFLite [`benchmark_model`](https://ai.google.dev/edge/litert/models/measurement) tool — 10 warm-up runs then 50 timed runs, reported as the tool's mean.

| Runtime | Backend | Graph on GPU | Latency |
|---|---|---|---|
| LiteRT `CompiledModel` (`LITERT_CL`) | GPU | 864 / 864 | ~8 ms |
| TFLite `benchmark_model` (`TfLiteGpuDelegateV2`) | GPU (OpenCL) | 864 / 864 | did not run |
| TFLite `benchmark_model` | CPU (XNNPACK, 4 threads) | — | 1093.3 ms |

**The two GPU rows are different runtimes, not a contradiction.** The `LITERT_CL` figure is the one recorded when this model shipped, taken through LiteRT's own `CompiledModel` accelerator — the path the Kotlin sample app and the LiteRT API use. The `TfLiteGpuDelegateV2` figure is the classic TFLite OpenCL delegate, measured with a tool anyone can download and re-run. They agree on how much of the graph the GPU takes; they disagree on speed, and the classic delegate is the slower of the two here. Read the `TfLiteGpuDelegateV2` row as a reproducible floor, not as this model's speed on LiteRT.

## Snapdragon NPU (Hexagon)

**The NPU is faster — once the GELU is written as an op.** The original file loses on
the NPU (85.9 ms vs 54.7 ms on the GPU), and the cause is not the architecture: it is
the tanh-GELU elementwise decomposition, which falls off the Hexagon compiler's fast
path. `dinov2_s_erf_fp16.tflite` is the same model with the MLP GELU emitted as the
builtin GELU op (exact erf — the activation the official DINOv2 uses), features
matching the original file at corr 0.99999:

| file | NPU (Hexagon v81) | GPU (Adreno) |
|---|---:|---:|
| `dinov2_s_fp16.tflite` (tanh-GELU) | 85.9 ms | 54.7 ms |
| `dinov2_s_erf_fp16.tflite` (GELU op) | **41.9 ms** | 55.2 ms |

Pick by accelerator: the erf file for the NPU (2.05x). It also ran at full speed on
this Adreno GPU, but the Mali path this card's main file was built for is unverified
with the builtin GELU op — keep `dinov2_s_fp16.tflite` for Mali.

Measured on a **Samsung Galaxy S26** (Snapdragon 8 Elite Gen 5 / SM8850, Hexagon v81,
Android 16), LiteRT `CompiledModel` 2.2.0, **on-device JIT compile** (first load
compiles in ~2-3 min and caches; later loads 0.2-0.4 s), one accelerator per process,
warm-up then N=50 timed runs, median reported, every row at thermal status `NONE`.
JIT matches the earlier ahead-of-time row (85.9 ms JIT vs 86.0 ms AOT). Latencies were
taken on the fp32 build of the same graph; storing the weights as fp16 (these files)
measured within run-to-run noise on this model (fp32-fold test: 85.9 -> 83.3/89.2 ms).
The runtime libraries the NPU needs are in the [NPU recipe](https://github.com/john-rocky/hf-to-litertlm/blob/main/docs/android-npu.md); GPU wiring is in the [GPU recipe](https://github.com/john-rocky/hf-to-litertlm/blob/main/docs/android-gpu.md).

## License

Apache-2.0 (DINOv2 / Meta). Converted with litert-torch.
