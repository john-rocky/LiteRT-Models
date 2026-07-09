---
license: apache-2.0
base_model: Tongyi-MAI/Z-Image-Turbo
tags:
  - litert
  - text-to-image
  - on-device
  - diffusion-transformer
  - int8
pipeline_tag: text-to-image
library_name: litert
---

# Z-Image-Turbo — LiteRT (on-device text-to-image)

Alibaba Tongyi-MAI **[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)**
(6B, Apache-2.0) — a Single-Stream Diffusion Transformer (S3-DiT) — converted to
**LiteRT `CompiledModel`** int8 graphs and generating images fully on the phone GPU.

![generated on a Pixel 8a Mali GPU](pixel8a_generated.png)

Prompt: *"a red apple on a wooden table, studio lighting"*. Generated end-to-end on a
Pixel 8a Mali GPU (8 GB) through LiteRT — the first 6B diffusion generator verified
producing a real image on a commodity 8 GB phone, via chunked sequential residency.

## What's here

The pipeline (Qwen3-4B text encoder → S3-DiT → VAE) is exported as **INTEGER-int8**
LiteRT graphs (int×int compute — the path the GPU delegate runs; weight-only-FLOAT
hangs on the GPU delegate). The 6 GB monolithic DiT is split into graphs that each
compile fully on the GPU delegate and load one at a time, so the peak footprint is a
single sub-1 GB graph:

| Graph | Role | int8 size | I/O (256 px) |
|-------|------|-----------|--------------|
| `z_embx` / `z_refx` | image embed + noise refiner | 0.3 / 363 MB | `img[1,256,64]` → `[1,256,3840]` |
| `z_embc` / `z_refc` | caption embed + context refiner | 10 / 355 MB | `cap[1,32,2560]` → `[1,32,3840]` |
| `zc_main0..5` | 5 S3-DiT layers each (30 total) | 866 MB ×6 | `hidden[1,288,3840]` → `[1,288,3840]` |
| `zc_final` | final adaLN + projection | 1.2 MB | `[1,288,3840]` → `[1,288,64]` |
| `zvae` | VAE decoder | 50 MB | `latent[1,16,32,32]` → `img[1,3,256,256]` |

Tensors are raw `float32`, little-endian, row-major. The pad-token mask, x/c concat,
classifier-free guidance and (un)patchify run on the host; see the conversion scripts
for the exact, reproducible reference loop.

## Usage (Python reference — reproduces the exact device loop)

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter

def load(path):
    it = Interpreter(model_path=path); it.allocate_tensors()
    ins = sorted(it.get_input_details(), key=lambda d: d["index"])
    out = it.get_output_details()[0]
    def run(*arrs):
        for d, a in zip(ins, arrs):
            it.set_tensor(d["index"], np.ascontiguousarray(a, "<f4"))
        it.invoke()
        return it.get_tensor(out["index"])
    return run

# host-precompute cap_feats / rope / per-step adaln / sigmas / initial latent (gen_prep.py),
# then per step, in latent space:
#   xt = patchify(latent)
#   pos = unpatchify(dit(xt, cap_cond, adaln[s], rope...))   # chunked DiT, cond
#   neg = unpatchify(dit(xt, cap_uncond, adaln[s], rope...)) # chunked DiT, uncond
#   noise_pred = -(pos + guidance * (pos - neg))             # Z-Image CFG (guidance=1)
#   latent += dsigma[s] * noise_pred
# image = vae(latent)
```

## Usage (Kotlin — on-device, LiteRT `CompiledModel` GPU)

```kotlin
// One shared Environment across every graph (a null Environment leaks the OpenCL
// context and aborts after ~20 FP32 compiles).
val env = Environment.create()
fun gpu(name: String, inputs: List<FloatArray>): FloatArray {
    val opts = CompiledModel.Options(Accelerator.GPU).apply {
        // FP32 compute: the adaLN/attention path overflows fp16 to NaN.
        gpuOptions = CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)
    }
    val model = CompiledModel.create(File(dir, name).absolutePath, opts, env)
    val ins = model.createInputBuffers(); val outs = model.createOutputBuffers()
    inputs.forEachIndexed { i, a -> ins[i].writeFloat(a) }
    model.run(ins, outs)
    val out = outs[0].readFloat()
    ins.forEach { it.close() }; outs.forEach { it.close() }; model.close()
    return out
}
// chunked DiT per step: embx -> [host pad mask] -> refx ; embc -> refc ;
// host concat -> zc_main0..5 -> zc_final -> [host unpatchify]. See ZImageGen.kt.
```

## Notes

- **Precision:** int8 (INTEGER-compute) renders a faithful image (DiT-only PSNR 28.3 dB;
  on-device sequential output corr 0.966 vs fp32). int4 is garbage (PSNR 18).
- **GPU-delegate-only fixes** (invisible to the desktop op-checker): move the pad-token
  MUL-after-FC to the host (`bc coord for BATCH axis` compile wall), force
  `precision = FP32` (fp16 adaLN NaN), share one `Environment` (OpenCL context leak).
- Weights are **never redistributed** here — the graphs are produced from the original
  Apache-2.0 checkpoint with the conversion scripts.

License: Apache-2.0 (inherited from Z-Image-Turbo).
