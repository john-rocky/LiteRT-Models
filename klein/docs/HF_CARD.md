---
license: apache-2.0
base_model: black-forest-labs/FLUX.2-klein-4B
tags:
  - litert
  - text-to-image
  - on-device
  - diffusion-transformer
  - int8
pipeline_tag: text-to-image
library_name: litert
---

# FLUX.2-klein-4B — LiteRT (on-device text-to-image)

Black Forest Labs **[FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)**
(Apache-2.0) converted to **LiteRT `CompiledModel`** int8 graphs and generating images
fully on a phone GPU. The upstream model card says klein "runs on consumer GPUs, with
as little as 13 GB VRAM". These graphs run it on a Pixel 8a's Mali-G715, which has no
dedicated VRAM at all.

![generated on a Pixel 8a Mali GPU](pixel8a_generated.png)

Prompt: *"a red apple on a wooden table, studio lighting"*. 4 steps, 256×256, generated
end-to-end on a Pixel 8a Mali GPU in 306 s. Matches the fp32 `diffusers` pipeline at
**PSNR 36.8 dB / corr 0.9987**.

klein is **step-wise distilled**, so the sampling loop is unusually plain: 4 steps, no
classifier-free guidance (one DiT pass per step, not two), no sign flip — just a
flow-matching Euler update `latents += dsigma[step] * noise_pred`.

## What's here

The pipeline (Qwen3-4B text encoder → rectified-flow DiT → VAE) is exported as
**INTEGER-int8** LiteRT graphs — int×int compute, the path the GPU delegate actually
runs; weight-only-FLOAT quantization hangs the GPU compile. The 4B DiT and the 4B
encoder each exceed both LiteRT's 2 GB flatbuffer load limit and a phone's GPU budget,
so they are **split into chunks that are resident one at a time**: peak footprint is a
single ~912 MB graph rather than the 6.2 GB total.

| Graph | Role | int8 size | I/O (256 px) |
|-------|------|-----------|--------------|
| `ke_enc0` / `ke_enc1` / `ke_enc2` | Qwen3-4B layers 1-9 / 10-18 / 19-27 | 912 MB each | `[1,512,2560]` → `[1,512,2560]` |
| `kc_prep` | image + context embedders, 3 modulation FCs | 166 MB | `img[1,256,128]`, `txt[1,512,7680]`, `temb[1,3072]` → hidden + 3 modulations |
| `kc_double0` / `kc_double1` | 3 + 2 double-stream blocks | 739 / 492 MB | `img[1,256,3072]`, `txt[1,512,3072]` → same |
| `kc_single0..3` | 5 single-stream blocks each (20 total) | 615 MB each | `joint[1,768,3072]` → `[1,768,3072]` |
| `kc_final` | adaLN-continuous norm + projection | 19 MB | `[1,768,3072]` → `[1,256,128]` |
| `kv_vae` | VAE decoder | 50 MB | `latent[1,32,32,32]` → `img[1,3,256,256]` |

The text encoder is included because klein conditions on Qwen3-4B **hidden states from
layers 9 / 18 / 27**, interleaved to 7680 channels — not on a pooled embedding, so there
is no smaller drop-in replacement.

Tensors are raw `float32`, little-endian, row-major. Tokenization, `embed_tokens`, the
causal+padding mask, both rotary tables, the scheduler and the two tail permutations run
on the host.

## Two things the graphs assume

Both come from the GPU delegate (ML Drift), and neither is visible to a desktop op check.

1. **The attention mask is pre-expanded across heads**: pass `[1, 32, 512, 512]`, not
   `[1, 1, 512, 512]`. A broadcast `ADD` whose left operand is a `BATCH_MATMUL` result is
   silently miscomputed — the probabilities still sum to 1 and still honour the causal and
   padding masks, but the logits are wrong.
2. **Compute must be FP32**: `GpuOptions(precision = FP32)`. The modulated (adaLN) blocks
   overflow fp16 and return NaN.

Also: create one `Environment` and share it across every `CompiledModel` (a null
environment leaks the OpenCL context), and close every `TensorBuffer` after each run.

## Usage — Python (reproduces the exact device loop)

```python
import numpy as np
from ai_edge_litert.compiled_model import CompiledModel


def run(path, *inputs):
    """Runs one chunk, then releases it — sequential residency, as on device."""
    model = CompiledModel.from_file(path)
    signatures = model.get_signature_list()
    key = list(signatures)[0]
    input_details = model.get_input_tensor_details(key)
    output_details = model.get_output_tensor_details(key)
    input_buffers = model.create_input_buffers(0)
    output_buffers = model.create_output_buffers(0)
    for name, buffer, value in zip(signatures[key]["inputs"], input_buffers, inputs):
        buffer.write(np.ascontiguousarray(value, np.dtype(input_details[name]["dtype"])))
    model.run_by_index(0, input_buffers, output_buffers)
    outputs = []
    for name, buffer in zip(signatures[key]["outputs"], output_buffers):
        detail = output_details[name]
        flat = buffer.read(int(np.prod(detail["shape"])), np.dtype(detail["dtype"]))
        outputs.append(flat.reshape(detail["shape"]).copy())
    return outputs


# Host prep (tokenizer, embed_tokens, mask, rotary tables, sigmas) omitted — see below.
hidden, taps = inputs_embeds, []
for i in range(3):
    hidden = run(f"ke_enc{i}.tflite", hidden, mask, enc_cos, enc_sin)[0]
    taps.append(hidden)
prompt_embeds = np.stack(taps, 1).transpose(0, 2, 1, 3).reshape(1, 512, 7680)

for step in range(4):
    image, text, mod_img, mod_txt, mod_single = run(
        "kc_prep.tflite", latents, prompt_embeds, temb[step:step + 1])
    for i in range(2):
        image, text = run(f"kc_double{i}.tflite", image, text, cos, sin, mod_img, mod_txt)
    joint = np.concatenate([text, image], axis=1)
    for i in range(4):
        joint = run(f"kc_single{i}.tflite", joint, cos, sin, mod_single)[0]
    latents = latents + dsigma[step] * run("kc_final.tflite", joint, temb[step:step + 1])[0]

latent = unpatchify(unpack(latents) * bn_std + bn_mean)   # two pure permutations
image = run("kv_vae.tflite", latent)[0]                   # [1,3,256,256] in [-1,1]
```

## Usage — Kotlin (Android, LiteRT GPU)

```kotlin
val environment = Environment.create()                    // create once, share

fun gpu(name: String, inputs: List<FloatArray>): List<FloatArray> {
    val options = CompiledModel.Options(Accelerator.GPU)
    options.gpuOptions = CompiledModel.GpuOptions(
        precision = CompiledModel.GpuOptions.Precision.FP32)
    val model = CompiledModel.create(File(dir, name).absolutePath, options, environment)
    val inputBuffers = model.createInputBuffers()
    val outputBuffers = model.createOutputBuffers()
    inputs.forEachIndexed { index, values -> inputBuffers[index].writeFloat(values) }
    model.run(inputBuffers, outputBuffers)
    val outputs = outputBuffers.map { it.readFloat() }
    inputBuffers.forEach { it.close() }
    outputBuffers.forEach { it.close() }
    model.close()                                          // one graph resident at a time
    return outputs
}

var hidden = inputsEmbeds
val taps = (0 until 3).map { gpu("ke_enc$it.tflite", listOf(hidden, mask, encCos, encSin))[0]
    .also { output -> hidden = output } }
// interleave the three taps -> [1, 512, 7680], then the 4-step DiT loop, then kv_vae
```

## Conversion

Quantization is `litert_torch` `full_dynamic_recipe(weight_dtype=INT8, granularity=CHANNELWISE)`.
The conversion scripts (`build_klein_enc.py`, `chunked_export_klein.py`,
`vae_deploy_klein.py`) and the host-prep / verification reference
(`gen_prep_klein.py`, `gen_verify_klein.py`) ship alongside the LiteRT sample app for
this model. Three rewrites are required for a GPU-clean graph, all exact:

- **RoPE without `GATHER_ND`** — bake the even/odd de-interleave into the rows of
  `to_q` / `to_k` and the fused `to_qkv_mlp_proj`, turning it into a contiguous
  half-split rotation. `q · k` is invariant to a permutation applied to both.
- **GQA `repeat_kv` as a `CONCATENATION`** — the stock `expand` is rank-5 *and* lowers
  to `BROADCAST_TO`, which the GPU delegate rejects outright.
- **Safe RMSNorm / LayerNorm** (max-normalized) and `ManualGroupNormND` in the VAE.

Note that the **desktop int8 path is a pessimistic proxy**: the same graphs score
36.4 dB through the host CPU int8 kernels and 44.1 dB on the device. Weights are never
redistributed here — the graphs are produced from the original Apache-2.0 checkpoint
with those scripts.

## License

Apache-2.0, inherited from `black-forest-labs/FLUX.2-klein-4B` (weights and text encoder).
