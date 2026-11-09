---
{GATE_FRONT_MATTER}
base_model: facebook/dinov3-vith16plus-pretrain-lvd1689m
base_model_relation: quantized
library_name: litert
pipeline_tag: image-feature-extraction
tags:
  - litert
  - tflite
  - android
  - on-device
  - gpu
  - dinov3
---

# DINOv3 ViT-H+/16 for Idol (LiteRT)

The image encoder of [litert-community/Idol](https://huggingface.co/litert-community/Idol) (one photo to a 3D Gaussian
splat on the phone): Meta's DINOv3 ViT-H+/16
([facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m))
as four LiteRT graphs for `CompiledModel` on the GPU, converted from the bf16 copy of those weights that Idol itself
loads (Comfy-Org/Idol `clip_vision/dino_v3_vit_h_bf16.safetensors`). It is its own repo because the weights are under the DINOv3
License, which this repo carries in full (`LICENSE.md`); the rest of Idol is Apache-2.0.

## Input and output

| | Shape | What |
|---|---|---|
| `dino_0` input `args_0` | [1, 3, 1024, 1024] float32 | RGB in [0, 1]; the graph applies the ImageNet mean / std |
| `args_1`, `args_2` of every graph | [1, 1, 4101, 64] float32 | RoPE cos / sin, `dino_rope_cos.f32` and `dino_rope_sin.f32` (raw little-endian float32) |
| `dino_0..2` output → next graph's `args_0` | [1, 4101, 1280] | the hidden states after 8 more layers each |
| `dino_3` output | [1, 4101, 1280] | the final norm, then a layer norm without weights, as Idol feeds it to its denoiser |

4,101 tokens = 1 class token, 4 register tokens and 64 × 64 patches. The last output is what Idol consumes, not the
plain DINOv3 feature: Idol adds the parameter-free layer norm on top.

## How close it is to the original

On the Mac (LiteRT GPU, Metal, fp16), the last output against the author's PyTorch pipeline (float32): {DINO_FACTS}.
Attention over the 4,101 tokens sums the product of the attention weights and the values over key chunks of 1,024,
after centring the values. Exported
with one fp16 sum instead, the last output was 4.8 % off (relative L2) on the first test image, against 1.5 % with the
chunks.

**Four graphs, not one.** In the feasibility run of the Idol pipeline on a Galaxy S26, the VAE encoder ran as one
graph, about 9 s per GPU call; the screen compositor stalled and the process was killed (signal 9) on its fifth call.
The encoder is four graphs of 8 layers for that reason, about 1.5 s per call each on the Galaxy S26. Keep that split
if you re-export.

## Minimal usage

### Python

```python
{DINO_PYTHON_SNIPPET}
```

### Kotlin (Android)

```kotlin
{DINO_KOTLIN_SNIPPET}
```

For the whole Idol pipeline, see [litert-community/Idol](https://huggingface.co/litert-community/Idol).

## Licence

DINOv3 License (Meta, last updated August 19, 2025), in full in `LICENSE.md`. You get these files under that
licence; it asks you to pass it on with any copy and lists the uses it does not allow.
