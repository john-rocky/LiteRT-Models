---
license: apache-2.0
base_model: speridlabs/idol
base_model_relation: quantized
library_name: litert
pipeline_tag: image-to-3d
tags:
  - litert
  - tflite
  - android
  - on-device
  - gpu
  - gaussian-splatting
  - 3d
---

# Idol on LiteRT: one photo to a 3D Gaussian splat on the phone

{SEAT}

[speridlabs/idol](https://huggingface.co/speridlabs/idol) turns one photo of an object into a 3D Gaussian splat of
262,144 Gaussians that you can turn around, including the sides the picture never showed. This repo holds the
author's denoiser with the [speridlabs/idol-turbo](https://huggingface.co/speridlabs/idol-turbo) LoRA (5 steps, no
guidance), its octree and
Gaussian decoders, the FLUX.2 VAE encoder, and the author's BiRefNet background remover. The DINOv3 image encoder is
in a separate, gated repo under its own licence:
[litert-community/DINOv3-ViT-H16plus-Idol-LiteRT](https://huggingface.co/litert-community/DINOv3-ViT-H16plus-Idol-LiteRT).

![The photo, and the splat made from it turned to the front, the side and the back](assets/hero.png)

*{HERO_CAPTION}*

## Speed

{SPEED_PARAGRAPH} Background removal (BiRefNet) runs on the CPU: {BIR_S26} s of that time on the Galaxy S26. The
other networks run on the GPU.

Speed still being tuned (updates follow).

{SPEED_TABLE}

## How close it is to the author's model

Each check runs the author's PyTorch pipeline (float32, CPU) and these graphs on the same input with the same random
numbers, against thresholds fixed before the first run. The inputs are two images from the author's gallery
(`nb-armchair`, `fig-axolotl`) and one photo (a succulent, Pexels). Per part, each graph gets the author's own input
for that part and its output is compared with the author's (cosine; the threshold is 0.99). The whole pipeline is compared
by rendering both splats from four sides with the same renderer (PSNR; the threshold is 25 dB).

{FACTS}

## Minimal usage

### Python: the whole pipeline

```bash
hf download litert-community/Idol --local-dir Idol
cd Idol
pip install -r python/requirements.txt
python python/run_idol.py photo.jpg --out out --models .   # writes out/gaussian.ply
```

`run_idol.py` downloads the DINOv3 repo (accept its licence on the Hub and `hf auth login` first) and runs every
graph with `CompiledModel`, the GPU for all but BiRefNet. The steps between graphs are in `python/idol_runner.py` and
`python/idol_glue.py`. On an M4 Max Mac (Metal GPU) one photo took 64 s and up to 22 GB of memory.

### Python: one graph with `CompiledModel` (background removal)

```python
{PYTHON_SNIPPET}
```

### Kotlin (Android)

```kotlin
{KOTLIN_SNIPPET}
```

The Android sample app (photo or camera → progress per stage → a splat you can turn with a finger) is in `android/`.
The graphs are too large for the APK; a script copies them into the app's files folder over adb:

```bash
hf download litert-community/Idol --local-dir ~/Downloads/Idol
hf download litert-community/DINOv3-ViT-H16plus-Idol-LiteRT --local-dir ~/Downloads/DINOv3-ViT-H16plus-Idol-LiteRT
cd ~/Downloads/Idol/android
chmod +x gradlew scripts/install_to_device.sh        # the Hub does not keep the executable bit
./gradlew :app:installDebug
IDOL_DIR=~/Downloads/Idol DINO_DIR=~/Downloads/DINOv3-ViT-H16plus-Idol-LiteRT ./scripts/install_to_device.sh
```

Close other apps before a run: one photo peaks at about 5.4 GiB of resident memory, and with another app holding
memory the system can close the app during background removal.

## What runs where

| Stage | Files | Runs on | Calls per photo |
|---|---|---|---|
| Background removal (BiRefNet) | `bir.tflite` | CPU | 1 |
| DINOv3 ViT-H+/16 image encoder | `dino_0..3.tflite` (DINOv3 repo) | GPU | 4 |
| FLUX.2 VAE encoder | `vae_0..3.tflite`, `vae_attn.tflite`, `vae_4.tflite` | GPU | 6 |
| Denoiser with the turbo LoRA | `den_ctx.tflite`, `den_in.tflite`, `den_joint_00..23.tflite` | GPU | 5 steps × 25, plus `den_ctx` once per LoRA phase |
| Octree decoder | `oct.tflite` (5 signatures by row count) | GPU | 8 levels |
| Gaussian decoder | `gs_0..7.tflite` | GPU | 8 |
| Octree sampling, Gaussian activation, PLY | host code | CPU | — |

`manifest.json` lists every graph with its signatures, accelerator and precision. On Android the GPU graphs run with
`FP16_WITH_FP32_ACCUM` precision.

## Notes

- **Background removal runs on the CPU.** BiRefNet's 20 deformable convolutions sample at positions computed from
  the image. They are rewritten as a gather with bilinear weights and one matmul (same arithmetic; 5.5e-6 from the
  author's map in PyTorch), and the LiteRT GPU has no kernel for that gather. The gathers run in row tiles one after
  another: each tile's offsets wait for the previous tile's result. Without that wait the CPU runtime ran every
  tile's gather before using any of them; the first export ran out of memory on the Galaxy S26, and this one peaks
  at {BIR_PEAK} on it.
- **The VAE's GroupNorm scales each group by its largest |x − mean| to stay in fp16 range.** The graph finds that
  maximum as a max pool, then a max over the 32 × 32 result (the same value). One REDUCE_MAX over the whole
  1024 × 1024 map took 8 s per GPU call on the Galaxy S26; this form takes 0.83 s.
- **Attention over 4,096 keys or more is summed in chunks.** The product of the attention weights and the values is
  summed over key chunks of 1,024 after centring the values (exact, since each softmax row sums to 1). One fp16 sum over 8,192 keys lost 18 % in the
  Gaussian decoder's first block; the chunked form loses 0.6 %. Keep it if you re-export.
- **Keep each GPU call short on the Galaxy S26.** In the feasibility run the VAE encoder ran as one graph, about 9 s
  per GPU call; the screen compositor stalled and the process was killed (signal 9) on its fifth call. Every stage here
  is split into several graphs for that reason ({CALL_LENGTHS}).

## Licence and provenance

| Part | Weights | Licence |
|---|---|---|
| Denoiser, octree decoder, Gaussian decoder | [speridlabs/idol](https://huggingface.co/speridlabs/idol) | Apache-2.0 |
| Turbo LoRA (applied to the denoiser graphs) | [speridlabs/idol-turbo](https://huggingface.co/speridlabs/idol-turbo) | Apache-2.0 |
| FLUX.2 VAE encoder | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) `vae/` | Apache-2.0 |
| BiRefNet background remover | `birefnet.safetensors` in speridlabs/idol ([BiRefNet](https://github.com/ZhengPeng7/BiRefNet)) | MIT |
| DINOv3 ViT-H+/16 (separate repo) | [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m), read from the bf16 copy Idol loads (Comfy-Org/Idol `clip_vision/dino_v3_vit_h_bf16.safetensors`) | DINOv3 License |

The graphs here are Apache-2.0 except `bir.tflite` (MIT). `LICENSES.md` records where each licence was read.
