# Licences of the parts of the Idol pipeline (read 2026-11-09)

Copies of the texts read are in `~/code/idol_work/licenses/` (SSD work dir). Line numbers below are in those copies.

| Part | Weights used here | Licence | Source read |
|---|---|---|---|
| Denoiser, octree decoder, Gaussian decoder | speridlabs/idol @2731c229 (`denoiser`, `decoder`, `decoder_gs` .safetensors) | Apache-2.0 | HF card front matter line 2 `license: apache-2.0`; idol-sdk LICENSE @4702a7f0 (Apache License 2.0, lines 1-2) |
| Turbo LoRA (slow / fast experts) | speridlabs/idol-turbo @45cdd41f (`idol_turbo_lora.safetensors`) | Apache-2.0 | HF card front matter `license: apache-2.0`; card §License "Apache-2.0, inherited from speridlabs/idol" |
| Background removal (shipped: `bir.tflite`, LiteRT CPU) | speridlabs/idol @2731c229 `birefnet.safetensors` (BiRefNet, Swin-L) | MIT (BiRefNet); the idol repo that packages it is Apache-2.0 | github ZhengPeng7/BiRefNet licence = MIT (GitHub API); idol HF card front matter line 2 |
| Background removal, alternative in the app (off by default) | litert-community/DIS-ISNet-LiteRT @1b966dbe `dis.tflite` (IS-Net DIS general-use, xuebinqin/DIS) | Apache-2.0 | litert-community/DIS-ISNet-LiteRT card `license: apache-2.0` (Hub API) |
| FLUX.2 VAE (encoder side used) | black-forest-labs/FLUX.2-klein-4B @e7b7dc27 `vae/` (ungated) | Apache-2.0 | black-forest-labs/flux2 README @50fe5162, line 111: "The autoencoder is released under Apache 2.0"; klein-4B card `license: apache-2.0` |
| DINOv3 ViT-H+/16 | Comfy-Org/Idol @e907d8dc `clip_vision/dino_v3_vit_h_bf16.safetensors` (bf16 repack of facebook/dinov3-vith16plus-pretrain-lvd1689m, gated) | DINOv3 License (Meta, "Last Updated: August 19, 2025") | facebookresearch/dinov3 LICENSE.md @6876159a |
| Idol acknowledges TripoSplat | — (no TripoSplat weights used) | MIT | github VAST-AI-Research/TripoSplat licence = MIT (GitHub API) |

## DINOv3 License — the clauses that decide redistribution (dinov3_LICENSE.md)

- Line 20: "By clicking "I Accept" below or by using or distributing any portion or element of the DINO Materials, you agree to be bound by this Agreement." → using the Comfy-Org repack binds the user of it to the Agreement as if accepted on the gated page.
- Line 24 (§1.a Grant of Rights): "non-exclusive, worldwide, non-transferable and royalty-free limited license … to use, reproduce, distribute, copy, create derivative works of, and make modifications to the DINO Materials." → a converted .tflite of the encoder is a derivative work and may be distributed.
- Line 28 (§1.b.i): "If you distribute or make the DINO Materials, or any derivative works thereof, available to a third party, you may only do so under the terms of this Agreement and you shall provide a copy of this Agreement with any such DINO Materials." → a hosted encoder file must carry the DINOv3 License text and be offered under it (not under Apache-2.0).
- Line 30 (§1.b.ii): acknowledgement in research publications.
- Line 34 (§1.b.iv): no use that involves or encourages reverse engineering / decompiling the DINO Materials.
- Line 36 (§1.b.v): trade-control limits (no ITAR / military / nuclear / espionage / weapons end uses; licensee not a target of trade controls).
- Line 52 (§5.a): derivative works made by the licensee are owned by the licensee (subject to Meta's ownership of the DINO Materials).
- No non-commercial clause anywhere in the text (66 lines read in full).

## FLUX.2 VAE

- flux2 README line 111 (commit 50fe5162): the FLUX.2 autoencoder is released under Apache 2.0. The FLUX.2 [dev] Non-Commercial License of the FLUX.2-dev repo covers the [dev] transformer, not the autoencoder.
- The klein-4B `vae/` dir is ungated and tagged apache-2.0; the Idol pipeline loads the VAE in bf16 (`conditioning.py` `torch_dtype=torch.bfloat16`) and klein-4B's file is bf16.
- Verified 2026-11-09 14:04 (`~/code/idol_work/probe/vae_equal.py`): klein-4B `vae/diffusion_pytorch_model.safetensors` (sha256 ca70d220…, bf16) = the FLUX.2-dev VAE cast to bf16. The FLUX.2-dev VAE was read from Comfy-Org/Idol `vae/flux2-vae.safetensors`, whose LFS sha256 868fe7b3… equals FLUX.2-dev's `ae.safetensors` (Hub API, FLUX.2-dev @26afe3a7): the same file. All 250 float tensors match bit for bit after the bf16 cast (84,046,372 values; 8 attention projections are [512, 512] in klein and [512, 512, 1, 1] in BFL, same values). The one other tensor differs: `bn.num_batches_tracked` (int64 training counter, 0 vs 400000), unused at inference.

## Hosting (window decision D-912, 2026-11-09)

- litert-community/Idol (ungated, Apache-2.0, with the per-part provenance above): the denoiser, octree and Gaussian
  decoder graphs, the VAE encoder graphs, BiRefNet (`bir.tflite`, MIT part named in the card), the side files, the
  manifest, and the Python / Android sources.
- The DINOv3 encoder graphs and their RoPE tables: a separate litert-community repo, gated (automatic approval), carrying
  the full DINOv3 License text and the provenance (base_model facebook/dinov3-vith16plus-pretrain-lvd1689m), the way
  litert-community/Llama-3.2-1B carries the Llama licence.

## Before D-912 (kept as written)

- Apache-2.0 parts (denoiser, decoders, turbo LoRA, VAE encoder) and MIT BiRefNet: may be converted and hosted on litert-community with their licence and attribution.
- DINOv3 encoder: the licence permits hosting a converted derivative with the DINOv3 License attached and offered under it (precedent on the Hub: needle-tools/IDOL-webgpu `imageEncoder/` and ddalcu/Idol-MLX-Serve-* `dinov3/`, both with the licence text). Whether litert-community (a Google org) carries a Meta-licensed file is the window's / user's call. Until it answers, the encoder conversion stays local and the day-0 repo ships the Idol graphs plus `scripts/convert_encoders.py` for the user to convert the encoder from weights they accepted.
