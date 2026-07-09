# FLUX.2-klein conversion scripts

Run in this order. Everything writes `.tflite` / `.bin` next to the scripts; nothing
is committed.

| Script | What it does |
|---|---|
| `build_klein.py` | The single-stream block, re-authored GPU-clean. Bakes the RoPE even/odd de-interleave into the fused `to_qkv_mlp_proj` rows; `safe_rms`, `safe_ln_noaffine`. |
| `build_klein_double.py` | The double-stream block. Same RoPE bake across `to_q`/`to_k`/`add_q_proj`/`add_k_proj` and the four qk-norm weights; joint RoPE over `cat([text, image])`. |
| `build_klein_dit.py` | Assembles the full DiT from the two block types; verifies against the stock `Flux2Transformer2DModel`. |
| `build_klein_real.py` | Loads the real weights and pins the fixed geometry (256 px → 16x16 image tokens, 512 text tokens, `latent_ids`, RoPE axes). |
| `chunked_export_klein.py` | Splits the DiT into `kc_prep`, `kc_double0/1`, `kc_single0-3`, `kc_final` and exports each as an INTEGER-int8 graph. Checks the sequential composition against the monolithic model first. |
| `build_klein_enc.py` | The Qwen3-4B text encoder as three 9-layer chunks (`ke_enc0/1/2`), tapped at layers 9 / 18 / 27. Contains the two device-only fixes: the concat-based `repeat_kv` and the head-expanded attention mask. |
| `vae_deploy_klein.py` | The VAE decoder (`kv_vae`), with `GroupNorm` → `ManualGroupNormND`. |
| `gen_prep_klein.py` | Runs the stock fp32 pipeline once: writes `ref_fp32.png` and every `.bin` the device needs (embeddings, mask, rotary tables, per-step `temb`, `dsigma`, initial latents, BN stats, the two gather maps). |
| `gen_verify_klein.py` | The device loop, on the host, driven **only** by the twelve `.tflite` graphs. If this matches `ref_fp32.png`, the Kotlin port is a transcription. |
| `generate_klein.py` | Quality gate: the stock pipeline with the fp32 transformer swapped for the int8 chunks. Isolates DiT quantization error from encoder quantization error. |
| `probe_enc_layer.py` | One encoder layer as an fp32 graph with taps at `q_rope`, `k_rope`, `key_rep`, raw logits, `probs`, `context`, layer output. The tool for "the device is wrong but nothing errors". |
| `check_ops.py` | Enumerates ops and flags the banned set (incl. `BROADCAST_TO`) and >4D tensors. |
| `probe_vae.py` | `ManualGroupNormND` and the `patch_groupnorm_nd` helper. |

Conversion needs roughly 40 GB of RAM (the fp32 pipeline plus a 912 MB export at a
time) and `litert-torch`, `diffusers`, `transformers`, `ai_edge_litert`.
