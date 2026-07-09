# VibeVoice-Realtime-0.5B on LiteRT CompiledModel (hybrid GPU/CPU)

Streaming, autoregressive **next-token-diffusion** text-to-speech running on-device with LiteRT
`CompiledModel`. The model is four LiteRT graphs — two Qwen2 transformer LMs (with KV cache), a
diffusion head, and a convolutional σ-VAE acoustic decoder — driven by host-side orchestration,
with **no FFT anywhere**. This is the first streaming AR-diffusion TTS and the first real-attention
autoregressive decoder with an on-device KV cache in this zoo.

| | |
|---|---|
| Model | [VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) (Microsoft, MIT) |
| Backbone | Qwen2.5-0.5B, split 4-layer text LM + 20-layer TTS LM |
| Acoustic | σ-VAE tokenizer (conv, 3200× / 7.5 Hz) + 4-layer DDPM diffusion head |
| Sample rate | 24 kHz mono |
| App | `com.vibevoice` — text in, audio out |

## Device placement (Pixel 8a / Mali ML Drift)

Each graph runs on whichever accelerator produces correct output on the device — verified on-device,
not just by desktop parity:

| Graph | Runs on | Why |
|---|---|---|
| base / TTS LM | **CPU (fp32)** | Mali rejects the KV-step `FULLY_CONNECTED` weights shape; and fp16 collapses the 20-layer stack to noise on ARM XNNPACK — so the LMs ship as **fp32** graphs on CPU. |
| diffusion head | **GPU (fp32 precision)** | Small, compiles and computes correctly on ML Drift. |
| σ-VAE decoder | **CPU (fp32)** | Compiles on GPU but ML Drift **miscomputes** it — see below. |

**The σ-VAE decoder is a confirmed ML Drift correctness bug, not a conversion problem.** On-device
probing with **single-output** sub-graphs (immune to the delegate's known output-buffer aliasing, so
any divergence is a genuine miscompute) shows every individual op of a ConvNeXt block — conv, RMSNorm,
depthwise conv, FFN `linear1 → tanh-GELU → linear2` — is *bit-exact* on GPU, yet the *assembled* block
miscomputes. The divergence first appears at the LayerScale broadcast-multiply that closes the block,
though the same multiply is correct earlier in the block, so the trigger is graph assembly/depth, not
any one op. It reproduces identically on OpenCL and OpenGL and under every buffer-storage / precision /
constant-sharing option, so it is a graph-assembly buffer/scheduling bug in ML Drift's shared
compilation layer — not precision (it fails at fp32), not a backend, not a kernel. Splitting the
decoder does not help (a single block already trips it), so the decoder runs on CPU, where it is
bit-exact with the reference. (A minimal reproducer has been kept for the LiteRT team.)

## Pipeline

```
text --BPE(host)--> token ids
     --host: per text token-->
         embed_tokens(host lookup) -> base_lm_kv(CPU) -> +type_embed -> tts_lm_kv(CPU) -> cond
     --host: per speech token (6 per text window)-->
         5-step DPM-Solver++ loop:
             diffusion_head(GPU) x2 (cond + null prompt) -> CFG blend -> v -> latent[64]
         acoustic_connector(host) + type_embed -> tts_lm_kv(CPU)  -> next cond
                                                -> neg tts_lm_kv(CPU) -> next null cond
         eos_classifier(host) > 0.5 -> stop
     --host: accumulate latents-->  acoustic_decoder(CPU) -> waveform[N*3200] @ 24 kHz
```

The two LMs keep their KV cache **host-side**: a packed `[1, L*nkv, Pmax, 64]` key/value tensor is
fed into the step graph and the new column read back out each token (the ML-Drift-safe "state as
graph I/O" pattern — Mali silently corrupts in-graph stateful cache). The **voice** is a precomputed
prompt KV cache (`voice_*.bin`); swapping voices swaps that file.

## Graphs (loaded from filesDir)

- **`vv_base_lm_kv_fp32.tflite`** (239 MB) — 4-layer Qwen2 text LM, one AR step. in: `x[1,1,896]`, `cos[1,1,1,64]`, `sin[1,1,1,64]`, `mask[1,1,1,129]`, `pk[1,8,128,64]`, `pv[1,8,128,64]`; out: `hidden[1,1,896]`, `k[1,8,1,64]`, `v[1,8,1,64]`.
- **`vv_tts_lm_kv_fp32.tflite`** (1193 MB) — 20-layer Qwen2 TTS LM, one AR step. Same I/O with `Pmax=384`, `L*nkv=40`.
- **`vv_diffhead_fp16.tflite`** (84 MB) — DDPM head, one denoise step (GPU, fp32 precision). in: `noisy[1,64]`, `t_freq[1,256]`, `cond[1,896]`; out: `v[1,64]`.
- **`vv_decoder_fp32.tflite`** (1378 MB) — σ-VAE conv decoder. in: `latent[1,64,128]`; out: `wav[1,1,409600]`.

Non-graph assets pushed alongside: `embed_tokens.f16` (272 MB, mmapped fp16 token embeddings),
`glue.f32` (6.5 MB, connector + type-embed + EOS weights), `voice_en-Emma_woman.bin` (2.7 MB).

## GPU re-authoring (what makes the graphs ML-Drift-clean)

The graphs are re-authored to compile under `CompiledModel` (all four *compile* on GPU; only the
decoder is moved to CPU because ML Drift miscomputes it at runtime, above):

- **Token embedding is `GATHER`** (banned) → looked up on the host from the mmapped `embed_tokens.f16`; the LMs take `inputs_embeds`.
- **Autoregressive KV cache** → packed into a 4D `[1, L*nkv, Pmax, 64]` tensor (all layers on dim 1 to stay ≤4D); the current token's key/value is **concatenated at the tail** and an additive mask blanks the padding slots, so there is no in-graph scatter. Keys are stored post-RoPE.
- **`scaled_dot_product_attention`** → manual matmul + softmax; GQA (14 Q / 2 KV heads) expanded by `cat` (no `BROADCAST_TO`).
- **RoPE** cos/sin computed on the host and fed per step (position-dependent → not a constant).
- **RMSNorm** → max-normalized safe form (fp16 sum-of-squares overflow guard).
- **σ-VAE decoder `ConvTranspose1d`** → `ZeroStuffConvT1d` (nearest-upsample × zero-stuff mask + flipped conv), avoiding the Mali-rejected `TRANSPOSE_CONV`; ConvNeXt GELU → tanh-GELU.
- **Diffusion head** sinusoidal timestep embedding computed on the host (fed as `t_freq`); `chunk` → slicing (no `SPLIT`).

See `../docs/LITERT_CONVERSION_GUIDE.md` for the full op-fix catalog.

## Build & run

```bash
cd vibevoice/
./gradlew :app:installDebug
# build the graphs + export assets (scripts/), then push them to the device:
./scripts/install_to_device.sh <dir-with-the-tflites-and-assets>
```

The first launch fails with "Model not found" until `install_to_device.sh` has been run.

## Notes

- **Stochastic, autoregressive**: each frame draws fresh diffusion noise, so two runs differ.
  Quality is judged by ear on-device; a short sentence generates in ~30 s on a Pixel 8a (the two
  fp32 LMs and the fp32 decoder are the cost — this is a correctness-first placement, not a
  real-time one, given the ML Drift decoder bug forces CPU).
- **Voice presets** are the reference model's precomputed prompt KV caches; this app ships the
  `en-Emma_woman` voice. The realtime checkpoint is decoder-only (no acoustic encoder), so voices
  cannot be cloned on-device — they are exported offline.
- Single speaker, English; ~10 min max context.
