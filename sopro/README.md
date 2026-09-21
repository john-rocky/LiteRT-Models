# Sopro v2 turbo on LiteRT CompiledModel (classic `.tflite`)

[Sopro v2 turbo](https://huggingface.co/samuel-vitorino/sopro-v2-turbo) (Samuel Vitorino, Apache-2.0)
is a zero-shot voice-cloning TTS for English, European Portuguese, French and German: a 10 s reference
clip plus text gives 24 kHz speech in that voice. This module holds the conversion and verification
code; the converted graphs and the full model card live on Hugging Face.

| | |
|---|---|
| Upstream | [samuel-vitorino/sopro](https://github.com/samuel-vitorino/sopro), `sopro==2.2.0`, HF revision `f747f9ed` |
| Converted | [litert-community/sopro-v2-turbo](https://huggingface.co/litert-community/sopro-v2-turbo) |
| Params | 121.6M core (AR LM + acoustic DiT) + 36M (speaker encoder, semantic encoder, vocoder) |
| Pipeline | reference → speaker encoder + semantic encoder (16 kHz) + reference mel → 12-layer **semantic AR LM** (FSQ tokens, 1 token = 1024 samples) → 8-layer **flow-matching DiT** (2 Euler steps, sway grid) → **Vocos** vocoder (causal ConvNeXt → log-mag/phase → host iSTFT) |
| Status | Mac CPU conversion + gates done (2026-09-21). Android app and on-device numbers: next run |

## Graphs

Eight static graphs for the offline path plus a three-graph streaming vocoder. All I/O is float32
except token ids and gather indices (int32); batch 1. Signature names, tensor order and every static
size are in [contract.json](contract.json).

| graph | inputs → outputs | bucket |
|---|---|---|
| `sopro_speaker_encoder` | speaker log-mel [1,80,1001] → id_emb [1,192], style_emb [1,128], style_ctrl [1,8], cond_vec [1,512] | 10 s reference |
| `sopro_semantic_encoder` | Whisper log-mel [1,80,1002] → tokens [1,235] int32 | 10 s reference |
| `sopro_style_prefix` | 160 token embeddings [1,160,512] → 8 prefix vectors [1,8,512] | fixed |
| `sopro_ar_prefill` | prefix emb [1,256,512] + additive bias [1,1,256,256] + last_index [1] → logits [1,4377] + packed k,v [1,96,256,64] | P_MAX 256 |
| `sopro_ar_step` | token emb [1,1,512] + cos/sin [1,1,1,64] + bias [1,1,1,1024] + pk/pv [1,96,1024,64] → logits + new k,v [1,96,1,64] | CAP 1024 (≤ 704 generated tokens) |
| `sopro_ar_merged` | the two above as `prefill` / `step` signatures of one file, weights shared (223 MB fp32 instead of 421) | |
| `sopro_acoustic_condition` | tokens [1,N] + token_mask [1,1,N] + frame_to_token [T] → mu [1,100,T] | (T,N) = (2048,512) or (4096,1024) |
| `sopro_acoustic_velocity` | x, mu, cond_mel [1,100,T] + t [1] + cond_vec [1,512] + cond_mask [1,1,T] + key_bias [1,1,1,T] → velocity [1,100,T] | one Euler step per call |
| `sopro_vocoder` | denormalized mel [1,100,1024] + frame_mask [1,1,1024] → iSTFT features [1,1024,1026] | offline, ≤ 10.9 s |
| `sopro_vocoder_stream_{start,step,flush}` | mel chunk [1,100,64] (+ conv states) → features [1,37/64/27,1026] (+ states) | any length, exact vs offline |

Host side (NumPy here, Kotlin in the app): resampling, the three mel front-ends (speaker: slaney mel +
per-frame LayerNorm; Whisper: 400-sample zero append, log10 floor, max−8 clamp; Vocos: log-mel for the
reference), the sentencepiece tokenizer + language tag, embedding tables (`text_tok_emb` [8192,512] and
`sem_in_proj∘semantic_tok_emb` folded to [4377,512]), RoPE tables, additive masks, the packed KV cache
(the step writes only its new slice), sampling (temperature 0.8 / top-k 25 / top-p 0.9, BOS masked, EOS
allowed from step 10), the 2-step Euler loop with prompt re-masking, exp/clamp + iSTFT (n_fft 1024, hop
256, periodic Hann), and the upstream post-processing chain (gain, trim, soft limit, fades). The
upstream `crop_on_pause` is bypassed (it appends random room tone); the reference is exactly 10 s.

## Build and verify

`scripts/` is the portable copy of the conversion code; the ordered commands (venv with the exact pins,
source download with sha256 checks, oracle from your own reference clips, exports, wfp16 / int8, gates)
are in the [REPRODUCE.md](https://huggingface.co/litert-community/sopro-v2-turbo/blob/main/REPRODUCE.md)
of the HF repo. Pins: torch 2.11.0 / torchaudio 2.11.0 / litert-torch 0.9.4 / ai-edge-litert 2.2.0 /
ai-edge-quantizer 0.9.0 / sopro 2.2.0, Python 3.12. No reference audio is bundled: supply clips ≥ 10 s
that you are allowed to clone.

## Results (Mac M4 Max CPU, 4 threads, ai-edge-litert 2.2.0, contended)

Two private reference voices × 12 sentences (6 en / 2 pt / 2 fr / 2 de) + 4 long (16–18 s) utterances.

| set | gate | result |
|---|---|---|
| fp32 | per-graph vs PyTorch | speaker id max err 3e-7; semantic tokens 235/235 ×2; AR greedy replay 2920/2920 (logit max err 3.3e-5); solved mel max err 1.5e-4 |
| fp32 | teacher-forced chain (oracle tokens + noise) | raw waveform corr ≥ 0.99999, final trim points exact, 24/24; long 4/4 |
| fp32 | streaming vocoder vs offline decode | max err 3.9e-4, 24/24 |
| fp32 | free-running (own sampler, same seed as the PyTorch logits) | identical token sequences 24/24; WER 2.06 % (Whisper turbo) vs 1.11 % for the PyTorch pipeline; speaker cosine 0.922 vs 0.926 |
| wfp16 + int8 AR (ship set) | domain gates | acoustic solved-mel corr ≥ 0.9999; vocoder waveform corr ≥ 0.9999 (single swap); chain log-mel corr ≥ 0.9999; HNR Δ ≤ 0.011 dB, 4–12 kHz band Δ ≤ 0.022 dB |
| wfp16 + int8 AR | free-running quality | WER 1.26 %, speaker cosine 0.926 (min 0.853); RTF 0.44 median (fp32 set: 0.34) |
| int8 AR (native PT2E) | oracle-path greedy replay | 2834/2920 (97.1 %), 55 MB vs 223 MB fp32 |

## What the conversion taught

- **Reduced-precision gates by domain.** Judge acoustic (flow-matching) graphs in mel space and vocoder
  graphs in waveform space. fp16 weights in the DiT turn into phase differences through the 2-step ODE:
  one utterance dropped to raw-waveform corr 0.977 while its log-mel corr stayed 0.9999 and WER / speaker
  cosine did not move. Absolute tensor rules on the vocoder's phase channel (absmax ≈ 125) are meaningless:
  a 0.6 rad error at a near-silent bin does not change the waveform (corr 0.99994).
- **Exact right-padding for causal conv stacks**: multiply the residual by the frame mask after the embed
  LayerNorm and after every ConvNeXt block (not only after the convolution — the LayerNorm bias would
  populate padded frames). fp64 padded-vs-unpadded difference 1e-13.
- **Host mel mirrors must use an fp32 FFT.** NumPy's default `rfft` dispatches to fp64 and drifts 1e-3 in
  quiet high bands against torch's fp32 STFT (the log amplifies it); `rfft(..., norm="forward") * 1024`
  selects the fp32 path bit-exactly. A Kotlin FFT needs its own parity check.
- **Runtime-input indices** for the token→frame upsampler (`frame_to_token`) and key-only additive biases
  `[1,1,1,T]` make one static graph exact for any valid length; the constant cache aliasing bug
  (litert-torch #1061) is avoided by cloning every parameter contiguous before export (alias probe 0).
- **Two findings for the toolchain**: litert-torch 0.9.4 native PT2E per-axis int8 on a rank-3 Conv1d
  weight (512×100×7) fails legalization (`stablehlo.uniform_dequantize`), so there is no native-int8
  vocoder; and macOS ai-edge-litert 2.2.0 GPU-only `CompiledModel` (Metal) crashes on a trivial
  Linear+ReLU graph, so no Mac GPU numbers exist.

**Original project**: [samuel-vitorino/sopro](https://github.com/samuel-vitorino/sopro) |
[Apache-2.0](https://huggingface.co/samuel-vitorino/sopro-v2-turbo)
