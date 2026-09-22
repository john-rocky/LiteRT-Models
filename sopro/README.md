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
| Status | Mac CPU gates and Galaxy S26 Android measurements complete. Pixel 8a: pending |

## Graphs

Eight static graphs for the offline path plus a three-graph streaming vocoder. All I/O is float32
except token ids and gather indices (int32); batch 1. Signature names, tensor order and every static
size are in [contract.json](https://huggingface.co/litert-community/sopro-v2-turbo/blob/b1d2073b8111950ebf35614f4fd70b01ddaac3fb/contract.json).

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
- **Two findings for the toolchain**: with litert-torch 0.9.4 native PT2E per-axis int8,
  the documented `fold_quantize=False` path converts rank-3 Conv1d (4/4 small/real-shape,
  dynamic/static cases); `True` fails all tested ranks with `stablehlo.uniform_dequantize`
  (Conv1d 4/4, Conv2d/Linear controls 8/8). This is a folded-weight recipe failure, not a
  general rank-3 Conv1d limitation. On macOS 27.0 / Apple M4 Max, ai-edge-litert 2.2.0
  crashes at CompiledModel creation (SIGSEGV) for GPU-only, GPU+CPU, and GPU FP32 on a
  trivial Linear64+ReLU graph; CPU succeeds. The same graph/script with the 2.1.6 wheel
  completes all four cases. Both wheels ran on the same Python 3.12.13 interpreter, so the wheel version is the
  only variable.

**Original project**: [samuel-vitorino/sopro](https://github.com/samuel-vitorino/sopro) |
[Apache-2.0](https://huggingface.co/samuel-vitorino/sopro-v2-turbo)

## Android app

The standalone `com.sopro` app turns English, European Portuguese, French, or German text
into 24 kHz speech using a reference voice. It supports the bundled CC0 demo voice, audio-file
import, a ten-second recording, streaming playback, WAV saving, and sharing. Use a voice you
are allowed to clone; do not impersonate people. **Pixel 8a: pending.**

### App structure and on-device host steps

Compose Material 1 and MVVM separate `MainActivity`, immutable `UiState`, `MainViewModel`,
and `view/SoproScreen.kt`. The ViewModel owns the engine and confines model/DSP work to one
limited-parallelism dispatcher; blocking AudioTrack writes run on a separate playback thread.
The app requires arm64 Android API 26 or newer. Build pins are JDK 17, AGP 8.9.1,
Kotlin 2.2.21, compile/target SDK 35, and LiteRT 2.2.0.

Kotlin decodes and downmixes a reference, resamples it with a windowed-sinc kernel, normalizes
its level, and crops or pads it to ten seconds. It computes fp32 FFT speaker/Whisper/acoustic
mel front ends and caches the prepared reference. Short-reference padding has not received
the full reference-quality gate. SentencePiece adds the language tag; the host assembles the
style/text/prompt prefix, samples semantic tokens, updates packed KV rows, and builds one-hot
index inputs. FSQ digit-to-token indexing runs on the host. The two-step acoustic Euler solve
finishes for the whole utterance before streaming vocoder start/step/flush, reference-context
removal, and overlap-add iSTFT emit PCM chunks.

Playback applies gain and soft limiting without trim or fade. Saved WAVs apply the complete
offline post-processing contract: gain, lead/trail trim, soft limiting, and an 80 ms fade-out.
`Ready in N ms` includes model loading and GPU compilation; no warm-up inference runs.
TTFA is Synthesize → first PCM chunk handed to the sink, excluding AudioTrack hardware latency.
TTFA-to-onset adds the saved waveform's lead-cut duration.

### Assets and installation

[`scripts/assets.json`](scripts/assets.json) is the fixed runtime list: **20 files,
447,323,510 bytes**, including 11 model files, three contracts, and six host/reference assets.
It records SHA256 and byte count for every file. The app uses the R6 semantic/AR/acoustic
rewrites, R9 fp32 style, the published speaker graph, and the original published streaming
vocoder on CPU. The R9 rank-four vocoder probes and folded wfp16 style are excluded.
The three contract files are resolved newest first; the shipment's `contract_r9.json` overrides
only fp32 style. Gate fixtures use a separate optional manifest and are not distributed.

| Installed file | Bytes | HF asset path |
|---|---:|---|
| `contract.json` | 98,047 | [`contract.json`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/b1d2073b8111950ebf35614f4fd70b01ddaac3fb/contract.json) |
| `contract_r6.json` | 93,961 | [`contract_r6.json`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/main/contract_r6.json) |
| `contract_r9.json` | 2,383 | [`contract_r9.json`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/main/contract_r9.json) |
| `host_assets/ar_tables_fp32.bin` | 25,741,312 | [`host_assets/ar_tables_fp32.bin`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/main/host_assets/ar_tables_fp32.bin) |
| `host_assets/ar_tables_fp32.json` | 801 | [`host_assets/ar_tables_fp32.json`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/main/host_assets/ar_tables_fp32.json) |
| `host_assets/demo_voice.wav` | 480,044 | [`voice-donations/0a67.wav`](https://huggingface.co/kyutai/tts-voices/blob/323332d33f997de8394f24a193e1a76df720e01a/voice-donations/0a67.wav) |
| `host_assets/dsp_constants_fp32.bin` | 446,224 | [`host_assets/dsp_constants_fp32.bin`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/b1d2073b8111950ebf35614f4fd70b01ddaac3fb/host_assets/dsp_constants_fp32.bin) |
| `host_assets/host_assets.json` | 6,665 | [`host_assets/host_assets.json`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/b1d2073b8111950ebf35614f4fd70b01ddaac3fb/host_assets/host_assets.json) |
| `host_assets/tokenizer.model` | 370,821 | [`host_assets/tokenizer.model`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/b1d2073b8111950ebf35614f4fd70b01ddaac3fb/host_assets/tokenizer.model) |
| `r6/int8/sopro_ar_merged_i8native.tflite` | 55,249,192 | [`int8/sopro_ar_merged_r6_int8.tflite`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/main/int8/sopro_ar_merged_r6_int8.tflite) |
| `r6/wfp16/sopro_acoustic_condition_t4096_wfp16.tflite` | 50,820,160 | [`wfp16/sopro_acoustic_condition_t4096_r6_wfp16.tflite`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/main/wfp16/sopro_acoustic_condition_t4096_r6_wfp16.tflite) |
| `r6/wfp16/sopro_acoustic_condition_wfp16.tflite` | 50,820,160 | [`wfp16/sopro_acoustic_condition_r6_wfp16.tflite`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/main/wfp16/sopro_acoustic_condition_r6_wfp16.tflite) |
| `r6/wfp16/sopro_acoustic_velocity_t4096_wfp16.tflite` | 65,712,384 | [`wfp16/sopro_acoustic_velocity_t4096_r6_wfp16.tflite`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/main/wfp16/sopro_acoustic_velocity_t4096_r6_wfp16.tflite) |
| `r6/wfp16/sopro_acoustic_velocity_wfp16.tflite` | 64,655,616 | [`wfp16/sopro_acoustic_velocity_r6_wfp16.tflite`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/main/wfp16/sopro_acoustic_velocity_r6_wfp16.tflite) |
| `r6/wfp16/sopro_semantic_encoder_wfp16.tflite` | 41,371,840 | [`wfp16/sopro_semantic_encoder_r6_wfp16.tflite`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/main/wfp16/sopro_semantic_encoder_r6_wfp16.tflite) |
| `r9/fp32/sopro_style_prefix_fp32.tflite` | 3,193,116 | [`fp32/sopro_style_prefix_r9_fp32.tflite`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/main/fp32/sopro_style_prefix_r9_fp32.tflite) |
| `wfp16/sopro_speaker_encoder_wfp16.tflite` | 6,516,144 | [`wfp16/sopro_speaker_encoder_wfp16.tflite`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/b1d2073b8111950ebf35614f4fd70b01ddaac3fb/wfp16/sopro_speaker_encoder_wfp16.tflite) |
| `wfp16/sopro_vocoder_stream_flush_wfp16.tflite` | 27,246,640 | [`wfp16/sopro_vocoder_stream_flush_wfp16.tflite`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/b1d2073b8111950ebf35614f4fd70b01ddaac3fb/wfp16/sopro_vocoder_stream_flush_wfp16.tflite) |
| `wfp16/sopro_vocoder_stream_start_wfp16.tflite` | 27,250,512 | [`wfp16/sopro_vocoder_stream_start_wfp16.tflite`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/b1d2073b8111950ebf35614f4fd70b01ddaac3fb/wfp16/sopro_vocoder_stream_start_wfp16.tflite) |
| `wfp16/sopro_vocoder_stream_step_wfp16.tflite` | 27,247,488 | [`wfp16/sopro_vocoder_stream_step_wfp16.tflite`](https://huggingface.co/litert-community/sopro-v2-turbo/blob/b1d2073b8111950ebf35614f4fd70b01ddaac3fb/wfp16/sopro_vocoder_stream_step_wfp16.tflite) |

The asset paths point at
[litert-community/sopro-v2-turbo](https://huggingface.co/litert-community/sopro-v2-turbo),
except the CC0 voice's original repository. A null manifest revision means the file is staged
for publication; use a matching local asset directory until that revision is available.
For a published snapshot, `--download --revision COMMIT` fetches those new files; existing
published dependencies retain their pinned revisions. At most four downloads run at once,
and all file sizes and SHA256 values are verified before installation.

No models, WAVs, test fixtures, local signing keys, or build outputs are in this source tree.
From this directory, prepare the licensed default reference and build:

```sh
python -B scripts/install_assets.py --asset-dir /path/to/assets --validate-only
python -B scripts/install_assets.py --asset-dir /path/to/assets --prepare-bundled-assets
./gradlew --no-daemon :app:assembleDebug :app:assembleRelease
export ANDROID_SERIAL=DEVICE_SERIAL
scripts/install_to_device.sh --asset-dir /path/to/assets \
  --apk app/build/outputs/apk/debug/app-debug.apk \
  --release-apk app/build/outputs/apk/release/app-release.apk
```

`--validate-only` never calls adb. The installer uploads its own archive into
`/data/local/tmp`, then extracts it into private `files/` with `run-as com.sopro`; the debug
APK is required for this copy. The same-checkout release APK uses the same local signing key
and is installed afterward without deleting the assets. `ANDROID_SERIAL` is honored.
Output transfer runs in the opposite direction through binary-safe `adb exec-out run-as
com.sopro tar`, or the release app's external files directory and `adb pull`.
An app UID cannot copy output into `/data/local/tmp` on this Galaxy S26.

JVM gates require external test data via `SOPRO_FIXTURES`; private recordings and their arrays
are not part of the distribution. Conversion and exact re-export sources remain in `scripts/`.
The default hybrid and all-CPU choices use exactly the runtime list above. Optional GPU-AR
experiments need the additional R6 wfp16 AR artifact and are outside that install list.

### Galaxy S26 placement and measurements

The automatic Galaxy S26 map recognizes its SM8850 hardware/SoC identity. It uses GPU FP32
for both encoders, default GPU precision for condition and velocity in both acoustic buckets,
and CPU for native-int8 AR, fp32 style, and the streaming vocoder. Unknown devices fall back
to all CPU; Pixel 8a tuning and measurements are pending. The selected map persists, and a
`placement` launch extra can override it with graph-to-`cpu`/`gpu`/`gpu32` JSON.

**Measurement conditions:** Galaxy S26 (SM-S942Q / Adreno), LiteRT 2.2.0, release build,
screen on and exclusively held for the device measurements. Hybrid-suite battery temperature
was **38.9–41.1 °C**; first-tap session endpoints were **39.2–39.9 °C**. Release APK SHA256:
`171a98ff4d1f819cfa7718bf0f249d0743e457c54b26c6aef6ee2f7ad22ea9c5`.
The correctness group probes used debug builds; latency rows below use release only.
Warm synchronized ms is median `run + read`, excluding the first call of each graph/signature.

| Graph | Storage | Accelerator | Precision | Gate | Warm synchronized ms |
|---|---|---|---|---|---:|
| speaker_encoder | wfp16 | GPU | FP32 | PASS, GPU FP32 encoder gate | 16.348 |
| semantic_encoder | wfp16 | GPU | FP32 | PASS, documented digit near ties | 50.623 |
| style_prefix | fp32 | CPU | float compute | CPU PASS; GPU compile rejected | 1.049 |
| ar_merged / prefill | int8 | CPU | float compute | CPU ship chain PASS | 47.461 |
| ar_merged / step | int8 | CPU | float compute | CPU ship chain PASS | 9.478 |
| acoustic_condition (T2048) | wfp16 | GPU | default | PASS, mel-domain group gate | 32.530 |
| acoustic_velocity (T2048) | wfp16 | GPU | default | PASS, mel-domain group gate | 177.214 |
| vocoder_stream_start | wfp16 | CPU | float compute | CPU PASS; GPU waveform gate failed | 7.542 |
| vocoder_stream_step | wfp16 | CPU | float compute | CPU PASS; GPU waveform gate failed | 7.422 |
| vocoder_stream_flush | wfp16 | CPU | float compute | CPU PASS; GPU waveform gate failed | 4.064 |
| acoustic_condition_t4096 (T4096) | wfp16 | GPU | default | Compile passed; no long GPU chain measurement | not measured |
| acoustic_velocity_t4096 (T4096) | wfp16 | GPU | default | Compile passed; no long GPU chain measurement | not measured |

The short suite uses T2048; T4096 compiled at Ready but has no release warm-call or GPU
long-chain measurement. Encoder warm medians have two calls each because references are cached.
These R10 vocoder timings used the rank-four probe files on CPU. The shipped rank-three files
have exact Mac CPU feature/state parity (24 utterances, 251 calls per storage), but were not
re-timed on the phone after the file selection. The numbers above remain the measured R10 data.

The CPU placements follow measured limitations:

- **AR:** GPU FP32 passed 2,920/2,920 greedy predictions, with small nonzero tensor error;
  the R8 wfp16 GPU step took 22.221 ms synchronized versus 9.621 ms for CPU int8
  (9.478 ms in the R10 hybrid). Packed KV buffers cross the CPU/GPU boundary each step,
  so the shipped map keeps AR on CPU.
- **Style:** fp32 and folded-wfp16 probes were rejected at both GPU precisions. The runtime
  reported `BATCH_MATMUL: Not supported batched mat mul case: non-constant tensor`.
- **Vocoder:** rank-four wfp16 single-swap gates passed 0/24 at default and FP32 GPU precision,
  with raw waveform correlations spanning −0.037 to 0.061, worst HNR change 5.712 dB and
  worst 4–12 kHz energy change 8.781 dB. One-call fp32-storage probes also miscomputed at
  both precisions. The offline default-GPU probe failed too (raw correlation −0.042),
  so this is not isolated to streaming state I/O. The ConvNeXt vocoder remains on CPU;
  the exact runtime cause is unresolved.

| Galaxy S26 placement | Release APK prefix | TTFA ms, median [min, max] | RTF, median [min, max] | Quality | Battery °C, start → end |
|---|---|---:|---:|---|---:|
| All CPU (R8) | `0c977ebccda0` | 3344 [1766, 4064] | 0.641 [0.358, 0.778] | PASS, 36/36 | 40.0 → 42.9 |
| All passing GPU groups (R8) | `0c977ebccda0` | 4546 [3978, 11142] | 0.887 [0.841, 2.167] | PASS, 36/36 | 42.7 → 41.7 |
| Shipped hybrid (R10) | `171a98ff4d1f` | 2066 [1130, 3766] | 0.414 [0.247, 0.794] | PASS, 36/36 | 38.9 → 41.1 |

The R8 all-passing-GPU row includes GPU AR; style and vocoder stayed on CPU. The R8 rows
were measured before the hybrid placement, with release APK
`0c977ebccda0408e09e82f341a9f865ad2141901e426403baffe1d50d4f4cc6b`.
These are separate sessions and builds; the table makes no paired speedup claim.
Hybrid TTFA-to-onset median was 2,096 ms; total synthesis median was 2,164 ms.
Its 36 utterances include 33 prepared-reference cache hits. RTF is synthesis wall time divided
by saved audio duration, excluding loading/compilation at Ready.

| Galaxy S26 hybrid reference | Utterances | Mean WER | Speaker cosine mean / minimum | Gate |
|---|---:|---:|---:|---|
| ref1 | 12 | 0.56% | 0.939 / 0.916 | PASS |
| ref2 | 12 | 3.70% | 0.916 / 0.849 | PASS |
| CC0 demo voice | 12 | 0.00% | 0.777 / 0.691 | PASS |

Each reference uses the same twelve short sentences. WER is scored with Whisper turbo.
Private-reference gates use their matched oracle mean cosine minus 0.03, a minimum of 0.80,
and oracle WER plus three percentage points. The demo voice uses its own torch baseline:
mean cosine ≥ 0.751165, minimum ≥ 0.600910, WER ≤ 4.0185%.
All 36 outputs were finite, RMS > 0.001, and peak ≤ 1.

Cold-launch first taps used the same release build, demo voice and prefilled English sentence.
Ready includes all GPU compiles; each second tap is in the same process with cached reference
preparation. Tap time starts after Ready, not at app launch.

| Galaxy S26 placement / cold launch | Ready ms | First TTFA / total ms | Second TTFA / total ms |
|---|---:|---:|---:|
| hybrid / 1 | 3549 | 1769 / 1829 | 1334 / 1389 |
| hybrid / 2 | 3563 | 2929 / 3045 | 1636 / 1709 |
| cpu / 1 | 592 | 3056 / 3131 | 2390 / 2462 |

**Pixel 8a: pending.** No Pixel 8a quality or latency result is claimed.

### Default voice and licenses

The default reference is `voice-donations/0a67.wav`, a volunteer donation from
[kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices/tree/323332d33f997de8394f24a193e1a76df720e01a/voice-donations),
licensed CC0 1.0. Its SHA256 is
`4bd75d0ef0ad3f4e82ac075eab2a132651d2463f83bec210edeeccaf69294886`.
Attribution is retained in `app/src/main/assets/ATTRIBUTION.md` and `licenses/Demo-voice-CC0.md`.
The model, conversion/app code and listed Android dependencies retain their Apache-2.0
licenses and notices in `LICENSE`, `NOTICE`, and `licenses/`. Use only a reference voice you
are allowed to clone; do not impersonate people.
