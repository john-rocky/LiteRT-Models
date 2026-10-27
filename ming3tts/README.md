# Ming-3.0-tts-0.5B on LiteRT CompiledModel (GPU + NPU)

Type text, and the phone speaks it in a 44.1 kHz voice generated on the device, with no network. [Ming-3.0-tts-0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B) (inclusionAI, MIT) runs here as LiteRT graphs: on a Galaxy S26 the language model and the aggregator run on the GPU at FP32, the flow head on the Qualcomm NPU and the decoder on the GPU, at 0.93× real time with the first audio after 1.87 s. A switch in the app runs every graph on the NPU instead: 1.35× real time and the first audio after 1.10 s, at lower precision ([below](#faster-placement-every-graph-on-the-npu)).

<a href="https://huggingface.co/litert-community/Ming-3.0-tts-0.5B/resolve/main/assets/demo.mp4"><img src="https://huggingface.co/litert-community/Ming-3.0-tts-0.5B/resolve/main/assets/demo.jpg" width="360" alt="Demo clip: the app on a Galaxy S26 in airplane mode reads a short story aloud on the NPU"></a> <br> *On a Galaxy S26 in airplane mode, with the switch on (every graph on the NPU), the app reads a 57-word story aloud. The app times it from the start of generation, just after the tap: the first audio chunk after 1.31 s, 17.0 s of speech generated in 14.7 s. Unlike the sentence in the speed tables below, the text is longer and a screen recorder was running. The clip has sound: click the picture to play it.*

| | |
|---|---|
| Model | [inclusionAI/Ming-3.0-tts-0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B) @ `8d66b70a`, code [inclusionAI/Ming-3.0-tts](https://github.com/inclusionAI/Ming-3.0-tts) @ `028c1eac` (MIT) |
| Converted | [litert-community/Ming-3.0-tts-0.5B](https://huggingface.co/litert-community/Ming-3.0-tts-0.5B) (graphs, host files, model card) |
| Language model | Qwen2.5-0.5B, 24 layers, KV cache as graph input and output |
| Flow head | DiT, 10 Euler solver steps with classifier-free guidance (scale 2.0) in one call |
| Decoder | 24-layer causal Qwen2 (attention window 64), 16 latents per call, ISTFT on the host |
| Rates | one 64-number latent per 80 ms · 44.1 kHz mono |
| Voices | the checkpoint's built-in voices (no voice file); 7 languages in the app |
| App | `com.ming3tts`: type text, pick language and voice, tap Speak; audio streams while decoding; a switch for the faster placement |

## Graphs and placement

| Graph | Signatures | With the Qualcomm NPU | Faster switch on | Without the NPU |
|---|---|---|---|---|
| `lm2_wfp16.tflite` | `prefill` (64 rows), `step` | GPU, FP32 | not used | GPU, FP32 |
| `lm2_epsdiv_prefill_wfp16.tflite`, `lm2_epsdiv_step_wfp16.tflite` | `prefill`, `step` (one per file) | not used | NPU | not used |
| `flow_r4b_s10_wfp16.tflite` | `flow` | NPU (compiled on the phone) | NPU | GPU, default precision |
| `agg_r4b_wfp16.tflite` | `agg` | GPU, FP32 | NPU | GPU, FP32 |
| `dec_c16_wfp16.tflite` | `dec` | GPU, default precision | NPU | GPU, default precision |

The host keeps the tokenizer, the prompt, the embedding lookup, RoPE tables, attention masks, both KV caches, the stop decision, the noise and the ISTFT (`MingEngine.kt`; `scripts/host_mirror.py` is the same in numpy). `MingGraphs.kt` is the default placement in one class, the model card's Kotlin example.

## Speed

| Phone | Where each graph runs | Real-time factor (audio ÷ generation time) | First audio | One 80 ms frame | Decoder call (16 latents) | Playback underruns |
|---|---|---|---|---|---|---|
| Galaxy S26 (Snapdragon SM8850, 12 GB) | LM and aggregator: GPU FP32 · flow: NPU · decoder: GPU | **0.93** (0.928 / 0.931 / 0.917) | 1.87 s | 78 ms (LM 31 + flow 37 + aggregator 10) | 111 ms | 6/6/6 |
| Pixel 8a (Tensor G3, Mali-G715, 8 GB) | no Qualcomm NPU: the flow moves to the GPU | 0.31 (0.312 / 0.307 / 0.308) | 5.2 s | 244 ms (94 + 127 + 23) | 264 ms | 7/7/7 |

The sample app's own loop with playback, the same English sentence (8.9 s of audio on the S26, 8.8 s on the Pixel 8a), one warm-up and 3 runs (median; the 3 values in parentheses). LiteRT 2.2.0, QAIRT 2.47.0 for the NPU. The first launch compiles the flow for the NPU (49–63 s over four first launches); later launches load the cached form in 0.35 s.

## Faster placement: every graph on the NPU

Where the app can use the Qualcomm NPU, the switch "Faster: run everything on the NPU" runs the language model, the flow head, the aggregator and the decoder on the NPU. The app remembers the choice; phones without a Qualcomm NPU, and APKs built without Qualcomm's libraries, do not show the switch.

| Placement on the Galaxy S26 | Real-time factor | First audio | One 80 ms frame | Decoder call (16 latents) | Playback underruns |
|---|---|---|---|---|---|
| default: LM and aggregator on the GPU at FP32, flow on the NPU, decoder on the GPU | 0.93 (0.928 / 0.931 / 0.917) | 1.87 s | 78 ms (LM 31 + flow 37 + aggregator 10) | 111 ms | 6/6/6 |
| faster: every graph on the NPU | **1.35** (1.358 / 1.332 / 1.347) | 1.10 s | 58 ms (LM 16 + flow 37 + aggregator 5) | 22 ms | 0/0/0 |

The NPU computes in fp16. Whole sentences stay within the audio limits (10 of 10: worst HNR change -0.83 dB, worst 4–12 kHz band change +1.22 dB, speaker cosine ≥ 0.966), but the language model (up to 8.0× its per-graph limit) and the aggregator (up to 1.39×) miss their per-graph limits. On the NPU the language model runs as `lm2_epsdiv_prefill_wfp16.tflite` and `lm2_epsdiv_step_wfp16.tflite` (one signature per file, one division in the normalization rewritten, the same in real arithmetic). The first switch compiles the four graphs the default placement does not run on the NPU, one at a time (63 s on the S26).

## Build and run

1. For the NPU, copy Qualcomm's runtime libraries into `app/src/main/jniLibs/arm64-v8a/` (not in git; the list is in `app/build.gradle.kts`): `libLiteRtDispatch_Qualcomm.so` and `libLiteRtCompilerPlugin_Qualcomm.so` from LiteRT's NPU runtime libraries (`litert_npu_runtime_libraries.zip`, `litert_npu_runtime_libraries_jit.zip` on the LiteRT GitHub releases) and the `libQnn*` files from QAIRT 2.47.0. Without them the app runs on the GPU and says so.
2. `./gradlew :app:installDebug`
3. Download the model files and copy them to the phone:
   ```sh
   hf download litert-community/Ming-3.0-tts-0.5B --local-dir Ming-3.0-tts-0.5B
   bash scripts/install_to_device.sh Ming-3.0-tts-0.5B
   ```
4. Open Ming TTS. Before step 3 it lists the missing files and how to copy them.

## Validation

| Check | Result |
|---|---|
| each graph on the S26 (default placement) against the float32 export on the Mac | all 5 signatures within their limits; every GPU node on the GPU, the flow as one NPU partition |
| whole sentences on the S26 against PyTorch fp32 (same noise) | 10 of 10 (6 English, 2 Japanese, 2 Chinese): worst HNR change -0.77 dB (limit ±1), worst 4–12 kHz band change +1.28 dB (limit ±1.5), speaker cosine ≥ 0.978, Whisper errors equal to PyTorch's |
| the same on a Pixel 8a (flow on the GPU) | 2 of 2 |
| the faster placement on the S26: whole sentences | 10 of 10 within the audio limits |
| the faster placement on the S26: each graph | 5 of 5 graphs on the NPU as one partition each; the language model (up to 8.0× its per-graph limit) and the aggregator (up to 1.39×) miss their per-graph limits |
| the app's switch on the S26 | turned on from the screen, it compiled the new graphs one at a time and spoke; a new launch kept it on and spoke; turned off, the default placement loaded and spoke; an APK without the Qualcomm libraries showed no switch, ran on the GPU and said so; with one of its files missing, the switch named the file and stayed off |
| the model card's Python example in a new environment | 8.7 s of audio in 9.7 s on an Apple M4 Max CPU; Whisper transcript = input text (WER 0.00) |
| the Python example with PyTorch's noise | the same 110 latents; waveform corr 0.9994 with PyTorch |
| the model card's Kotlin example on the S26 | E2, Z2 with the NPU: identical to the app, 2/2 within the limits; without the Qualcomm libraries: flow on the GPU, 2/2 within the limits; missing files: its install message |
| rebuilding the files from the checkpoint (REPRODUCE.md) | 14 of 14 files byte-identical; 204 s on an Apple M4 Max (macOS 27.0, Python 3.12.11) in a new environment (step 2 from a local copy of the same revision) |

## Conversion

`scripts/` holds the conversion as it ran: `ming_modules.py` and `r4_modules.py` rewrite the PyTorch modules for export (same arithmetic, caches and masks as inputs), `export_graphs.py` and `r4_export.py` export them with litert-torch and store the weights as float16, `write_assets.py` and `make_app_assets.py` write the host files, and `export_npu_lm.py` writes the faster placement's language model files. `requirements.txt` pins the versions. The model repository's REPRODUCE.md has the commands in order; the rebuilt files have the same SHA256 as the published ones.
