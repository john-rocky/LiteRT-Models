# Nemotron-3-Diarization — streaming speaker diarization (LiteRT CompiledModel GPU)

NVIDIA's [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) (streaming Sortformer, 100M,
up to 8 speakers, OpenMDW-1.1) on Android: record or pick a clip and a per-speaker timeline grows every 0.72 s.
The 31-layer encoder and its output head run fully on the LiteRT CompiledModel GPU; the speaker cache and FIFO
(the streaming state) are Kotlin host code ported from transformers' `Nemotron3DiarizationSpeakerCache`.

```
16 kHz pcm --[Kotlin log-mel, 128 bins]--> mel [1,104,128] --[GPU graph A, FP32]--> 13 rows [1,13,512]
  --[pack: speaker cache | FIFO | chunk + look-ahead, zero rows to T=541]--> packed [1,541,512] + attn_bias + RoPE
  --[GPU graph B: 31 layers + head]--> logits [1,4328,8] --[host: sigmoid, mean of 8, cache / FIFO update]--> timeline
```

## On-device (Galaxy S26 — verified)

Galaxy S26 (SM8850, Adreno), Android 16, LiteRT 2.2.0: graph A 2 / 2 and graph B 2,915 / 2,915 nodes on
`LITERT_CL`, one partition each. The 97.6 s example clip pushed 0.1 s at a time at audio rate (a microphone's
pace), compared with transformers FP32 streaming on the same clip:

| graph B precision | latency per 0.72 s step, median / p95 | RTF | agreement @ 0.5 | segments (reference 37) | compile A / B |
| --- | --- | --- | --- | --- | --- |
| FP32 (app default) | 170.7 / 175.5 ms | 0.238 | 100 % | identical | 72 / 1,477 ms |
| FP16 (GPU default) | 115.1 / 120.0 ms | 0.160 | 99.973 % (21 flips) | 40 (2 added, 1 split, 8 moved 10 ms) | 73 / 1,434 ms |
| FP16 + FP32 accumulation | 140.3 / 144.7 ms | 0.196 | 99.997 % (2 flips) | 38 (1 split, 1 moved 10 ms) | 71 / 1,382 ms |

- Graph A always runs with GPU precision FP32 (its rows stay in the cache for the whole session). FP32 is the app
  default: the only setting whose cache contents and segments equal the reference.
- Offline file mode (`Nemotron3Diarizer.runFile`, graph B T=684): the same 97.6 s clip in 0.99–1.00 s (FP32,
  29 / 29 segments identical to transformers' offline forward) or 0.64–0.65 s (FP16, 7 boundaries moved by 10 ms).
- Back to back (a file through the streaming path as fast as it runs) the GPU thermal governor caps the clock after
  about 8 s (1300 → 500 MHz) and graph B goes 135 → 288 ms (first / last 10 steps); at audio rate it stays at
  136 ms.
- Conditions: screen on, unlocked, USB power, battery 96–98 %, 35.8–37.9 °C, thermal status 0 at every start.
  Details and raw numbers: the HF card, `scripts/` reports.

## Build & run

```bash
cd nemotron3diar
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir> nemotron3_diar_frontend.tflite nemotron3_diar_encoder_low_latency_fp16.tflite
```

The .tflite files are on Hugging Face (`litert-community/Nemotron-3-Diarization-LiteRT`), 2.1 MB + 198.7 MB; add
`nemotron3_diar_encoder_offline_fp16.tflite` for the offline file mode. The first launch fails with "model not
found" until they are pushed. The mel filter bank, window and silence embedding ship as APK assets.

The app: **Record** (microphone, up to 5 min, Stop ends the stream) or **Pick clip** (any audio / video, decoded to
16 kHz mono, then the same streaming loop), graph B precision switch (FP32 / FP16 / FP16 + FP32 acc), per-step ms,
real-time factor, time to the first chunk, and a talk-time summary per speaker at the end.

## Scripts (in order)

All take `--run-dir <dir>` (with `fixtures/`, `results/`, `exports/`); `--model-dir` = the upstream checkpoint
directory (config.json, processor_config.json, model.safetensors).

| # | script | what |
| --- | --- | --- |
| 1 | `make_reference.py` | transformers FP32 reference, offline + low_latency streaming, per-step capture (`--offline-capture` for offline chunks) |
| 2 | `nemotron3diar_model.py`, `gate_reauthor.py` | the re-authored PyTorch model (fixed T, safe LayerNorm, row mask) vs the reference, negative controls |
| 3 | `build_nemotron3diar.py` | litert-torch export, float32 + float16-weight files (`--mode low_latency|offline`, `--ln plain|safe`) |
| 4 | `gate_tflite_cpu.py`, `tap_ln.py` | exported files on the Mac CPU vs the reference, op census; LayerNorm input magnitudes |
| 5 | `extract_frontend.py` | mel / window tables, numpy log-mel vs the processor |
| 6 | `nemotron3_diar_litert.py`, `gate_reference_impl.py` | the host as an importable Python reference (CompiledModel API) and its gate on the shipped files |
| 7 | `host_loop.py` | diagnostic streaming loop with torch / tflite backends |
| 8 | `gate_kotlin_parity.py`, `jvm/KotlinParity.kt` | the app's Kotlin host on the Mac JVM with replayed graph outputs |
| 9 | `install_to_device.sh`, `gate_device.py`, `gate_closed_loop.py`, `gate_runfile_device.py` | device: single steps, closed loop, offline file mode + GPU clock trace |
| 10 | `make_hero_app.py` | the card image from a real run |

## Device self-test

The app runs a device check instead of the demo when `files/selftest.json` exists; it renames the file to
`selftest.done` first, so a relaunch from recents never repeats it. `{"mode": "closed_loop", "wav":
"wav/<clip>.wav", "b_precision": "fp32|default|accum", "realtime": true|false, "file_mode": true|false}` runs
`ClosedLoopTest` (streaming or offline file mode) and writes per-step logits, `steps.json`, `timing.json` and
`status.txt` to `/sdcard/Android/data/com.nemotron3diar/files/n3d/`; the form with `"variants"` runs `SelfTest`
(single steps from pushed inputs). The `gate_*.py` scripts push the request, start the app, read the log by process
id, pull the results and compare with transformers. `am start -n com.nemotron3diar/.MainActivity --es clip
<file under files/>` runs the Pick path on a pushed file; `--ei record_seconds N` records N seconds.
