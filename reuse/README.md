# RE-USE speech restoration on LiteRT (CompiledModel GPU)

An Android sample that restores noisy, reverberant, clipped, band-limited or codec-damaged 16 kHz speech with
NVIDIA NemotronLabs RE-USE ([nvidia/NVIDIA-NemotronLabs-RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE) @31f82f2, OpenMDW-1.1) running as a
LiteRT `CompiledModel` on the phone's GPU.

The model runs as one fixed-length chunk graph. The STFT, the inverse STFT and the overlap-add of the chunks run on
the host in Kotlin (`Spectral.kt`, `SpeechRestorer.kt`), as in NVIDIA's `inference_chunk.py`.

## Model files (not in git)

The `.tflite` files are not committed. Download one or both from
[litert-community/NVIDIA-NemotronLabs-RE-USE-LiteRT](https://huggingface.co/litert-community/NVIDIA-NemotronLabs-RE-USE-LiteRT) and put them in `app/src/main/assets/` before you build:

| file | bytes | ops | use |
|---|---|---|---|
| `reuse_16k_2s_fp32.tflite` | 54,155,412 | 91,014 | main: start with this one |
| `reuse_16k_2s_fastload_fp32.tflite` | 44,289,500 | 33,114 | fastload: loads several times faster and uses less memory, on a GPU each chunk takes longer; for phones with less memory |

Both take 2 s chunks, one every 1.75 s, with the same input and output. The app lists the
`reuse_*.tflite` files in its assets in reverse name order, so with both files the main file comes first and is the
default (the list also leaves out a system `.tflite` that a Galaxy S26 adds to every app's asset list); with none it
says so and does not crash (see When something is missing). `SpeechRestorer` takes the chunk length from the graph's input shape, so either file works
without code changes (both ran on a Pixel 11 Pro with the commands below, 2026-11-03).

In both files 2 dilated convolutions of the encoder run in 2 tiles each: without that,
LiteRT 2.2.0 on Android stops before the GPU takes the graph with `Dilated im2col buffer size overflowed`.

## Input and output

- 16 kHz mono. The app decodes WAV itself and other formats with `MediaCodec`, and resamples to 16 kHz.
- STFT: n_fft 640, hop 80, periodic Hann window 640, centered with reflect padding, 321 bins. A
  2 s chunk is 401 frames.
- Graph input `args_0` `[1, 2, frames, 321]`: `log1p(|X|)` and `angle(X)`, frames on the third axis. Signature
  `serving_default`.
- Graph output `output_0` `[1, 3, frames, 321]`: the compressed magnitude `m` (`log1p` of the magnitude) and the phase as `(x_r, x_i)`. The host takes
  `expm1(relu(m))` as the magnitude, normalizes `(x_r, x_i)` to unit length, zeroes the frames where more than half of
  the bins are zero, and runs the inverse STFT.
- Chunks of 2 s every 1.75 s, the last one aligned to the end of the clip. Each chunk is weighted
  by a Hann window (at least 1e-7) and the sum is divided by the sum of the windows.

## Build and run

```bash
./gradlew :app:assembleDebug        # LiteRT 2.2.0, minSdk 26, arm64-v8a
./gradlew :app:installDebug
```

Open the app, pick a model and an accelerator (GPU or CPU), pick an audio file (up to one minute), then play the original and the
restored audio.

Headless mode (for measurements, one model load for all files) takes 16 kHz WAV files from the app's files dir
and writes `files/out/<tag>/<clip>.wav` plus `files/out/<tag>/result.json` (per-chunk times, seconds per second of audio, errors).
Its extras:

- `--es input <a.wav,b.wav>`: paths relative to the app's files dir.
- `--es model <asset>`: default = the first in the list above (the main file when both are there).
- `--es accel gpu|gpu_fp32|cpu`: default `gpu`; `gpu_fp32` runs the GPU in float32, `--ei threads <n>` sets the CPU
  threads.
- `--es hop <step>`: the step between chunks as a fraction of the chunk; default 0.875 for 2 s chunks.
- `--ez gpu_cache true`: LiteRT's GPU program cache in `files/gpu_cache` (off by default; see Measured);
  `--ez clear_gpu_cache true` deletes it first.
- `--es out <tag>`: the output dir under `files/out/`.

Put a 16 kHz WAV into the files dir of the debug build, then run. `MainActivity` reads the extras when it is
created, so stop the app before each run:

```bash
adb push clip.wav /data/local/tmp/clip.wav
adb shell chmod 644 /data/local/tmp/clip.wav
adb shell run-as com.reuse mkdir -p files/in
adb shell run-as com.reuse cp /data/local/tmp/clip.wav files/in/clip.wav
adb shell rm /data/local/tmp/clip.wav
adb shell am force-stop com.reuse
adb shell am start -W -n com.reuse/.MainActivity --es input in/clip.wav \
  --es model reuse_16k_2s_fp32.tflite --es accel gpu --ei threads 4 --es out run1
```

The run is done when `result.json` is there; then copy the restored audio to the computer:

```bash
adb shell run-as com.reuse cat files/out/run1/result.json
adb exec-out run-as com.reuse cat files/out/run1/clip.wav > run1_clip.wav
```

## Measured

The app restoring the 10 clips of the model card's quality set back to back with `reuse_16k_2s_fp32.tflite` (2026-11-03):
graph and host per chunk (medians over the 30 chunks after each clip's first), the time to restore one second
of audio, and the load. The run used `--ez gpu_cache true`: the first load wrote a 166.5 MB cache on the
Galaxy S26 (167.4 MB on the Pixel 11 Pro), and the second load, in a new process, could not read it
back (`Deserialization failed: DATA_LOSS`) and compiled the graph again.

| phone | backend | graph (ms) | host (ms) | per audio second (s) | load (s) | load again (s) |
|---|---|---|---|---|---|---|
| Galaxy S26 | GPU | 4,225.6 | 59.17 | 2.448 | 80.0 | 97.5 |
| Pixel 11 Pro | GPU | 6,729.4 | 116.97 | 3.912 | 73.9 | 160.9 |

The commands under Build and run, as printed, also ran on a Pixel 11 Pro (2026-11-03): the main file's output
was byte-identical to that phone's run in the table, the fastload file loaded in 10.3 s against
73.8 s and its output was 46.73 dB SI-SDR from the main file's, and without `--es model` the
app took the main file.

Slower than real time: restore recordings, not a live call. The graph alone on three phones (both files), the
quality scores and the Mac numbers are on the [model card](https://huggingface.co/litert-community/NVIDIA-NemotronLabs-RE-USE-LiteRT).

## When something is missing

- No model in the assets: the screen shows "Failed: No model in this app: put the RE-USE .tflite in app/src/main/assets/
  and rebuild (see README).", and a headless run writes "Model not found: assets/<name> is not in this app (models: none).
  Put the RE-USE .tflite in app/src/main/assets/ and rebuild (see README)." to `result.json` (both seen on a
  Pixel 11 Pro, 2026-11-03; the app stays open).
- A model name that is not there: "Model not found: assets/<name> is not in this app (models: …). Put the RE-USE .tflite
  in app/src/main/assets/ and rebuild (see README)."
- No GPU accelerator: "GPU unavailable: LiteRT found no GPU accelerator on this phone (available: …). Choose CPU to run
  the model there, much slower."

## License

The `.tflite` files are under OpenMDW-1.1, the checkpoint's license: keep the `LICENSE` and `NOTICE` of the Hugging
Face repo with them. The Mamba scans in the graphs follow the reference scan of mamba_ssm (Apache-2.0).
