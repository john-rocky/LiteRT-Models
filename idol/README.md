# Idol: one photo to a 3D Gaussian splat on the phone

This Android app turns one photo of an object into a 3D Gaussian splat. The splat has 262,144 Gaussians and is saved
as a standard 3DGS `.ply`, so the object can be turned to the sides and back the photo never showed. The pipeline is
[speridlabs/idol](https://huggingface.co/speridlabs/idol) with its turbo LoRA (5 steps). It runs as 46 LiteRT graphs
through `CompiledModel`. Background removal (the author's BiRefNet) runs on the CPU and the other 45 graphs on the GPU
(ML Drift, OpenCL). The steps between graphs are Kotlin ports of `scripts/idol_runner.py`, `scripts/idol_glue.py` and
`scripts/idol_graphs.py`, in the same order and with the same float32 arithmetic.

Targets: Galaxy S26 (12 GB) and Pixel 11 Pro. arm64 only. The phone needs about 8.2 GB free to install the models.

## Get the model files

| Folder | Hugging Face repo | Files |
|---|---|---|
| `IDOL_DIR` | [litert-community/Idol](https://huggingface.co/litert-community/Idol) | `manifest.json`, every graph except `dino_*` and `dis`, `den_pos_emb.f32`, `gs_perturbation.f32`, `vae_bn.json`, `gs_rep.json` |
| `DINO_DIR` | [litert-community/DINOv3-ViT-H16plus-Idol-LiteRT](https://huggingface.co/litert-community/DINOv3-ViT-H16plus-Idol-LiteRT) (gated: accept the DINOv3 License on the Hub first) | `dino_0..3.tflite`, `dino_rope_cos.f32`, `dino_rope_sin.f32` |
| `DIS_DIR` (optional) | [litert-community/DIS-ISNet-LiteRT](https://huggingface.co/litert-community/DIS-ISNet-LiteRT) | `dis.tflite`, only for the GPU background removal setting |

```bash
hf download litert-community/Idol --local-dir ~/Downloads/Idol
hf download litert-community/DINOv3-ViT-H16plus-Idol-LiteRT --local-dir ~/Downloads/DINOv3-ViT-H16plus-Idol-LiteRT
hf download litert-community/DIS-ISNet-LiteRT --local-dir ~/Downloads/DIS-ISNet-LiteRT   # optional
```

The app needs 53 files, 7.2 GB in total. `manifest.json` lists each graph with its size, SHA-256, signatures,
accelerator and precision.

## Build and install

```bash
cd idol
./gradlew :app:installDebug
IDOL_DIR=~/Downloads/Idol DINO_DIR=~/Downloads/DINOv3-ViT-H16plus-Idol-LiteRT ./scripts/install_to_device.sh
```

`install_to_device.sh` reads the file list from `manifest.json` and checks every source file's size before it runs any
device command. It then copies one file at a time into the app's private `files/models/` (`adb push` to
`/data/local/tmp`, `run-as cp`, remove the temporary copy), so the phone never holds a second copy of more than one
file. After each copy it checks the SHA-256 on the phone, and a file the phone already holds with the right SHA-256
is not copied again. Without `IDOL_DIR` it uses this folder's `models/v2/`. It
needs the debug APK installed first (`run-as` only works with a debuggable app) and `python3` on the computer.
`ANDROID_SERIAL` selects the phone.

On its first start after an install the app checks every file's size and each graph's SHA-256 ("Checking model
files k/n"), then compiles the graphs.

## Use

1. **Home.** Choose "Photo" for a photograph or "Render or illustration" for a synthetic picture. The choice sets the
   crop margin around the object (1.7 or 1.2, the author's values). Then tap "Pick a photo" or "Take a photo".
2. **Run.** The screen shows the picture, then the cut-out the 3D model is made from, and a list of stages with their
   seconds: Removing background → Image features (DINOv3) → Image latent (VAE) → Generating the 3D shape (step k of
   5) → Placing points (level l of 8) → Decoding Gaussians → Saving the .ply. The total is shown at the end. Back
   stops the run before the next graph.
3. **Result.** A splat viewer turns the object a full turn every 12 s until you touch it. Drag to rotate, pinch to
   zoom. "Share .ply" sends the file. "New" goes back to Home. The last 5 results stay in the app's `files/results/`.

"Background removal on the GPU (IS-Net DIS)" at the bottom of Home switches the matting from BiRefNet (CPU, the
author's) to DIS (GPU, with the alpha floor of 8 from `scripts/idol_glue.py`). It is off by default and needs
`dis.tflite`.

### Messages

| When | Message |
|---|---|
| A model file is missing or has the wrong size or SHA-256 | "Model files are missing. Run scripts/install_to_device.sh (see README)." and the list of files |
| The picture has no object (all black, one flat colour, no alpha value of 128 or more) | "No object found in this picture." |
| Camera permission denied | "Camera permission is off. You can still pick a photo." |
| A graph does not fit in memory even after the others are closed | "Not enough memory for <graph> ... Close other apps and try again." |

## Memory and loading

The 45 GPU graphs do not all fit in the phone's memory at once (one denoiser block alone takes about 0.7 GB of GPU
memory). At start the app compiles them in the order a run uses them, until `MemAvailable` would drop below a reserve
of 2,500 MB ("Loading models k/45"). The rest are compiled during the run. When the next graph does not fit, the app
closes the resident graph whose next use is farthest away in the run's order, so the graphs used next stay resident.
The reserve is large because `MemAvailable` counts the page cache of the mapped graph files: with 1,000 MB, a compile
burst outran its reclaim and lmkd killed the app on the Galaxy S26 at about 1.9 GB available. After closing a graph
the app waits up to 1.5 s for its memory to show up (the Pixel 11 Pro returns GPU memory a few seconds late), and up
to 8 s before it gives up with "Not enough memory".
BiRefNet (930 MB, CPU) is compiled for each picture and closed right after the background removal, before the DINOv3
graphs run.

Each graph runs as its own GPU call. One long GPU call starves the phone's compositor, and the process gets killed
(`F0.md` finding 2). Outputs pass to the next graph through host memory (`readFloat` / `writeFloat`), and inputs that
did not change since the last write are not written again: the RoPE tables, the position embedding, the LoRA gate, the
conditioning of the current LoRA phase and the time modulation of the current step. The AutoRun plan key
`"handoff": "direct"` passes the output buffer itself to the next graph instead. That path is not verified on a phone
yet, and on the first error the app falls back to host copies and records why.

## Speed and memory

| Galaxy S26 | |
|---|---|
| Photo to `.ply` (generate: preprocessed picture → `.ply`, graphs loaded) | TODO(numbers) s |
| Background removal (BiRefNet, CPU) | TODO(numbers) s |
| DINOv3 / VAE / denoiser / octree / Gaussian decoder / PLY | TODO(numbers) s |
| Load and compile at start | TODO(numbers) s |
| Peak memory (VmHWM) / GPU pages | TODO(numbers) |

## AutoRun (debug builds)

AutoRun runs a plan of rows through the app without touching the screen. Its outputs are the files the contract
lines in `CONTRACT.md` are read from.

```bash
FIXTURES=1 ./scripts/install_to_device.sh       # also copies device/fx/<tag> (armchair, axolotl, succulent, black) and plans/*.json
adb shell am start -n com.idol/.MainActivity --es autorun plans/e2e_armchair.json --es out runs/e2e1
adb logcat -s IdolAutoRun                        # START, ROW, RUN, DONE lines
adb shell run-as com.idol cat files/runs/e2e1/DONE.json
adb exec-out run-as com.idol cat files/runs/e2e1/e2e_armchair.1.ply > e2e_armchair.1.ply
adb shell run-as com.idol touch files/runs/e2e1/STOP   # ends the plan after the current run
```

Paths are under the app's `files/`. `device/fx/` is gitignored; `scripts/make_fixtures.py` writes the fixtures.

| Plan | Rows |
|---|---|
| `plans/e2e_armchair.json` | `e2e`: the whole pipeline from the armchair fixture's raw picture with its noise arrays, 4 runs (run 0 is the warm-up), graphs preloaded |
| `plans/parity_armchair.json` | `parity`: each part on the fixture's reference input for it, outputs as `<id>.<name>.f32` |
| `plans/fail_black.json` | `image`: an all-black PNG, which should end with "No object found in this picture." |
| `plans/snippet.json` | `snippet`: the model card's Kotlin block (`CardSnippet.kt`) on the armchair picture |

Plan keys (all optional; a row can override `preload`): `preload` (default true), `reserve_mb` (2500), `mem_check`
(true), `handoff` (`host` or `direct`), `matting` (`bir` or `dis`; default from `manifest.json`), `cpu_threads` (6),
`gpu_priority` (`low`, `normal`, `high`; default unset), `dis_floor` (8).

Output files in the `out` folder:

- `header.json`: device, Android and LiteRT versions, memory, thermal status, the plan, and under `load` one record
  per preload: seconds per graph, `compile_s`, `total_s`, how many graphs fit, where it stopped.
- `e2e`: `<id>.<run>.ply` and `<id>.<run>.json` per run. The json has `warmup` (true for run 0), `stages` (matting,
  preprocess, dinov3, vae, denoiser, octree, gaussian_decoder, ply), `generate` (preprocessed pixels ready → `.ply`
  written), `total`, `thermal_start` / `thermal_end`, `vmhwm_kb` / `vmrss_kb` / `vmswap_kb` at the end of the run
  (`vmhwm_reset` says whether the peak was reset at its start), `gpu_page_alloc_max_bytes` (−1 when unreadable),
  `compiles_in_run` (graphs compiled during the run, with seconds), `memory_events` (compiles, closes, shortfalls),
  `box`, `handoff`. `<id>.alpha.u8` is the raw uint8 matting alpha at the resized picture's size, and `<id>.json`
  holds the median `generate` of runs 1 and up.
- `parity`: `<id>.<name>.f32` (little-endian float32): `feature1`, `vae_moments`, `vae_tokens`, `v_latent_<i>` /
  `v_camera_<i>` for steps 0 to 4, `oct_logits_<l>` for levels 1 to 8, `gs_features`, and the activated Gaussians
  `g_xyz`, `g_features_dc`, `g_scaling`, `g_rotation`, `g_opacity`. Also `<id>.alpha.u8` and `<id>.json` with the
  shapes.
- `image`: `<id>.json` with `status` and `message_on_screen`.
- `snippet`: `<id>.json` with `line`, also in `DONE.json` and logcat.
- `DONE.json`: one summary per row.

## Differences from the Python steps

- The app decodes pictures with Android's `BitmapFactory`; the reference uses Pillow. On a Mac, the armchair fixture
  (lossy WebP) decoded by libwebp's `dwebp` and run through the app's preprocessing code gives the reference's
  1024 × 1024 input exactly. Whether the phone's decoder gives the same pixels is not verified yet. The EXIF rotation
  is not applied, as in the author's pipeline. Palette and 16-bit images become 8-bit RGB(A) first; Pillow resizes a
  palette image with nearest-neighbour sampling instead.
- Three no-object guards the author's code does not have: a picture of one flat colour, an alpha with no value of
  128 or more, and a crop under one pixel.
- `exp`, `log`, `tanh`, `sin` and `cos` come from Java's math library, not torch's. Their results differ by 1–3 float32
  ulps in a fraction of the values, and the inverse sigmoid of opacities near 1 magnifies that up to 2e-4.
- App runs draw their random numbers with `java.util.Random` (seed 42); AutoRun uses the fixture's arrays, which are
  torch's draws.

## Licences

- App code: the repository's `LICENSE`.
- `app/src/main/java/com/idol/Pil.kt` ports parts of Pillow 13.0.0 (MIT-CMU License, `licenses/Pillow-MIT-CMU.txt`).
- The splat viewer in `app/src/main/assets/viewer/` is [antimatter15/splat](https://github.com/antimatter15/splat) at
  commit `ccb6eecf42d6a3aa13db0adbee92c7175849828b` (MIT License, Copyright (c) 2023 Kevin Kwok, `LICENSE` beside it).
  The changes are marked `idol:`: the contract's camera, a full-turn orbit, and the help text hidden.
- Model weights: `LICENSES.md`. The DINOv3 graphs are under the DINOv3 License; BiRefNet is MIT; the other graphs are
  Apache-2.0.
