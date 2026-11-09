# Idol: one photo to a 3D Gaussian splat on the phone

This Android app turns one photo of an object into a 3D Gaussian splat, saved as a standard 3DGS `.ply`, so the
object can be turned to the sides and back the photo never showed. The pipeline is
[speridlabs/idol](https://huggingface.co/speridlabs/idol) with its turbo LoRA (5 steps). It runs as LiteRT graphs
through `CompiledModel`, in either of the two forms that
[litert-community/Idol](https://huggingface.co/litert-community/Idol) holds:

- **The default form** (the repo's root): 65,536 Gaussians per splat, 20 graphs. The denoiser's 22 middle blocks share
  one graph and the Gaussian decoder's six middle graphs share another; before each call the app copies that block's
  weights from a side file into the graph's weight inputs. GPU precision FP16.
- **The first release's form** (`262k/`): 262,144 Gaussians, the author's default size, 46 graphs (a graph per block).
  GPU precision FP16 with fp32 accumulation.

In both, background removal (the author's BiRefNet) runs on the CPU and the other graphs on the GPU (ML Drift,
OpenCL); `manifest.json` tells the app which form it has. The steps between graphs are Kotlin ports of
`scripts/idol_runner.py`, `scripts/idol_glue.py` and `scripts/idol_graphs.py`, in the same order and with the same
float32 arithmetic.

Targets: Galaxy S26 (12 GB) and Pixel 11 Pro. arm64 only. The phone needs about 8.5 GB free to install the default
form's models (8.2 GB for the `262k/` form).

## Get the model files

| Folder | Hugging Face repo | Files |
|---|---|---|
| `IDOL_DIR` | [litert-community/Idol](https://huggingface.co/litert-community/Idol): the downloaded folder for the default form, its `262k/` for the other | `manifest.json`, every graph except `dino_*` and `dis`, `den_pos_emb.f32`, `gs_perturbation.f32`, `vae_bn.json`, `gs_rep.json`; the default form also `den_joint_w.f32` and `gs_w.f32` |
| `DINO_DIR` | [litert-community/DINOv3-ViT-H16plus-Idol-LiteRT](https://huggingface.co/litert-community/DINOv3-ViT-H16plus-Idol-LiteRT) (gated: accept the DINOv3 License on the Hub first) | `dino_0..3.tflite`, `dino_rope_cos.f32`, `dino_rope_sin.f32` |
| `DIS_DIR` (optional) | [litert-community/DIS-ISNet-LiteRT](https://huggingface.co/litert-community/DIS-ISNet-LiteRT) | `dis.tflite`, only for the GPU background removal setting, which needs the `262k/` form (the default form's manifest has no `dis` graph) |

```bash
hf download litert-community/Idol --exclude "262k/*" --local-dir ~/Downloads/Idol   # the default form; drop --exclude to get 262k/ too
hf download litert-community/DINOv3-ViT-H16plus-Idol-LiteRT --local-dir ~/Downloads/DINOv3-ViT-H16plus-Idol-LiteRT
hf download litert-community/DIS-ISNet-LiteRT --local-dir ~/Downloads/DIS-ISNet-LiteRT   # optional, 262k/ only
```

The app needs 29 files, 7.2 GB in total, with the default form (53 files, 7.2 GB, with the `262k/` form).
`manifest.json` lists each graph with its size, SHA-256, signatures, accelerator and precision.

## Build and install

```bash
cd idol
./gradlew :app:installDebug
IDOL_DIR=~/Downloads/Idol DINO_DIR=~/Downloads/DINOv3-ViT-H16plus-Idol-LiteRT ./scripts/install_to_device.sh
# the first release's form instead: IDOL_DIR=~/Downloads/Idol/262k (same DINO_DIR)
```

`install_to_device.sh` reads the file list from `manifest.json` and checks every source file's size before it runs any
device command. It then copies one file at a time into the app's private `files/models/` (`adb push` to
`/data/local/tmp`, `run-as cp`, remove the temporary copy), so the phone never holds a second copy of more than one
file. After each copy it checks the SHA-256 on the phone, and a file the phone already holds with the right SHA-256
is not copied again. Set `IDOL_DIR`: its default, this folder's `models/v2/`, is a local export that
is not in the repository. It needs the debug APK installed first (`run-as` only works with a debuggable app) and `python3` on the computer.
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
`dis.tflite` and the `262k/` form: the default form's manifest has no `dis` graph.

### Messages

| When | Message |
|---|---|
| A model file is missing or has the wrong size or SHA-256 | "Model files are missing. Run scripts/install_to_device.sh (see README)." and the list of files |
| The picture has no object (all black, one flat colour, no alpha value of 128 or more) | "No object found in this picture." |
| Camera permission denied | "Camera permission is off. You can still pick a photo." |
| A graph, or background removal, does not fit in memory even after the others are closed | "Not enough memory for <graph> ... Close other apps and try again." (`<graph>` = "background removal (bir)" for BiRefNet) |

## Memory and loading

With the `262k/` form the 45 GPU graphs do not all fit in the phone's memory at once (one denoiser block alone takes
about 0.7 GB of GPU memory). The default form has 19 GPU graphs and compiles each once per photo: the two shared graphs
take the next block's weights as inputs instead of being closed and compiled again. With either form, at start the
app compiles the graphs in the order a run uses them, until `MemAvailable` would drop below a reserve of
2,500 MB plus the room background removal needs (below) ("Loading models k/n"). The rest are compiled during the run.
When the next graph does not fit, the app closes the resident graph whose next use is farthest away in the run's order,
so the graphs used next stay resident. The reserve is large because `MemAvailable` counts the page cache of the mapped
graph files: with 1,000 MB, a compile burst outran its reclaim and lmkd killed the app on the Galaxy S26 at about 1.9 GB
available. After closing a graph the app waits up to 1.5 s for its memory to show up (the Pixel 11 Pro returns GPU
memory a few seconds late), and up to 8 s before it gives up with "Not enough memory". BiRefNet (930 MB, CPU) is
compiled for each picture and closed right after the background removal, before the DINOv3 graphs run. Its call needs
far more memory than its file: on the Galaxy S26 one call lowered `MemAvailable` by about 2,670 MB (a standalone runner
leg, `device/g2/bir3`). The manifest carries that measured peak (`run_peak_bytes`; without it the app assumes 3 × the
file size). Before compiling BiRefNet, the app closes graphs until `MemAvailable` covers the peak plus the reserve, or
stops with "Not enough memory for background removal (bir) ... Close other apps and try again." The preload leaves the
same room free, so the graphs compiled ahead stay resident through background removal (on the Galaxy S26 in the take
below, about 5,190 MB free, that left no room to compile any graph ahead). During the call the app reads `MemAvailable`
every 50 ms and logs the drop (`peak` events, tag `IdolGraphs`); a larger drop than the manifest's raises the estimate
for later pictures while the app runs. With this check a photo ran through to the viewer on the Galaxy S26 (the call
lowered `MemAvailable` by 1,911 MB) and on the Pixel 11 Pro (2,475 MB), and neither phone closed the app. The stop
itself was run on the Pixel 11 Pro with a reserve no phone can meet (`plans/fail_bir_memory.json`): the message above,
the app still running.

Each graph runs as its own GPU call. One long GPU call starves the phone's compositor, and the process gets killed
(`F0.md` finding 2). Outputs pass to the next graph through host memory (`readFloat` / `writeFloat`), and inputs that
did not change since the last write are not written again: the RoPE tables, the position embedding, the LoRA gate, the
conditioning of the current LoRA phase and the time modulation of the current step. The AutoRun plan key
`"handoff": "direct"` passes the output buffer itself to the next graph instead. That path is not verified on a phone
yet, and on the first error the app falls back to host copies and records why.

## Speed and memory

The default form (the Hub repo's root) is the fastest of the forms measured on a Galaxy S26: it makes a splat in 158 s against 267 s for the first release's form in `262k/`, while its renders agree less closely with the author's default output: 32.3–34.4 dB PSNR against 41.4–46.6 dB (higher is closer). Of the other forms tried, int8 weights did not run on the phone GPUs, a 512 × 512 input made even the author's own PyTorch pipeline miss the 25 dB bar (13.5–22.9 dB against its 1,024 × 1,024 output), the denoiser did not compile for the Galaxy S26's NPU on the phone, and the CPU alone was slower.

| | default (the Hub repo's root) | first release (`262k/`) |
|---|---|---|
| Gaussians per splat | 65,536 (2,048 anchors) | 262,144 (8,192 anchors, the author's default) |
| GPU precision | FP16 | FP16 with fp32 accumulation |
| Denoiser blocks 1–22 and Gaussian decoder graphs 1–6 | one shared graph per chain; the app copies each block's weights in before the call | a graph per block |
| Galaxy S26: the first photo after start, cool phone (thermal status 0 at the start) | **158 s** | 267 s |
| Galaxy S26: photos on a phone already warm | 329–351 s (4 photos of the armchair, thermal status 3) | 339–449 s (5 photos of 3 inputs, thermal status 2) |
| Pixel 11 Pro: the first photo after start | 343 s | not measured |
| 4-view render PSNR against the author's default on the Galaxy S26 (threshold 25 dB) | 32.3–34.4 dB | 41.4–46.6 dB |

Thermal status is Android's: 0 none, 1 light, 2 moderate, 3 severe. Both forms run with this app (`IDOL_DIR` = the downloaded folder or its `262k/`) and the Hub repo's Python runner (`--models`, the same folders).

The first release's card gives 303 s for the `262k/` form: that photo started on a warmer phone (thermal status 1, skin 38.5 °C, against 34.6 °C here). With this app the `262k/` form makes the same splat as with the first release's, byte for byte.

The rest of this card describes the default form; the first release's card ([commit e7ee1226](https://huggingface.co/litert-community/Idol/blob/e7ee12262a3ed622fa44fb8df8ff4f335dd77064/README.md)) has the
per-stage tables of the `262k/` form.

On a Galaxy S26, the sample app turns one photo into a splat of 65,536 Gaussians in 158 s, starting from a cool phone (the first photo after start). The phone throttles its GPU during that photo: the GPU clock cap fell from 1,300 to 500 MHz (thermal status 0 → 2). After 12 minutes of back-to-back runs the phone was at thermal status 3; through the next 4 photos the GPU clock cap stayed at 222–282 MHz (one reading of 191 MHz), and they took 329–351 s. At start the app compiles 1 graph before the photo (2 s) and 19 more during it (20 s): the denoiser's 22 middle blocks share one graph and the Gaussian decoder's six middle graphs share another, so each is compiled once per photo. The first release's form in `262k/` (262,144 Gaussians, a graph per block, GPU FP16 with fp32 accumulation) took 267 s the same way, also from thermal status 0 (skin 34.6 °C at the start against 33.5 °C). Background removal (BiRefNet) runs on the CPU: 7.4 s of that time on the Galaxy S26. The
other networks run on the GPU at FP16.

| Galaxy S26 (SM-S942Q), the sample app, LiteRT 2.3.0; from a cool phone (GPU clock cap 1,300 MHz, thermal status 0 at the start) | first photo after start |
|---|---|
| **photo → .ply written** | **158 s** |
| background removal (BiRefNet, CPU) | 7.4 s |
| DINOv3 encoder | 10.6 s |
| VAE encoder | 5.2 s |
| denoiser, 5 steps | 119.8 s |
| octree decoder | 9.1 s |
| Gaussian decoder | 5.5 s |
| PLY writer | 0.1 s |
| graphs compiled at app start, before the first photo | 1 (2 s) |
| graphs compiled during the photo | 19 (20 s) |
| peak resident memory (VmHWM) | 5.42 GiB |
| peak GPU memory (kgsl) | 5.73 GiB |

| Galaxy S26 (SM-S942Q), the sample app, LiteRT 2.3.0; right after the cool photo's hold (thermal status 3 at the start) | first photo after start | photo 2 | photo 3 | photo 4 |
|---|---|---|---|---|
| **photo → .ply written** | **351 s** | **351 s** | **341 s** | **329 s** |
| background removal (BiRefNet, CPU) | 14.9 s | 18.9 s | 15.2 s | 15.4 s |
| DINOv3 encoder | 22.5 s | 23.6 s | 25.1 s | 20.6 s |
| VAE encoder | 10.7 s | 8.1 s | 8.0 s | 7.5 s |
| denoiser, 5 steps | 278.1 s | 280.3 s | 272.6 s | 265.4 s |
| octree decoder | 14.3 s | 11.7 s | 12.1 s | 12.2 s |
| Gaussian decoder | 9.2 s | 7.7 s | 7.6 s | 7.4 s |
| PLY writer | 0.1 s | 0.0 s | 0.0 s | 0.0 s |
| graphs compiled at app start, before the first photo | 3 (9 s) | — | — | — |
| graphs compiled during the photo | 18 (29 s) | 18 (21 s) | 19 (25 s) | 18 (20 s) |
| peak resident memory (VmHWM) | 5.30 GiB | 4.58 GiB | 5.10 GiB | 4.63 GiB |
| peak GPU memory (kgsl) | 5.73 GiB | 5.12 GiB | 5.13 GiB | 5.04 GiB |

| Pixel 11 Pro, the sample app, LiteRT 2.3.0; thermal status 0 at the start | first photo after start |
|---|---|
| **photo → .ply written** | **343 s** |
| background removal (BiRefNet, CPU) | 17.4 s |
| DINOv3 encoder | 8.5 s |
| VAE encoder | 36.1 s |
| denoiser, 5 steps | 242.3 s |
| octree decoder | 12.9 s |
| Gaussian decoder | 23.7 s |
| PLY writer | 0.2 s |
| graphs compiled at app start, before the first photo | 6 (60 s) |
| graphs compiled during the photo | 15 (75 s) |
| peak resident memory (VmHWM) | 5.96 GiB |

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
