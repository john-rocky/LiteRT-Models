# Timestral Forecast — probabilistic time-series forecasts on Android with Timestral-209M

Paste a series of numbers (or open a CSV file, or pick a bundled example), choose how many steps to forecast (up to 1,024) and the accelerator (CPU, GPU, or the Qualcomm NPU), and tap **Forecast**. The app draws the end of your series, the forecast's median and its 10–90 % band, and shows how long the model call took. Everything runs on the phone: the whole model as one LiteRT graph, and in Kotlin the pre- and post-processing of the official Python forecaster (scaling the series, un-scaling and capping the forecast, averaging it with the forecast of the negated series, clipping at zero), which the JVM tests below check against the Python reference.

## Model and requirements

- Model: [mistralai/Timestral-209M](https://huggingface.co/mistralai/Timestral-209M), revision `9874d252ab0aae16f7f7c999f98f98d6ad7d8365`, weights under the Modified MIT License (see [License](#license)) — condition 2: no rights under the license if the global consolidated monthly revenue of your company or your employer exceeded $20 million in the preceding month; it covers the converted graphs too. Code: [github.com/mistralai/timestral](https://github.com/mistralai/timestral), commit `a534bdbd244d`, Apache-2.0.
- Converted files (the graphs and tables the app loads): [mlboydaisuke/Timestral-209M-LiteRT](https://huggingface.co/mlboydaisuke/Timestral-209M-LiteRT) on Hugging Face; the next section has the download command, and the model card has their speed on phones and their accuracy against the official forecaster. To build them yourself from the official checkpoint, see [Rebuilding the graphs](#rebuilding-the-graphs); [Files on the device](#files-on-the-device) lists them.
- What is forecast (the official defaults of `TimestralForecaster.forecast()` for one series without covariates): the forecast of the series averaged with the forecast of its negation, clipped at zero when no observed value is negative, quantiles capped near the range of your series (the context's range ± 10⁴ × its last scale), and one model pass of up to 1,024 steps (32 patches of 32 steps); the official forecaster's pass-by-pass loop for longer horizons is not part of this sample. Of a longer series the last 8,192 steps are the context (the official forecaster with `max_context=8192`).
- Output: 21 quantiles per step (levels 0.01, 0.05, 0.1, 0.15, …, 0.9, 0.95, 0.99).
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile/target SDK 35.
- Runtime: LiteRT **2.2.0** `CompiledModel`; Material 1 Compose with MVVM.

## Download, build and install

```bash
cd timestral/
# Optional, for the NPU choice: the Qualcomm libraries go into the APK (the repository README's
# section "Running on the NPU" lists them). v81 is the Hexagon version of the SM8850 (Galaxy S26);
# use v73 for an SM8550, v75 for an SM8650, v79 for an SM8750.
./scripts/fetch_npu_libs.sh /path/to/npu-libraries v81
./gradlew :app:installDebug
# Optional when several devices are connected:
# export ANDROID_SERIAL=your-device-serial
hf download mlboydaisuke/Timestral-209M-LiteRT --local-dir ~/Downloads/Timestral-209M-LiteRT
./scripts/install_to_device.sh ~/Downloads/Timestral-209M-LiteRT
```

`install_to_device.sh` copies every file that `app/src/main/assets/timestral_forms.json` names from that directory into the app's private `files/` directory, one file at a time, checking each copy's size: the eight graphs and three small `.f32` tables the app reads (the decoder token and the two RoPE tables), 2.5 GB in all. To install the files of some accelerators only, name them:

```bash
# GPU and NPU (1.6 GB; the NPU runs the two c2048 graphs); for the CPU (0.9 GB) name the four
# -int8 graphs and the three tables.
FILES="timestral-209m-c2048-h256.tflite timestral-209m-c2048-h1024.tflite \
timestral-209m-c8192-h256.tflite timestral-209m-c8192-h1024.tflite \
forecast_query.f32 rope_cos.f32 rope_sin.f32" ./scripts/install_to_device.sh ~/Downloads/Timestral-209M-LiteRT
```

The screen counts the files that are not installed yet, and a forecast whose graph or tables are missing names them instead of running.

## Using the app

- **Series**: numbers separated by commas, spaces, semicolons or new lines; or a table (CSV or tab-separated) whose last column is the series, with or without a header line. An empty field, `NaN`, `NA`, `N/A`, `null`, `none`, `-`, `?` or an infinite value is a missing step: the model reads only the observed steps. Full-width digits, commas, periods and minus signs (what a CJK keyboard in full-width mode types) are read as their ASCII forms (Unicode NFKC). **Open file** reads a text or CSV file; **Examples** loads one of three synthetic series (hourly temperature, daily orders, a sensor with gaps).
- **Horizon**: a whole number of steps from 1 to 1,024.
- **Accelerator**: CPU, GPU or NPU. The NPU can be chosen only when the APK carries the Qualcomm libraries and the phone has a Qualcomm Snapdragon; otherwise the screen says why. The NPU takes a context of up to 2,048 steps (see [Accelerators](#accelerators)).

The screen refuses a forecast, with a sentence saying why, when:

| Input | Sentence |
|---|---|
| fewer than 16 values | "The series has N values; this sample needs at least 16." |
| fewer than 16 observed values | "Only N of M values in the context are numbers; this sample needs at least 16 observed values …" |
| a value that is not a number | "Value N ("…") is not a number. Use numbers, or leave a field empty or write NaN for a missing value." |
| a horizon outside 1 to 1,024 | "The horizon must be a whole number of steps from 1 to 1,024 …" |
| the NPU and a series of more than 2,048 values | "The NPU takes a context of up to 2,048 steps, and this one has N. The 8,192-step graphs run on the GPU and the CPU only: on a Galaxy S26, 8,192-step builds of this model ran 82 to 97 ms per forecast on the NPU against 69 to 73 ms for these graphs on the GPU. Choose another accelerator, or keep the last 2,048 values of the series." |
| model files missing | "Model file not found: files/… Install the model with scripts/install_to_device.sh …" |
| the graph does not compile on the chosen accelerator | "The GPU could not compile the graph: … The sample does not fall back to another accelerator" |
| the NPU compiler plugin did not take the graph | "The NPU did not take the graph (no DispatchDelegate line in this app's log …)" |
| the graph output is not finite | "The forecast is not finite on the GPU (forecast produced N NaN values). Choose another accelerator." |

The DispatchDelegate line is the line LiteRT writes to the app's log when the NPU takes the graph ("Replacing N out of M node(s) with delegate (DispatchDelegate)"). Without that line LiteRT would run the graph on the CPU, so the app stops instead.

A constant series is forecast (it stays at its level); the screen says so under the chart. A series longer than 8,192 values is forecast from its last 8,192 values, also with a note.

## How a forecast runs

This section is for readers who port the host code; using the app needs none of it.

1. **Context** (`TimestralHost.prepare`): missing steps are filled forward (they stay marked as missing), the last 8,192 steps are kept and the front is padded with missing steps to whole patches of 32. The forecast's two rows are this context and its negation (the mirror row).
2. **Scaling**: per row the official causal statistics (float64 running mean and Bessel standard deviation over observed steps, floored at 10⁻⁶); each patch takes the values at its last step; observed steps become asinh((x − loc) / scale) in float32.
3. **Graph inputs** (`TimestralHost.buildInputs`) for the graph the forecast takes (`GraphSpec.choose`): of the graphs the accelerator runs, the smallest graph that fits (fewest 32-step patches of context plus horizon). A graph has L context patches, H decoder patches and R rows per call: R = 2 runs the context and its negation in one call, R = 1 runs one call for each. The shipped graphs have L = 64 (a context of up to 2,048 steps) or L = 256 (8,192 steps), H = 8 (a horizon of up to 256 steps) or H = 32 (1,024 steps), and R = 2, so 120 values and a horizon of 24 take `c2048-h256`, the smallest.

   | Input | Shape | Content |
   |---|---|---|
   | `patches` | [R, L, 64] | per patch 32 normalized values, then 32 missing flags (1 = missing); padding patches in front |
   | `dec` | [R, H, 768] | the checkpoint's `forecast_query` vector for every decoder token |
   | `bias` | [1, 1, S, S] | additive attention mask, S = L + H: 0 where a token may attend, the graph's mask value where it may not (causal context, decoder tokens reading every valid context patch, causal decoder) |
   | `cos`, `sin` | [1, 1, S, 64] | RoPE rows of the positions counted from the earliest observed patch |
   | output `output_0` | [R, H, 32, 21] | the normalized quantiles of each decoder step, monotone (each quantile is at least the one below it) |

4. **Graph calls**: write the five inputs, `run()`, read the output back — once for a two-row graph, once per row for a one-row graph (the time on screen is their sum).
5. **Finish** (`TimestralHost.finish`): decoder patches 1 to ⌈horizon / 32⌉ of each row are denormalized (sinh, float64 to float32) at the last context patch's loc and scale, moved to the advancing refine frame (float64 Welford over the context and each decoded patch's median; Welford is a running mean and variance updated one value at a time), capped, checked for NaN, averaged with the reversed negation of the mirror row, and clipped at 0 when every observed value is ≥ 0.

All CompiledModel calls run on one thread; one graph is loaded at a time (choosing another accelerator, or a series that needs another graph, closes it and compiles the new one).

## Accelerators

`app/src/main/assets/timestral_forms.json` lists the graphs (`name`, `file`, `context_patches`, `horizon_patches`, `rows`, the attention `mask` value, the `backends` that run it and an optional `gpu_precision`) and each accelerator's CPU threads and GPU precision, so the graphs and the defaults can follow the device measurements without a code change; `make_test_vectors.py` checks that each entry's patches, rows and mask value agree with its file name. Every shipped graph is a float16-safe form of the model: the same function in float32, rewritten so that float16 arithmetic stays finite (the rewrites are listed under [Rebuilding the graphs](#rebuilding-the-graphs)). The GPU and NPU graphs have float16 weights. The CPU graphs (`-int8`) have int8 weights in 199 of their 217 fully connected layers: those of transformer blocks 1 to 23 of 24, whose activations XNNPACK (LiteRT's CPU backend) quantizes at run time. The last block (11 layers), the patch embedding (3) and the quantile head (4) keep float16 weights: a copy with int8 weights in every fully connected layer raised the CRPS (the error of the whole quantile forecast, here the weighted quantile loss of its nine deciles; lower is better) by 2.3 % in a desktop CPU run over the 84 test windows of the conversion run (72 windows of public datasets and 12 synthetic ones, each with its true continuation; not redistributed), past the run's 1 % limit (a shipped graph's MASE and CRPS may be at most 1 % above the official forecaster's).

- **CPU**: XNNPACK, 4 threads, the `-int8` graphs.
- **GPU**: ML Drift (LiteRT's GPU backend), `CompiledModel.GpuOptions(precision = FP16)` for the float16-safe graphs (the exact forms need `FP32`: with float16 arithmetic their attention scores and RMSNorm sums leave the float16 range); no CPU fallback — a graph that does not compile is an error on screen.
- **NPU**: LiteRT's on-device (JIT) compiler plugin for the Qualcomm NPU (the HTP), with the libraries in the APK, the Environment's dispatch and plugin directories set to the APK's native library directory, and BURST performance mode (the NPU's highest clock setting) on every graph of the process; the two `c2048` graphs only. The 8,192-step graphs run on the GPU and the CPU: on the Galaxy S26, 8,192-step builds of this model ran 82 to 97 ms per forecast on the NPU against 69 to 73 ms for the `c8192` files on the GPU, so a longer context on the NPU stops with a sentence instead of moving to the GPU. The initial compile of a graph on the phone takes long (17 and 22 s for the two `c2048` graphs on the Galaxy S26); LiteRT keeps the result in the app's cache, and a later compile of the same graph in the same app run took 0.14 s ([Speed in this app](#speed-in-this-app)). Because LiteRT runs a graph the plugin does not take on the CPU without an error, the app reads its own log for the `DispatchDelegate` line and stops with a sentence when there is none.

## Files on the device

| File | Bytes | Context / horizon (steps) | Accelerators |
|---|---:|---|---|
| `timestral-209m-c2048-h256.tflite` | 405,041,792 | ≤ 2,048 / ≤ 256 | GPU, NPU |
| `timestral-209m-c2048-h1024.tflite` | 405,041,792 | ≤ 2,048 / ≤ 1,024 | GPU, NPU |
| `timestral-209m-c8192-h256.tflite` | 405,041,792 | ≤ 8,192 / ≤ 256 | GPU |
| `timestral-209m-c8192-h1024.tflite` | 405,041,792 | ≤ 8,192 / ≤ 1,024 | GPU |
| `timestral-209m-c2048-h256-int8.tflite` | 222,336,912 | ≤ 2,048 / ≤ 256 | CPU |
| `timestral-209m-c2048-h1024-int8.tflite` | 222,336,912 | ≤ 2,048 / ≤ 1,024 | CPU |
| `timestral-209m-c8192-h256-int8.tflite` | 222,336,912 | ≤ 8,192 / ≤ 256 | CPU |
| `timestral-209m-c8192-h1024-int8.tflite` | 222,336,912 | ≤ 8,192 / ≤ 1,024 | CPU |
| `forecast_query.f32` | 3,072 | the decoder token (768 float32) | all |
| `rope_cos.f32`, `rope_sin.f32` | 532,224 each | RoPE tables, positions −1,023 to 1,055 (2,079 rows of 64 float32) | all |

## Speed in this app

This app's debug build with the files above on a Galaxy S26 (SM-S942Q, Android 16), LiteRT 2.2.0: a typed series of 120 values and a horizon of 300 steps, which takes `c2048-h1024` on the NPU and the GPU and `c2048-h1024-int8` on the CPU. A forecast is one graph call (the series and its negation are the call's two rows), and its time is the screen's "Graph call": write the inputs, run the graph, read the output back. Forecast 1 on an accelerator also compiles the graph; the last column is the median of the 10 forecasts after it, tapped one after another in the app (not a benchmark run). The accelerators ran in the order of the table, in one app process, with the phone's Android thermal status 0 (no throttling) before and after each of them.

| Accelerator | Graph | Forecast 1 (compile) | Forecasts 2–11: median (min–max) |
|---|---|---|---:|
| NPU | `c2048-h1024` | 21.9 ms (21.8 s) | 21.1 ms (20.9–21.4) |
| GPU (FP16) | `c2048-h1024` | 31.1 ms (1.0 s) | 30.8 ms (28.7–34.8) |
| CPU (4 threads) | `c2048-h1024-int8` | 53.4 ms (0.34 s) | 57.5 ms (50.9–58.0) |

The host steps around the call (statistics, inputs, refine, averaging) added 3 to 25 ms to each of these forecasts. In every NPU forecast the whole graph ran on the NPU (one partition, nothing left to the CPU; the app's NPU line: "the HTP took the whole graph (1 of 1 nodes)"). The NPU's initial compile of `c2048-h256` took 16.9 s; when the app came back to that graph later in the same run, LiteRT loaded it from its cache in 0.14 s.

The app has not run these files on a Pixel 8a yet.

## Tests

```bash
./gradlew :app:testDebugUnitTest
```

- `HostVectorsTest`: the Kotlin host against the Python reference (`scripts/host_forecast.py`) on 19 synthetic vectors (`app/src/test/resources/vectors/`, written by `scripts/make_test_vectors.py`): two-row graphs with 32 and with 8 decoder patches, one-row graphs (the reference's `mirror_in_one_call=False` path, one call per row), L = 64 and L = 256, a series longer than 8,192 steps, missing steps, non-negative clipping, a constant series, horizons of 1 to 32 patches, saturated and NaN graph outputs. The graph is replaced by a float32 formula both sides compute; the graph inputs of every call must be bit-equal (sha256) and every quantile within a relative difference of 10⁻⁶. `-Ptimestral.host=<dir with the host tables>` also compares `dec`, `cos` and `sin`.
- `GraphSpecTest`: which graph a forecast takes.
- `SeriesInputTest`: the input rules and the sentences of the table above.
- `FixtureParityTest`: the 84 test windows of the conversion run (not redistributed), read from outside the module: `-Ptimestral.fixturePack=<dir>` (each window's context), `-Ptimestral.rows=<dir>` (the graph inputs the Python host built for each window, and the graph's desktop CPU output on them), `-Ptimestral.host=<dir>` (the host tables) and `-Ptimestral.out=<dir>` (where the test writes each window's quantiles). Without them the test is skipped, and a skipped run is no evidence of a match.

Result on a desktop JVM (Temurin 17.0.16, 2026-10-23): the graph inputs of every call are bit-equal on all 19 vectors and all 84 test windows, and every quantile is bit-equal to the Python reference (45,948 values of 17 vectors; 339,024 values of the 84 windows when both hosts finish the same graph outputs). The two vectors with a NaN graph output give the reference's errors, "forecast produced 1009 NaN values" (two rows per call) and "forecast produced 799 NaN values" (one row per call).

Debug builds keep each forecast under `files/forecast_dumps/` (the series, the sha256 of each call's graph inputs, the raw graph output of both rows and the quantiles) so a phone run can be compared with the Python reference on the same graph output. On the Galaxy S26 run of [Speed in this app](#speed-in-this-app) and the 2,400-value hourly example on the `c8192` graphs, the graph inputs and the quantiles of all 44 forecasts were bit-equal to the Python reference.

## Rebuilding the graphs

The shipped graphs come from the official checkpoint through litert-torch. `scripts/timestral_graph.py` is a re-implementation of the official model with fixed shapes; in float32 it matches the official forecaster to ×1.0000002 (MASE, the scaled error of the median) and ×1.0000001 (CRPS) on the 84 test windows (a desktop CPU run of its float32 export). `scripts/r4_graph.py` adds a last block that computes only the decoder tokens, `scripts/r4_export.py` exports a graph and its float16-weight copy, and `scripts/r4_quant.py drq` (ai-edge-quantizer) writes the CPU copy from the float32 file, with int8 weights in the fully connected layers of blocks 1 to 23 (its `Block_\d+/` scope does not match the re-authored last block). The `r4_` scripts keep their names from the conversion run: the exported files' tensor names depend on them.

The float16-safe form comes from these rewrites, each exact in float32: every RMSNorm pre-scaled by a power of two (`--norm pow2 --norm-k scripts/normk_fixture.json`, the exponent of each RMSNorm); attention logits computed at 1/16 with the row maximum subtracted (`--attn-k 4`); the softmax input clamped at −3·10⁴ (`--attn-clamp`), with −3·10⁴ as the mask value (the exact forms take −10³⁰); and a last transformer block that computes only the decoder tokens the quantile head reads (`--dec-only-last`; the other tokens' last-block outputs are never read).

Run these from `timestral/`. The scripts read the official checkpoint from the Hugging Face cache, so fetch it first:

```bash
hf download mistralai/Timestral-209M config.json model.safetensors --revision 9874d252ab0aae16f7f7c999f98f98d6ad7d8365
for shape in 64x8 64x32 256x8 256x32; do
  python scripts/r4_export.py --bucket "$shape" --rows 2 --norm pow2 \
    --norm-k scripts/normk_fixture.json --attn-k 4 --attn-clamp --dec-only-last --fp16
done
for f in out/tflite/timestral209m_r2_l*_pow2knormk_fixture_att4c_dl_fp32.tflite; do
  python scripts/r4_quant.py drq --src "$f" --sel blocks --rest-fp16
done
# The host tables (every scripts/export_tflite.py run also writes them):
python -c "import sys; sys.path.insert(0, 'scripts'); import common, export_tflite; \
export_tflite.write_host_tables(common.load_official(), common.OUT + '/host')"
```

In the output names, `r2` is `--rows 2`, `l64` / `l256` and `h8` / `h32` are the context and decoder patches of `--bucket`, `pow2knormk_fixture` is `--norm pow2` with `--norm-k scripts/normk_fixture.json`, `att4c` is `--attn-k 4` with `--attn-clamp`, `dl` is `--dec-only-last`, `fp32` / `wfp16` are the float32 file and its float16-weight copy, and `blocksint8w16` is `r4_quant.py drq --sel blocks --rest-fp16`. Copy the outputs into one directory under the app's file names (the directory `install_to_device.sh` reads):

| Output | File |
|---|---|
| `tflite/timestral209m_r2_l64_h8_pow2knormk_fixture_att4c_dl_wfp16.tflite` | `timestral-209m-c2048-h256.tflite` |
| `tflite/timestral209m_r2_l64_h32_pow2knormk_fixture_att4c_dl_wfp16.tflite` | `timestral-209m-c2048-h1024.tflite` |
| `tflite/timestral209m_r2_l256_h8_pow2knormk_fixture_att4c_dl_wfp16.tflite` | `timestral-209m-c8192-h256.tflite` |
| `tflite/timestral209m_r2_l256_h32_pow2knormk_fixture_att4c_dl_wfp16.tflite` | `timestral-209m-c8192-h1024.tflite` |
| the same four with `_dl_blocksint8w16` in place of `_dl_wfp16` | the same four names with `-int8` |
| `host/forecast_query.f32` | `forecast_query.f32` |
| `host/rope_cos_p-1023_1055.f32`, `host/rope_sin_p-1023_1055.f32` | `rope_cos.f32`, `rope_sin.f32` |

The conversion run used Python 3.12.11 with torch 2.13.0, numpy 2.5.4, litert-torch 0.9.4, ai-edge-litert 2.2.0, ai-edge-quantizer 0.9.0 and the official `timestral` package at commit `a534bdb`, with the checkpoint's revision `9874d252` in the Hugging Face cache.

The `a16w8` path of `scripts/r4_quant.py` (int16 activations, int8 weights) is an experiment that was not shipped; it needs a script that is not in this repository.

## Files

| Path | Role |
|---|---|
| `app/src/main/java/com/timestral/TimestralHost.kt` | the host steps (Android-free) |
| `app/src/main/java/com/timestral/GraphSpec.kt` | a graph's shape and which graph a forecast takes (Android-free) |
| `app/src/main/java/com/timestral/SeriesInput.kt` | reading the series and the horizon, the refusal sentences |
| `app/src/main/java/com/timestral/TimestralGraph.kt` | one compiled graph: options per accelerator, buffers, the NPU check |
| `app/src/main/java/com/timestral/TimestralRuntime.kt` | the LiteRT thread and Environment, the NPU libraries and log |
| `app/src/main/java/com/timestral/TimestralConfig.kt` | `timestral_forms.json` |
| `app/src/main/java/com/timestral/ForecastEngine.kt` | files, host tables, the resident graph, one forecast |
| `app/src/main/java/com/timestral/MainViewModel.kt`, `UiState.kt`, `MainActivity.kt`, `view/` | the Compose screen and chart |
| `app/src/main/assets/timestral_forms.json` | the graphs and each accelerator's settings |
| `app/src/main/assets/examples/` | three synthetic example series |
| `scripts/install_to_device.sh` | copies the model files into the app's `files/` |
| `scripts/fetch_npu_libs.sh` | copies the Qualcomm NPU libraries into the APK's sources |
| `scripts/host_forecast.py` | the host in Python (NumPy), the reference the Kotlin host is tested against |
| `scripts/timestral_graph.py`, `scripts/r4_graph.py`, `scripts/r4_export.py`, `scripts/r4_quant.py`, `scripts/export_tflite.py`, `scripts/common.py`, `scripts/litert_helpers.py`, `scripts/normk_fixture.json` | the graph's re-implementation, its export and the CPU copy (see [Rebuilding the graphs](#rebuilding-the-graphs)) |
| `scripts/make_test_vectors.py` | writes the test vectors and the examples |

## License

The sample is licensed under the Apache License 2.0 (`LICENSE`), as it ports the official inference code (Apache-2.0); see `NOTICE`. The Timestral-209M weights, and the graph files converted from them, are licensed by Mistral AI under the Modified MIT License (`licenses/Timestral-209M-weights-Modified-MIT.txt`). Attribution notice: 2026 - Mistral AI. Its condition 2 reads:

> 2. You are not authorized to exercise any rights under this license if the global consolidated monthly revenue of your company (or that of your employer) exceeds $20 million (or its equivalent in another currency) for the preceding month. This restriction in (b) applies to the Model and any derivatives, modifications, or combined works based on it, whether provided by Mistral AI or by a third party. You may contact Mistral AI (sales@mistral.ai) to request a commercial license, which Mistral AI may grant you at its sole discretion, or choose to use the Model on Mistral AI's hosted services available at https://mistral.ai/.
