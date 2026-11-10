# LiteRT 1.4.0 Interpreter vs LiteRT 2.3.2 CompiledModel (Galaxy S26)

Latency and output agreement of three vision models on one phone, called two ways. 1.4.0 is the version that
react-native-fast-tflite 3.0.1 pins; 2.3.2 is the latest LiteRT release on 2026-11-10.

- **A rows, LiteRT 1.4.0**: the TensorFlow Lite C API of `libtensorflowlite_jni.so` (and `libtensorflowlite_gpu_jni.so`)
  in the call order of react-native-fast-tflite 3.0.1's Android code: model from bytes, interpreter options, an optional
  delegate, interpreter, allocate; per call copy in, invoke, copy out. A1d, A2d and A3n are the calls that code makes
  (CPU, `android-gpu`, `nnapi`); the other A rows set options it does not set: a thread count (A1t4), the XNNPACK
  delegate (A1x), both (A1xt4), GPU options that allow precision loss (A2h).
- **B rows, LiteRT 2.3.2**: the CompiledModel C API of `libLiteRt.so` (from `litert` 2.3.2; the GPU library
  `libLiteRtClGlAccelerator.so` is in `litert-gpu` 2.3.2, the Qualcomm dispatch library and compiler plugin in
  `litert-npu-runtime-qualcomm` 2.3.2) on the CPU, the GPU and the Qualcomm NPU (compiled on the
  phone at load, Qualcomm's QAIRT 2.50). The GPU and NPU rows also pass the CPU, which runs the ops they do not take;
  B2g passes the GPU alone. Buffers come from the compiled model's requirements, as the C++ `CreateInputBuffers()`
  makes them.
- **X1t4** makes the A rows' TFLite C API calls against 2.3.2's `libLiteRt.so` (CPU, 4 threads).

Row names: A = 1.4.0, B = 2.3.2; 1 = CPU, 2 = GPU, 3 = NNAPI (A) or NPU (B); d = default, t4 = 4 threads,
x = XNNPACK, h = fp16 (precision loss allowed), f = fp32, g = GPU alone, H = host-memory buffers, c = no compile cache,
b = burst mode, n = NNAPI.

Models: det = EfficientDet-Lite0 int8 (uint8 input), pose = MoveNet SinglePose Lightning int8 v4 (uint8 input),
seg = MediaPipe Selfie Segmenter float16 256x256 (float32 input); all Apache 2.0. Five photos from Wikimedia Commons,
CC0.

Galaxy S26 (SM-S942Q, Android 16, build BP4A.251205.006.S942QOPS1AZH9), 2026-11-10.

## Results

Median ms per call (write the input, run, read every output back); each row was run once. FAIL = the outputs do not
agree with the reference (below). "does not load" = the load step returns an error. "-" = not run.
"failed" = the model loads, but a call returns an error.
"Wrapper change needed" = what code that calls the TFLite C API as react-native-fast-tflite 3.0.1 does would change
to get the row, read from the runner (not built in the wrapper).

| row | runtime and options | wrapper change needed | det | pose | seg |
|---|---|---|---|---|---|
| A1d | 1.4.0 Interpreter, CPU, threads unset | none (3.0.1 makes this call) | 24.87 | 7.82 | does not load |
| A1t4 | 1.4.0 Interpreter, CPU, 4 threads | set a thread count (3.0.1 does not) | 27.11 | 8.86 | does not load |
| A1x | 1.4.0 Interpreter + XNNPACK delegate, the delegate's default threads | add the XNNPACK delegate (3.0.1 does not) | 10.31 | 2.32 | 1.61 |
| A1xt4 | 1.4.0 Interpreter + XNNPACK delegate, 4 threads | add the XNNPACK delegate and a thread count (3.0.1 does neither) | 9.01 | 1.92 | 1.18 |
| A2d | 1.4.0 GPU delegate, default options (max precision first) | none (3.0.1 makes this call) | 19.54 | 7.08 | does not load |
| A2h | 1.4.0 GPU delegate, precision loss allowed (min latency first) | set the GPU options above (3.0.1 does not) | 13.71 FAIL | 6.21 | does not load |
| A3n | 1.4.0 NNAPI delegate | none (3.0.1 makes this call) | 24.58 | 7.78 | does not load |
| B1d | 2.3.2 CompiledModel, CPU, default | rewrite to the CompiledModel API | 8.68 | 2.23 | 1.69 |
| B1t4 | 2.3.2 CompiledModel, CPU, 4 threads | rewrite to the CompiledModel API | 8.93 | 1.89 | 1.42 |
| B2d | 2.3.2 CompiledModel, GPU + CPU, default precision | rewrite to the CompiledModel API | 13.20 FAIL | 2.63 FAIL | 1.00 |
| B2h | 2.3.2 CompiledModel, GPU + CPU, fp16 | rewrite to the CompiledModel API | 13.27 FAIL | 2.65 FAIL | 1.01 |
| B2f | 2.3.2 CompiledModel, GPU + CPU, fp32 | rewrite to the CompiledModel API | 18.98 FAIL | 3.27 FAIL | 1.23 |
| B2g | 2.3.2 CompiledModel, GPU alone, default precision | rewrite to the CompiledModel API | does not load | does not load | 0.99 |
| B2dH | 2.3.2 CompiledModel, GPU + CPU, default precision, host-memory buffers | rewrite to the CompiledModel API; host-memory tensor buffers | 13.36 FAIL | 2.65 FAIL | 1.08 |
| B2hH | 2.3.2 CompiledModel, GPU + CPU, fp16, host-memory buffers | rewrite to the CompiledModel API; host-memory tensor buffers | 13.25 FAIL | 2.65 FAIL | 1.11 |
| B2fH | 2.3.2 CompiledModel, GPU + CPU, fp32, host-memory buffers | rewrite to the CompiledModel API; host-memory tensor buffers | 18.77 FAIL | 3.32 FAIL | 1.34 |
| B3c | 2.3.2 CompiledModel, NPU + CPU, compiled on the phone at load, no cache | rewrite to the CompiledModel API; Qualcomm's QAIRT libraries in the app | 17.37 | 16.59 FAIL | failed |
| B3 | 2.3.2 CompiledModel, NPU + CPU, compiled on the phone, read from B3c's cache | rewrite to the CompiledModel API; Qualcomm's QAIRT libraries in the app | 18.05 | 17.46 FAIL | - |
| B3b | 2.3.2 CompiledModel, NPU + CPU, compiled on the phone at load, HTP burst mode | rewrite to the CompiledModel API; Qualcomm's QAIRT libraries in the app; the HTP burst mode in the Qualcomm options | 4.15 | 2.38 FAIL | - |
| X1t4 | 2.3.2 libLiteRt.so through the TFLite C API, CPU, 4 threads | version bump, the new .so name in CMake, headers kept from 1.4.0, a thread count; no GPU or NNAPI delegate (their C functions are not in the 2.3.2 libraries) | 10.88 | 3.37 | does not load |

Per model, the fastest row of each version whose outputs agree:

- det: 1.4.0 A1xt4 9.007 ms (p10-p90 7.58-9.73); 2.3.2 B3b 4.146 ms (p10-p90 4.00-4.28); time 1.4.0 ÷ 2.3.2 = 2.17.
- pose: 1.4.0 A1xt4 1.919 ms (p10-p90 1.80-2.01); 2.3.2 B1t4 1.893 ms (p10-p90 1.74-1.99); time 1.4.0 ÷ 2.3.2 = 1.01.
- seg: 1.4.0 A1xt4 1.182 ms (p10-p90 1.06-1.22); 2.3.2 B2g 0.992 ms (p10-p90 0.67-1.02); time 1.4.0 ÷ 2.3.2 = 1.19.

p10-p90, load and first-call times, CPU threads used and every agreement number: [TABLE.md](TABLE.md),
[TABLE.json](TABLE.json) (the segmentation verdicts are under `agreement_vs_mac`).

## Notes

- **B2 rows on the two int8 models.** On detection, every B2 row that loads gives the same output bytes for all five
  photos, while the 2.3.2 CPU and NPU rows give different outputs per photo and agree. On pose, the B2 rows' outputs
  change per photo but fail on all five (0 to 78 % of the counted keypoints within 0.05). On segmentation (float32
  input) every B2 row agrees, B2g included. The GPU runs through OpenCL here (`LITERT_CL` takes 266 of the 267
  detection nodes, 112 of the 157 pose nodes and all 246 segmentation nodes). The B2 rows also fail on both int8
  models on a Pixel 11 Pro (PowerVR; B2d, B2dH, B2hH, B2fH) and, with an earlier build of this runner, on a Pixel 8a
  (Mali; B2d, B2h, B2f); those runs are not in this table. Possibly related public issue:
  [LiteRT #10800](https://github.com/google-ai-edge/LiteRT/issues/10800) (UINT8 tensors on the OpenCL GPU path).
- **B2g** (the GPU alone) does not load the detection or the pose model: `LiteRtCreateCompiledModel` returns status 504
  (`kLiteRtStatusErrorCompilation`), and the log lists `TFLite_Detection_PostProcess` (detection) and `GATHER_ND` (pose)
  as not supported by the GPU. Possibly related public issue:
  [LiteRT #7936](https://github.com/google-ai-edge/LiteRT/issues/7936). B2g loads the segmentation model, whose 246
  nodes all go to the GPU.
- **B3c, B3, B3b (NPU + CPU) on pose** fail: the outputs change per photo, but none of the 53 keypoints that count
  (A1d score >= 0.3) lies within 0.05 of A1d's. On detection they agree. B3c and B3 differ only in load time (729 and
  170 ms on detection): B3 reads the compile cache B3c wrote.
- **B3b** sets the HTP performance mode to burst. It was given B3c's cache, but its log shows a new QNN compile and no
  cache line, so its load time (734 ms detection, 495 ms pose) includes the compile.
- **Precision.** A2d asks for max precision and A2h allows precision loss; B2d gives the same output bytes as B2h. Like
  for like: A2d with B2f, A2h with B2h. A2h fails one detection photo: its box above score 0.5 gets another class than
  A1d's, at a near-equal score.
- **B2dH, B2hH, B2fH** give each tensor a host-memory buffer of its own size instead of the buffer the compiled model
  asks for. On detection it asked for host memory already; on pose it asked for an OpenCL buffer for the
  input, on segmentation for the input and the output. The outputs are the same bytes as those of B2d, B2h and B2f.
- **A3n**: the NNAPI delegate is created but takes no node on detection and pose; the outputs are the same bytes as
  A1d's.
- **Segmentation**: the 1.4.0 rows without XNNPACK (A1d, A1t4, A2d, A2h, A3n) and X1t4 do not load the model; their
  log reports the MediaPipe custom op `Convolution2DTransposeBias` as unresolved. A1x and A1xt4 load it (the XNNPACK
  delegate takes all 246 nodes). Every segmentation row is compared with the Mac CPU output. B3c loads (the NPU
  compile takes 245 of the 246 ops), but its first call fails: `LiteRtRunCompiledModel` returns status 3 and the log
  says `Convolution2DTransposeBias` failed to invoke ("Stub operation invoked"). B3 and B3b, which are given B3c's
  compile cache, were not run.

## How it was measured

- One native runner (`runner/`) for both versions. Each row is its own process that loads the libraries of one
  version only, with `dlopen`. Building `runner/` with NDK 29.0.13113456 (`build.sh`) gives the binary whose sha256 is
  in `manifest.json`.
- Each row runs `ftfl_bench --accel <accel> [--gpu-precision <p>] --burst 1 --libdir <dir> --fixtures <dir> --count 5
  --warmup 10 --rounds 100 --ab-fixture 001 --dump full --out <dir> --model <file> --inputs in0 --form <row>` plus the
  row's flags in `forms.tsv`, with `LD_LIBRARY_PATH` set to the directory of the runtime libraries and
  `ADSP_LIBRARY_PATH` (used by the NPU) to that directory followed by
  `;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp`. `--burst` and `--form` are only recorded.
- Median of 100 calls after 10 warm-up calls on one input, p10-p90 next to it. Load = from opening the runtime library
  to tensors or buffers ready (the model file read, the model, options, delegate or compile). In every B row, the CPU
  rows included, the load also covers loading the GPU accelerator, the Qualcomm compiler plugin and two QNN libraries
  from the library directory. First call = the first call after the load.
- CPU threads used: the threads of the process whose CPU time grew during the timed calls (`/proc/self/task/*/stat`),
  not the thread count that was asked for.
- CPU clock: a row starts after a check reads thermal status 0 and no CPU frequency cap. A cap can return within
  seconds, so the runner also reads each CPU policy's frequency ceiling before the first timed call and after every
  call, outside the timed span; the lowest reading in the table is 91.5 % of the maximum (det A1d).
- Output agreement on five photos, against A1d (1.4.0, CPU, no options). The segmentation model, which A1d does not
  load, is compared with the Mac CPU output of `ai_edge_litert` 2.3.0, which agrees with A1d on the other two models.
  - detection: every box with score >= 0.5 is matched one to one by a box with score >= 0.4, the same class and
    IoU >= 0.5, both ways;
  - pose: at least 90 % of the keypoints whose reference score is >= 0.3 lie within 0.05 of the reference (input
    side = 1);
  - segmentation: at least 99 % of the pixels agree after a 0.5 threshold, and the mask IoU is >= 0.95 when the
    reference foreground is at least 1 % of the image.

  A row that fails any photo is FAIL. The summary lines above use only rows that pass.
- A1x, A1xt4 and the host-memory rows were added after a first Pixel 8a run, which showed that the 1.4.0 C API does
  not create XNNPACK on its own and that the B2 rows do not agree on the int8 models.

## Files

- `runner/`: `ftfl_bench.cc`, `tfl_path.cc` (the TFLite C API: A rows, X1t4), `lrt_path.cc` (the CompiledModel C API:
  B rows), `bench_common.h` and `build.sh` (NDK clang; no LiteRT library is linked; headers from the 1.4.0 AARs and
  `litert_cc_sdk.zip` listed in `manifest.json`).
- `tools/`: model inputs from the photos (`make_fixtures.py`), the Mac reference outputs (`mac_ref.py`), the scorer
  (`score.py`) and the table (`table.py`). The scorer also reads a record of each run (`<model>_<row>.leg.json`)
  that the phone-side script writes; that script is not included.
- `manifest.json`: the models, the Maven artifacts, the header zip and the photos, each with its URL and sha256; none
  of them is in this directory. The QAIRT libraries (2.50.0.260828, from Qualcomm) are not included.
- `forms.tsv`: the runner flags of each row. `TABLE.md`, `TABLE.json`: the table and every number in it.
