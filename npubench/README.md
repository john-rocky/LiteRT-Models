# npubench

Latency harness for the Qualcomm Hexagon NPU, run as an instrumentation test so it
works on a cloud device farm where there is no shell. It reports median and minimum
over 50 iterations, plus the device's thermal headroom at the start of each
measurement. Background and results: the **Snapdragon NPU** section of
[MODELS.md](../MODELS.md#snapdragon-npu-hexagon).

## What you have to fetch

The model files and the vendor runtime libraries are not in git — `libQnnHtpPrepare.so`
alone is 82 MB. Put them here before building:

```
app/src/main/assets/
  ssdlite_npu_aot_sm8650.tflite     # AOT-compiled, see root README
  ssdlite_stock.tflite              # the stock fp16 file, for GPU and CPU
  twinlite_npu_aot_sm8650.tflite
  twinlite_stock.tflite
  silentface_npu_aot_sm8650.tflite
  silentface_stock.tflite

app/src/main/jniLibs/arm64-v8a/
  libLiteRtDispatch_Qualcomm.so     # litert_npu_runtime_libraries.zip (GitHub Release)
  libQnnHtp.so                      # QAIRT lib/aarch64-android/
  libQnnSystem.so                   #  "
  libQnnHtpV75Stub.so               #  "   — v75 is SM8650; match your target SoC
  libQnnHtpPrepare.so               #  "
  libQnnHtpV75Skel.so               # QAIRT lib/hexagon-v75/unsigned/
```

`_stock` files are the ones from each model's own module under this repo. The AOT
artifacts are produced by the compile step in the root README.

## Running it

```bash
./gradlew :app:assembleDebug :app:assembleDebugAndroidTest

gcloud firebase test android run --project <your-project> --type instrumentation \
  --app app/build/outputs/apk/debug/app-debug.apk \
  --test app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk \
  --device model=SC-51E,version=36,locale=en,orientation=portrait \
  --test-targets "class com.litertzoo.npubench.NpuBenchmarkTest#a01_npu_ssdlite"
```

`SC-51E` is a physical Galaxy S24. Results land in logcat under the `NpuBench` tag; pull
it from the run's GCS bucket and grep for `BENCH`.

**Run one accelerator per invocation.** LiteRT's `Environment` is shared across a
process and the first model load fixes the dispatch options for every later one, so a
CPU case running first leaves the NPU without its Qualcomm options — that cost 6.2x on
ssdlite before we pinned it. `@FixMethodOrder` alone does not fix it: pinning the NPU
first corrected the NPU and made the GPU 29% slower instead. The `d01`–`d03` methods
exist to measure that drift directly.

**Read every output back, warm-ups included.** On the GPU `run()` returns before the
work is done (on the CPU it returns when done; the NPU was not checked), so
`NpuBenchmark` reads all outputs back after every warm-up and every timed call. A
warm-up loop without the read-back leaves forwards queued, and the first timed call
pays for them: with the 1 s graphs of the `reuse/` speech model (15,102–56,562 nodes)
on the GPU (LiteRT 2.2.0, 5 warm-ups), the first timed call took as long as about 6
forwards on a Pixel 11 Pro, 4 on a Galaxy S26 and 1.1–1.5 on a Pixel 8a. If your own
harness skips the read-back, time at least 2 iterations and quote the second.

⛔ **A run that produces output has not necessarily used the NPU.** LiteRT logs a
warning and continues on CPU when the dispatch library directory is missing. Judge from
`Replacing N out of M node(s) with delegate (DispatchDelegate)` in logcat, with N
equal to M. `NPU accelerator registered.` is not enough: it also appears on a Pixel 11
Pro, which has no Qualcomm NPU, and on a Galaxy S26 run whose NPU compile had failed
and whose 15,102 nodes all went to XNNPACK.
