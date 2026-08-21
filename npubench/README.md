# npubench

Latency harness for the Qualcomm Hexagon NPU, run as an instrumentation test so it
works on a cloud device farm where there is no shell. It reports median and minimum
over 50 iterations, plus the device's thermal headroom at the start of each
measurement. Background and results: the **Snapdragon NPU** section of the root
[README](../README.md).

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

⛔ **A run that produces output has not necessarily used the NPU.** LiteRT logs a
warning and continues on CPU when the dispatch library directory is missing. Judge from
`NPU accelerator registered.` in logcat.
