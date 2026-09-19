# GLiNER2.5 Small — Android entity extraction

Enter English text to find people, organizations, locations, products and dates on your phone.
The Compose app highlights entities, lists confidence and Unicode character offsets, and reports
separate tokenization/embedding, graph and decoding times. Choose GPU or CPU and tap **Extract**.
For the prefilled sentence, “Maya Chen from Orvane Robotics demonstrated the Veltrix 9 in Lisbon
on March 12, 2025.”, the five entities are Maya Chen, Orvane Robotics, Veltrix 9, Lisbon and
March 12, 2025.

## Model and requirements

- Model: [litert-community/GLiNER2.5-Small-LiteRT](https://huggingface.co/litert-community/GLiNER2.5-Small-LiteRT),
  revision `8c4759df4feb91497e7f49212abf056d79789cac`.
- Upstream: [fastino/gliner2.5-small-v1](https://huggingface.co/fastino/gliner2.5-small-v1), Apache-2.0.
  Retained dependency notices are in `licenses/`.
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile/target SDK 35.
- Runtime: LiteRT **2.2.0** `CompiledModel`; Material 1 Compose with MVVM.

**Explicit GPU FP32 precision is mandatory:** default GPU precision produces NaN on this graph.
The app uses `CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)`.
`wfp16` describes stored weights; host tensors and computation remain float32.

## Download, build and install

Use JDK 17, Android SDK platform 35 / build-tools 35.0.0, Android platform-tools (`adb`), and the
Hugging Face CLI (`hf`). Open this directory in Android Studio or build from its root:

```bash
export HF_HUB_DISABLE_XET=1
hf download litert-community/GLiNER2.5-Small-LiteRT \
  --local-dir "$HOME/Downloads/GLiNER2.5-Small-LiteRT"
./gradlew :app:assembleDebug
# Optional when multiple devices are connected:
# export ANDROID_SERIAL=your-device-serial
adb install app/build/outputs/apk/debug/app-debug.apk
./scripts/install_to_device.sh "$HOME/Downloads/GLiNER2.5-Small-LiteRT"
adb shell am start -n com.gliner25/.MainActivity
```

The install script's argument defaults to the download directory above. It validates all nine
files before contacting the device, pushes through `/data/local/tmp/gliner25/`, copies with
`run-as com.gliner25` into private `files/`, removes temporary files, and lists the installed files.
It uses plain `adb`, including its optional `ANDROID_SERIAL` selection. Install the debug APK first;
`run-as` requires a debuggable package. Model files are external and are not needed to build the APK.
The Gradle project creates its own local debug signing key. Missing assets appear as inline errors
naming the first missing file and `scripts/install_to_device.sh`.

## External files

The app loads graphs by path and memory-maps the embedding table. The installer flattens
`host_assets/` into the app's `files/` directory; all nine files total **407,515,254 bytes**.

| Downloaded file | Bytes | Purpose |
|---|---:|---|
| `gliner25_small_s128_wfp16.tflite` | 54,051,424 | 128 encoded slots / 48 text words |
| `gliner25_small_s256_wfp16.tflite` | 63,906,288 | 256 encoded slots / 192 text words |
| `gliner25_small_s512_wfp16.tflite` | 84,111,584 | 512 encoded slots / 384 text words |
| `host_assets/word_embeddings_fp32.bin` | 196,624,896 | Float32 `[128011,384]` embedding table |
| `host_assets/tokenizer.json` | 8,341,713 | SentencePiece Unigram vocabulary and normalization |
| `host_assets/sparse_decoder_fp32.safetensors` | 467,956 | 16 sparse-decoder tensors |
| `host_assets/graph_contract_s128.json` | 3,772 | 17 packed output slices |
| `host_assets/graph_contract_s256.json` | 3,807 | 17 packed output slices |
| `host_assets/graph_contract_s512.json` | 3,814 | 17 packed output slices |

The 3,152-byte checkpoint configuration is bundled as `app/src/main/assets/gliner_config.json`.
The phone does not need Python or `tokenizer_config.json`.

## Architecture and limits

```text
Text + five-label schema → Kotlin word splitter / Unigram tokenizer
  → padded masks and routing + memory-mapped embedding lookup
  → LiteRT dense graph (GPU FP32 or CPU) → unpack 17 float32 tensors
  → Kotlin sparse candidate pooling / scoring / decoding → spans and Compose highlights
```

The host ports follow gliner2 2.0.0 and the published
[host contract](https://huggingface.co/litert-community/GLiNER2.5-Small-LiteRT/blob/main/HOST_CONTRACT.md).
`MainActivity` observes immutable state; `MainViewModel` owns the helper and confined dispatcher.
One LiteRT Environment is shared per process. s128 loads at startup; s256/s512 compile on first use
and remain resident. Each window/backend receives an untimed warm-up. Graph timing includes the
first input-buffer write through output readback because `run()` is asynchronous.

English and the five fixed labels are supported: person, organization, location, product, date.
The smallest fitting N/T window is chosen from 128/48, 256/192, 512/384. Both schema-plus-text encoded
length and text word count must fit; input above **512 encoded tokens / 384 words** is rejected
without truncation. Offsets are half-open Unicode code-point indices, not UTF-16 indices. Confidence
threshold is 0.5. Overlapping highlights prefer higher confidence. Arbitrary schemas, multilingual
quality, batching and NPU execution are outside the validated configuration.

## Reproduce parity

The debug APK bundles reference spans and captured tokenizer positions. Stop this app before
starting a new check so previous Activity extras are cleared:

```bash
adb shell am force-stop com.gliner25
adb shell am start -W -n com.gliner25/.MainActivity --ez gate true
# Extended set: append --es set f1; single backend: append --es accel GPU or --es accel CPU.
adb exec-out run-as com.gliner25 cat files/gate/gate_GPU.json > gate_GPU.json
```

The default checks ten texts, GPU FP32 then CPU, with one warm-up and five timed runs per text.
`--es set f1` checks 70 texts, one warm-up and three timed runs, plus element-wise token IDs and
text/query positions. Reports land in private `files/gate/gate_GPU.json` / `gate_CPU.json`, or
`gate_f1_GPU.json` / `gate_f1_CPU.json`; completion lines use log tag `GLINER_GATE`.
Span sets must match the reference, confidence differences must be at most `5e-3`, and outputs
must be finite. A normal launch without extras displays the interactive screen.

`./gradlew :app:testDebugUnitTest` skips the six external-data JVM tests unless
`-Pgliner.fixtures=/path/to/data` (or `-Dgliner.fixtures=...`) is supplied. Downloaded `host_assets/`
and separate parity fixtures belong under that directory. Packed references are **not distributed**;
[scripts/TEST_DATA.md](scripts/TEST_DATA.md) explains their layout and regeneration with the published
Python runtime. Test reports are written only under `app/build/`. With complete reference data,
six tests pass: 80 fixture files / 70 unique texts / 225 file-window checks and 195/195 decoder pairs.
Maximum confidence differences are `4.172325134277344e-7` against Python on identical packed data
and `0.002693772315979004` against the official fp32 reference.

## Verified on

**Samsung Galaxy S26 SM-S942Q, Android 16, LiteRT 2.2.0, debug build**, 2026-09-19.
GPU uses explicit **FP32 precision**; CPU uses four threads. Start: **100% battery, 32.0 °C**, USB
connected, warm device after earlier validation. GPU then CPU ran in one process; **70 inputs,
one warm-up + three timed runs per input**. The validation texts use invented person names.
Medians below include all timed runs per window; compilation and input inspection are excluded.

| Backend | Window N / T | Inputs | Timed runs | Tokenize + embed (ms) | Graph to readback (ms) | Decode (ms) |
|---|---|---:|---:|---:|---:|---:|
| GPU FP32 | 128 / 48 | 60 | 180 | 1.53 | 12.01 | 58.49 |
| GPU FP32 | 256 / 192 | 5 | 15 | 4.74 | 29.55 | 78.37 |
| GPU FP32 | 512 / 384 | 5 | 15 | 9.71 | 94.13 | 82.09 |
| CPU | 128 / 48 | 60 | 180 | 1.93 | 23.46 | 84.43 |
| CPU | 256 / 192 | 5 | 15 | 4.72 | 50.55 | 83.49 |
| CPU | 512 / 384 | 5 | 15 | 10.31 | 111.92 | 89.07 |

Both backends pass **70/70 exact tokenizer inputs**, **70/70 span sets / 400 spans**, and all
280 warm-up/timed finite-output checks. Maximum confidence differences are
**0.00269240140914917 (GPU)** and **0.002690911293029785 (CPU)**. All three windows compiled,
including lazy s256/s512. GPU delegation is 1120/1120 nodes at s128 and 1121/1121 at s256/s512,
each in one LITERT_CL partition. Temperature after both backends was 35.1 °C; battery stayed 100%.
The launch screen shows the invented-name example and Ready status, with all five label chips.
NPU execution has not been tested.
