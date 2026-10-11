# Phocinae decide — typed decisions about a text, on the phone

Type or paste a situation (a deploy run, a support ticket, an invoice, a security alert), pick a question set and tap
**Decide**. Each question gets one answer with its probability: a choice between options you define, a level on a
rubric, or yes / no. Four question sets are bundled with invented examples; the state text is yours to edit. The model
is [Phocinae-Largha-150M-v1](https://huggingface.co/Phocinae/Phocinae-Largha-150M-v1) (the Phocinae project,
Apache-2.0), a 144 M-parameter decision model on mmBERT-small that answers from one forward pass per question and
generates no text. The same graph runs on the GPU (FP32 or FP16), the Qualcomm Hexagon NPU, or the CPU.

## Model and requirements

- Model download: [litert-community/Phocinae-Largha-150M-LiteRT](https://huggingface.co/litert-community/Phocinae-Largha-150M-LiteRT)
  (its card holds the conversion notes, the checks and the Galaxy S26 numbers).
- Upstream: [Phocinae/Phocinae-Largha-150M-v1](https://huggingface.co/Phocinae/Phocinae-Largha-150M-v1) v1.1, commit
  `38071574`, `model.safetensors` sha256 `b6472511…`; the publisher's runtime is
  [phocinae-server](https://github.com/Phocinae/phocinae-server), the reference of every check.
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile/target SDK 35.
- Runtime: LiteRT **2.2.0** `CompiledModel`; Kotlin and Material 1 Compose with MVVM.
- NPU (optional): Qualcomm runtime libraries packaged in the APK; see [NPU](#npu).
- Build: JDK 17, Android SDK platform 35 and build-tools 35.0.0.

The accelerator is an explicit choice: **GPU FP32** requests `GpuOptions(precision = FP32)`, **GPU FP16** requests
`Precision.FP16` (the fastest form that passed the parity bar on a Galaxy S26, with less margin on near ties), **NPU**
compiles the graph on the device (JIT, cached after the first launch) and computes in fp16, **CPU** uses XNNPACK with
four threads. A compile error appears inline with the runtime message; **Use CPU** is an explicit user choice. The app
remembers the selected accelerator; the first launch selects the NPU when its libraries are packaged.

## Download, build and install

Commands run from this module directory. Model files are external and are not required to build the APK. Install
Android platform-tools (`adb`) and the Hugging Face CLI (`hf`) for device installation. The installer uses Python 3.9+
and SDK Build Tools 35.0.0 `aapt` to verify that the APK is the debuggable `com.phocinae` app. Set `ANDROID_HOME` to
your SDK directory, or provide the `AAPT` executable explicitly.

```bash
export HF_HUB_DISABLE_XET=1
export HF_HOME="$PWD/.cache/huggingface"
export PHOCINAE_MODEL_DIR="$PWD/model-files"
hf download litert-community/Phocinae-Largha-150M-LiteRT \
  phocinae_largha_150m_L256_fp16fc_embeds.tflite phocinae_largha_150m_L384_fp16fc_embeds.tflite \
  phocinae_act_head_fp32.tflite rl_agent_config.json \
  tokenizer/tokenizer.json tokenizer/tokenizer_config.json tables/tok_embeddings_fp16.bin tables/tok_embeddings.json \
  --local-dir "$PHOCINAE_MODEL_DIR"
mv "$PHOCINAE_MODEL_DIR"/tokenizer/* "$PHOCINAE_MODEL_DIR"/tables/* "$PHOCINAE_MODEL_DIR"/   # the installer reads a flat directory
./scripts/fetch_npu_libs.sh /path/to/npu-libraries   # optional; see NPU below
./gradlew --no-daemon clean :app:assembleDebug
./scripts/install_to_device.sh --assets "$PHOCINAE_MODEL_DIR" --windows 256,384 --validate-only
export ANDROID_SERIAL=your-device-serial
adb -s "$ANDROID_SERIAL" install -r app/build/outputs/apk/debug/app-debug.apk
./scripts/install_to_device.sh --assets "$PHOCINAE_MODEL_DIR" --windows 256,384
adb -s "$ANDROID_SERIAL" shell am start -n com.phocinae/.MainActivity
```

The installer validates the selected files against `scripts/assets.json` (sizes and SHA-256 of the published files)
before contacting the device, stages them and copies them into this app's private files with `run-as com.phocinae`.
Install the debug APK first because `run-as` needs a debuggable package. `--windows` selects the graphs to install
(256, 384, 512); the editor uses the 384-token window.

## External files

The 384-token graph and the shared assets total **324,776,658 bytes**; the 256-token graph adds 92,680,128 bytes. The
embedding table is memory-mapped read-only. Hashes and exact sizes are checked by the installer manifest.

| File | Bytes | Purpose |
|---|---:|---|
| `phocinae_largha_150m_L256_fp16fc_embeds.tflite` | 92,680,128 | Window 256 main graph; FP16 fully connected weights; `inputs_embeds` |
| `phocinae_largha_150m_L384_fp16fc_embeds.tflite` | 93,400,928 | Window 384 main graph (the editor's window) |
| `phocinae_largha_150m_L512_fp16fc_embeds.tflite` | 94,384,064 | Window 512 main graph (optional) |
| `phocinae_act_head_fp32.tflite` | 402,560 | The act head (escalation probability) |
| `rl_agent_config.json` | 596 | The publisher's temperatures per question type and the head budget |
| `tokenizer.json`, `tokenizer_config.json` | 34,363,188 / 553 | The checkpoint tokenizer (256k Gemma-style BPE) |
| `tok_embeddings_fp16.bin` | 196,608,000 | Little-endian FP16 token table, shape `[256000, 384]`, the checkpoint's rows unchanged |
| `tok_embeddings.json` | 833 | Table shape, hash, source and PAD policy |

For the debug numerical gate, also pass `--gate-rows gate_rows_s384.json` to the installer (1,597 public test decisions
with the reference logits, 7.3 MB; see [scripts/GATE_ROWS.md](scripts/GATE_ROWS.md)). The four question sets are in
`app/src/main/assets/presets.json`.

## Architecture and behavior

```text
State + preset questions → Kotlin BPE tokenizer and prompt builder (the laya contract, head 192 tokens)
  → memory-mapped token embedding lookup → LiteRT main graph (window 384)
  → raw-probability features + LiteRT act head
  → temperature decoder → immutable UI state → Compose cards
```

`MainActivity` observes immutable state and `MainViewModel` owns the engine on a serialized model dispatcher. A complete
tokenizer → builder → embedding → graph → decoder pass warms the pipeline before **Ready**; the status header reports
loading, compilation and launch-to-Ready time. Each question set runs three question rows sequentially; the header's
token count sums their real sequence lengths. The state text is right-truncated after the question and its options
are placed (the publisher's runtime cuts the same way at 512 tokens).

An answer card shows the most probable option and its probability: the choice label, the rubric level's text, or Yes /
No, with the other options' probabilities in one line below. The publisher's server answers the same way (choice =
the most probable option, score = the most probable level, yes/no = p(true) ≥ 0.5). **T=1 (uncalibrated)** switches the
temperatures off; the act features use the raw probabilities in both modes. Confidence is the contract's entropy or
binary confidence, not a measured correctness rate.

On an NPU run the main graph runs on the NPU and the act head stays on the CPU. Graph timings cover input writes,
`run()` and output readback for the main and act graphs; the embedding lookup is measured separately; the per-question
time on a card also includes host preparation and decoding.

## Reproduce the device gate

Install the debug APK, the models and the gate rows, then:

```bash
export ANDROID_SERIAL=your-device-serial
adb -s "$ANDROID_SERIAL" shell am force-stop com.phocinae
adb -s "$ANDROID_SERIAL" shell am start -W -n com.phocinae/.MainActivity \
  --ez gate true --es window 384 --es accel npu
adb -s "$ANDROID_SERIAL" exec-out run-as com.phocinae cat files/phocinae_gate_npu_384.json > gate_npu_384.json
```

`--es accel gpu`, `gpu_fp16`, `npu` or `cpu`; force-stop the app between configurations. The report holds, per row, the
on-device token ids and marker positions (compared with the fixture's), the raw logits and their difference from the
reference, and the timings; completion is logged under `PHOCINAE_GATE`. The gate entry is debug-only; release launches
the UI. `app_gate_score.py` in the model repository's `conversion/` scores a report with the lane's bar.

## Verified on

(filled from the device gate reports: see the section below)

## NPU

The NPU needs Qualcomm runtime libraries that are not in git. Collect the ten files listed in
[scripts/fetch_npu_libs.sh](scripts/fetch_npu_libs.sh) (two LiteRT GitHub Release assets and the QAIRT libraries for
the phone's Hexagon version, v81 for the Galaxy S26) and run `scripts/fetch_npu_libs.sh <directory>` before building.
The root README section *Running on the NPU* explains where each file comes from. Without them the NPU option is
disabled and the GPU and CPU paths are unchanged.

The first NPU launch compiles the graph on the phone (about 11 s at the 384-token window) and caches it; later launches
load it in about 0.2 s. The NPU computes in fp16, so the graph carries an exact rewrite: the LayerNorms of encoder
layers 12–21 and the final norm compute on their input scaled by 2⁻⁶ with epsilon scaled by 2⁻¹² (their squares reach
2.0e7, over float16's 65,504), and the attention masks use −1e4 instead of −1e9. In fp32 the outputs are bit-identical
to the plain graph.

## License

This sample is [Apache-2.0](LICENSE). Phocinae-Largha-150M-v1 (model, tokenizer, config) is Apache-2.0 with the
publisher's NOTICE; mmBERT-small declares MIT; the host-side Kotlin port follows the Laya 0.3.4 protocol (Apache-2.0).
License texts and attributions are retained under [licenses/](licenses/README.md) and in [NOTICE](NOTICE).
