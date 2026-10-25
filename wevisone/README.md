Pick a photo and type a word: the phone draws the box around that thing and cuts it out with the background removed (a transparent PNG), all on the device; the same screen also detects, points, counts and reads text.

# WeVisOne Cut-out — WeVisOne-VL-2B on LiteRT (Android)

[WeVisOne-VL-2B](https://huggingface.co/tencent/WeVisOne-VL-2B) (Tencent, Apache-2.0) is a 2B vision-language model that answers with boxes, points and text, and marks each object it should cut out with an `<OBJ>` token that a small alpha-matte decoder turns into a mask. This app runs the whole path as LiteRT graphs on `CompiledModel`: the vision encoder, the language model (greedy decoding with its KV cache), and the matte decoder. LiteRT-LM cannot return the `<OBJ>` hidden state the matte needs, so the app drives the raw graphs itself.

<img src="https://huggingface.co/litert-community/WeVisOne-VL-2B/resolve/main/assets/demo.gif" width="300" alt="The app on a Galaxy S26: the kitten cut out of a photo of a kitten and a puppy"> <br> *On a Galaxy S26: a photo of a kitten and a puppy (by blende12 on Pixabay, CC0, from Wikimedia Commons), the words "the kitten", and the cut-out after 4.9 s. It is not the app's first run after the install, which builds the GPU graphs (35.6 s on this install; the table's 35.7 s is another install).*

| Phone | Photo + words → cut-out | Steps | Memory |
|---|---|---|---|
| Galaxy S26 (12 GB), GPU | **4.7 s** (640 × 480 synthetic picture), 5.8 s (3000 × 2029 photo) | vision encoder 0.54 s, adapter 0.06 s, prompt 296 tokens 1.17 s, answer 13 tokens 1.19 s (91 ms a token), matte decoder 1.57 s, matte to photo size 0.04 s | all three graphs loaded: free memory 6.6 → 1.6 GB, app peak 6.1 GB |
| Galaxy S26, first run after install | 35.7 s | building the graphs: vision 8.6 s, language model 20.3 s, matte decoder 1.6 s | |
| Pixel 8a (8 GB) | does not run | the app stops before loading anything: 3.3 GB free, 4.1 GB needed (an earlier build that checked one stage at a time started the vision graphs, and they alone took the free memory from 3.0 to 0.3 GB on this phone's GPU) | |

Galaxy S26 SM-S942Q (Android 16), LiteRT 2.3.0, 2026-10-25: the median of four runs after the first, every graph on the GPU (OpenCL); the language model with int8 weights and float32 arithmetic (`GPU (int8 weights, fp32)`, the app's default). Memory is in GB of 2^30 bytes, the unit of the app's messages. With fp16 weights the language model takes 4.5 GB instead of 3.0 GB, the vision graphs no longer fit beside it and are rebuilt for every photo: 15.7 s per cut-out. On the CPU (XNNPACK unpacks the fp16 weights to float32) the app refuses the language model on a Galaxy S26 (8.7 GB needed, 7.3 GB free); with the check turned off it gives the reference's answer token for token but takes 132 s, 5.4 s a token, while the phone swaps. The fp16 number was measured with an earlier build of the app (the same graphs), the CPU and Pixel 8a numbers with the first version of this sample; the app's model code is unchanged since, and only the Stop button, the name shown for a picked photo and the way the app finds the camera's photo again after Android has restarted it changed.

## Use

1. Build and install the debug build of the app (below).
2. Download the model files (Files, below).
3. Copy them to the phone: `./scripts/install_to_device.sh`. It reads `~/Downloads/WeVisOne-VL-2B-LiteRT/cutout` unless you give another directory, installs the int8 language model (`FORM=fp16` the fp16 one, `FORM=both` both), and downloads three NASA example photos unless `EXAMPLES=0`.
4. Open **WeVisOne Cut-out**. Choose a photo (**Photos**, **Camera**, or **Example**, which also fills in a task and words), pick a task, type the words, tap **Run**.
5. Boxes, points and labels appear on the photo; a cut-out is tinted. Tap the photo to see the cut-out alone on a checkerboard, **Save PNG** writes it with its transparency to `Pictures/WeVisOne`.

| Task | Type | The instruction the model gets (the authors' `build_prompt`) |
|---|---|---|
| Cut out | what to cut out: `the man on the left` | `Locate the instances that match the following description and provide the alpha mattes: the man on the left.` |
| Detect | labels, one per line or comma-separated: `person, hat` | `Locate all the instances that match the following description: person. hat.` |
| Find one | a description: `the yellow rectangle` | `Locate the instances that match the following description: the yellow rectangle.` |
| Point | labels: `airplane, windsock` | `point to: airplane. windsock` |
| Count (boxes / points) | what to count: `blue circles` | `How many blue circles in the image? Please count them with boxes.` |
| Read text | nothing | `detect all the word in the box format` |
| Find text | the text: `EXIT 12` | `Locate the following text in the image: EXIT 12` |

The app stops with a message, and loads nothing, when no photo is chosen, when a model file is missing or its SHA-256 differs, and when the phone does not have the free memory a stage needs; when the answer names nothing it says "No match".

## How it runs

| Step | Where | Notes |
|---|---|---|
| Photo → 512 × 512 and 1024 × 1024 | Kotlin | Pillow 12.2.0's bicubic resampler ported byte for byte (`PilBicubic.kt`): the vision encoder answers differently on other resamplers |
| Vision encoder (re-authored Qwen3-VL tower, 1024 patches) + adapter (merger and the three DeepStack mergers side by side, 8196 wide) | GPU, fp16 weights, fp32 | one graph each |
| Prompt | Kotlin | the fixed system turn `You are a helpful assistant.`, the image first, the instruction; byte-level BPE from the checkpoint's `tokenizer.json` (`QwenBpeTokenizer.kt`) |
| Token embeddings | Kotlin | rows of the checkpoint's table as float16 (`wevisone_embed_tokens_fp16.bin`, memory-mapped) |
| Language model (28 layers, 7 signatures: prefill 1024 / 256 / 64 / 16 / 4 / 1 and decode, KV cache 2048) | GPU, int8 weights (dynamic range, per channel), fp32; or fp16 weights with fp16 storage and fp32 accumulation; or the CPU | the DeepStack features are added inside the graph; the derived M-RoPE positions are computed inside the graph from the image flag |
| `<OBJ>` hidden state | Kotlin | the decode step whose input token is `<OBJ>` returns it |
| Box encoding | Kotlin | Fourier features of x1, y1, 0.2 log w + 0.5, 0.2 log h + 0.5 up to 2^31 cycles: the float32 phase of torch, and the log values torch gives (a 28 KB table), since one float32 ulp of the box value moves the matte |
| Matte decoder (Deformable-DETR-style decoder over the vision features at 1024 × 1024) | GPU, fp16 | GroupNorm written as two-level means, its one-query attention as a plain projection (both exact) |
| Matte → photo size | Kotlin | sigmoid, bilinear (half-pixel) like `torch.nn.functional.interpolate` |

Graphs stay loaded while there is memory for them: before each stage the app reads `MemAvailable` and closes the graphs the stage does not use until the stage fits, or stops with a message when it would not fit with all of them closed.

## Files

The files are in the [`cutout/`](https://huggingface.co/litert-community/WeVisOne-VL-2B/tree/main/cutout) folder of [litert-community/WeVisOne-VL-2B](https://huggingface.co/litert-community/WeVisOne-VL-2B); the `.litertlm` file at the top of that repository is the LiteRT-LM build of the model (boxes, points, counts and text, no cut-outs) and is not used by this app. The default install (the int8 language model, 3,257,599,697 bytes):

```bash
export HF_HUB_DISABLE_XET=1
hf download litert-community/WeVisOne-VL-2B \
  cutout/tokenizer.json cutout/wevisone_embed_tokens_fp16.bin \
  cutout/wevisone_vision_encoder_512_fp16.tflite cutout/wevisone_vision_adapter_512_fp16.tflite \
  cutout/wevisone_mask_decoder_gs16.tflite cutout/wevisone_decoder_b_int8.tflite \
  --local-dir "$HOME/Downloads/WeVisOne-VL-2B-LiteRT"
```

Add `cutout/wevisone_decoder_b_fp16.tflite` to the list for `FORM=fp16` or `FORM=both`.

| File | Bytes | SHA-256 |
|---|---|---|
| `wevisone_vision_encoder_512_fp16.tflite` — vision encoder (fp16 weights) | 611,814,624 | `47d4089213cfdb6da006263ac92cfee92e8e2ff36c4e59027e4fe16dce2e837a` |
| `wevisone_vision_adapter_512_fp16.tflite` — vision adapter (fp16 weights) | 201,460,224 | `d0c8dc73b2a5a358a3bb06438d643cadbc454245f011d89853d184d9d646c530` |
| `wevisone_decoder_b_int8.tflite` — language model, int8 weights (the default form) | 1,785,462,176 | `f93bf881dccd203a1e55d8fa43cf51798739a5aa5fe441ad7cb3b53808579a6f` |
| `wevisone_decoder_b_fp16.tflite` — language model, fp16 weights (the fp16 and CPU forms) | 3,458,517,312 | `56dc97e27115ab8be102c8e8eb543895735a0eb2d15367fd7a44a736b3ee9864` |
| `wevisone_mask_decoder_gs16.tflite` — matte decoder (fp16 weights) | 21,878,000 | `ac0765dc75ce81708564421bfe930a7714b589bae43edac2ddc810c57fa5f7e1` |
| `wevisone_embed_tokens_fp16.bin` — token embeddings, float16 [152680, 2048] | 625,377,280 | `5ff0fcb29afce21c5efc2c717fe723e4329443373513ba53b7c33d18508050fe` |
| `tokenizer.json` — the checkpoint's tokenizer | 11,607,393 | `3bb172429b952469b51c64783f52fa01c26d1b30f1f31a4693e385a43f9cfc5e` |

## Build

Android Studio, or from this directory: `./gradlew :app:installDebug` (the install script needs the debug build: `run-as` copies the files into the app's private storage). `minSdk` 26, `arm64-v8a`, LiteRT 2.3.0 (`litert` and `litert-gpu`), Kotlin 2.2.21 with `-Xskip-metadata-version-check` (LiteRT 2.3.0's Kotlin API carries Kotlin 2.4 metadata).

JVM tests: `./gradlew :app:testDebugUnitTest` runs the prompt and parsing checks on the committed public rows; with `-Pwevisone.golden=<dir>` (the conversion run's reference data, `scripts/TEST_DATA.md`) they also check the tokenizer on all 29 fixture rows, the resampler against Pillow 12.2.0 byte for byte, the box encoding against torch and the matte resize against torch.

## Limits

- One photo and one instruction per run: no follow-up turns, the system turn is fixed, and a conversation of text alone is not what this app is for.
- The model sees the photo at 512 × 512, not at the authors' default resolution, so its answers can differ from the authors' package: on 29 test requests, 47 of the 56 boxes the fp32 model gives at the package's resolution (greedy) are matched at 512 × 512.
- Memory: the app needs about 4.1 GB of free memory before it starts (the language model and its buffers take 3.0 GB on a Galaxy S26). It runs on the 12 GB Galaxy S26; the 8 GB Pixel 8a stops with the message.
- The answers follow the authors' fp32 model on the rows checked below, not everywhere: lower-precision weights move a coordinate by one bin where the model's own top two choices are nearly tied.

## How the answers were checked

Against the authors' model in float32 (PyTorch, the same 512 × 512 view of the photo, greedy decoding):

- The prompt: the app's token ids equal the reference processor's on 29 of 29 requests (JVM test).
- The pixels: the app's photo decoding and its 512² and 1024² resizes equal PIL's byte for byte (JVM test on 17 images; on the phone the decoded photos of 17 requests, one of them a JPEG, equal the reference pixels).
- Cut-outs on the Galaxy S26 (six requests, among them the synthetic `the green circle` and the NASA photo's `the man on the left`): alpha MAE at most 0.0032 and IoU(alpha > 0.5) at least 0.984 against the reference matte; the box token sequence is the reference's on two of the six (int8 weights move a box edge by one bin where the reference's top two coordinates are nearly tied).
- Boxes, points, counts and text on the Galaxy S26 (ten requests): 18 of 18 reference boxes matched at IoU ≥ 0.75, 9 of 9 points within 2 % of the side, 3 of 3 counts, character error rate 0 on the text.

## License

The sample's code: Apache-2.0 (`LICENSE`); what it ports and the example images: `NOTICE`. The model: WeVisOne-VL-2B, Apache-2.0 (`licenses/WeVisOne-VL-APACHE-2.0.txt`).
