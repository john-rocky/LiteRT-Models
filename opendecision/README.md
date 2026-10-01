# Open Decision — typed decisions on Android (DeBERTa-v3-large)

Type a short text (the *state*) and a few typed questions about it, one per line, and tap **Decide**.
A `choice` question returns the winning option with its probability, a `score` question the expected
level on an ordered scale, and a `noul` question the probability of yes. One forward pass answers every
question. For the prefilled support ticket (an invented message) the app answers team `shipping`,
frustration `frustrated` (expected level 1.86 of 0–3) and refund p(yes) 0.12, the same answers as the
author's implementation on the desktop.

## Model and requirements

- Model: [litert-community/Open-Decision-DeBERTa-v3-Large-LiteRT](https://huggingface.co/litert-community/Open-Decision-DeBERTa-v3-Large-LiteRT)
  (float16-weight graphs for 256 and 512 tokens, the float16 word table, the source `tokenizer.json`).
- Upstream: [com-kotobalabs/open-jev-deberta-v3-large](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large)
  revision `188ee67a5c93122b916e5acd5bdb0cb3623e380a`, Apache-2.0; DeBERTa-v3-large encoder (MIT). English only.
- Semantics: the checkpoint's `typed_decisions` package (`Collator.encode_one`, span-pool head, `decide()` read-out
  at temperature 1.05).
- Android: arm64-v8a, Android 8.0 / API 26 or newer; compile/target SDK 35.
- Runtime: LiteRT **2.2.0** `CompiledModel`; Material 1 Compose with MVVM.

**Explicit GPU FP32 precision is required.** At the default GPU precision every output of these graphs is
non-finite (the attention mask constant is −3.4e38, which is −inf in fp16). The app uses
`CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP32)`. `wfp16` describes the
stored weights; host tensors and computation stay float32.

## Download, build and install

Use JDK 17, Android SDK platform 35 / build-tools 35.0.0, Android platform-tools (`adb`) and the Hugging
Face CLI (`hf`). The app needs four files of the model repository (1,796,900,898 bytes):

```bash
export HF_HUB_DISABLE_XET=1
hf download litert-community/Open-Decision-DeBERTa-v3-Large-LiteRT \
  deberta_v3_large_decision_s256_wfp16.tflite deberta_v3_large_decision_s512_wfp16.tflite \
  word_embeddings_fp16.bin tokenizer.json \
  --local-dir "$HOME/Downloads/Open-Decision-DeBERTa-v3-Large-LiteRT"
./gradlew :app:assembleDebug
# Optional when multiple devices are connected:
# export ANDROID_SERIAL=your-device-serial
adb install app/build/outputs/apk/debug/app-debug.apk
./scripts/install_to_device.sh "$HOME/Downloads/Open-Decision-DeBERTa-v3-Large-LiteRT"
adb shell am start -n com.opendecision/.MainActivity
```

The install script pushes through `/data/local/tmp/opendecision/`, copies with `run-as com.opendecision`
into private `files/` and removes the temporary files. Install the debug APK first; `run-as` requires a
debuggable package. Model files are external and are not needed to build the APK.

| File | Bytes | Purpose |
|---|---:|---|
| `deberta_v3_large_decision_s256_wfp16.tflite` | 712,780,432 | 256-token window |
| `deberta_v3_large_decision_s512_wfp16.tflite` | 813,443,728 | 512-token window |
| `word_embeddings_fp16.bin` | 262,348,800 | Float16 `[128100,1024]` word table, widened to float32 on lookup |
| `tokenizer.json` | 8,657,170 | SentencePiece Unigram vocabulary, precompiled charsmap, added tokens |

## How it works

1. **Tokenizer** (`DecisionTokenizer.kt`): the published `tokenizer.json` without JNI. Added tokens are matched
   first, each segment is stripped and normalized with the SentencePiece precompiled charsmap (a double-array
   trie, applied per grapheme as Hugging Face tokenizers does), then Metaspace and the Unigram Viterbi lattice.
   This is the first DeBERTa-v3 tokenizer in this repository whose `tokenizer.json` keeps the `Precompiled`
   normalizer; the GLiNER2.5 samples' files use `Replace` + `NFC` instead.
2. **Sequence** (`DecisionInputs.kt`): `[CLS] [STATE] state[:256] ([Q] instructions ([OPT] option)+)+ [SEP]`, the
   text span of every question and option (markers excluded), the smallest window (256 or 512) that fits, and the
   four float32 inputs: the table rows, the attention mask, and two routing matrices `[128, N]` whose row j holds
   1/len over the text tokens of option j's question and of option j. Longer requests and more than 128 options
   are rejected, never truncated.
3. **Graph** (`DecisionModel.kt`): one `CompiledModel` per window and backend, named buffers `inputs_embeds`,
   `attention_mask`, `q_routing`, `o_routing` → `logits [1,1,1,128]`, one native thread, one process-wide
   `Environment`.
4. **Read-out** (`DecisionDecoder.kt`): per question, softmax at temperature 1.05 over its option slots;
   `choice` = best option, `score` = Σ i·pᵢ, `noul` = p(yes).

## Verification

- JVM (`scripts/TEST_DATA.md`): the Kotlin tokenizer and builder reproduce the official `Collator`'s ids, question
  spans and option spans on all 1,809 requests of the conversion run's fixtures (the author's public test files
  plus examples); edge strings match the official fast tokenizer; the read-out reproduces the official answers.
- On device (Galaxy S26, LiteRT 2.2.0, debug build with `--ez gate true --es accel GPU`, 160 fixture requests / 420 questions): the on-device tokenizer and builder produced the same ids and spans as the official `Collator` on 160/160 requests, and every question got the same winning option as the author's implementation (max probability difference 0.00069); graph interval median 301 ms over mixed windows (s256 299 ms, s512 620 ms) on an already warm phone (thermal status 2).
- The graphs themselves: on the Galaxy S26 GPU with explicit FP32 (LiteRT 2.2.0, debug gate app, 500 requests /
  1,302 questions of the author's public test files, 754 boundary questions) every question gets the same
  winning option as the author's implementation on CPU FP32; max probability difference 0.00083; 1785/1785 ops in
  one `LITERT_CL` partition; compile 3.7 s; warm median 697 ms per 512-token request [576, 1202] with the phone
  heating from thermal status 0 to 3 over the 6-minute run.

## Debug extras

`adb shell am start -n com.opendecision/.MainActivity --ez gate true --es accel GPU` runs the fixture gate
(`files/gate_fixtures.json`, written by the conversion run's `scripts/app_gate_fixtures.py`) and writes
`files/gate/gate_gpu.json`: ids and spans equal to the captured Python ones, same winning option per question,
max probability difference, and the graph interval per request.
