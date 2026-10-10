# Bonsai Image 4B — type a prompt, get a picture, on an Android phone

Type a sentence, tap **Generate**, and the phone draws it: a 256×256 image in 23.5 s on a Galaxy S26
(CPU, LiteRT `CompiledModel` with XNNPACK, 4 sampling steps, measured 2026-10-09 by this app's device
check). Nothing leaves the phone: the Qwen3 tokenizer, the text encoder, the diffusion transformer
(DiT) loop and the VAE decoder all run on the device, and the APK has no `INTERNET` permission.
512×512 works the same way with its own pair of graphs.

The model is [Bonsai Image 4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B), PrismML's
ternary-weight text-to-image model on the FLUX.2-klein-4B architecture, converted to three fixed-shape
`.tflite` graphs.

## Model files

From [litert-community/Bonsai-Image-ternary-4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B)
(sizes as of revision `8878f895`):

| File | Bytes | Used for |
|---|---|---|
| `textenc_int4.tflite` | 1,798,100,240 | text encoder (Qwen3-4B, top 9 layers pruned), both sizes |
| `dit_256_int4b32.tflite` | 2,267,356,304 | DiT, 256×256 |
| `vae_dec_256_fp32.tflite` | 198,831,180 | VAE decoder, 256×256 |
| `dit_int4b32.tflite` | 2,267,355,872 | DiT, 512×512 (optional) |
| `vae_dec_fp32.tflite` | 198,831,180 | VAE decoder, 512×512 (optional) |
| `pipeline_meta.json`, `tokenizer/vocab.json`, `tokenizer/merges.txt` | 6,577 / 2,776,833 / 1,671,853 | bundled in the APK by `prep_assets.sh` |

The three 256×256 graphs are 4.26 GB. They are pushed to the app's files folder and opened by path, never
copied into the APK.

## Run it in 5 lines

From this directory, with a phone on `adb`:

```bash
hf download litert-community/Bonsai-Image-ternary-4B textenc_int4.tflite dit_256_int4b32.tflite vae_dec_256_fp32.tflite pipeline_meta.json tokenizer/vocab.json tokenizer/merges.txt --local-dir bonsai-models
./prep_assets.sh bonsai-models
./gradlew :app:installDebug && adb shell am start -W -n com.bonsai.imagegen/.MainActivity
adb push bonsai-models/textenc_int4.tflite bonsai-models/dit_256_int4b32.tflite bonsai-models/vae_dec_256_fp32.tflite /sdcard/Android/data/com.bonsai.imagegen/files/
# 5. On the phone: type a prompt, tap Generate.
```

Line 3 starts the app once so that Android creates its files folder; the push in line 4 needs that folder.
The app reads the folder again when Generate is tapped, so it does not need a restart after the push.
For 512×512, push `dit_int4b32.tflite` and `vae_dec_fp32.tflite` to the same folder.

## Galaxy S26 numbers

Galaxy S26 (SM-S942Q, Snapdragon 8 Elite Gen 5, Android 16), LiteRT 2.1.3 `CompiledModel` on the CPU with
6 XNNPACK threads, 4 steps. The seconds are the app's own measurements: the total runs from the press of
Generate until the image is on screen, so it includes the three graph loads.

| Size | Text encoder | DiT, per step | VAE decoder | Graph loads | Total | Measured |
|---|---|---|---|---|---|---|
| 256×256 | 1.00 s | 3.71 / 3.79 / 3.77 / 4.71 s | 1.20 s | 4.77 s | **23.5 s** | 2026-10-09, device check, phone cold (thermal status 0) |
| 512×512 | 1.00 s | 11.23 / 14.73 / 14.76 / 14.74 s | 5.24 s | 3.79 s | **66.2 s** | 2026-09-28, earlier build of this app (same pipeline code) |

A second run straight after another is slower: the CPU clock ceiling drops after about 20 s of full load
(in the device check, the run after a cancelled one took 29.0 s). Peak memory (VmHWM) is about 4.6 GiB at 256×256 and 5.0 GiB at
512×512. The graphs load and close one at a time, so the peak stays near the DiT's size instead of the sum;
even so, Android closes background apps during a run. Treat 12 GB of RAM as the practical target and 8 GB
as the floor (a Pixel 8a finishes 512×512 in about 7 minutes).

## What the screen says when something is missing

| Situation | What the app shows |
|---|---|
| Graphs not on the phone | The image area lists the missing files and the folder (the pill reads MODELS MISSING when no size can run). Tapping a size whose pair is missing names its files, and the other size still runs. A run that finds a graph gone names it before loading anything. |
| APK built without `prep_assets.sh` | "pipeline_meta.json missing from the APK assets (run prep_assets.sh)" or "Tokenizer tables missing from the APK (run prep_assets.sh)." |
| Empty prompt | "Type a prompt first." under the prompt box; nothing starts. |
| Prompt too long | The line under the box counts tokens against what the text encoder reads (244 for the prompt) and turns red past it; Generate is refused instead of cutting off the end of the prompt. |
| Cancel | The run stops at the next stage boundary (a DiT step is never interrupted; 4.9 s from Cancel to READY in the device check at 256×256), and Generate runs again right away. |

The device check below covers the empty prompt, the long prompt, Cancel followed by Generate, a missing
512×512 pair and a run pointed at an empty folder. The two texts for an APK without assets are not checked
on a phone.

## Device check

`app/src/androidTest/.../BonsaiDeviceCheck.kt` uses the app the way a person does: it types into the prompt
box, presses Generate and Cancel, taps a size, and reads the pill, the line under the prompt box, the PNG in
`files/outputs` and the run record. One `RESULT step=<name> ok=<bool> …` line per step goes to logcat under
the tag `bonsai-check`. The steps: assets, models, missing-model, launch, empty-prompt, long-prompt,
size-unavailable, generate, cancel, regenerate. With the three 256×256 graphs pushed:

```bash
./gradlew :app:installDebug :app:installDebugAndroidTest
adb shell am instrument -w -e class com.bonsai.imagegen.BonsaiDeviceCheck com.bonsai.imagegen.test/androidx.test.runner.AndroidJUnitRunner
adb logcat -d -s bonsai-check | grep RESULT | tail -11
adb pull /sdcard/Android/data/com.bonsai.imagegen/files/outputs/
```

The last line of the Galaxy S26 run on 2026-10-09:

```
RESULT ok=true steps=10 failed=none failure_paths_ok=6 line_pass=true generate_total_s=23.54
```

The generate step uses a fixed prompt and seed, and the check also reports whether the pixels match the
image the same inputs gave on a Galaxy S26 before (`same_as_s26_reference`; they did).

## How it works

| File | What it does |
|---|---|
| `MainActivity.kt` | One screen: prompt box and token count, size, steps, Generate / Cancel, status pill, stopwatch, the image and a table of the measured seconds of every stage |
| `BonsaiPipeline.kt` | The three graphs on LiteRT `CompiledModel` (CPU, XNNPACK threads from `CpuOptions`), opened by file path, one at a time |
| `QwenTokenizer.kt` | Byte-level BPE in Kotlin plus the chat template, token-exact against the Python tokenizer on a 26-case golden set |
| `PromptRules.kt` | The prompt limit and the empty-prompt text, shared by the screen and the device check |
| `BonsaiMath.kt` | Sigma schedule, position ids, seeded noise and latent unpatchify on the host |
| `ImageArea.kt`, `TabularSpan.kt` | The square image area with one progress segment per step; fixed-width digits for the stopwatch |

- Inputs are fed by signature name (`args_<n>`), never by shape: at 256×256 the DiT's image and text position
  ids are both (256, 4).
- The prompt is encoded as `[<|im_start|>]` + BPE(`"user\n"` + prompt) + a fixed assistant suffix, padded
  to 256 tokens, which is what `apply_chat_template(..., enable_thinking=False)` produces.
- The noise generator (SplitMix64 + Box-Muller) is the one the iOS app uses, so the same seed starts from
  the same noise on both.
- XNNPACK runs through `CompiledModel`; the classic `Interpreter` path ran this model on one thread on the S26.

`./gradlew :app:testDebugUnitTest` checks the tokenizer against the golden set (`app/src/test/resources`),
the prompt limit against the tokenizer's own cut-off, and the host math against recorded fixtures (the
fixture tests are skipped when the fixtures are not on the machine).

## Recording and unattended runs

For screen recordings, intent extras type the prompt and press Generate without touches, and each launch
writes a JSON record (device, settings, every stage's milliseconds) to `files/Documents/`:

```bash
adb shell am start -n com.bonsai.imagegen/.MainActivity --ez autorun true \
  --es prompt "'a red fox sitting in fresh snow, soft morning light'" --el seed 7 --ei steps 4 --es sizes 256
adb logcat -d -s BonsaiDemo:V | grep -E "AUTORUN_(DONE|FAILED)"
```

The full list of extras is at the top of `MainActivity.kt`.

## License

The code is Apache-2.0. The model is Apache-2.0 from
[prism-ml/bonsai-image-ternary-4B-unpacked](https://huggingface.co/prism-ml/bonsai-image-ternary-4B-unpacked),
built from FLUX.2 [klein] 4B (Black Forest Labs) and Qwen3-4B (Alibaba Cloud). Created using Bonsai Image
by Prism ML.
