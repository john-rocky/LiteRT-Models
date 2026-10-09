# ASR LiteRT-LM — speech to text on Android with three LiteRT-LM bundles

Hold the button, speak for up to 30 s, and the app writes what you said. On the CPU of a Galaxy S26
(4 threads, measured 2026-10-09), Qwen3-ASR-1.7B turned a 6.4 s English clip into text in 2.49 s and
Fun-ASR-Nano-2512 in 1.60 s. The app runs three speech recognition models in the LiteRT-LM `.litertlm`
format through one code path: pick a model, then talk or play one of three bundled clips. Everything
runs on the phone; the app asks for the microphone and nothing else (no network permission).

## Models

| Model | Hugging Face | File | Size | License | Languages |
|---|---|---|---|---|---|
| Qwen3-ASR-1.7B | [litert-community/Qwen3-ASR-1.7B](https://huggingface.co/litert-community/Qwen3-ASR-1.7B) | `Qwen3-ASR-1.7B.litertlm` | 2,693,064,192 B | Apache-2.0 | 30 languages (the card scores English, Mandarin, Japanese) |
| Fun-ASR-Nano-2512 | [litert-community/Fun-ASR-Nano-2512](https://huggingface.co/litert-community/Fun-ASR-Nano-2512) | `Fun-ASR-Nano-2512.litertlm` | 1,255,894,736 B | Apache-2.0 | Chinese, English, Japanese |
| Confucius4-R2T2 | [mlboydaisuke/Confucius4-R2T2-LiteRT](https://huggingface.co/mlboydaisuke/Confucius4-R2T2-LiteRT) | `Confucius4-R2T2.litertlm` | 2,693,064,192 B | [NetEase Youdao Model Use License Agreement](https://huggingface.co/mlboydaisuke/Confucius4-R2T2-LiteRT/blob/main/MODEL_LICENSE) | Chinese, English; a fine-tune of Qwen3-ASR-1.7B |

Each file holds the audio encoder, the language model, the tokenizer and the prompt template. Read
the Confucius4-R2T2 agreement before you use that file: it has user-count and revenue thresholds
(clause 2.2) and limits on using the model to improve other models (clause 3.4).

## Run it in 5 lines

From this directory, with the phone connected over adb:

```bash
hf download litert-community/Qwen3-ASR-1.7B Qwen3-ASR-1.7B.litertlm --local-dir bundles
hf download litert-community/Fun-ASR-Nano-2512 Fun-ASR-Nano-2512.litertlm --local-dir bundles
hf download mlboydaisuke/Confucius4-R2T2-LiteRT Confucius4-R2T2.litertlm --local-dir bundles
./gradlew :app:installDebug && adb shell am start -n com.asrlitertlm/.MainActivity
adb push bundles/*.litertlm /sdcard/Android/data/com.asrlitertlm/files/
```

Reopen the app, tap a model in the picker and hold **Hold to talk**. The first launch creates the
folder the push writes to. One bundle is enough; the picker marks the others `missing`.

- Storage: the three files take 6.6 GB. The first load of each file writes a runtime cache into the
  app's cache directory: 3.0 GB for each Qwen3-ASR file, 1.5 GB for Fun-ASR.
- Run `adb shell touch /sdcard/Android/data/com.asrlitertlm/files/*.litertlm` after the push. adb push
  writes whole-second modification times, and with those a later launch can miss the runtime cache
  and write it again ([LiteRT-LM #3772](https://github.com/google-ai-edge/LiteRT-LM/issues/3772)).

## Galaxy S26 numbers

Galaxy S26 (SM-S942Q, Android 16), `litertlm-android` 0.17.1, language model and audio encoder on the
CPU with 4 threads, 2026-10-09. "Transcribe" is the wall time of `sendMessage()`, from sending the
clip to the full answer; RTF is that time over the clip's length. The three clips are the same sentence
in three languages (below).

| Model | zh, 9.7 s | en, 6.4 s | ja, 11.0 s | Microphone, 9 s | CPU limit at the start (big / prime cores) |
|---|---|---|---|---|---|
| Qwen3-ASR-1.7B | 2.38 s, RTF 0.25 | 2.49 s, RTF 0.39 | 3.56 s, RTF 0.32 | 2.97 s, CER 0 | 3.51 / 4.19 GHz |
| Confucius4-R2T2 | 4.03 s, RTF 0.41 | 3.06 s, RTF 0.48 | 3.83 s, RTF 0.35 | 2.93 s, CER 0 | 2.75 / 2.67 GHz |
| Fun-ASR-Nano-2512 | 1.26 s, RTF 0.13 | 1.60 s, RTF 0.25 | 2.29 s, RTF 0.21 | 1.75 s, CER 0 | 2.23 / 2.23 GHz |

| Model | Engine load, first (writes the cache) | Engine load, cache in place | Runtime cache | Peak resident memory, first load |
|---|---|---|---|---|
| Qwen3-ASR-1.7B | 7.35 s | 0.72 s | 3.0 GB | 5.6 GB |
| Confucius4-R2T2 | 4.27 s | 1.33 s | 3.0 GB | 5.7 GB |
| Fun-ASR-Nano-2512 | 1.39 s | 1.13 s | 1.5 GB | 3.5 GB |

- Every clip's text equals the Mac CPU answer of the same file (LiteRT-LM 0.17.1, Python) after
  removing punctuation and spaces. The three models write the same sentences for these clips.
- The phone was warm: its CPU limit fell during the runs (the maximum is 3.63 / 4.74 GHz), and the
  Confucius4-R2T2 and Fun-ASR rows ran under the lower limits in the last column. Confucius4-R2T2 has
  the same architecture and file size as Qwen3-ASR-1.7B. In a run that started at the maximum,
  Qwen3-ASR-1.7B took 2.13 / 2.14 / 3.19 s for the three clips.
- "Engine load" is `Engine.initialize()` plus the first conversation, which creates the audio encoder.
  The Qwen3-ASR-1.7B cached load is a new launch; the other two are a second load in the same process.
- Microphone: the phone plays clip en through its speaker and records it with its microphone
  (media volume 3 of 15), then transcribes the recording. CER is the character error rate against
  the clip's sentence.
- Language model on the GPU (launch extra `--es backend gpu`), Fun-ASR-Nano-2512: 1.69 / 1.68 / 2.29 s,
  the same text. All nodes of the prefill and decode graphs ran on the GPU (LITERT_CL); the audio
  encoder stays on the CPU (the model cards: the GPU delegate does not take it). Engine load 2.61 s.
  It was not faster than the CPU here.

## How the app calls a model

The three bundles share one path ([`AsrEngine.kt`](app/src/main/kotlin/com/asrlitertlm/AsrEngine.kt)):
one `Engine`, a new `Conversation` per clip, and one user message that holds only the audio file. Each
bundle's template adds the rest: Fun-ASR's default instruction `语音转写：`, Qwen3-ASR's audio markers.

```kotlin
val engine = Engine(EngineConfig(
    modelPath = bundle.path,                      // /sdcard/Android/data/com.asrlitertlm/files/<file>
    backend = Backend.CPU(threadCount = 4),       // the language model
    audioBackend = Backend.CPU(threadCount = 4),  // the audio encoder
    cacheDir = context.cacheDir.path,
))
engine.initialize()

val raw = engine.createConversation(ConversationConfig(
    samplerConfig = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0),
    maxOutputToken = 512,
)).use { conversation ->
    conversation.sendMessage(Message.user(Contents.of(Content.AudioFile(wav.absolutePath)))).toString()
}
// Qwen3-ASR-1.7B and Confucius4-R2T2: "language English<asr_text>Everything in the universe ..."
// Fun-ASR-Nano-2512: "Everything in the universe ..."
```

Only the answer differs: [`ModelProfile.kt`](app/src/main/kotlin/com/asrlitertlm/ModelProfile.kt)
splits the Qwen3-ASR answer at `<asr_text>` and shows the named language; Fun-ASR's answer is the text.

- One file is loaded at a time: picking another model closes the engine first. A first load of a
  Qwen3-ASR file peaked at 5.6–5.7 GB resident (VmHWM, the memory-mapped file included) on this phone
  with 11.4 GB of memory.
- `Engine.close()` throws when called twice, so the app drops its reference before closing; releasing
  twice is harmless.
- All model calls run on one worker thread.

## Microphone

Hold **Hold to talk** and speak; releasing the button ends the recording, and so does the 30 s limit.
Each model reads one window of about 30 s per message, so the app sends at most 30 s. The app records
16 kHz mono 16-bit PCM (the `VOICE_RECOGNITION` source) and writes it as a WAV file for the runtime.
A recording quieter than −60 dBFS RMS is reported as "No speech heard" and is not sent to the model.
Without the microphone permission the app shows a sentence that says how to allow it; the clips still work.

## The three clips

`app/src/main/res/raw/` holds FLEURS test sentence 1698 read in Mandarin, English and Japanese
([google/fleurs](https://huggingface.co/datasets/google/fleurs) revision `70bb2e84`, test split, id 1698;
FLEURS, Conneau et al. 2022, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). Changes:
converted to 16-bit PCM and given one linear gain each, so that all three have an active RMS of −20 dBFS.
The English reading is "Everything in the Universe is made of matter. All matter is made of tiny
particles called atoms."

## Device check

[`AsrDeviceCheck.kt`](app/src/androidTest/kotlin/com/asrlitertlm/AsrDeviceCheck.kt) runs on a phone
with at least one bundle pushed (as above). Turn the media volume up first: the microphone step plays a
clip through the speaker.

```bash
./gradlew :app:installDebug :app:assembleDebugAndroidTest
adb install -r -t app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell pm revoke com.asrlitertlm android.permission.RECORD_AUDIO
adb shell am instrument -w -e class com.asrlitertlm.AsrDeviceCheck#check com.asrlitertlm.test/androidx.test.runner.AndroidJUnitRunner
adb logcat -d -s asr-check | grep RESULT
```

It prints one line per step, `RESULT step=<name> ok=<true|false> model=<id> ...`, then
`RESULT ok=<all>`. It starts with the microphone permission revoked and checks the sentence the app
shows; then it grants the permission. For each bundle on the phone: load, the three clips (text equal to
the expected sentence after removing punctuation and spaces), the microphone (CER ≤ 0.15), release
twice, and a second load. Then a 33.6 s clip (cut to 30 s), a silent recording and a file that is not
on the phone. With two or more bundles it also switches models without releasing. Add
`-e class com.asrlitertlm.AsrDeviceCheck#gpu -e gpu_model fun-asr-nano-2512` for the GPU run.

## Launch extras

For recordings and tests; the app works without them.

| Extra | Effect |
|---|---|
| `--es model <id>` | `qwen3-asr-1.7b`, `fun-asr-nano-2512` or `confucius4-r2t2` (default: the model used last) |
| `--es backend gpu` | language model on the GPU; the audio encoder stays on the CPU |
| `--ez autoplay true` | after loading: clips zh, en, ja, each played and then transcribed (`--ei delay_ms`, `--ei gap_ms`) |
| `--ei mic_test_ms 9000` | after loading and `delay_ms`: record that long, then transcribe (`--ez mic_test_play true` plays clip en meanwhile) |

The app logs one `STATE <name>` line per screen state and `LOAD` / `TRANSCRIBED` lines with the times
under the tag `AsrLitertlm`.

## Files

| File | Role |
|---|---|
| `app/src/main/kotlin/com/asrlitertlm/AsrEngine.kt` | Engine and one Conversation per clip: load, transcribe (cut at 30 s), release (safe twice) |
| `app/src/main/kotlin/com/asrlitertlm/ModelProfile.kt` | the three bundles: file name, Hugging Face id, answer format |
| `app/src/main/kotlin/com/asrlitertlm/AsrSession.kt` | the app's one engine on one worker thread, events for the screen, the no-speech check |
| `app/src/main/kotlin/com/asrlitertlm/MainActivity.kt` | picker, hold to talk, clip buttons, launch extras |
| `app/src/main/kotlin/com/asrlitertlm/MicRecorder.kt` | 16 kHz mono recording, up to 30 s |
| `app/src/main/kotlin/com/asrlitertlm/ClipPlayer.kt`, `PlayBar.kt` | clip playback through the speaker and its progress bar |
| `app/src/main/kotlin/com/asrlitertlm/Wav.kt`, `TextMatch.kt` | WAV read, write and cut; transcript comparison (normalization and CER) |
| `app/src/main/res/raw/fleurs_*_1698.wav` | the three clips |
| `app/src/androidTest/kotlin/com/asrlitertlm/AsrDeviceCheck.kt` | device check |
| `app/src/test/kotlin/com/asrlitertlm/AsrLogicTest.kt` | JVM tests: answer parsing, transcript comparison, WAV cut and levels |

Build: AGP 9.3.1 with Kotlin 2.4.0 (the `litertlm-android` 0.17.1 AAR carries Kotlin 2.4 metadata),
compileSdk 36, minSdk 31, arm64-v8a. The runtime version is pinned in `gradle.properties`.

## License

The app code is under the repository's MIT license. The clips are CC BY 4.0 (FLEURS, see above). Each
model file has its own license (table above); none of them is in this directory.
