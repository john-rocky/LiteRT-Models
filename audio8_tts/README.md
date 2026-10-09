# Audio8 TTS — type a sentence, hear it in a chosen voice, on the phone

Type a sentence, pick a voice and tap **Speak**: the phone turns the sentence into speech in that
voice and plays it, in a little more time than the speech lasts. The voice is one of the model's
two sample voices or your own, recorded once for up to 10 seconds. On a Galaxy S26 a typed
18-token English sentence became 6.0 s of speech in 6.7 s (RTF 1.12) with the app on screen, and
the device check measured RTF 1.50 for a Japanese sentence and 1.42 for an English one
(2026-10-10). Everything runs on the phone through LiteRT `CompiledModel` from Kotlin: the two
autoregressive graphs on the CPU with 4 threads, and the codec decoder on the GPU when its output
passes a check, else on the CPU. The app has no network permission.

## Model

[litert-community/Audio8-TTS-Preview-0.6b](https://huggingface.co/litert-community/Audio8-TTS-Preview-0.6b)
(revision `880dfa08`), converted from
[Edge0/Audio8-TTS-Preview-0.6b](https://huggingface.co/Edge0/Audio8-TTS-Preview-0.6b), Apache-2.0:
a DualAR speech model with zero-shot voice cloning in 11 languages (en, zh, yue, ja, ko, de, fr,
es, it, nl, pl) at 44.1 kHz. The app reads these files of the model repository:

| File | Bytes | Role | Runs on |
|---|---:|---|---|
| `slow_ar_int8.tflite` | 551,890,704 | slow AR, 24 layers: one semantic token per 46 ms frame | CPU, 4 threads |
| `fast_ar_int8.tflite` | 67,780,320 | fast AR, 4 layers: the frame's 10 codebook codes, 10 calls per frame | CPU, 4 threads |
| `codec_decoder_fp16_T128.tflite` | 261,602,768 | codec decoder, up to 128 frames (5.9 s) per call | GPU, if it passes the output check |
| `codec_decoder_int8_T128.tflite` | 132,429,760 | the same decoder, int8 | CPU (reference for the check, and the fallback) |
| `codec_decoder_fp16_T192.tflite` | 261,799,376 | decoder for up to 192 frames (8.9 s); longer speech in windows | GPU (only when the GPU passed) |
| `codec_encoder_fp16_10s.tflite` | 419,269,280 | codec encoder: 10 s of audio to codes, for "Record my voice" | CPU, created per recording |
| `tokenizer.json` | 12,217,872 | the vendor's Qwen2 BPE tokenizer | Kotlin |
| `voices/ja_funasr_example/`, `voices/en_librispeech_1272/` | 7,310 | the two sample voices: codec codes and transcript | — |

Total 1,706,997,390 bytes. The sample voices come from the Fun-ASR-Nano example clip (Apache-2.0)
and LibriSpeech dev-clean speaker 1272 (CC BY 4.0).

## Run it in 5 lines

Android 8.0 (API 26) or newer, arm64-v8a, about 1.8 GB free; [`hf`](https://huggingface.co/docs/huggingface_hub/guides/cli) and `adb` on the computer.

```bash
hf download litert-community/Audio8-TTS-Preview-0.6b --local-dir audio8 --exclude slow_ar_int4.tflite
cd audio8_tts && ./gradlew :app:installDebug && adb shell am start -n com.audio8tts/.MainActivity
adb push ../audio8/*.tflite ../audio8/tokenizer.json /sdcard/Android/data/com.audio8tts/files/
for v in ja_funasr_example en_librispeech_1272; do adb push ../audio8/voices/$v/* /sdcard/Android/data/com.audio8tts/files/voices/$v/; done
adb shell am start -S -n com.audio8tts/.MainActivity
```

Line 2 starts the app once without models: it shows which files are missing and creates the
folders the pushes fill (a folder that `adb` creates belongs to the shell user, and the app may not
be allowed to open it). After line 5, pick a voice, type a sentence and tap **Speak**. The
sentence plays from the speaker and is saved as a wav in
`Android/data/com.audio8tts/files/Documents/`.

## Galaxy S26 (2026-10-10)

SM-S942Q, Android 16, LiteRT 2.2.0, slow and fast AR on the CPU with 4 threads, codec decoder
CPU int8 (the GPU decoder failed its check on this phone, see below), seed 42.

| Text | Voice | Tokens | Frames | Audio | Generate | RTF | Where |
|---|---|---:|---:|---:|---:|---:|---|
| Good morning. I typed this sentence on the phone, and it is spoken right here. | English sample | 18 | 129 | 5.99 s | 6.73 s | 1.12 | app on screen, thermal status 1 |
| This is the voice I just recorded. | My voice (10 s take) | 8 | 117 | 5.43 s | 5.19 s | 0.95 | app on screen, thermal status 1 |
| 今日は天気が良いので、公園まで散歩に行きましょう。 | Japanese sample | 14 | 74 | 3.44 s | 5.14 s | 1.50 | device check, first speak, thermal 0 |
| Hello from LiteRT. This voice was made on the phone, with no network at all. | English sample | 19 | 136 | 6.32 s | 8.99 s | 1.42 | device check, second speak, thermal 0 |

Generate is text in, samples out: prompt, prefill, the frame loop and the codec decode. RTF is
generate / audio. Two earlier runs of the device check gave RTF 1.46 and 1.43 for the Japanese
sentence and 1.45 and 1.46 for the English one. The device check runs in the instrumentation
process, which the S26 places in the `/foreground` cpuset; the app on screen is in `/top-app`.
There the frame loop takes about 25 ms per frame against about 33 ms in the check; the cause of
that gap was not isolated.
Speech longer than 128 frames (5.9 s) needs a second codec window: on the CPU each window takes
about 1.6 s, so the 129-frame sentence spent 3.2 s of its 6.7 s in the codec, while the 117-frame
one needed a single window and came out faster than real time.

Load time: 10.1 to 10.6 s when the app creates the GPU codec and compares its output with the CPU
int8 decoder (measured in the device check; 11.7 s the very first time, when the GPU program is
compiled). On the S26 the GPU decoder returns the same wav for every input (correlation 0.0097
with the CPU decoder, 1.0 with its own output for all-zero codes), so the app uses the CPU decoder
and remembers the failure for this OS build, LiteRT version and decoder file; later launches load
in 2.8 to 3.4 s. Peak memory (VmHWM): 2.4 GiB while speaking, 4.1 GiB while registering a voice.

Whisper large-v3-turbo transcribes the outputs as typed: every Japanese output with no character
error, the typed English sentences with no word error, and the check's English sentence with one
("LiteRT" heard as "LightRT").

## Record your own voice

Tap **Record my voice (10 s)**, allow the microphone, and read the sentence the screen shows,
exactly as written (it becomes the voice's transcript; the sentence is Japanese when the text box
holds Japanese, English otherwise). **Stop and save** ends the take early; the recording also ends
at 10.03 s, the encoder's input length. The app cuts the silence before and after the speech to
0.25 s, sets the peak to -3 dBFS, runs the codec encoder (created for this call and closed after
it) and saves the voice in `Android/data/com.audio8tts/files/voices/my_voice/` (`codes.npy`,
`meta.json`, `reference.wav`). "My voice" is then selected, and stays available on later
launches. On the S26 a 10.03 s take registered in 2.6 s (encoder created, run and closed) and the
new voice spoke the sentence in the table above; that take held only room sound, so it shows the
path and its timing, not how close the voice gets. Record only voices whose owners agreed, and say
that the audio is synthetic when you share it.

## When something is missing or wrong

| Situation | What the app does |
|---|---|
| A required model file is missing | ERROR, and the list of the missing files with the folder they belong in; Speak stays off |
| Empty text | "Type a sentence first." |
| More than 80 tokens | "Too long: N tokens. One Speak reads up to 80 tokens (about 20 s of speech); split the text." The count under the text box turns red first. One call makes at most 512 frames (23.8 s); 80 tokens end within that |
| Microphone permission refused | "Microphone permission denied: recording a voice is off. The sample voices still work." |
| Cancel while generating | stops before the next frame (17 ms in the device check) and returns to READY; the next Speak runs |
| Cancel while playing or recording | stops the playback; drops the recording |
| A voice folder is missing | that voice shows "(not installed)" and cannot be picked |
| The encoder file is missing | "Record my voice" says which file to push; speaking still works |

## Device check

`Audio8DeviceCheck` (`app/src/androidTest`) runs the same `Audio8Tts` the screen uses on the
pushed files and logs one line per step under the tag `audio8-check`: models, load, speak-ja,
speak-en, empty-text, long-text, cancel, regenerate (the same frames as speak-ja after the
cancel), register (a voice registered from the speak-ja wav and its text, read back from its
folder, then used for the English sentence), release (a second close is a no-op), missing-model
and mic-denied. The last line, `RESULT ok=<all>`, passes when every step is ok; its `line_pass`
needs the core steps, RTF ≤ 2.0 on the first speak and at least 3 failure-path steps. After
`./gradlew :app:assembleDebug :app:assembleDebugAndroidTest` and the pushes above:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell am instrument -w -e class com.audio8tts.Audio8DeviceCheck com.audio8tts.test/androidx.test.runner.AndroidJUnitRunner
adb logcat -d -s audio8-check | grep RESULT
```

On the S26 on 2026-10-10: `RESULT ok=true steps=12 failed=none core_ok=true first_rtf=1.496
failure_paths_ok=6 line_pass=true`. The check writes its wav files to
`Android/data/com.audio8tts/files/check/`. `./gradlew :app:testDebugUnitTest` runs the JVM tests
(text rules, the `.npy` writer, the recording clean-up).

## Recording tool (autorun)

The launch extras of the demo this app grew from remain for screen recordings and timing runs, for
example `adb shell am start -n com.audio8tts/.MainActivity --ez autorun true --es ref_mode bundled`
(take a sample voice, then speak a Japanese and an English sentence with no taps), `--ez kv_probe
true` and `--ez codec_probe true`. `--es text "..."` fills the text box of a normal launch, and
`--es codec cpu|gpu|auto` overrides the codec choice. None of these extras was run on the phone for
this release. Each launch writes its numbers to
`Android/data/com.audio8tts/files/Documents/audio8-demo-<epoch>.json`; the full list of extras is
in the KDoc of `MainActivity.kt`.

## How it works

The Kotlin code reproduces the model repository's host loop (`audio8_tts_litert.py`) on the
`CompiledModel` API. The prompt is the vendor's chat format: the reference transcript and its 10
rows of codec codes, then the typed text. The slow AR prefills it in chunks of 256 and decodes one
frame at a time against a 2,048-position KV cache kept in two sets of tensor buffers (each call
reads one set and writes the other, so no cache bytes cross JNI per step). For every frame the fast
AR makes 10 calls for the 10 codebook codes. Sampling is the vendor's top-k 50, top-p 0.9,
temperature 0.7 with a repetition-aware redraw. The codec decoder turns the frames into 44.1 kHz
audio, in one call up to 128 frames and in windows with 64 frames of left context beyond that on
the CPU decoder (128 frames with the T192 GPU decoder).

| File | Role |
|---|---|
| `MainActivity.kt` | the input screen (voice picker, text box, Speak / Cancel, footer with device, runtime, backends, load time and `generate / audio = RTF`), the microphone recording, playback, and the demo's autorun and probe modes |
| `Audio8Tts.kt` | text + voice → samples and times, shared by the screen and the device check: missing files, voices, text rules, cancellation, a close that may be called twice, the remembered GPU codec failure |
| `Audio8Engine.kt` | the graphs: prefill and decode with the ping-pong KV cache, the fast AR, the codec decoder windows, the encoder, the vendor sampler |
| `TextRules.kt` | empty and too-long texts (80 tokens) |
| `QwenBpeTokenizer.kt` | byte-level BPE from `tokenizer.json`, checked at load against the test vectors in `assets/prompt_constants.json` |
| `PromptConstants.kt` | the fixed prompt fragments and the tokenizer test vectors |
| `ReferenceAudio.kt` | silence trim and peak normalization of a recording |
| `Wav.kt`, `Npy.kt`, `PlayBar.kt` | wav files, the voices' `codes.npy`, the progress and level bar |
| `Audio8DeviceCheck.kt` | the device check above |

## License

The code in this folder is MIT (the repository's license). The model is Apache-2.0 (Edge0); the
sample voices are Apache-2.0 (Fun-ASR-Nano example) and CC BY 4.0 (LibriSpeech). The model card
lists the vendor's limitations: a preview checkpoint, limited dialect coverage, and sensitivity to
noisy or mis-transcribed references.
