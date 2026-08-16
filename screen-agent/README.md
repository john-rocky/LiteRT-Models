# Screen Agent — LFM2.5-VL-3B

An Android app that operates the phone by looking at it. It takes a screenshot,
sends it to LFM2.5-VL running on the device, gets back the coordinates to press,
and presses them. The app being driven needs no integration — no accessibility
tree, no selectors, no test hooks, no cooperation of any kind.

Everything runs on the phone. The app holds no `INTERNET` permission.

| | |
|---|---|
| Model | [LiquidAI/LFM2.5-VL-3B](https://huggingface.co/LiquidAI/LFM2.5-VL-3B), int4 `.litertlm` |
| Runtime | [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) `litertlm-android`, **CPU** |
| Device measured | Pixel 8a, 8 GB |
| Per turn | 11–23 s (screenshot → prompt → coordinates → press) |
| Bundle size | 2.35 GB |

This module does **not** use LiteRT `CompiledModel` GPU like the rest of this
zoo: LFM2.5-VL runs through the LiteRT-LM engine, and on this 8 GB device the
GPU backend dies during engine init. CPU is the floor, not the ceiling.

## What it does

You type a goal. The loop is: capture the screen → ask the model where to press
→ press → capture again → stop when the screen stops changing.

| mode | what it does |
|---|---|
| **Point** | ground once and draw a marker. Nothing is touched |
| **Tap** | ground once and press the answer |
| **Agent** | a goal over several screens, tap only |
| **Act** | the same loop with scroll / back / type — the model plans the action first, then the coordinate |
| **Ask** | your text alone, no grounding prompt |

Measured examples, all from a Pixel 8a with the model on CPU:

- `Point to the Battery row.` — one call, one press.
- `open the notification history` — Settings → Notifications → Notification
  history. 3 turns, 3 calls; the third changed nothing, which is what stops it.
- `search settings for wifi` — focus the search field and type into it. 2 steps,
  3 calls, 2 min 10 s.

## How it works

The model is never asked "what is on this screen". It is asked, in the vendor's
own grounding prompt, to return a point:

```
[{"image_id": 0, "point_2d": [500, 96], "label": "search box"}]
```

Coordinates are normalized to 0–1000, so the answer survives the resize between
the screen and the model's 512×512 input. `500, 96` on a 1080×2400 screen is
`dispatchGesture(540, 230)`.

Two prompts, not one. The grounding prompt is a fixed vendor contract and is
sent verbatim; **Act** mode adds a second, separate prompt that chooses the
action (`tap` / `scroll` / `back` / `type`) before anything is grounded. See
[`sdk/README.md`](sdk/README.md).

The loop stops on the screen, not on the model. Asked whether the goal is met,
the model invents a target rather than returning `[]`, so `Agent` compares
consecutive frames and stops when a press changes nothing.

## The bundle has to be repaired

Against a stock `litert-community/LFM2.5-VL-*` bundle this app runs, returns
well-formed JSON, and is **wrong below the top quarter of the screen**.

LiteRT-LM derives `patch_num_shrink_factor = encoder_input_patches /
adapter_output_tokens` (1024/256 = 4) and forwards only `input_patches / shrink`
= 256 encoder rows into the adapter. It assumes the *encoder* already pooled.
LFM2.5-VL does its 2×2 pixel-unshuffle in the *adapter*, so 768 of 1024 rows
stay zero and only the top quarter of the picture reaches the model. Captioning
still works; anything positional does not, silently.

Reported upstream as
[LiteRT-LM#3246](https://github.com/google-ai-edge/LiteRT-LM/issues/3246). Until
that lands, move the pooling into the exported encoder and repack the bundle:

```bash
python3 tools/reexport_vision_unshuffle.py LiquidAI/LFM2.5-VL-3B out_vision
python3 tools/repack_vision.py --src LFM2.5-VL-3B_int4.litertlm \
    --vision-dir out_vision --out LFM2.5-VL-3B_int4_fixB.litertlm
```

The 12-second check that this worked, with no reliance on the model being good
at coordinates: display a 512×512 image of 16 numbered bands and ask it to list
every number top to bottom.

```bash
python3 tools/make_grid_fixture.py --out fixtures/grid2x2 --rulers
# repaired -> 1 … 16    stock -> 1, 2, 3, 4
```

Two more things that are not obvious, both silent when wrong:

- **Send the image content part before the text.** `Content.ImageFile` then
  `Content.Text`. The other order describes the screen correctly and cannot
  locate anything on it. The desktop CLI puts attachments first, so desktop
  testing never shows this.
- **Grounding is a 3B-only capability in this family.** The 450M and 1.6B emit
  round numbers (0, 100, 200, 500) for every target, repaired or not.

## Build

Bazel, not Gradle — unlike the other modules in this repo. Prereqs: Android SDK
(API 35, build-tools 35.0.0), NDK, `bazelisk`. The SDK and NDK paths in
`MODULE.bazel` are machine-specific; set them first.

```bash
bazelisk build //app:screen_grounding \
  --android_platforms=@rules_android//:arm64-v8a \
  --cxxopt=-std=c++17 --host_cxxopt=-std=c++17
adb install -r -i com.android.vending bazel-bin/app/screen_grounding.apk
```

Then push a bundle into the app's own external files directory, so the engine
opens it in place instead of copying it:

```bash
adb shell am start -n com.edgeagent.lab/.MainActivity
adb push LFM2.5-VL-3B_int4_fixB.litertlm \
    /sdcard/Android/data/com.edgeagent.lab/files/
```

**Acting needs an accessibility service you enable by hand.** A sideloaded
package has the toggle silently reverted by Android's restricted-settings
protection, so install with `-i com.android.vending` as above, then enable it in
Settings → Accessibility. Full notes, including the `adb` equivalents and why
reinstalling turns it back off, are in [`app/README.md`](app/README.md).

Choose **Share entire screen** in the capture-consent dialog. It defaults to
"Share one app", which captures this app rather than what is behind it.

## Layout

| | |
|---|---|
| [`sdk/`](sdk/README.md) | the agent itself: framing, the two prompts, the parsers, the loops. No Android services |
| [`app/`](app/README.md) | the host: MediaProjection, the accessibility service, the overlay, the UI |
| `tools/` | the bundle repair scripts and the visibility-ruler fixture generator |

The split is the point: `sdk` takes a `ScreenSource` (anything that yields a
`Bitmap`) and an `ActionExecutor` (anything that can press a coordinate), so the
same loop runs against a PNG and a recorder in a test.

**Original project**: [LiquidAI/LFM2.5-VL-3B](https://huggingface.co/LiquidAI/LFM2.5-VL-3B)
| [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-VL-3B/blob/main/LICENSE)
