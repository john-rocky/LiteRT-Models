# Screen Grounding (Android)

Screenshot → local LFM2.5-VL → tap point → real tap → overlay. No network.

This is the demo host. The agent itself is the library next door,
[`//sdk:screen_agent`](../sdk/README.md); everything here is the two Android
services it cannot contain (MediaProjection, accessibility) plus the UI.

**Acting needs an accessibility service the user enables by hand.** The app
shows an "Enable tapping" button until it is on. Two things bite when testing
from `adb`: a sideloaded package (`installer=null`) has the toggle silently
reverted by Android's restricted-settings protection, so install with
`adb install -r -i com.android.vending`; and the service must be switched on
*after* that, either in Settings → Accessibility or with

```bash
adb shell settings put secure enabled_accessibility_services \
    com.edgeagent.lab/com.edgeagent.lab.TapService
adb shell settings put secure accessibility_enabled 1
```

To revoke: `adb shell settings put secure enabled_accessibility_services ""`, or
turn it off in Settings → Accessibility.

Reinstalling the app after changing `accessibility_config.xml` switches the
service back off — Android will not carry a grant across a capability change.
The symptom is "Agent mode needs the accessibility service" from a build that
worked five minutes ago; re-run the two `settings put` lines above.

The service declares `canRetrieveWindowContent` only so that **Act** mode can
type: there is no gesture that puts characters into a field. It is used to write
into the input-focused node and for nothing else — no window enumeration, no
reading node text. The screen is still read from pixels by the model.

## Requires a repaired bundle

Against a stock `litert-community/LFM2.5-VL-*` bundle this app runs, returns
well-formed JSON, and is **wrong below the top quarter of the screen** — see
[the repaired-bundle note](../README.md#the-bundle-has-to-be-repaired) and
[LiteRT-LM#3246](https://github.com/google-ai-edge/LiteRT-LM/issues/3246). Build a repaired bundle first:

```bash
python3 ../tools/reexport_vision_unshuffle.py LiquidAI/LFM2.5-VL-3B out_vision
python3 ../tools/repack_vision.py --src LFM2.5-VL-3B_int4.litertlm \
    --vision-dir out_vision --out LFM2.5-VL-3B_int4_fixB.litertlm
```

**Send the image content part before the text.** `Content.ImageFile` then
`Content.Text`. The other order describes the screen correctly and cannot locate
anything on it, with no error — see [../README.md](../README.md). The desktop CLI puts
attachments first, so desktop testing will not reveal it.

Model choice is not free: **grounding coordinates are a 3B-only capability in
this family.** The 450M and 1.6B emit round numbers (0, 100, 200, 500) for every
target, repaired or not, which matches ScreenSpot-v2 80.7 being published for
the 3B alone. They are fine for the visibility rulers and useless for this app.

## Build

Prereqs: Android SDK (API 35, build-tools 35.0.0), NDK, `bazelisk`. The SDK and
NDK paths in `MODULE.bazel` are machine-specific — the NDK one in particular
moves whenever the SDK manager updates it, and the failure is an opaque
`can't readdir()` at analysis time.

```bash
bazelisk build //app:screen_grounding \
  --android_platforms=@rules_android//:arm64-v8a \
  --cxxopt=-std=c++17 --host_cxxopt=-std=c++17
adb install -r -i com.android.vending bazel-bin/app/screen_grounding.apk
```

The Bazel workspace is the repository root, not this directory — `//app` is the
demo, `//sdk` is the library.

macOS: if `xcode-select` points at an Xcode beta, host-tool compiles fail with
"absolute path inclusion(s)". The committed `.bazelrc` pins `DEVELOPER_DIR` for
repo rules, target actions **and** host actions — `--host_action_env` is the one
people miss.

## Run

Push a bundle into the app's own external files directory. `adb push` can write
there directly, so the engine opens the file in place:

```bash
adb shell am start -n com.edgeagent.lab/.MainActivity   # creates the directory
adb push LFM2.5-VL-3B_int4_fixB.litertlm \
    /sdcard/Android/data/com.edgeagent.lab/files/
```

That is deliberate. A `content://` file picker forces a copy into the app cache,
which needs twice the model's size free — the difference between fitting on a
nearly-full device and not.

The face at the top answers as you type — it says back what it understood and
what it will do with it, which changes with the mode. It is the same character
that floats over the app being driven.

Then: pick a backend → **Load** → pick a mode → type an instruction → **Run**.
The app asks for screen-capture consent once, puts itself in the background,
and works on whatever is underneath. Tap its notification to come back to the
overlay.

Choose **Share entire screen** in the consent dialog — it defaults to
"Share one app", which captures this app rather than what is behind it.

Four modes:

| mode | what it does |
|---|---|
| **Point** | ground once and draw the marker. Nothing is touched |
| **Tap** | ground once, press the first target, show the screen that produced |
| **Agent** | a goal, several steps: capture → ground → tap → repeat until the screen stops changing |
| **Act** | the same loop with the full action set — the model plans tap / scroll / back / type first, then acts. One extra model call per step |
| **Ask** | your text alone, no grounding prompt, reply printed |

*Ask* is how the visibility rulers get run against a real screen — a check that
does not depend on the model being good at coordinates:

```bash
python3 ../tools/make_grid_fixture.py --out fixtures/grid2x2 --rulers
adb push fixtures/rows16.png /sdcard/Pictures/
# display it full-screen, then Ask: "List every number you can see, from top to bottom."
# repaired bundle -> 1…16   |   stock bundle -> 1, 2, 3, 4
```

Measured on a Pixel 8a: the 3B int4 on CPU loads in 9–24 s with warm caches and
takes ~24 s per capture; the 1.6B int4 on text-GPU + vision-CPU loads in 27–30 s
and takes 8–9.4 s, but cannot ground. The 3B on the GPU backend kills the
process during engine init on this 8 GB device.

The shape of that flow is Android policy, not preference: a foreground service
of type `mediaProjection` must already be running before `getMediaProjection()`,
and a background app may not launch itself — hence the notification instead of
an automatic return. The consent token is single-use on API 34+, but the
projection it creates is not, so `CaptureService` holds one for the whole
session and an agent loop asks once, not once per step.

## Storage and memory

The 3B int4 is 2.35 GB and the GPU path writes compile caches next to it that
have run to multiple GB. On a nearly-full device that shows up as a failure at
kernel init that looks like a memory error but is `ENOSPC`. Budget several GB
free, or run CPU.

## Layout

Only what has to be an app lives here. Each file below fills one of the seams
the SDK declares in `Seams.kt`.

| file | role |
|---|---|
| `AgentFace.kt` | the agent as a robot's face, drawn with Canvas primitives — no bitmap, and the demo's build script draws the same figure |
| `AgentOverlay.kt` | that face floating over the app being driven; walks to the point it presses |
| `ShownExecutor.kt` | wraps any `ActionExecutor` so the badge travels with the gesture |
| `CaptureService.kt` | holds the MediaProjection for the session; `captureScreen()` is the `ScreenSource` |
| `TapService.kt` | the accessibility service; `AccessibilityExecutor` is the `ActionExecutor` |
| `OverlayView.kt` | the marker rendering |
| `MainActivity.kt` | wiring and the four modes |

Framing, the prompt and parser, the engine and the loop are in
[`../sdk`](../sdk/README.md).
