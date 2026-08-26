# screen-agent SDK

The middle of the screen agent, extracted: framing, grounding, and the loop.

```kotlin
val agent = ScreenAgent(
    screen   = ScreenSource { myCaptureService.grab() },
    executor = myExecutor,
    grounder = LiteRtGrounder.create(bundle, "cpu", cacheDir)!!,
    cacheDir = cacheDir,
)

agent.locate("the Notifications row")        // coordinates only
agent.tapOnce("the Notifications row")       // find it and press it
agent.pursue("open the notification history")// goal, several taps
agent.operate("turn on notification history")// goal, tap/scroll/back/type
```

## What is deliberately not in here

No MediaProjection, no AccessibilityService. The library takes a
`ScreenSource` (anything that yields a `Bitmap`) and an `ActionExecutor`
(anything that can press a coordinate). The demo app supplies those from the
two Android services; a test can supply a PNG and a recorder instead, and a
rooted host could supply `input tap`.

| file | role |
|---|---|
| `Seams.kt` | `ScreenSource`, `ActionExecutor`, `Grounder` — the three things the host provides |
| `Plan.kt` | the action vocabulary, the planner prompt, and its parser |
| `Grounding.kt` | the vendor prompt and JSON parser, plus the `Grounded` data contract |
| `Framing.kt` | screen bitmap → model view, and the transform back |
| `LiteRtGrounder.kt` | `Grounder` backed by LiteRT-LM, with the backend cascade |
| `Agent.kt` | the two goal loops (`run` taps only, `operate` plans first) and their stopping rules |
| `ScreenAgent.kt` | the front door that ties them together |
| `ScreenFrames.kt` | `Image` → cropped `Bitmap` (padded row strides) |

## Two loops

`pursue` grounds the goal directly and taps the answer. One model call per
step, and it can only reach what is already on the screen.

`operate` asks what to do before asking where — `{"action":"scroll"}`,
`{"action":"back"}`, `{"action":"type","text":"…"}`, `{"action":"tap",
"target":"…"}` — so it can scroll to something out of view or back out of a
dead end. Two model calls per step, except for the actions that need no
coordinate, which take one.

The planner has its own prompt because the grounding prompt is a fixed vendor
contract that must not be paraphrased. They are two questions about the same
screen, not one prompt doing both jobs.

Adding an action is one method on `ActionExecutor` (with a default that
declines, so existing hosts still compile), one verb in `Planning`, one branch
in `Agent.perform`.

## Three things that are not obvious

**Image content part before text.** `Grounding.promptFor` is used with the image
placed first. The other order describes a screen correctly and cannot locate
anything on it, silently. See [../README.md](../README.md).

**`Framing` emits the fixed point of the runtime's own resize** — for a
1080×2400 screen that is 320×768 — so LiteRT-LM's resize is a no-op and the
image the engine saw can be reproduced off-device.

**The loop stops on the screen, not on the model.** The model does not reliably
report "done"; it invents a target instead. `Agent` compares consecutive frames
and stops when a tap changes nothing.

## Model

Grounding coordinates are a 3B-only capability in the LFM2.5-VL family, and the
bundle must be one where the vision adapter takes pooled tokens (see
[../README.md](../README.md)). A stock bundle runs and returns well-formed JSON that is
wrong below the top quarter of the screen.
