# LFM Tools — an on-device agent behind Apple's Foundation Models API

**This is a personal project, not a Google product.**

An iPhone app where `LanguageModelSession` — Apple's own API — is driven by
**LFM2.5 running on the device through LiteRT-LM**, not by Apple's system model.
It is given 27 tools onto iOS and asked to use them.

```
You: turn the flashlight on and tell me how much battery I have left
     🔧 called set_torch {"on": true}
     🔧 set_torch returned: torch on
     🔧 called get_battery {}
     🔧 get_battery returned: 63%, on battery
LFM: Torch is on, and you have 63% battery left.
```

Nothing leaves the phone. The model file is a `.litertlm` bundle you drop into
the app's Documents folder.

## Why this is not just a chat demo

iOS 27's Foundation Models framework lets a third party supply the model:
conform to `LanguageModel` + `LanguageModelExecutor` and every FM feature —
sessions, transcripts, `@Generable` guided generation, tool calling — runs on
your weights. The conformance used here is
[`LiteRTLMFoundationModels`](https://github.com/google-ai-edge/LiteRT-LM/tree/main/swift/apple_fm)
in the LiteRT-LM Swift package.

So the app itself contains no inference code and no agent loop. It contains 27
`Tool` structs and a text field.

## The tools

**No permission needed** — `get_current_time`, `get_device_info`, `get_battery`,
`get_storage`, `get_power_state`, `get_locale`, `get_network_status`,
`calculate`, `get_brightness`, `get_volume`

**Changes something** — `set_brightness`, `set_torch`, `speak`, `vibrate`,
`read_clipboard`, `write_clipboard`, `open_url`, `schedule_notification`

**Asks permission** — `get_location`, `describe_location`, `search_places`,
`list_calendar_events`, `create_calendar_event`, `list_reminders`,
`create_reminder`, `search_contacts`, `photo_library_summary`

Each group can be switched off from the toolbar. Start with the first group when
you are checking whether a model calls tools at all: a permission dialog in the
middle of a run looks like the model failed when it did its part.

## Tool calling on a 1–3B model

A small model asked in prose to "reply with only this JSON" will add a sentence
in front of it perhaps a fifth of the time, and that is a failed tool call. The
adapter this demo depends on therefore constrains decoding twice per turn:

1. **Route.** Grammar: `{"tool": <enum of tool names + "none">, "answer": string}`.
   The model cannot name a tool that does not exist, and "none" is how it
   answers without calling anything — so a plain chat turn still costs one
   generation.
2. **Arguments.** Only if a tool was picked, a second constrained pass against
   *that tool's own* parameter schema, which is exactly the schema FM already
   derived from its `@Generable Arguments`.

The split is deliberate: one combined schema would need `anyOf` to switch the
argument shape on the chosen name, and `anyOf` is not safe to assume of a
JSON-Schema-to-grammar compiler. Both grammars here stay inside
object/properties/required/enum/string.

## Build

Needs Xcode 27 (iOS 27 SDK) and [XcodeGen](https://github.com/yonaskolb/XcodeGen).

```bash
xcodegen generate
open LFMTools.xcodeproj
```

`project.yml` points at `../../litert-lm-ios-host` — a local checkout of
LiteRT-LM (branch `apple-fm-guided-constrained-decoding`, pinned at the
last commit that builds against the v0.16.0 prebuilt) carrying the
constrained tool routing described above. Point it somewhere else, or at
the released package, if you have that branch elsewhere.

Then put a text `.litertlm` in the app's Documents folder:

```bash
xcrun devicectl device copy to --device <id> \
  --domain-type appDataContainer --domain-identifier com.lfmtools.app \
  --source LFM2.5-1.2B-Instruct_int4.litertlm --destination Documents/
```

or drag it in through Files.app (`UIFileSharingEnabled` is on).

Models: any text `.litertlm`, e.g.
[litert-community/LFM2.5-1.2B-Instruct](https://huggingface.co/litert-community).
CPU is the default and the safe choice; the GPU backend is a toggle on the model
picker.

## Trying things fast: Apple's model, any pack, spoken

The quickest way to see whether a tool or a wording works is Apple's own
model — no bundle, an answer in a second. Same session, same tools, same
cards; only who generates changes.

```bash
# chat: Apple's model, every tool in the sheet, mic button in the composer
xcrun devicectl device process launch --terminate-existing --device <id> \
  com.lfmtools.app --model apple

# stage: a scripted pack on Apple's model
xcrun devicectl device process launch --terminate-existing --device <id> \
  com.lfmtools.app --autorun --backend apple --scenario handoff

# stage: same pack, but each beat is whatever you say into the mic
xcrun devicectl device process launch --terminate-existing --device <id> \
  com.lfmtools.app --autorun --backend apple --scenario sensors --voice
```

Packs: `photo` (editing on a stage photo), `focus` (timer + notifications
+ brightness), `report` (photo OCR → note → reminder), `briefing` (time,
battery, calendar, reminders, steps), `sensors` (location, heading,
motion, sound level, altitude), `handoff` (torch, sounds, badge,
clipboard, notification, note), `chains` / `compound`, `vision` / `look` /
`polish` (the photo goes in as an attachment; the model routes on what it
sees), `video` (a CapCut-style menu — trim, split, speed, crop 9:16,
caption, fade, stabilise, volume, export — on the newest library video;
the app's state opens every message and the model never sees a frame),
`store` (a Shopify admin's menu over canned products and orders: filter →
selection → bulk action; records, not pixels).
Without `--scenario` the stage runs the
coffee-run beats; without `--backend apple` it runs on the newest LiteRT
bundle (or `--model <substring>`).

## Not included

- **WeatherKit** — needs a paid team and a capability, which would stop this
  from building on a personal team.
- **HealthKit** — same, plus an App ID with the capability enabled.
- **Parallel tool calls** — the router picks one tool per turn.
