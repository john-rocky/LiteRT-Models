# Demo asset provenance (ormbg)

The demo clip is **Pexels License** — free for commercial and non-commercial use,
**no attribution required**, modification allowed (our matte and background
replacement are a derivative). The file itself is not committed; `**/assets/*.mp4`
is gitignored.

| file | source | author |
|---|---|---|
| `app/src/main/assets/demo_person.mp4` | https://www.pexels.com/video/a-man-talking-while-holding-a-cup-of-coffee-6930967/ | Mikhail Nilov |
| `app/src/androidTest/assets/person.jpg` (committed, 24 KB) | the frame at t = 3 s of the same clip, scaled to 480×854, JPEG — the fixture for `BgRemoverTest` | Mikhail Nilov |

Fetched via the Pexels API on 2026-08-23. Constraints we stay inside: no selling of
unaltered copies, no implication that the person depicted endorses anything, neutral
framing. Both build flavors read the same file, so the GPU and NPU runs see identical
frames and the comparison does not depend on where a camera was pointed.

## ⚠ Screen recording distorts the GPU run — do not record a GPU-vs-NPU comparison

Measured on the S26, same app, same build, one variable changed:

| condition | GPU inference | NPU inference |
|---|---|---|
| `adb shell screenrecord` running | **192 ms** | 25 ms |
| no recording | **75 ms** | 26 ms |

The control was taken at thermal status 1 — *hotter* than the recorded run — and still
returned 75 ms, so this is the recorder, not heat. The NPU is unaffected because it does
not share the GPU with the encoder.

Consequence: a screen-recorded side-by-side would show the GPU 2.6x slower than it is,
in our own favour. Record the **NPU** demo only (its on-screen figures are accurate) and
take GPU-vs-NPU numbers from the benchmark harness instead.
