# SAM 3 on iPhone — LiteRT CompiledModel (Metal GPU)

Text-prompted detection + instance segmentation with the image side of
`facebook/sam3.1` (ViT-L/14 @1008 → tri-neck → text-conditioned DETR head →
288×288 masks), fully on device:

| graph | accelerator | why |
|---|---|---|
| `sam3_vision.tflite` (930 MB fp16) | Metal GPU, fp16 | detection-invariant feature error; fastest |
| `sam3_text.tflite` (607 MB fp16) | CPU (XNNPACK) | CLIP-L residual stream hits \|x\|≈1.2e3 — fp16 GPU corrupts some prompts; graph is fast anyway |
| `sam3_head.tflite` (68 MB fp16) | Metal GPU, enforce_f32 | exact scores/masks (fp16 leaves borderline \|Δprob\| up to 0.15) |

Host side: CLIP BPE tokenizer (Swift port), fp16 token-embedding lookup
(101 MB table, memory-mapped), score thresholding, mask sigmoid + overlay.

## Build

```bash
brew install xcodegen   # once
cd sam3/ios
xcodegen                # generates SAM3.xcodeproj
open SAM3.xcodeproj
```

Depends on the local `swift-litert-lm` package (`~/code/swift-litert-lm`),
which supplies the LiteRT Next C API via the prebuilt `CLiteRTLM.xcframework`
from the official LiteRT-LM releases. The GPU (Metal) accelerator is statically
linked in that framework — no dylib staging needed.

## Model files

Never committed. After building `sam3/models/out/` on the Mac
(`scripts/build_sam3.py`), copy these six files onto the phone — either into
the app's **Documents** folder (Finder → iPhone → Files → SAM3; the app also
supports the Files app / AirDrop → Save to SAM3) or into `Resources/` before
building to embed them in the app bundle:

```
sam3_vision.tflite
sam3_text.tflite
sam3_head.tflite
sam3_token_embed.bin
sam3_tokenizer/vocab.json   -> vocab.json
sam3_tokenizer/merges.txt   -> merges.txt
```

First launch compiles the graphs on the Metal GPU (tens of seconds); later
launches are faster. Re-prompting the same photo skips the vision graph
(features are cached), so it runs at head+text speed.

## Numbers to expect

M4 Max (same graphs, Python runner): vision fp16 ≈ 560 ms, text ≈ 12 ms,
head f32 ≈ 175 ms. iPhone 16-class Metal estimate: vision 2–4 s, re-prompt
well under a second. Parity vs PyTorch fp32 was verified on Mac and Pixel 8a
(kept-set equal, mask IoU ≥ 0.97; see `sam3/PRECHECK_2026-08-19.md`).

## Tracker autotest (video, stage 2)

The Object-Multiplex tracker (host state machine + 4 tracker graphs sharing the
vision_tri trunk) is ported to Swift: `Sources/{Sam3Tracker,TrackerMath,
TrackerConsts,TrackerAutotest}.swift` — a 1:1 port of the Kotlin/Python host
loop (spec: `sam3/TRACKER_HOST_PORT.md`).

Stage the tracker payload with `./stage_models.sh tracker`, then drag BOTH the
six image-side files AND the `tracker` folder plus `sam3_vision_tri.tflite`
into Finder → iPhone → Files → SAM3. While `Documents/tracker/expected/
manifest.json` exists the app boots into the tracker autotest (runs the bundled
clip with the manifest prompt, compares per frame against the Mac host-loop
fixtures, writes `Documents/tracker_result.txt`); delete the `tracker` folder
to return to image mode. The text encoder is loaded, run once for the prompt,
and released before the GPU graphs compile — keep that order if you touch the
loading code (it is the difference between ~3.5 GB and ~5.5 GB peak).

Verified on the Mac (same sources, all graphs on the CPU accelerator — the mac
prebuilt's GPU is WebGPU and cannot run these graphs): ids identical on all 8
frames vs the f32 host-loop fixtures, min mask IoU 0.998 with decode-matched
frames (0.979 across the ImageIO-vs-PIL JPEG decoder gap), max |Δprob| 0.007.

## Tracker demo mode (hands-free showcase)

For screen recordings: when `Documents/trackerdemo/demo.json` exists the app boots
into an automatic video-tracking showcase — compiles the tracker, types the prompt
character-by-character, tracks it through the staged clip (raw frames advance with
progress chips while the GPU works; final per-frame outputs only exist at the end
of the run because of the tracker's hotstart delay), then loops the composited
overlay playback forever. No touches needed. Takes precedence over the autotest.

Payload layout (in the app's Documents):

```
tracker/graphs/*.tflite      tracker graphs (same as the autotest; expected/ not needed)
tracker/consts/  tracker/flags.json
trackerdemo/frames/0.jpg …   the clip, numbered, 1280×720
trackerdemo/demo.json        { "prompt": "person", "fps": 7.5, "startDelay": 2.0 }
```

Plus the image-side root files (`sam3_vision_tri.tflite`, `sam3_text.tflite`,
`sam3_head.tflite`, `sam3_token_embed.bin`, `vocab.json`, `merges.txt`). Delete
`Documents/trackerdemo` to return to the autotest / image modes.
