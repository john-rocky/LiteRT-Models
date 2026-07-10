# RAM++ — On-device multi-label image tagging (LiteRT GPU/CPU hybrid)

[RAM++ (Recognize Anything Plus)](https://github.com/xinyu1205/recognize-anything) (ICLR-workshop
2024, Apache-2.0) is an **open-vocabulary image tagger**: give it a photo and it returns the tags it
recognizes from a 4,585-tag vocabulary ("dog", "living room", "picture frame", …) — no fixed class
head, per-tag sigmoid. The Swin-L encoder stages 0-2 and the Query2Label tag head run on the LiteRT
CompiledModel **GPU**; the deep Swin block + the 479 MB frozen tag bank run on **CPU** (see below).

## On-device (Pixel 8a, Tensor G3 — verified)

| graph | in → out | delegate | note |
|---|---|---|---|
| G1 Swin stages 0-2 | image [1,3,384,384] → feat [1,144,1536] | **GPU** | corr 0.998 |
| C2 Swin stage 3 + norm + proj | feat → image_embeds [1,145,512] | **CPU** (exact) | fp16-fragile on GPU |
| R  reweight (multi-grained) | cls [1,512] → tag queries [1,4585,768] | **CPU** (exact) | 479 MB tag bank → 229 MB fp16 |
| B  Query2Label tag head | queries + image_embeds → logits [1,4585] | **GPU** | corr 0.9987, ~270 ms |

End-to-end the sample photo (living room, dog on a couch) tags in **~2 s** →
`dog · couch · living room · sit · carpet · picture frame · plant · armchair · lamp · pillow …`
(14 tags, all correct). The Mali fp16 GPU stages drop a handful of the reference's most borderline
descriptors (probabilities within ~0.01 of their threshold).

```
image →[ImageNet norm]→ [GPU Swin 0-2]→ feat →[CPU Swin-3 + norm + proj]→ image_embeds[1,145,512]
       token0 = cls →[CPU reweight over 4585×51 tag bank]→ queries[1,4585,768]
       (queries, image_embeds) →[GPU Q2L tag head]→ logits →[sigmoid + per-class threshold]→ tags
```

## Why the split — a Mali fp16 finding

The Swin-L encoder is fully GPU-convertible (window-partition re-authored ≤4D, qkv 3D-BMM attention,
baked relative-position bias, cyclic-shift as slice+concat, PatchMerging without strided slices,
tanh-GELU, SafeLayerNorm). But its **last stage miscomputes in fp16 on the Mali delegate**. Bisecting
the four stages on-device: stage 0 = 0.9999, stage 1 = 0.9999, stage 2 = 0.9983, **stage 3 = 0.709**.

It is **not** head_dim (stage 2 shares head_dim 32 and is fine) and **not** overflow — every stage-3
intermediate is < 848 (fp16 max 65504), and a round-to-fp16-between-ops simulation reproduces the
fp32 result at corr 0.99999997. It is Mali's **fp16 matmul accumulation** in the deep, high-magnitude
blocks (the residual stream grows to absmax 847; the 6144-wide fc2 and 48-head attention accumulate
in fp16). Down-scaling can't fix relative accumulation error, so those 2 blocks run on CPU — the same
fp16-fragile-part-on-CPU split as CLIPSeg and diarization. Everything else stays on GPU.

The reweight (RAM++'s multi-grained head: softmax over 51 text descriptions per class) bakes the
frozen tag bank **once** as fp16 with a runtime cast, keeping that graph at 229 MB instead of 686 MB.

## Build & run

```bash
python build_swin.py convert     # G-encoder pieces  (in ~/…/ramplus-work; needs the RAM++ checkpoint)
python build_hybrid.py           # -> ram_swin_s012_fp16.tflite + ram_stage3_tail_fp16.tflite
python build_head.py convert     # -> ram_taghead_fp16.tflite
python build_head.py reweight    # -> ram_reweight_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-the-4-tflites>
```

Launch the app — it tags the bundled sample on start; tap **Pick image** for your own photos.

Model bank ≈ 769 MB across 4 graphs (staged into the app's `filesDir` by the install script — never
bundled in the APK). Upstream: [xinyu1205/recognize-anything](https://github.com/xinyu1205/recognize-anything)
(Apache-2.0), `xinyu1205/recognize-anything-plus-model`.
