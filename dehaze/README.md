# DehazeFormer-MCT — Image dehazing (LiteRT GPU)

Real-time **image dehazing** with the network fully on the LiteRT `CompiledModel` GPU.
[DehazeFormer](https://github.com/IDKiro/DehazeFormer) (TIP 2023, MCT curve-mapping variant,
trained on a mixed dataset for real-world haze) removes fog / haze / smoke and restores
contrast and color.

- **Model:** [IDKiro/DehazeFormer_Demo](https://huggingface.co/spaces/IDKiro/DehazeFormer_Demo) (author-hosted) · MIT · 1.2M params
- **HF:** [litert-community/DehazeFormer-MCT-LiteRT](https://huggingface.co/litert-community/DehazeFormer-MCT-LiteRT)
- **Input:** `[1, 3, 256, 256]` NCHW, RGB in `[-1, 1]`
- **Output:** `[1, 72, 256, 256]` curve parameters (3 out × 3 in × 8 levels)
- **Size:** 17 MB · Swin-style windowed attention

The MCT design is mobile-ideal: the network runs at a fixed 256×256 regardless of frame
size; the predicted per-pixel curves are applied to the **full-resolution** frame host-side
(`Dehazer.applyCurves`, the exact official grid_sample mapping — verified corr 1.0000000
against PyTorch).

## GPU conversion

Fully GPU-resident (**2042/2042 nodes, 1 partition**; Pixel 8a corr 0.999998, E2E vs the
official pipeline corr 0.999997) with exact re-authors:

- reflect pads → slice+concat (litert-torch lowers `reflection_pad2d` to `GATHER_ND`,
  rejected by the delegate — including `padding_mode='reflect'` convs with padding=0)
- Swin window partition/reverse → ≤4D reshape/permute; qkv → channel slices; relative
  position bias baked from the meta MLP
- **RLN global norm / SKFusion global pool → hierarchical means**: a single `MEAN` over
  C·H·W (1.5M elements) overflows the Mali fp16 accumulator → NaN output; equal-window
  `avg_pool` stages + small MEANs are mathematically identical and fp16-safe
- SKFusion 5D view+softmax → 4D pairwise softmax; Conv+PixelShuffle → zero-stuff
  ConvTranspose (per-subpixel bias as constant map)

## Build & run

```bash
cd dehaze/
./gradlew :app:installDebug
```

The 17 MB `dehazeformer_base.tflite` is bundled in `app/src/main/assets/` (build it with
`scripts/build_dehaze.py`; not committed). Point the camera at a hazy scene — the dehazed
frame is shown full-screen; tap to compare with the original.

## Regenerate the model

```bash
pip install torch litert-torch huggingface_hub numpy
python scripts/build_dehaze.py   # fetches code+weights from the author's HF Space
cp dehazeformer_base.tflite app/src/main/assets/
```

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
- ~255 ms/frame network + host curve mapping at frame resolution (~4 fps live).
- Output resolution = input frame resolution (curves applied full-res, not 256).
