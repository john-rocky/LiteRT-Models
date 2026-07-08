# DINOv2 — dense feature visualization (LiteRT GPU)

Run the self-supervised [DINOv2](https://github.com/facebookresearch/dinov2)
ViT-S/14 backbone on the LiteRT `CompiledModel` GPU and visualize its **dense
patch features** — a top-3 PCA of the tokens mapped to RGB. Semantically similar
patches (object parts vs background) land near each other in feature space, so
they share a color; the object "pops out" without any labels or segmentation.

- Model: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
  ViT-S/14 (via `timm`), Apache-2.0.
- Graph: `image[1,3,448,448]` → `tokens[1,1024,384]` (32×32 patch grid).
  864/864 ops on the GPU delegate, 1 partition, **~8 ms** (fp16, Pixel 8a).
- Device fp16 patch features vs desktop fp32: corr 0.996.

## GPU re-authoring (proven ViT recipes)

- **4D attention (C12)**: the fused-qkv attention is split into q/k/v and reshaped
  to `[1, heads, N, d]` (≤4D), then manual `softmax(qkᵀ/√d)·v` — the delegate
  rejects the native 5D head-split reshape.
- **SafeLayerNorm**: the deviation is scaled by 1/64 before squaring so the
  per-token sum of squares stays within fp16 range on DINOv2's massive
  activations, then rescaled (algebraically identical).
- **LayerScale** (`ls1`/`ls2`) is baked into the following projection weights.
- **tanh-GELU** (`0.5x(1+tanh(…))`) — near-exact and delegate-friendly (the
  sigmoid approximation drifts to feature corr 0.968 over 12 blocks; tanh → 0.99999).
- The **pos_embed is baked at a fixed 448 grid** by timm at model creation, so
  there is no runtime interpolation (no `GATHER_ND`).

Host side: a top-3 PCA of the 1024×384 token matrix (power iteration on the
384×384 covariance) → per-patch RGB → upscaled overlay.

## Build & run

The model is not committed (45 MB). Get it from Hugging Face
([`litert-community/DINOv2-ViT-S14-LiteRT`](https://huggingface.co/litert-community/DINOv2-ViT-S14-LiteRT))
or build it, and place it in `app/src/main/assets/`:

```bash
cd scripts/
python build_dinov2.py all      # parity → convert → fp16 → device → viz
cp dinov2_s_fp16.tflite ../app/src/main/assets/
```

Then open this directory in Android Studio, build, and run. Pick a photo (or use
the bundled sample) to see the image and its DINOv2 feature-PCA side by side.

## Files

| File | Role |
|------|------|
| `app/src/main/java/com/dinov2/Dinov2Features.kt` | CompiledModel run + host PCA → RGB |
| `app/src/main/java/com/dinov2/MainActivity.kt` | image picker + side-by-side view |
| `scripts/build_dinov2.py` | ViT re-authoring, parity, convert, fp16, device, viz |
