# Vision-RWKV (VRWKV-S) — ImageNet classification on LiteRT GPU

The first **RWKV-style vision backbone running fully on the LiteRT `CompiledModel`
GPU delegate**. [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV) (ICLR 2025,
Apache-2.0) replaces softmax self-attention with a **bidirectional WKV** linear-
attention scan. This demo runs the VRWKV-S ImageNet-1K classifier (80.1% top-1) —
the vision companion to the RWKV-7 language-model demo in [`rwkv7/`](../rwkv7/).

- Model: [OpenGVLab/Vision-RWKV](https://huggingface.co/OpenGVLab/Vision-RWKV) VRWKV-S,
  Apache-2.0. 23.8M params, 224×224, 14×14 = 196 tokens.
- Graph: `image[1,3,224,224]` + `dist[1,1,196,196]` → `logits[1,1000]`.
  1371/1371 ops on the GPU delegate, 1 partition, **~28 ms/inference** (fp16, Pixel 8a).
- Device fp16 top-1 matches desktop fp32 (bundled Samoyed sample; logits corr 0.9989).

## The Bi-WKV re-authoring

VRWKV's token mixer is a CUDA `bi_wkv` kernel. Because the token count is fixed
(196), the bidirectional WKV is exactly a **per-channel decay-biased attention**:

```
L[c,t,i] = k[c,i] − (spatial_decay[c]/T)·|t−i| + (spatial_first[c]/T)·δ(t,i)
y[c,t]   = Σ_i softmax_i(L[c,t,·]) · v[c,i]
```

i.e. C independent [T,T] attention matrices → plain 4D `softmax` + `matmul`, no
sequential scan. Two details make it GPU-clean and small:

- The token-distance matrix `dist[t,i] = |t−i|` is a **runtime input** (not a
  frozen constant), so the per-channel `[C,T,T]` decay bias is computed live
  instead of being const-folded into a ~59 MB-per-block flatbuffer constant
  (that folding makes an unshippable 1.5 GB model that fp16 can't shrink; the
  runtime-input form is 48 MB). `eye = relu(1 − dist)` avoids a second input.
- VRWKV-S uses **post-norm** (norm after the mixer); the LayerScale gamma is
  baked into the following norm's affine params, q-shift is pad+slice+concat (≤4D).

## Build & run

The model is not committed (48 MB). Get it from Hugging Face
([`litert-community/Vision-RWKV-S-LiteRT`](https://huggingface.co/litert-community/Vision-RWKV-S-LiteRT))
or build it, and place it in `app/src/main/assets/`:

```bash
cd scripts/
# vrwkv_s_in1k_224.pth (OpenGVLab) + imagenet_classes.txt + dog.jpg next to the script
python build_vrwkv.py all      # parity → convert → fp16 → device check
cp vrwkv_s_fp16.tflite ../app/src/main/assets/
```

Then open this directory in Android Studio, build, and run. Pick a photo (or use the
bundled sample) to get the top-5 ImageNet predictions.

## Files

| File | Role |
|------|------|
| `app/src/main/java/com/vrwkv/VrwkvClassifier.kt` | CompiledModel 2-input run, ImageNet preprocessing, top-5 |
| `app/src/main/java/com/vrwkv/MainActivity.kt` | Image picker + top-5 display |
| `scripts/build_vrwkv.py` | Bi-WKV re-authoring, parity, convert, fp16, device check |
