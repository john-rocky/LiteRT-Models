# HSEmotion — facial emotion recognition (LiteRT GPU)

Recognize the **8 AffectNet emotions** (anger, contempt, disgust, fear, happiness,
neutral, sadness, surprise) from a face, fully on the LiteRT `CompiledModel` GPU.
[HSEmotion](https://github.com/av-savchenko/face-emotion-recognition) (EmotiEffLib,
Apache-2.0) is an EfficientNet-B0 fine-tuned on AffectNet.

- Model: [av-savchenko/face-emotion-recognition](https://github.com/av-savchenko/face-emotion-recognition)
  `enet_b0_8_best_afew.pt`, Apache-2.0. EfficientNet-B0, 224×224.
- Graph: `image[1,3,224,224]` → `logits[1,8]`. 342/342 ops on the GPU delegate,
  1 partition, **~2 ms/inference** (fp16, Pixel 8a).
- Device fp16 top-1 matches desktop fp32 (logits corr 0.99997).

## Two conversion hurdles

1. **Old-timm pickle.** The released weights are a pickled model built with an old
   timm whose forward is broken under current timm (missing `conv_s2d`). The fix:
   lift the state dict into a fresh timm `tf_efficientnet_b0` (num_classes=8,
   remapping `classifier.0.*` → `classifier.*`), which has a working forward and
   358/360 tensors matching by name+shape.
2. **fp16 SqueezeExcite mean → NaN.** ⭐The SE block's global mean `x.mean((2,3))`
   over the 112×112 stem map is a single fp16 reduction whose partial sum overflows
   65504 → the delegate emits **all-NaN** (it computes in fp16 even for an fp32
   graph; desktop fp16 CPU is fine). Replaced by a **hierarchical mean** — repeated
   `avg_pool2d` over equal-size tiling windows (≤ 49 elements each) — which is
   mathematically identical but keeps every accumulation small.

## Build & run

The model is not committed (8 MB). Get it from Hugging Face
([`litert-community/HSEmotion-B0-LiteRT`](https://huggingface.co/litert-community/HSEmotion-B0-LiteRT))
or build it, and place it in `app/src/main/assets/`:

```bash
cd scripts/
# enet_b0_8_best_afew.pt + happy.jpg next to the script
python build_hsemotion.py all      # parity → convert → fp16 → device check
cp hsemotion_b0_fp16.tflite ../app/src/main/assets/
```

Then open this directory in Android Studio, build, and run. Pick a face photo (or
use the bundled sample); the app detects the face with `android.media.FaceDetector`,
crops it, and shows the emotion distribution.

## Files

| File | Role |
|------|------|
| `app/src/main/java/com/hsemotion/EmotionClassifier.kt` | face crop + CompiledModel run + softmax |
| `app/src/main/java/com/hsemotion/MainActivity.kt` | image picker + emotion display |
| `scripts/build_hsemotion.py` | timm rebuild, safe-SE mean, parity, convert, fp16, device |
