# RTMPose-Face (WFLW) — on-device face alignment (LiteRT GPU, fully GPU)

[RTMPose](https://github.com/open-mmlab/mmpose) (mmpose, Apache-2.0) **face alignment** trained on **WFLW**:
**98 dense facial landmarks** (contour, eyebrows, eyes, nose, mouth, pupils), running **fully on the LiteRT
CompiledModel GPU**. The dense complement to the 5-point [YuNet](../yunet/) detector — detect a face, then align.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **333 / 333** LITERT_CL (full residency) |
| inference | **~4 ms** (256×256) |
| fp16 size | 33.6 MB |
| accuracy | device-vs-PyTorch SimCC corr **0.9995**, 98 landmarks |

```
face[1,3,256,256] (mmpose mean/std) →[GPU: RTMPose-m]→ simcc_x[1,98,512], simcc_y[1,98,512]
```

output[0] = simcc_x, output[1] = simcc_y; each landmark = `argmax` over its 1D SimCC (bins = pixels × 2).

## How it converts (litert-torch) — the RTMPose recipe, unchanged

Same model family as the human-pose RTMPose; only the config/checkpoint change to WFLW. The two on-device-only
Mali fixes transfer **without modification**: **`ScaleNorm` → SafeRMSNorm** (the RMS-norm channel `Σx²`
overflows fp16 on Mali → scale down by 64 before squaring, then rescale) and **GAU `act@act` BMM →
broadcast-reduce**. Result: banned ops NONE, all tensors ≤4D, tflite-vs-torch corr **1.0**, device-vs-torch
corr **0.9995**.

## Build & run

```bash
# weights auto-download via mmpose (WFLW rtmpose-m); needs the mm-stack (see ../rtmpose/README.md)
python scripts/build_rtm_face.py     # produces rtm_face_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-rtm_face_fp16.tflite>
```

The first launch fails with "Model not found" until the model is pushed.

**Preprocessing**: center-crop to a (centered) face, resize 256×256, mmpose mean/std (RGB, 0-255 scale), NCHW.

Model: `litert-community/RTMPose-Face-WFLW-LiteRT` (Hugging Face). Upstream:
[open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) (Apache-2.0); dataset
[WFLW](https://wywu.github.io/projects/LAB/WFLW.html).
