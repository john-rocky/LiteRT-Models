# RTMPose-Animal (AP-10K) — on-device animal pose (LiteRT GPU, fully GPU)

[RTMPose](https://github.com/open-mmlab/mmpose) (mmpose, Apache-2.0) **animal pose estimation** trained on
**AP-10K**, running **fully on the LiteRT CompiledModel GPU**. Top-down: a square animal crop → **17 AP-10K
keypoints** (eyes, nose, neck, tail root, and the four limbs' shoulder/hip → elbow/knee → paw).

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **333 / 333** LITERT_CL (full residency) |
| inference | **~5 ms** (256×256) |
| fp16 size | 27.5 MB |
| accuracy | device-vs-PyTorch SimCC corr **0.999**, 17/17 keypoints |

```
image[1,3,256,256] (mmpose mean/std) →[GPU: RTMPose-m]→ simcc_x[1,17,512], simcc_y[1,17,512]
```

Output[0] = simcc_x, output[1] = simcc_y; each keypoint = `argmax` over its 1D x/y SimCC (bins = pixels × 2).

## How it converts (litert-torch) — the RTMPose recipe, unchanged

This is the **same model family** as the [RTMPose-s](../rtmpose/) human-pose sample — only the config/checkpoint
change to AP-10K. The two on-device-only Mali fixes (both monkey-patched at conversion time) transfer **without
modification**:

1. **`ScaleNorm` (RMS norm) → SafeRMSNorm** — the RTMCC `ScaleNorm` input reaches large magnitudes, so its
   channel `Σx²` overflows fp16 on the Mali delegate (→ `norm=∞` → `x/∞=0`, an all-zero head). Scale `x` down by
   64 **before** squaring, then rescale (math-identical).
2. **GAU `act@act` BMM → broadcast-reduce** — the Gated Attention Unit's `q@kᵀ` and `kernel@v` are
   activation×activation batch-matmuls the Mali delegate mis-computes; the exact replacement is
   `(q[:,:,None,:]·k[:,None,:,:]).sum(-1)`.

Result: banned ops NONE, all tensors ≤4D, tflite-vs-torch corr **1.0**, device-vs-torch corr **0.999**.

## Build & run

```bash
# weights auto-download via mmpose (AP-10K rtmpose-m); needs the mm-stack (see ../rtmpose/README.md)
python scripts/build_rtm_animal.py     # produces rtm_animal_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-rtm_animal_fp16.tflite>
```

The first launch fails with "Model not found" until the model is pushed.

**Preprocessing**: center-crop to a square, resize 256×256, mmpose mean/std (RGB, 0-255 scale), NCHW.

Model: `litert-community/RTMPose-Animal-AP10K-LiteRT` (Hugging Face). Upstream:
[open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) (Apache-2.0); dataset
[AP-10K](https://github.com/AlexTheBad/AP-10K).
