# RTMPose-Hand — 21-keypoint Hand Pose on-device (LiteRT GPU, fully GPU)

[RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) (mmpose, CSPNeXt + RTMCC/SimCC head)
**hand** pose, running **fully on the LiteRT CompiledModel GPU**. It predicts the **21 standard hand
keypoints** (wrist + 4 joints × 5 fingers) for a single centered hand — useful for gesture/hand tracking.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **333 / 333** LITERT_CL (full residency) |
| inference | **~4 ms** (256×256) |
| fp16 size | 28 MB |
| accuracy | device-vs-PyTorch SimCC corr **0.999**, 21/21 keypoints |

```
image[1,3,256,256] (ImageNet 0-255) →[GPU: CSPNeXt + RTMCC]→ simcc_x[1,21,512], simcc_y[1,21,512]
```

## How it converts (litert-torch)

Identical RTMPose-family recipe to the body model ([`../rtmpose/`](../rtmpose/)) — the two re-authorings are
shared and numerically exact:

1. **`ScaleNorm` (RMS) → SafeRMSNorm** — the RTMCC `ScaleNorm` input overflows fp16 on Mali → all-zero head;
   scale `x` down by S=64 before squaring.
2. **GAU `act@act` BMM → broadcast-reduce** — the Gated Attention Unit attention over the keypoint tokens.

No PixelShuffle (no neck) — the plain RTMCC head converts as-is. Result: banned ops NONE, all tensors ≤4D,
tflite-vs-torch corr **1.0**, device-vs-torch corr **0.999**.

## Build & run

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer torch mmengine mmcv-lite mmpose --no-deps munkres json_tricks
RTM_CFG="hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py" \
RTM_CKPT="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth" \
RTM_H=256 RTM_W=256 RTM_NAME=rtmhand python scripts/build_rtmhand.py all
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-rtmhand_fp16.tflite>
```

**Preprocessing**: center-crop to square, resize to 256×256, ImageNet 0-255 normalize, NCHW. Top-down — one
centered hand. Output = two 1D SimCC distributions per keypoint; argmax (÷ split=2) → pixel.

Model: `litert-community/RTMPose-Hand-LiteRT` (Hugging Face). Upstream:
[open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) RTMPose-Hand (Apache-2.0).
