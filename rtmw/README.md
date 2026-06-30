# RTMW-m — Whole-Body Pose on-device (LiteRT GPU, fully GPU)

[RTMW](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) (mmpose, CSPNeXt + CSPNeXtPAFPN neck +
RTMW/SimCC head) **whole-body** 2D pose, running **fully on the LiteRT CompiledModel GPU**. It predicts **133
COCO-WholeBody keypoints** — 17 body + 6 feet + 68 face + 42 hands — for a single centered person. The model
ControlNet/animation pipelines want, device-verified end-to-end on a Pixel 8a.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **531 / 531** LITERT_CL (full residency) |
| inference | **~6 ms** (256×192) |
| fp16 size | 66 MB |
| accuracy | device-vs-PyTorch SimCC corr **0.999**, keypoints within **0.2 px** |

```
image[1,3,256,192] (ImageNet 0-255) →[GPU: CSPNeXt + PAFPN + RTMW]→ simcc_x[1,133,384], simcc_y[1,133,512]
```

## How it converts (litert-torch)

Same RTMPose-family recipe as the body model ([`../rtmpose/`](../rtmpose/)), **plus one** extra fix for RTMW's
neck/head:

1. **`ScaleNorm` (RMS) → SafeRMSNorm** — the RTMCC `ScaleNorm` input overflows fp16 (`Σx²`≈3.6M > 65504) on
   Mali → `norm=∞` → all-zero head; scale `x` down by S=64 before squaring (exact).
2. **GAU `act@act` BMM → broadcast-reduce** — the Gated Attention Unit's `q@kᵀ`/`kernel@v` over the keypoint
   tokens.
3. **`PixelShuffle` → depth-to-space `ZeroStuffConvT2d`** — the RTMW head upsamples the top feature map with
   `nn.PixelShuffle`, which litert-torch lowers to a 6D tensor (>4D, GPU-rejected). Replacing it with a fixed
   depth-to-space `ConvTranspose2d` (the PixelShuffle permutation as the kernel) wrapped in `ZeroStuffConvT2d`
   keeps it 4D and exact (reused from NAFNet/Metric3D).

Result: banned ops NONE, all tensors ≤4D, tflite-vs-torch corr **1.0**, device-vs-torch corr **0.999**.

## Build & run

The conversion needs the mmpose stack (lightweight stubs let `mmcv-lite` suffice — see `build_rtmw.py`):

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer torch mmengine mmcv-lite mmpose --no-deps munkres json_tricks
RTM_CFG="wholebody_2d_keypoint/rtmpose/cocktail14/rtmw-m_8xb1024-270e_cocktail14-256x192.py" \
RTM_CKPT="https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122.pth" \
RTM_H=256 RTM_W=192 RTM_NAME=rtmw python scripts/build_rtmw.py all
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-rtmw_fp16.tflite>
```

**Preprocessing**: center-crop to 3:4, resize to 192×256, ImageNet 0-255 normalize, NCHW. Top-down — one
centered person. Output = two 1D SimCC distributions per keypoint; argmax (÷ split=2) → pixel.

Model: `litert-community/RTMW-m-WholeBody-LiteRT` (Hugging Face). Upstream:
[open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) RTMW (Apache-2.0).
