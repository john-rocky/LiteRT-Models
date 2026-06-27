# RTMPose-s — Real-time 2D Human Pose on-device (LiteRT GPU, fully GPU)

[RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) (mmpose, CSPNeXt backbone +
RTMCC/SimCC head) top-down 2D human pose, running **fully on the LiteRT CompiledModel GPU** (ML Drift). It
estimates 17 COCO keypoints for a single person and draws the skeleton — the SOTA real-time pose model,
device-verified end-to-end on a Pixel 8a.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **256 / 256** LITERT_CL (full residency) |
| inference | **~4 ms** (256×192) |
| fp16 size | 11.1 MB |
| accuracy | device-vs-PyTorch SimCC corr **0.999**, keypoints within **0.3 px** (max 1 px) |

```
image[1,3,256,192] (ImageNet 0-255 norm) →[GPU: CSPNeXt + RTMCC]→ simcc_x[1,17,384], simcc_y[1,17,512]
```

The SimCC head emits two 1D distributions per keypoint; argmax over the bins (÷ split=2) gives the pixel
coordinate. The decode + skeleton run in the app.

## How it converts (litert-torch) — two numerically-exact re-authorings

The CSPNeXt backbone (SiLU, pure CNN) and the diffusers-free RTMCC head convert clean, **except** two
on-device-only Mali issues (both pass the desktop op-check and report full LITERT_CL residency, yet the
device output was wrong until fixed — *residency ≠ correctness*):

1. **`ScaleNorm` (RMS norm) fp16 overflow → all-zero head.** The RTMCC `ScaleNorm` input reaches ≈ |274|, so
   its channel `Σ x²` ≈ 3.6M **overflows fp16 (max 65504)** on the Mali delegate (which reduces in fp16 even
   for an fp32 graph) → `norm = ∞` → `x/∞ = 0` → the entire head collapses to zero. Fix: scale `x` down by
   S=64 **before** squaring, then rescale (math-identical). This is the same class as the NAFNet
   SafeLayerNorm finding, here in an RMS norm.
2. **GAU attention `act@act` BMM → broadcast-reduce.** The Gated Attention Unit's `q@kᵀ` and `kernel@v` are
   activation-by-activation batch-matmuls, which the Mali delegate mis-computes. With only K=17 tokens, the
   exact replacement is `(q[:,:,None,:]·k[:,None,:,:]).sum(-1)` (broadcast-multiply + reduce-sum).

Result: banned ops NONE, all tensors ≤4D, tflite-vs-torch corr **1.0**, device-vs-torch corr **0.999**.

## Files

| file | what |
|---|---|
| `app/src/main/java/com/rtmpose/RtmPoseEstimator.kt` | model wrapper (norm → CompiledModel GPU → SimCC argmax) |
| `app/src/main/java/com/rtmpose/MainActivity.kt` | image picker + COCO skeleton overlay |
| `scripts/build_rtmpose.py` | conversion: mmpose RTMPose-s + the two re-authorings + op-check + fp16 + parity |
| `scripts/device_gate_rtm.py` | real-image torch-vs-tflite parity + keypoint decode fixture |

## Build & run

The conversion needs the mmpose stack (the script sets up lightweight stubs so `mmcv-lite` suffices — no
compiled ops; see the header of `build_rtmpose.py`):

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer torch mmengine mmcv-lite mmpose --no-deps munkres json_tricks
python scripts/build_rtmpose.py        # downloads the RTMPose-s checkpoint; emits rtmpose_s_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-rtmpose_s_fp16.tflite>
```

The first launch fails with "Model not found" until the install script has been run.

**Preprocessing**: center-crop to 3:4, resize to 192×256, ImageNet 0-255 normalize (mean [123.675, 116.28,
103.53], std [58.395, 57.12, 57.375]), NCHW planar. Top-down — expects one roughly-centered person.

Model: `litert-community/RTMPose-s-LiteRT` (Hugging Face). Upstream:
[open-mmlab/mmpose](https://github.com/open-mmlab/mmpose) RTMPose-s (Apache-2.0).
