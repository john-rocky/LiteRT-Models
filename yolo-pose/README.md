# YOLO26 Pose Estimation

Real-time human pose estimation with YOLO26n-pose on LiteRT CompiledModel GPU.

## Features

- 17 COCO keypoints + skeleton overlay
- Per-person bounding box, NMS in Kotlin
- CameraX preview, FILL_CENTER overlay scaling

## Setup

1. Convert the model:

   ```bash
   cd yolo-pose/scripts
   pip install ultralytics litert-torch
   python convert_yolo26n_pose.py
   ```

   The script wraps the Ultralytics YOLO26 pose model so the head returns the
   raw one-to-many output (`end2end=False, export=True, format='tflite'`),
   then converts via `litert_gpu_toolkit` (litert-torch backend). The result
   is written directly to `app/src/main/assets/yolo26n_pose.tflite`.

2. Open this directory in Android Studio and run.

## Architecture

| File | Description |
|------|-------------|
| `PoseEstimator.kt` | CompiledModel GPU inference + raw output decoding + NMS |
| `PoseCanvasView.kt` | Skeleton + keypoint overlay matching PreviewView FILL_CENTER |
| `MainActivity.kt` | CameraX preview wiring |
| `CocoKeypoints.kt` | 17 COCO keypoint names + skeleton edges + colors |
| `Pose.kt` | Per-person data class (bbox + 17 keypoints) |

## Model Details

- Input: NCHW `[1, 3, 384, 384]`, RGB / 255 (no ImageNet mean/std), planar layout
- Output: `[1, 56, 3024]` — `4 bbox (x1,y1,x2,y2) + 1 person conf + 17*3 keypoints`
- Bbox / keypoint xy are in input image pixel space (0..384), rescaled to the
  original bitmap in `PoseEstimator.postProcess`. Person confidence and
  per-keypoint visibility are sigmoid-activated (0..1).
- File size: ~12 MB (FP32)

## Conversion Notes

- The default YOLO26 end-to-end head uses `torch.topk` (`TOPK_V2/GATHER`),
  rejected by CompiledModel GPU. We bypass it via `head.end2end=False`,
  `head.export=True`, `head.format='tflite'`, which exposes the raw
  one-to-many head output `[1, 56, N]`.
- `onnx2tf` (Ultralytics' default TFLite export path) breaks YOLO26's
  Window Attention / C2PSA block with a channel-tracking bug at
  `model.2/m.0/Add`. We sidestep it by going **PyTorch → litert-torch directly**,
  the same path used for `mobilesam`, `rmbg`, and `dsine` in this repo.
- `BATCH_MATMUL` from C2PSA attention is reported as "incompatible" by the
  toolkit checker, but the existing `yolo26n.tflite` in this repo also has
  4 BATCH_MATMUL ops and runs fine on GPU via the LiteRT delegate.
