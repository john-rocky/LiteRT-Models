# YOLOX-Nano Object Detection

Real-time object detection using **YOLOX-Nano** (Apache-2.0) with LiteRT CompiledModel GPU.

YOLOX is a permissive (Apache-2.0) alternative to the AGPL YOLO family, and a pure CNN —
so it converts cleanly to a **GPU-native** TFLite with zero GATHER / TopK / Cast ops.

## Features

- **~21 FPS** real-time camera detection on Pixel 8a (CompiledModel GPU)
- 80 COCO class labels with colored bounding boxes
- Letterbox preprocessing + grid/stride decode + NMS in Kotlin
- Apache-2.0 model — no AGPL constraints

## Setup

1. Download the model from Hugging Face: **[mlboydaisuke/yolox-nano-litert](https://huggingface.co/mlboydaisuke/yolox-nano-litert)**
   - `yolox_nano.tflite` (3.7 MB, FP32)
2. Place it in `app/src/main/assets/`
3. Open this directory in Android Studio and run (back camera + grant permission)

First launch compiles GPU shaders (~15 s, shown as "Compiling GPU model…"); subsequent frames run at full speed.

## Architecture

| File | Description |
|------|-------------|
| `YoloxDetector.kt` | CompiledModel GPU inference: letterbox preprocess, grid+stride box decode, per-class NMS |
| `RealtimeCameraPipeline.kt` | Reusable two-thread camera pipeline (preview + analysis share 4:3) |
| `DetectionCanvasView.kt` | Bounding-box overlay matching PreviewView's FILL_CENTER scaling |
| `MainActivity.kt` | CameraX preview + async model load |
| `CocoLabels.kt` | 80 COCO class names and colors |
| `Detection.kt` | Detection data class |

## Model Details

- **Input** `images [1, 416, 416, 3]` NHWC, **BGR, 0-255, no normalization**, letterboxed (pad 114).
- **Output** `Identity [1, 3549, 85]` raw head output, anchor-major. `85 = 4 box (x,y,w,h) + 1 obj + 80 class`.
  - obj/class are sigmoid'd in-graph; **box decode is NOT** — done in Kotlin: for strides `[8,16,32]`
    (52²+26²+13² = 3549 anchors) `cx=(x+gx)·s, cy=(y+gy)·s, w=exp(w)·s, h=exp(h)·s`; `score = obj·clsmax`.

## Conversion

`scripts/convert_yolox.py` — downloads the official YOLOX-Nano ONNX (Megvii, Apache-2.0) and converts
to GPU-clean TFLite via onnx2tf (pure CNN → no op rewrites). Verified GPU-clean: 0 banned ops, 0 Flex,
0 >4D tensors, 0 dynamic dims.
