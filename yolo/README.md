# YOLO Object Detection

Real-time object detection using YOLO11n / YOLO26n with LiteRT CompiledModel GPU.

## Features

- **18+ FPS** real-time camera detection on Pixel 8a
- YOLO11n and YOLO26n switchable via spinner
- 80 COCO class labels with colored bounding boxes
- NMS post-processing in Kotlin

## Setup

1. Download models from [Releases](https://github.com/john-rocky/LiteRT-Models/releases/tag/v1):
   - `yolo11n.tflite` (10 MB)
   - `yolo26n.tflite` (9.3 MB)
2. Place in `app/src/main/assets/`
3. Open this directory in Android Studio and run

## Architecture

| File | Description |
|------|-------------|
| `ObjectDetector.kt` | CompiledModel GPU inference + NMS. Handles both YOLO11 (xywh) and YOLO26 (xyxy) bbox formats |
| `DetectionCanvasView.kt` | Bounding box overlay matching PreviewView's FILL_CENTER scaling |
| `MainActivity.kt` | CameraX preview + model selector spinner |
| `CocoLabels.kt` | 80 COCO class names and colors |
| `Detection.kt` | Detection data class |

## Model Details

Both models output `[1, 84, 3024]` — 4 bbox coords + 80 class scores per detection.

- **YOLO11n**: Standard Ultralytics export. Bbox format: `[cx, cy, w, h]` normalized 0-1.
- **YOLO26n**: NMS-free head's top-k removed for GPU compatibility. Bbox format: `[x1, y1, x2, y2]` normalized 0-1.

Preprocessing: RGB / 255 (no ImageNet normalization).
