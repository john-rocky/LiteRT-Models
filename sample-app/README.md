# DepthAnything V2 Sample App

Real-time monocular depth estimation using the camera.

## Setup

1. Download the model from [GitHub Releases](../../releases) and place it in `app/src/main/assets/`:
   - `depth_anything_v2_keras.tflite` (518x518, 99 MB)

2. Open the `sample-app/` directory in Android Studio and run on a device with GPU support.

## Requirements

- Android 8.0+ (API 26)
- Device with ML Drift GPU support (Pixel 6+, Samsung with Mali/Adreno)
- LiteRT 2.1.3

## Architecture

- **CameraX** for camera preview and frame capture
- **LiteRT CompiledModel** (ML Drift GPU engine) for inference
- **Inferno colormap** overlay on camera feed
- ~9ms inference on Pixel 8a (Tensor G3)
