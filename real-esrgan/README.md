# Real-ESRGAN Super Resolution

4x image upscaling using Real-ESRGAN-General-x4v3 with LiteRT CompiledModel GPU.

## Features

- **4x super resolution** with Real-ESRGAN (1.21M params, 4.7 MB)
- Before/after comparison slider UI
- Tile-based processing for arbitrary image sizes
- Progress bar during upscaling

## Setup

1. Download model from [Releases](https://github.com/john-rocky/LiteRT-Models/releases/tag/v1):
   - `real_esrgan_x4v3.tflite` (4.7 MB)
2. Place in `app/src/main/assets/`
3. Open this directory in Android Studio and run

## Architecture

| File | Description |
|------|-------------|
| `Upscaler.kt` | CompiledModel GPU inference with 128x128 tile processing and overlap stitching |
| `CompareView.kt` | Draggable before/after comparison slider |
| `MainActivity.kt` | Image picker + upscale pipeline with progress |

## Model Details

- **Input**: `[1, 128, 128, 3]` float32 NHWC, RGB normalized to 0-1
- **Output**: `[1, 512, 512, 3]` float32 NHWC, 4x upscaled RGB (0-1 range)
- **Tiling**: Images larger than 128x128 are processed as overlapping tiles (16px overlap) and stitched
- **Ops**: Only 7 TFLite ops (ADD, CONV_2D, PRELU, RESHAPE, RESIZE_NEAREST_NEIGHBOR, TRANSPOSE) — fully GPU compatible
