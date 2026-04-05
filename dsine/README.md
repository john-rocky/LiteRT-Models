# DSINE Surface Normal Estimation

Monocular surface normal estimation using DSINE (CVPR 2024) with LiteRT CompiledModel GPU.

## Features

- **Per-pixel surface normal estimation** from a single image
- EfficientNet-B5 encoder + custom decoder (72.6M params)
- RGB normal map visualization (R=X, G=Y, B=Z)
- Original / Normals toggle comparison

## Setup

1. Download model from [Releases](https://github.com/john-rocky/LiteRT-Models/releases/tag/v2):
   - `dsine.tflite` (282 MB)
2. Push to device (too large for APK assets):
   ```bash
   adb push dsine.tflite /data/local/tmp/
   adb shell "run-as com.dsine cp /data/local/tmp/dsine.tflite /data/data/com.dsine/files/"
   ```
3. Open this directory in Android Studio and run

## Architecture

| File | Description |
|------|-------------|
| `NormalEstimator.kt` | CompiledModel GPU inference with center-crop preprocessing |
| `MainActivity.kt` | Image picker + normal map display with original/normals toggle |
| `convert_dsine.py` | Converts DSINE v02 via litert-torch with TFLite-safe op replacements |

## Model Details

- **Input**: `[1, 3, 480, 640]` float32 NCHW, ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Output**: `[1, 3, 480, 640]` float32 NCHW, unit normal vectors in [-1, 1]
- **Visualization**: `RGB = (normal + 1) / 2 * 255`

### Conversion Notes

Encoder + decoder initial prediction only — iterative refinement (ConvGRU) skipped due to TFLite-incompatible ops (boolean masking, axis-angle rotation, 7D tensors). Patches applied:

| Issue | Fix |
|-------|-----|
| GroupNorm (GPU unsupported) | Replaced with 4D-only manual mean/var computation |
| Conv2d_WS (runtime weight standardization) | Pre-computed standardized weights baked into regular Conv2d |
| F.normalize (div broadcast fail) | Manual sqrt + div implementation |
| align_corners=True (GPU incompatible) | Forced to False |
| Camera intrinsics input | Fixed 60° FOV baked as constant buffers |
| Learned convex upsampling (F.unfold + 7D) | Replaced with bilinear interpolation |

**Original project**: [baegwangbin/DSINE](https://github.com/baegwangbin/DSINE) | MIT License
