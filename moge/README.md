# MoGe-2 Monocular Geometry Estimation

Monocular 3D geometry estimation using MoGe-2 (CVPR'25 Oral) with LiteRT CompiledModel GPU.

## Features

- **Affine 3D point map, surface normals, depth, and metric scale** from a single image
- DINOv2 ViT-S encoder + ConvStack multi-scale decoder (35M params)
- Four visualization modes: Normal map, Depth heatmap, 3D point cloud (rotatable), Info
- ~522 ms inference on Pixel 8a GPU

## Setup

1. Download model from [Releases](https://github.com/john-rocky/LiteRT-Models/releases/tag/v3):
   - `moge.tflite` (136 MB)
2. Push to device (too large for APK assets):
   ```bash
   adb push moge.tflite /data/local/tmp/
   adb shell "run-as com.moge cp /data/local/tmp/moge.tflite /data/data/com.moge/files/"
   ```
3. Open this directory in Android Studio and run

## Architecture

| File | Description |
|------|-------------|
| `MoGePredictor.kt` | CompiledModel GPU inference with letterbox preprocessing |
| `PointCloudRenderer.kt` | OpenGL ES 2.0 point cloud viewer with touch rotation |
| `MainActivity.kt` | Image picker + 4-mode visualization (Normal / Depth / 3D / Info) |
| `scripts/convert_moge.py` | Converts MoGe-2 ViT-S via litert-torch with 9 GPU-compat patches |

## Model Details

- **Input**: `[1, 3, 448, 448]` float32 NCHW, RGB normalized to [0, 1] (DINOv2 applies ImageNet stats internally)
- **Outputs**:
  - `points [1, 448, 448, 3]` — affine point map (exp remap: `[xy*exp(z), exp(z)]`)
  - `normal [1, 448, 448, 3]` — L2-normalized surface normals
  - `mask [1, 448, 448, 1]` — sigmoid confidence (>0.5 = valid)
  - `scale [1, 1, 1, 1]` — metric scale factor

### Conversion Notes

Single combined TFLite with all 835 ops running on GPU. Nine patches applied to make DINOv2 + ConvStack fully GPU-compatible:

| Issue | Fix |
|-------|-----|
| LayerScale MUL causes FC 2D↔3D shape conflict | Bake gamma into preceding Linear weight/bias |
| Fused qkv Linear → 5D reshape + GATHER_ND | Decompose into 3 separate q/k/v Linear |
| Multi-layer feature stack → 5D | Element-wise add instead of torch.stack+sum |
| Position embedding bicubic interpolation → GATHER_ND | Pre-compute for fixed 32×32 patch grid |
| ConvTranspose2d (GPU unsupported on device) | Bilinear 2x upsample + Conv2d(1x1) with mean-weight transfer |
| Constant UV buffer → Conv2d rejects constant input | Add negligible image-dependent epsilon |
| nn.Upsample(scale_factor) → dynamic RESIZE | Replace with fixed-size F.interpolate |
| Conv2d padding_mode='replicate' | Switch to 'zeros' |
| F.interpolate bicubic (GPU unsupported) | Force bilinear |

**Original project**: [microsoft/MoGe](https://github.com/microsoft/MoGe) | MIT License (DINOv2: Apache-2.0)
