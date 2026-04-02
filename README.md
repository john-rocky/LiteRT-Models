# LiteRT-Models

Converted TFLite Model Zoo for Android GPU inference with [LiteRT](https://ai.google.dev/edge/litert) (CompiledModel / ML Drift).

## Models

### Depth Estimation

| Model | Input | Speed (Pixel 8a) | Quality | Size | Download |
|-------|-------|-------------------|---------|------|----------|
| **DepthAnything V2 Small** | 518x518 | **9ms** | corr=0.9995 | 99 MB | [Release](../../releases) |
| DepthAnything V2 Small | 392x518 | ~7ms | corr=0.9999 | 98 MB | [Release](../../releases) |

- Native Keras NHWC conversion (not onnx2tf) — preserves PyTorch quality
- LiteRT 2.1.3 CompiledModel API with ML Drift GPU (FP32 precision)
- Tested on Pixel 8a (Tensor G3)

## Sample App

[**sample-app/**](sample-app/) — Real-time camera depth estimation with CameraX + ML Drift GPU.

### Quick Start

1. Download model from [Releases](../../releases)
2. Place `depth_anything_v2_keras.tflite` in `sample-app/app/src/main/assets/`
3. Open `sample-app/` in Android Studio and run

## Conversion Scripts

[**scripts/**](scripts/) — Python scripts for model conversion.

| Script | Description |
|--------|-------------|
| `convert_keras_native.py` | Native Keras DepthAnything V2 → TFLite (recommended) |
| `compare_quality.py` | Quality comparison vs PyTorch ground truth |
| `convert_nhwc_gpu.py` | onnx2tf conversion (legacy) |
| `convert_v129.py` | onnx2tf 1.29 variant testing |

### Convert a model

```bash
cd scripts
pip install torch transformers tensorflow tf_keras numpy
python convert_keras_native.py --output_dir ../sample-app/app/src/main/assets/

# For 392x518 (iOS-equivalent resolution):
python convert_keras_native.py --output_dir ../sample-app/app/src/main/assets/ \
    --input_height 392 --input_width 518
```

## Technical Details

See [INVESTIGATION.md](INVESTIGATION.md) for:
- Conversion pipeline comparison (9 approaches tested)
- Native Keras reimplementation know-how
- TFLite GPU compatibility guide
- Weight transposition rules
- FP16 overflow analysis

## License

MIT
