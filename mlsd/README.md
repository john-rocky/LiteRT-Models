# M-LSD-tiny — Line Segment Detection on-device (LiteRT GPU, fully GPU)

[M-LSD](https://github.com/navervision/mlsd) (NAVER, AAAI 2022) **line segment detection** — a light-weight,
real-time detector of straight line segments (building edges, document borders, wireframes, room layout). The
**tiny** variant (MobileNetV2 backbone, 0.62M params) runs **fully on the LiteRT CompiledModel GPU**. At **1.4
MB** fp16 it is the **smallest model in the zoo**.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **99 / 99** LITERT_CL (full residency) |
| inference | **~2 ms** (512×512) |
| fp16 size | **1.4 MB** |
| accuracy | device-vs-PyTorch corr **0.997**, 127 vs 128 lines decoded |

```
image[1,4,512,512] (RGB + ones channel, scaled to [-1,1]) →[GPU: MobileNetV2 U-Net]→ tpMap[1,9,256,256]
```

The output is a "TP map": channel 0 = line-center heatmap, channels 1–4 = start/end displacement. The decode
(sigmoid + 3×3 NMS over centers, displacement → endpoints, ×2) runs in the app.

## How it converts (litert-torch)

Pure CNN encoder-decoder (MobileNetV2 + bilinear-upsample decoder). A **single** re-authoring: the decoder's
`F.interpolate(..., mode='bilinear', align_corners=True)` → **`align_corners=False`** (the Mali delegate bans
`align_corners=True` + half-pixel). Everything else is GPU-clean — MobileNetV2 has no max-pool (strided convs,
so no `PADV2`), and the upsample is `RESIZE_BILINEAR`, not a transposed conv. Result: banned ops NONE, all
tensors ≤4D, tflite-vs-torch corr **1.0**, device-vs-torch corr **0.997**.

## Files

| file | what |
|---|---|
| `app/src/main/java/com/mlsd/MlsdDetector.kt` | model wrapper + TP-map decode (sigmoid + NMS + displacement → segments) |
| `app/src/main/java/com/mlsd/MainActivity.kt` | image picker + line-segment overlay |
| `scripts/build_mlsd.py` | conversion: load weights + align_corners fix + op-check + fp16 + parity |
| `scripts/device_gate_mlsd.py` | real-image torch-vs-tflite parity + line decode fixture |

## Build & run

```bash
python scripts/build_mlsd.py all       # uses the committed mlsd_tiny_512_fp32.pth; produces mlsd_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-mlsd_fp16.tflite>
```

The first launch fails with "Model not found" until the install script has been run.

**Preprocessing**: resize to 512×512, append a 4th channel of ones, scale `(x/127.5) - 1`, NCHW. Decode:
sigmoid the center map, 3×3 max NMS, threshold (0.10), displacement → endpoints, filter by length, ×2 to
512-space.

Model: `litert-community/M-LSD-tiny-LiteRT` (Hugging Face). Upstream:
[navervision/mlsd](https://github.com/navervision/mlsd) (Apache-2.0); PyTorch port
[lhwcv/mlsd_pytorch](https://github.com/lhwcv/mlsd_pytorch).
