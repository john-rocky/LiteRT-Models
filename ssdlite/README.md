# SSDLite320 MobileNetV3 Object Detection

Real-time object detection using **SSDLite320-MobileNetV3-Large** (torchvision, BSD-3)
on LiteRT CompiledModel GPU.

A lightweight (**0.59 GMACs**) detector that converts **patch-free** through
litert-torch — the cleanest path for an official LiteRT sample — and fills the
GPU detector slot with a model fast enough for live camera use.

## Features

- **~30 FPS** live camera detection on Pixel 8a (Tensor G3) — CompiledModel GPU / ML Drift, all 286 graph nodes on OpenCL (no CPU fallback)
- 90 COCO class labels with colored bounding boxes
- Anchor (default-box) decode + multiclass NMS in Kotlin
- BSD-3 model + weights (auto-downloaded by torchvision, no Google Drive)

## Setup

1. Get the model **`ssdlite_mobilenetv3_320_fp16.tflite`** (7.2 MB, FP16) — either:
   - download from Hugging Face: **[mlboydaisuke/ssdlite320-mobilenetv3-litert](https://huggingface.co/mlboydaisuke/ssdlite320-mobilenetv3-litert)** (`ssdlite_mobilenetv3_320_fp16.tflite`), or
   - reproduce it with `scripts/convert_ssdlite.py` (litert-torch).
2. Place it in `app/src/main/assets/`
3. Open this directory in Android Studio and run (back camera + grant permission)

First launch compiles GPU shaders (~15 s, shown as "Compiling GPU model…"); subsequent frames run at full speed.

## Architecture

| File | Description |
|------|-------------|
| `SSDLiteDetector.kt` | CompiledModel GPU inference: normalize+resize, default-box build, BoxCoder decode, per-class NMS |
| `RealtimeCameraPipeline.kt` | Reusable two-thread camera pipeline (preview + analysis share 4:3) |
| `DetectionCanvasView.kt` | Bounding-box overlay matching PreviewView's FILL_CENTER scaling |
| `MainActivity.kt` | CameraX preview + async model load |
| `CocoLabels.kt` | 91-entry torchvision COCO label map (index 0 = background) and colors |
| `Detection.kt` | Detection data class |

## Model Details

- **Input** `[1, 3, 320, 320]` **NCHW, RGB**, normalized `pixel/127.5 - 1` → `[-1, 1]`
  (torchvision `GeneralizedRCNNTransform`, mean = std = 0.5), bilinear stretch-resize (not letterbox).
- **Output** 12 raw head tensors — `(cls, box)` per 6 feature levels (`H = W = 20, 10, 5, 3, 2, 1`):
  `cls[i] = [1, 6·91, H, W]`, `box[i] = [1, 6·4, H, W]` (NCHW). 6 anchors/location, 91 classes.
- **Decode in Kotlin** mirrors torchvision `SSD.postprocess_detections` + `BoxCoder(10,10,5,5)`:
  the 3234 default boxes are rebuilt from the `DefaultBoxGenerator` formula (scales
  `0.2…0.95`, aspect ratios `{1, 2, 3, 1/2, 1/3}`); per anchor, softmax over 91 → best
  non-background class → threshold → decode `(dx,dy,dw,dh)` against its default box → per-class NMS.

## Why the raw-head export (4D-head-tap)

SSD's built-in postprocess (`DefaultBoxGenerator` + box decode + NMS) lowers to
`GATHER_ND` / `TOPK` / `>4D` tensors that the GPU delegate rejects. Instead, the
export taps each FPN level's **final head conv outputs** (4D, NCHW) and moves
decode + NMS to Kotlin — the same "choose the output point" technique as the
YOLOX raw-head and U²-Net `d0` samples. This rewrites **no model-internal op**:
keeping **NCHW** I/O (no `to_channel_last_io`) avoids the channel-last × MobileNetV3
SqueezeExcitation `GATHER_ND` blow-up, so the model converts stock-clean.

Verified GPU-clean: **BANNED NONE, Flex/Custom NONE, max tensor ndim 4, 0 dynamic dims**
(ops include `SUM×8` = the SE global pools, `TRANSPOSE×11`). FP16 is bit-faithful to
PyTorch (per-output corr ≥ 0.99999); end-to-end decode matches stock torchvision
**298/300 boxes @ IoU 0.99** on the FP16 tflite.

**On-device verified (Pixel 8a, Tensor G3 / Mali):** CompiledModel GPU compiles the whole graph —
`Replacing 286 out of 286 node(s) with delegate (LITERT_CL)`, 1 partition, no CPU fallback — and
runs the live camera at **~30 FPS** with correct detections (e.g. person @ 84%). So the `SUM`/`TRANSPOSE`
ops that pass the desktop checker also pass Mali ML Drift on device.

## Conversion & validation

- `scripts/convert_ssdlite.py` — exports the raw 4D heads via litert-torch, op-checks,
  parity vs PyTorch, and emits the FP16 (`float_casting`) model. No model patches.
- `scripts/validate_decode.py` — proves the Kotlin decode against stock torchvision:
  rebuilds anchors, runs the FP16 tflite, decodes, and box-matches the references.
