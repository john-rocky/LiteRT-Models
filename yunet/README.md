# YuNet — On-device face detection (LiteRT GPU, fully GPU)

[YuNet](https://github.com/ShiqiYu/libfacedetection) (ShiqiYu/libfacedetection, **BSD-3-Clause**) — a tiny,
fast **face detector** (faces + 5 landmarks), running **fully on the LiteRT CompiledModel GPU**. At **0.076 M
params / 0.3 MB fp16** it is the **smallest model in this repo**.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **146 / 146** LITERT_CL (full residency) |
| inference | **~4 ms** (640×640) |
| fp16 size | **0.3 MB** |
| accuracy | device-vs-PyTorch corr **0.9999** (all 12 outputs) |

```
image[1,3,640,640] (BGR, 0-255) →[GPU: YuNet]→ 12 outputs: cls/obj/bbox/kps × strides {8,16,32}
```

## How it converts (litert-torch) — clean, no re-authoring

Pure CNN (depthwise-separable `ConvDPUnit`) + a **nearest-upsample** neck (`F.interpolate(mode="nearest")` →
`RESIZE_NEAREST_NEIGHBOR`, **no transposed conv** → no ZeroStuff). The `MaxPool2d` is non-padded → no `PADV2`.
Convert the backbone+neck+head; the head's `permute/reshape/sigmoid` per stride is baked in (output[0..2]=cls,
[3..5]=obj, [6..8]=bbox, [9..11]=kps). Result: banned ops NONE, all tensors ≤4D, tflite-vs-torch corr **1.0**,
device-vs-torch corr **0.9999**.

## Decode (host-side)

Anchor-free, priors at each stride with offset 0 (`px = col·s, py = row·s`):
- **score** = `cls · obj` (both sigmoid-baked).
- **box**: `cx = bbox₀·s + px`, `cy = bbox₁·s + py`, `w = exp(bbox₂)·s`, `h = exp(bbox₃)·s` → (x1,y1,x2,y2).
- **5 landmarks**: `kx = kpsₓ·s + px`, `ky = kpsᵧ·s + py`.
- then **NMS** (IoU 0.45). See `FaceDetector.kt`.

## Build & run

```bash
# weights weights/yunet_n.pth ship in the libfacedetection.train repo (BSD-3)
python scripts/build_yunet.py all      # produces yunet_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-yunet_fp16.tflite>
```

The first launch fails with "Model not found" until the model is pushed.

**Preprocessing**: resize to 640×640, **BGR**, **0-255 (no normalization)**, NCHW.

Model: `litert-community/YuNet-Face-LiteRT` (Hugging Face). Upstream:
[ShiqiYu/libfacedetection](https://github.com/ShiqiYu/libfacedetection) (BSD-3-Clause).
