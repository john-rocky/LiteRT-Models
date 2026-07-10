# XFeat — On-device image matching / local features (LiteRT GPU, fully-GPU)

[XFeat](https://github.com/verlab/accelerated_features) (CVPR 2024, Apache-2.0, 1.5 M params)
**local feature extraction + matching** running **fully on the LiteRT CompiledModel GPU**: pick two
photos of the same scene and see the matched keypoints — the building block for AR, panorama
stitching, SLAM and image registration.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **72 / 72** LITERT_CL (full residency, 1 partition) |
| inference | **~0.4 ms** per 640×480 image |
| fp16 size | 1.4 MB |
| accuracy | device-vs-PyTorch corr **0.9999** |

```
gray[1,1,480,640] (host instance-norm) →[GPU: XFeat CNN]→ feats[1,64,60,80] +
keypoints[1,65,60,80] + heatmap[1,1,60,80] →[host: cell-softmax decode + 5×5 NMS +
bilinear descriptor sampling + mutual-nearest-neighbor]→ matches
```

## How it converts (litert-torch) — numerically-equivalent

- Input grayscale + **InstanceNorm moved host-side** (its H·W spatial reduction would overflow
  fp16 on the delegate).
- `_unfold2d(x, 8)` (space-to-depth via unfold → >4-D / GATHER_ND) → a **one-hot
  `Conv2d(1,64,k=8,s=8)`** (exact, single CONV_2D).
- Result: zero GATHER/SELECT/TOPK/CAST, all tensors ≤4-D → full GPU residency.

## Build & run

```bash
python scripts/convert_xfeat.py            # -> xfeat_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-xfeat_fp16.tflite>
```

Pick two photos of the same scene (different angles) → side-by-side match visualization
(green = confident). Keypoint decode (8×8-cell logits + dustbin, reliability-weighted), NMS,
descriptor sampling and MNN matching run in Kotlin.

Model: `litert-community/xfeat-litert` (Hugging Face). Upstream:
[verlab/accelerated_features](https://github.com/verlab/accelerated_features) (Apache-2.0).
