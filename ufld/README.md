# Ultra-Fast-Lane-Detection (ResNet18, CULane) — LiteRT GPU

Real-time **lane detection** running fully on the LiteRT `CompiledModel` GPU.
[Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection) (ECCV
2020) reformulates lane detection as fast **row-wise classification** — the network
runs on the GPU, and a tiny host-side arg/expectation decode turns the grid into lane
points. First lane-detection model in the zoo. ~20 ms/frame on a Pixel 8a.

- **Model:** [cfzd/Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection) (CULane, ResNet18) · MIT
- **HF:** [litert-community/Ultra-Fast-Lane-Detection-LiteRT](https://huggingface.co/litert-community/Ultra-Fast-Lane-Detection-LiteRT)
- **Input:** `[1, 3, 288, 800]` NCHW, RGB, `x/255` then ImageNet-normalized
- **Output:** `[1, 201, 18, 4]` = `(griding+1, row_anchors, lanes)` row-wise class logits
- **Size:** 178 MB · pure CNN

## GPU conversion

UFLD is a pure CNN (ResNet18 + row-classification head). It converts fully
GPU-compatible (**41/41 nodes on the delegate, 1 partition**; device corr 0.999982,
~20 ms) with **one patch**: the ResNet18 stem `MaxPool2d(padding=1)` lowers to a `-inf`
PADV2 (rejected by Mali), replaced by a 0-pad + unpadded maxpool (exact post-ReLU).
CPU-exact vs PyTorch (corr 0.9999999999996).

## Host-side decode (`LaneDetector.kt`)

For each of the 4 lanes and 18 row anchors: softmax over the 200 grid cells, take the
expectation → column; drop it if the argmax over all 201 is the last index (200 = "no
lane"). Map the column to an x-pixel via `linspace(0, 799, 200)` and the row anchor to
a y-pixel (CULane row anchors, from 288).

## Build & run

```bash
cd ufld/
./gradlew :app:installDebug
```

The 178 MB `ufld.tflite` is bundled in `app/src/main/assets/` (build it with
`scripts/build_ufld.py`; not committed). Point the camera forward from a vehicle —
detected lane points are overlaid, colored per lane.

## Regenerate the model

```bash
pip install torch litert-torch gdown
git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection.git
python scripts/build_ufld.py    # fetches the MIT CULane ResNet18 weights
cp ufld.tflite app/src/main/assets/
```

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
- CULane-trained — works best on forward dashcam highway views (the training distribution).
