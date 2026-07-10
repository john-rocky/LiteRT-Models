# TwinLiteNet — Drivable-area + lane segmentation (LiteRT GPU)

Real-time **drivable-area and lane-line segmentation** running fully on the LiteRT
`CompiledModel` GPU. [TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet) (2023) is
an ultra-light ESPNet-based network with two segmentation heads — the ADAS perception
building block "where can I drive" + "where are the lanes". Only **3.1 MB**, ~44 ms/frame
on a Pixel 8a.

- **Model:** [chequanghuy/TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet) (BDD100K) · MIT · ESPNet-based
- **HF:** [litert-community/TwinLiteNet-LiteRT](https://huggingface.co/litert-community/TwinLiteNet-LiteRT)
- **Input:** `[1, 3, 360, 640]` NCHW, RGB, `x/255`
- **Outputs:** two `[1, 2, 360, 640]` logit maps — drivable_area + lane_line (argmax over 2 classes)
- **Size:** 3.1 MB · pure CNN

## GPU conversion

TwinLiteNet is a pure CNN. It converts fully GPU-compatible (**270/270 nodes on the
delegate, 1 partition**; device corr 0.99997 / 0.99998 on the two heads, ~44 ms) with
**one patch**: the `ConvTranspose2d` upsamplers → **ZeroStuffConvT2d** (nearest-upsample
+ stride zero-stuff mask + flipped conv; the Mali delegate rejects `TRANSPOSE_CONV`).
CPU-exact vs PyTorch (corr 1.0).

## Build & run

```bash
cd twinlite/
./gradlew :app:installDebug
```

The 3.1 MB `twinlite.tflite` is bundled in `app/src/main/assets/` (build it with
`scripts/build_twinlite.py`; not committed). Point the camera forward from a vehicle —
the drivable area is shaded green and lane lines red.

## Regenerate the model

```bash
pip install torch litert-torch numpy
git clone https://github.com/chequanghuy/TwinLiteNet.git
python scripts/build_twinlite.py    # uses the bundled MIT BDD100K weights (pretrained/best.pth)
cp twinlite.tflite app/src/main/assets/
```

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
- BDD100K-trained — works best on forward dashcam driving views. `argmax` each head for the mask.
