# DewarpNet — Document unwarping / rectification (LiteRT GPU)

Real-time **document dewarping** running fully on the LiteRT `CompiledModel` GPU.
[DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) (ICCV 2019) flattens a
photographed, curved/folded document — the core of a document scanner. Two CNNs predict
a backward-mapping grid; the network runs on the GPU and the `grid_sample` unwarp is a
tiny host-side step. First document-processing model in the zoo. ~24 ms/frame on a Pixel 8a.

- **Model:** [cvlab-stonybrook/DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet) (doc3d) · MIT
- **HF:** [litert-community/DewarpNet-LiteRT](https://huggingface.co/litert-community/DewarpNet-LiteRT)
- **Input:** `[1, 3, 256, 256]` NCHW, **BGR**, `x/255`
- **Output:** `[1, 2, 128, 128]` backward-mapping grid (~`[-1,1]`)
- **Size:** 189 MB · pure CNN (WCNet UNet + BMNet DenseNet)

## GPU conversion

DewarpNet is a pure CNN. It converts fully GPU-compatible (**371/371 nodes on the
delegate, 1 partition**; device corr 0.999866, ~24 ms) with **two exact patches**:
(1) the UNet/DenseNet `ConvTranspose2d` → **ZeroStuffConvT2d** (nearest-upsample +
stride zero-stuff mask + flipped conv; Mali rejects `TRANSPOSE_CONV`), and (2)
`Hardtanh(0,1)` → `relu(x) - relu(x-1)` (Mali rejects `RELU_0_TO_1`). CPU-exact vs
PyTorch (corr 0.9999999999).

## Host-side unwarp (`DocumentDewarper.kt`)

The model outputs a backward-mapping grid. For each output pixel: bilinearly read the
map (over the 128×128 grid), convert the `[-1,1]` coord to a source pixel, and
bilinearly sample the source — i.e. a `grid_sample` implemented in Kotlin.

## Build & run

```bash
cd dewarp/
./gradlew :app:installDebug
```

The 189 MB `dewarp.tflite` is bundled in `app/src/main/assets/` (build it with
`scripts/build_dewarp.py`; not committed). Point the camera at a curved/folded document
— the flattened, rectified page fills the screen (live preview in the corner).

## Regenerate the model

```bash
pip install torch litert-torch gdown numpy
git clone https://github.com/cvlab-stonybrook/DewarpNet.git dewarp-src
python scripts/build_dewarp.py    # fetches the MIT doc3d weights (WCNet + BMNet)
cp dewarp.tflite app/src/main/assets/
```

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
- Input channel order is **BGR** (`x/255`, no mean/std). Output is a backward map — apply grid_sample host-side.
