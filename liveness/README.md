# Silent-Face Anti-Spoofing (MiniFASNetV2) — Face liveness (LiteRT GPU)

Real-time **face liveness / anti-spoofing** running fully on the LiteRT `CompiledModel`
GPU. [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
detects **presentation attacks** — a printed photo or a replayed screen — so a live face
passes and a fake is rejected. The anti-fraud building block for face login / e-KYC. Tiny
(**1.85 MB**), ~5 ms/frame on a Pixel 8a.

- **Model:** [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) (`2.7_80x80_MiniFASNetV2`) · Apache-2.0
- **HF:** [litert-community/Silent-Face-Anti-Spoofing-LiteRT](https://huggingface.co/litert-community/Silent-Face-Anti-Spoofing-LiteRT)
- **Input:** `[1, 3, 80, 80]` NCHW, **BGR**, `x/255` (a face crop)
- **Output:** `[1, 3]` softmax — class 1 = live, 0 & 2 = spoof (print / replay). Live = `output[1]`.
- **Size:** 1.85 MB · pure CNN

## GPU conversion

MiniFASNetV2 is a pure CNN → fully GPU-compatible (**168/168 nodes on the delegate, 1
partition**; device corr 1.0, ~5 ms) with **zero patches** (PReLU lowers to GPU-clean
relu ops). CPU-exact vs PyTorch (corr 1.0).

## Build & run

```bash
cd liveness/
./gradlew :app:installDebug
```

The 1.85 MB `silentface.tflite` is bundled in `app/src/main/assets/` (build it with
`scripts/build_silentface.py`; not committed). Center your face in the box — a live face
shows **LIVE**; a photo/screen shown to the camera shows **SPOOF**.

## Regenerate the model

```bash
pip install torch litert-torch numpy
git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git silentface-src
python scripts/build_silentface.py    # uses the bundled Apache-2.0 MiniFASNetV2 weights
cp silentface.tflite app/src/main/assets/
```

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
- Feed a detected face crop (~2.7× the face box). The full repo ensembles a 2nd MiniFASNet (scale 4.0).
