# 3DDFA_V2 — On-device 3D face alignment (LiteRT GPU)

[3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) (Guo et al., ECCV 2020, MIT) fits a **3D morphable
face model** to a photo: a MobileNetV1 regresses **62 3DMM parameters** (pose + 40 shape + 10
expression), from which the 68 3D face landmarks (and a dense mesh) are reconstructed. The regressor
runs fully on the LiteRT CompiledModel **GPU**; the BFM reconstruction is a small host-side matmul.

## On-device (Pixel 8a, Tensor G3 — verified)

| stage | in → out | where |
|---|---|---|
| face box | photo → box | android.media.FaceDetector (frontal) |
| 3DMM regressor | crop [1,3,120,120] → 62 params | **GPU** 6.3 MB fp16 |
| BFM reconstruction | 62 params → 68 3D landmarks | host (u + w_shp·α_shp + w_exp·α_exp, posed by R·v + offset) |

fp16 tflite vs PyTorch: 62-param corr **0.999999**, and the reconstructed landmarks match to
**0.02 px** (the fp16 error concentrates on the depth-scale parameter, which does not move the 2D
landmarks). ~68 landmarks in well under a second.

```
photo →[FaceDetector box → parse_roi → crop 120² (BGR, (x−127.5)/128)]→ [GPU MobileNetV1]→ 62 params
      → denorm → R,offset,α_shp,α_exp → BFM 68 verts → R·v+offset → similar_transform → 68 (x,y)
```

## How it converts

The regressor is a plain MobileNetV1 → 62 outputs, so it converts through litert-torch with no
re-authoring and every op rides the GPU delegate. Three details matter for the host code: the model
was trained on **cv2 BGR** input; the BFM bases are **interleaved** `[x0,y0,z0,x1,…]` (the reference
reconstructs with `reshape(3,-1, order='F')`); and `android.media.FaceDetector` needs an **even
width**. The BFM 68-keypoint bases + parameter mean/std are tiny (~42 KB, bundled).

## Build & run

```bash
python build_tddfa.py           # -> tddfa_mb1_fp16.tflite + recon-asset bins (needs ~/clipconv, litert-torch)
# put the .tflite + tddfa_*.bin in app/src/main/assets/  (bundled in the APK; also on Hugging Face)
./gradlew :app:installDebug
```

Launch the app — it detects the bundled sample's face and draws the 68 landmarks; tap **Pick face
photo** for your own frontal-face photos.

Model on Hugging Face: `litert-community/3DDFA-V2-LiteRT`. Upstream:
[cleardusk/3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) (MIT).
