# 6DRepNet — Head pose estimation (LiteRT GPU)

Real-time **6-DoF head pose estimation** running fully on the LiteRT `CompiledModel` GPU.
[6DRepNet](https://github.com/thohemp/6DRepNet) (ICIP 2022) regresses a continuous 6D
rotation from a face crop — yaw / pitch / roll for driver-monitoring, AR, and attention.
~21 ms/frame on a Pixel 8a.

- **Model:** [thohemp/6DRepNet](https://github.com/thohemp/6DRepNet) (300W-LP) · MIT · RepVGG-B1g2
- **HF:** [litert-community/6DRepNet-HeadPose-LiteRT](https://huggingface.co/litert-community/6DRepNet-HeadPose-LiteRT)
- **Input:** `[1, 3, 224, 224]` NCHW, RGB, ImageNet-normalized (a **face crop**)
- **Output:** `[1, 6]` continuous 6D rotation → Gram-Schmidt → yaw/pitch/roll (host-side)
- **Size:** 157 MB · pure CNN

## GPU conversion

6DRepNet in **deploy** mode (RepVGG re-parameterized to plain 3×3 convs + ReLU) is a pure
CNN → fully GPU-compatible (**36/36 nodes on the delegate, 1 partition**; device corr
0.9993, ~21 ms) with **zero patches**. The 6D→rotation→Euler decode runs host-side. Use the
**deploy** weights (fused `rbr_reparam`), not the training-mode branches. CPU-exact vs
PyTorch (corr 1.0).

## Host-side decode (`HeadPoseEstimator.kt`)

Gram-Schmidt the 6D into a 3×3 rotation matrix, then read Euler angles:
`x = normalize(v[0:3]); z = normalize(cross(x, v[3:6])); y = cross(z, x)`; then
`pitch = atan2(R21,R22)`, `yaw = atan2(-R20, sqrt(R00²+R10²))`, `roll = atan2(R10,R00)`.

## Build & run

```bash
cd sixdrepnet/
./gradlew :app:installDebug
```

The 157 MB `6drepnet.tflite` is bundled in `app/src/main/assets/` (build it with
`scripts/build_6drepnet.py`; not committed). Center your face in the frame — the 3D head
pose axes are drawn (the demo uses a centered crop; add a face detector for full framing).

## Regenerate the model

```bash
pip install torch litert-torch sixdrepnet huggingface_hub
python scripts/build_6drepnet.py    # loads the MIT deploy weights (osanseviero mirror)
cp 6drepnet.tflite app/src/main/assets/
```

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
- Trained on face crops — feed a detected+cropped face for best accuracy (demo uses a centered crop).
