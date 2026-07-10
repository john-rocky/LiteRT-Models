# YOLACT-ResNet50 — Real-time instance segmentation (LiteRT GPU)

Per-object **instance segmentation** (COCO 80 classes) running fully on the LiteRT
`CompiledModel` GPU. [YOLACT](https://arxiv.org/abs/1904.02689) (ICCV 2019): the
network (ResNet50 + FPN + protonet + heads) runs on the GPU; the lightweight decode
(NMS + linear-combination masks) runs host-side. The first instance-segmentation
model in this zoo. ~41 ms/graph on a Pixel 8a.

- **Model:** [dbolya/yolact](https://github.com/dbolya/yolact) (`yolact_resnet50`) · MIT
- **HF:** [litert-community/YOLACT-ResNet50-LiteRT](https://huggingface.co/litert-community/YOLACT-ResNet50-LiteRT) (`yolact.tflite` + `priors.bin`)
- **Input:** `[1, 3, 550, 550]` NCHW, **BGR**, `(x - [103.94,116.78,123.68]) / [57.38,57.12,58.40]` (no /255)
- **Raw outputs:** `loc [1,19248,4]`, `conf [1,19248,81]`, `mask [1,19248,32]`, `proto [1,138,138,32]`

## GPU conversion

Base YOLACT (no deformable conv) is a pure CNN, so the graph converts fully
GPU-compatible (**138/138 nodes on the delegate, 1 partition**; device corr
0.99999–1.0 on all four raw outputs) with **one patch**: the ResNet50 stem
`MaxPool2d(padding=1)` lowers to a `-inf` PADV2 (rejected by Mali), replaced by a
0-pad + unpadded maxpool (exact post-ReLU). The scripted FPN is made traceable by
disabling YOLACT's JIT (`use_jit=False`). The 3D `[1,19248,C]` heads survive the
Mali delegate. CPU-exact vs PyTorch (corr 1.0).

## Host-side decode (`YolactSegmenter.kt`)

1. **Boxes:** SSD `decode(loc, priors, variances=[0.1,0.2])` against the baked
   `priors.bin` (19248 anchors).
2. **NMS:** per-class greedy NMS (score-threshold 0.3, IoU 0.5, top-k 40).
3. **Masks (lincomb):** per kept detection, `sigmoid(proto @ coeff)` (`> 0`) cropped
   to the box → colored instance overlay.

## Build & run

```bash
cd yolact/
./gradlew :app:installDebug
```

The 125 MB `yolact.tflite` + `priors.bin` are bundled in `app/src/main/assets/`
(build with `scripts/build_yolact.py` / download from HF; not committed). Point the
camera at people/objects — colored instance masks + boxes + COCO labels are overlaid.

## Regenerate the model

```bash
pip install torch litert-torch huggingface_hub pycocotools
git clone https://github.com/dbolya/yolact.git
python scripts/build_yolact.py    # loads dbolya/yolact-resnet50 weights from HF
```

`scripts/build_yolact.py` disables YOLACT's CUDA/JIT assumptions, bypasses NMS, applies
ZeroPadMaxPool, and exports the raw-head graph with litert-torch.

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
- Base YOLACT only — **not** YOLACT++ (DCNv2 deformable conv = GATHER_ND, GPU-incompatible).
