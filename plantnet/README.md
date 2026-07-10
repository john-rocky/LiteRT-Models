# PlantNet-300K — Fine-grained plant species identification (LiteRT GPU)

Identify **1081 plant species** from a photo, fully on the LiteRT `CompiledModel`
GPU. A [PlantNet-300K](https://github.com/plantnet/PlantNet-300K) (NeurIPS 2021)
ResNet18 — the first fine-grained classifier in this zoo. ~16 ms/frame on a Pixel 8a.

- **Model:** [cpoisson/plantnet300k-resnet18](https://huggingface.co/cpoisson/plantnet300k-resnet18) · Apache-2.0
- **Input:** `[1, 3, 224, 224]` NCHW, RGB, ImageNet-normalized
- **Output:** `[1, 1081]` species logits (Latin names)
- **Size:** 47 MB · torchvision ResNet18 · pure CNN

## GPU conversion

Plain torchvision ResNet18 — a pure CNN that converts to a fully GPU-compatible
graph (**37/37 nodes on the delegate, 1 partition**; device corr 0.99999, top-1
match) with **one patch**: the ResNet stem `MaxPool2d(padding=1)` lowers to a PADV2
with `-inf` padding (`PADV2: src has wrong size` on the Mali delegate), replaced by
an explicit 0-pad + unpadded maxpool — exact, since the maxpool input is post-ReLU
(≥ 0). CPU-exact vs PyTorch (corr 0.99999999999).

**Labels**: class index `i` (0–1080) maps to the `i`-th species when the PlantNet-300K
species-id strings are sorted (torchvision `ImageFolder` order); names from
`plantnet300K_species_id_2_name.json`. Bundled as `PlantNetLabels.kt`.

## Build & run

```bash
cd plantnet/
./gradlew :app:installDebug
```

The 47 MB `plantnet.tflite` is bundled in `app/src/main/assets/` (build it with
`scripts/build_plantnet.py`; the file is not committed). Point the camera at a plant
/ flower — the top-5 species (Latin names) with confidences are shown.

## Regenerate the model

```bash
pip install torch torchvision litert-torch huggingface_hub
python scripts/build_plantnet.py
cp plantnet.tflite app/src/main/assets/
```

`scripts/build_plantnet.py` loads the Apache-2.0 ResNet18 weights from
`cpoisson/plantnet300k-resnet18`, applies the ZeroPadMaxPool patch, and converts.

## Notes

- `minSdk 26`, `arm64-v8a`, LiteRT `com.google.ai.edge.litert:litert:2.1.3`.
