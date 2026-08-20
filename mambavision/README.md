# MambaVision-T — on-device GPU (LiteRT CompiledModel)

The first **Mamba / state-space vision model running fully on the mobile GPU** via LiteRT
`CompiledModel` (ML Drift). MambaVision-T (NVIDIA) ImageNet classifier: pick any image from the photo
library, run on-device GPU inference, and see the top-5 predictions.

- Input `[1,3,224,224]` NCHW, ImageNet-normalized → output `[1,1000]` logits → softmax top-5.
- FP16, ~68 MB, `arm64-v8a`. Built once on first use and reused.

## Why this is interesting

A Mamba **selective scan** (`h_t = ΔA_t·h_{t-1} + ΔB_t·u_t`) has no compact, GPU-clean lowering out of
the box: the manual per-step recurrence unrolls to *O(L)* ops, and `torch.associative_scan` emits
`GATHER_ND` (GPU-rejected). The fix used here expresses the scan as a **Hillis-Steele parallel scan with
contiguous shift-slices** (`cat[identity, x[:, :L-d]]`, `log2(L)` levels), which lowers to `SLICE` only —
**GPU-clean (no `GATHER_ND`), compact, and numerically exact** (corr 1.0 vs the reference recurrence).

The full model is made GPU-compatible with four patches (see `scripts/make_mambavision_gpu.py`):
1. selective scan → Hillis-Steele parallel scan (the key one above);
2. attention `qkv` 5D + SDPA → 3 Linears + 4D manual attention;
3. `softplus` (`where(x>20,…)` → `SELECT`) → `relu(x)+log1p(exp(-|x|))` (no `SELECT`);
4. `window_partition`/`reverse` 6D view → flatten (windows are full-resolution = single window).

Result: `GPU_BAD=NONE, >4D=0, GATHER_ND=0`; FP32→tflite fidelity corr **1.00000**; FP16 68 MB.

## Model files (not committed)

Per repo policy the `.tflite` is never committed. Generate it and place it in `app/src/main/assets/`:

```bash
cd scripts
python make_mambavision_cls.py            # -> mambavision_t_cls_gpu_fp16.tflite + imagenet_labels.txt
cp mambavision_t_cls_gpu_fp16.tflite imagenet_labels.txt ../app/src/main/assets/
```

(`make_mambavision_gpu.py` produces the feature-backbone variant; `make_mambavision_cls.py` the 1000-class
classifier used by this app.) Requires a `torch ≥ 2.11` env with litert-torch; `mamba_ssm`/`causal_conv1d`
are CUDA-only and are stubbed (the scan is pure-torch), so no GPU is needed to convert.

## Build & run

```bash
./gradlew :app:installDebug
```

Tap **Select image**, pick a photo; the status bar shows the accelerator (GPU/CPU) and latency.

### Converting your own fine-tuned checkpoint

The GPU re-authoring is architecture-level, so a MambaVision-T fine-tune saved with
`save_pretrained()` (HF `AutoModelForImageClassification`, `trust_remote_code=True`)
converts the same way (defaults reproduce the official ship exactly):

```bash
MAMBAVISION_MODEL_ID=/path/to/your_saved_model python scripts/make_mambavision_cls.py
```

Class count and names flow from the model's `config.json` (`id2label`) into the logits
width — update the app's label table to match. T-size only; other sizes have not been
device-verified.
