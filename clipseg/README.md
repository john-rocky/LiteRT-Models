# CLIPSeg — On-device text-prompted segmentation (LiteRT GPU vision + CPU decoder)

[CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined) (CVPR 2022, Apache-2.0) **open-vocabulary
segmentation**: type what you want to segment ("a cat", "the sky", "a red car") and get a mask — no
fixed class list. The CLIP text + vision encoders run on the LiteRT CompiledModel **GPU**; the tiny
decoder runs on **CPU** (its small-head attention fp16-miscomputes on the Mali delegate — see below).

## On-device (Pixel 8a, Tensor G3 — verified)

| stage | in → out | delegate | latency |
|---|---|---|---|
| text encoder | token-emb [1,77,512] → hidden [1,77,512] | **GPU** 761/761 | ~8.7 ms |
| vision encoder | image [1,3,352,352] → t3,t6,t9 [1,485,768] | **GPU** 613/613 | ~8.2 ms |
| decoder | t3,t6,t9,cond → logits [1,352,352] | **CPU** (exact) | ~few ms |

End-to-end device-vs-PyTorch logits corr **0.99998**, mask IoU **0.9986**.

```
prompt →[BPE + emb lookup]→ [GPU text enc]→ EOT row @ text_proj → cond[512]
image  →[CLIP norm]→ [GPU vision enc]→ t3,t6,t9 →[CPU decoder + FiLM(cond)]→ logits → sigmoid mask
```

## How it converts (litert-torch) — numerically-equivalent

- CLIP ViT-B/16 vision @352 (484+1 tokens): qkv-decompose to 3D-BMM attention, quick-GELU
  (`x·σ(1.702x)`), pos-embed baked interpolated 14²→22², SafeLayerNorm. **484-token global attention
  survives fp16 on Mali at corr 0.998** (contrast: DA-V2 784 tokens walls) — a useful data point on
  the CLIP-B/32 (49✓) → DA-V2 (784✗) boundary.
- Text encoder: token embedding lookup host-side (avoids GATHER), causal mask baked, **⭐`safe_ln_up`**
  (up-scale small-variance inputs so the LN eps lands in the fp16-normal range — exact to 1e-6).
- **Decoder → CPU**: bisection pinned a device miscompute to the decoder's first attention layer
  (post-FiLM corr 0.999 → dec-layer-0 0.864). The decoder uses **4 heads × 485 tokens × head_dim 16**;
  the small head dim is the fp16-fragile axis (the 12-head/head_dim-64 vision encoder survives). The
  decoder is only 3 layers of 64-dim, so running it on CPU is exact and fast — the same fp16-fragile-
  part-on-CPU split as the diarization sample. `convT4x4` re-authors the two non-overlapping
  ConvTranspose upsamples as an exact 1×1-conv + 4-D interleave (no ZeroStuff).

## Build & run

```bash
python scripts/build_clipseg.py       # -> clipseg_{vision,text}_fp16.tflite + clipseg_decoder.tflite + host assets
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-the-artifacts>
```

Pick an image, type a prompt, tap **Segment** → red mask overlay. Re-prompting the same image only
re-runs the text + decoder graphs (vision features are cached).

Model: `litert-community/CLIPSeg-rd64-LiteRT` (Hugging Face). Upstream:
[CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined) (Apache-2.0).
