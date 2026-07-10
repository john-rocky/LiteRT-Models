# Parakeet ASR (FastConformer-CTC) — on-device, fully GPU

NVIDIA **parakeet-tdt_ctc-110m** (the CTC branch) running speech-to-text **entirely on the LiteRT
`CompiledModel` GPU (ML Drift / LITERT_CL)** on a Pixel 8a. The 17-layer FastConformer encoder + CTC head
run as a single GPU graph; only the log-mel front-end and the greedy-CTC + SentencePiece decode are on the
host.

Device-verified (LibriSpeech sample, 7.4 s): the on-device transcript matches PyTorch exactly, real-frame
logits corr **0.99997**, **3105/3105 ops on LITERT_CL** (1 partition), ~**330 ms GPU + ~70 ms host mel ≈
0.4 s end-to-end** per 16 s window (device-app measured).

| | |
|---|---|
| Model | `nvidia/parakeet-tdt_ctc-110m` (CTC head only; the RNN-T branch is skipped) |
| License | CC-BY-4.0 (attribute NVIDIA) |
| Input | log-mel `[1,80,1601]` (~16 s window, zero-padded) + frame mask `[1,201]` |
| Output | CTC logits `[1,201,1025]` (1024 SentencePiece pieces + blank) |
| Precision | fp16, ~226 MB |
| applicationId | `com.parakeet` |

## How it runs

```
mic / wav ─▶ host log-mel ─▶ [GPU] FastConformer encoder + CTC Conv1d ─▶ host greedy-CTC + SentencePiece
```

- **Fixed window + masking.** `CompiledModel` needs static shapes, so audio (≤16 s) is padded to a
  1601-frame mel window. The encoder's length masking is folded into the graph as a GPU-clean **additive
  attention bias** (`scores += (1-mask)·-3e4`) plus a **conv frame-mask**, so padded frames never leak into
  real ones. The host decodes only the real frames.
- **Host log-mel** matches NeMo's `AudioToMelSpectrogramPreprocessor`: preemphasis 0.97, center zero-pad,
  hann(400) in a 512-pt FFT, `|·|²`, slaney mel filterbank (`assets/mel_fb.bin`, the model's own buffer),
  `log(x + 2⁻²⁴)`, per-feature normalization. See `MelSpectrogram.kt`.
- **Decode** is greedy CTC (argmax per frame, drop blank=1024 and repeats) + manual SentencePiece
  detokenize from `assets/tokens.txt` (concatenate pieces, `▁`→space).

## GPU compatibility notes (conversion)

Converted with **litert-torch** (NCHW preserved). The re-authoring that makes the FastConformer GPU-clean:

- Subsampling run static & mask-free; encoder `_create_masks` → none (full attention, masking re-added as
  the additive bias above for the shippable window).
- `RelPositionMultiHeadAttention` re-authored as manual ≤4D matmuls (q±pos_bias_u/v, `matrix_ac`,
  `rel_shift(matrix_bd)`, softmax·v) — no SDPA, no cache.
- GLU → `a·sigmoid(b)` via slicing (SPLIT is banned); BatchNorm folds; CausalConv1d uses symmetric
  zero-padding; CTC `ConvASRDecoder` (Conv1d 512→1025) fused into the graph.
- **fp16-safe LayerNorm.** The subsampling front-end emits very large pre-norm activations (|x|≈7000); the
  usual `var = mean(d²)·S²` rescaling overflows fp16 on Mali (S²≈8e5, var≈2.5e7 > 65504) → the norm
  collapses and the transcript goes blank. The fix keeps the whole reduction in the down-scaled domain and
  never rebuilds the large variance (`y = d/√(mean(d²)+ε)`, the scale cancels) — exact, and fp16-safe at any
  magnitude.

Result is GPU-clean (no banned ops, ≤4D) and fully delegated to LITERT_CL.

## Build & run

```bash
cd parakeet/
./gradlew :app:assembleDebug
./gradlew :app:installDebug
```

The model is too large to bundle, so push it once into the app's private storage:

```bash
# parakeet_ship_fp16.tflite from scripts/build_parakeet_ship.py or Hugging Face
./scripts/install_to_device.sh <dir-with-the-tflite>
```

Then launch **Parakeet ASR**: tap **Record** (auto-stops at 16 s) or **Transcribe sample** (bundled clip).

## Reproduce the model (`scripts/`)

NeMo and litert-torch can't share a process, so conversion is split (each ends with `os._exit(0)`):

1. `build_parakeet_A.py` — NeMo only: load the `.nemo`, `torch.save` the encoder+CTC modules
   (`parakeet_modules.pt`, what step 3 converts). `build_parakeet_A2.py` additionally saves a real-speech
   reference (mel/logits/ids) for parity checking.
2. `extract_prep.py` — save the exact mel filterbank (`mel_fb.bin`) and window from the preprocessor.
3. `build_parakeet_ship.py` — litert-torch: load the saved modules, apply the GPU patches above, pad to the
   16 s window with the additive mask, convert to tflite.
4. `fp16_parakeet.py` — fp16 quantize (`ai_edge_quantizer` FLOAT_CASTING) + op-check.
5. `validate_mel.py` — verify the host log-mel against NeMo's reference.
