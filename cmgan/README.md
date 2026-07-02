# CMGAN — On-device speech enhancement / noise suppression (LiteRT GPU, fully-GPU)

[CMGAN](https://github.com/ruizhecao96/CMGAN) (TASLP 2024, MIT) **speech enhancement** running
**fully on the LiteRT CompiledModel GPU**: record in a noisy place (or pick a clip) and A/B the
denoised result. One 1.83 M-param dual-path conformer (4× time+freq blocks) processes 2 s 16 kHz
chunks; the **STFT and the mag^0.3 power compression run inside the GPU graph**, the host does
only reflect-padding, un-compression, inverse STFT and overlap-add.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **1651 / 1651** LITERT_CL (full residency, 1 partition) |
| inference | **~20 ms** per 2 s chunk (RTF ≈ 0.01) |
| fp16 size | 4.2 MB |
| quality | SI-SNR +7.2 dB on a 6.6 dB noisy sample (PyTorch +9.6 dB); device-vs-torch wav corr 0.997 |

```
wav[1,32400] (2 s chunk, host reflect-pad) →[GPU: DFT-conv STFT → mag^0.3 → dense encoder →
4× (time-conformer + freq-conformer) → mask + complex decoders]→ (real, imag)[1,1,321,201]
→[host: mag^(1/0.3) + iSTFT + OLA]→ denoised wav
```

## How it converts (litert-torch) — all numerically-equivalent

1. **The phase path cancels algebraically**: `mask·mag·cos(∠x) ≡ mask·x_r` (and sin → x_i) — no
   `atan2`/`cos`/`sin` in the graph at all.
2. **Shaw relative positional embedding** (`nn.Embedding(dist)` = banned GATHER): the distance
   matrix is constant for a fixed chunk → baked to a constant, applied as a 2D `FULLY_CONNECTED`
   (pre-flipped columns) + the pad/reshape **skew** realignment — no runtime gather, ≤4D.
3. Conformer folded batches `(b·f, t, c)` → batch-1 4D `[1, c, A, n]`: LayerNorm → channel-LN per
   position, Linear → 1×1 Conv2d, depthwise Conv1d → `(1,k)` Conv2d; attention as 3D BMM with
   heads folded into the batch, 1/√d folded into Q.
4. `mag^0.3` → `exp(0.3·ln(max(mag², 1e-6)))` (`POW` is banned; fp16-safe floor).
5. `SPConvTranspose2d` 5-D view → `[b,r·c,H,W] → [b,r,c,H·W] → permute → [b,c,H,W·r]` (exact).
6. InstanceNorm2d → Safe per-channel spatial norm; BatchNorm1d (eval) → constant scale/shift;
   PReLU → `relu(x) − w·relu(−x)`; GLU → slices; **all norm eps ≥ 1e-4** (fp16 min-normal, C38)
   and **no dim-1 broadcasts** (C39).

fp32/fp16 tflite-vs-torch corr 0.999999; the remaining device gap (spec corr 0.991) is fp16
precision compounding across the 8 conformer blocks (bisect: front-end 0.999997, encoder 0.99998,
after TSCB4 0.989) — task metric stays strong (+7.2 dB SI-SNR).

## Build & run

```bash
python scripts/build_cmgan.py        # -> cmgan_fp16.tflite (+ parity gates)
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-cmgan_fp16.tflite>
```

The first launch fails with "Model not found" until the model is pushed. **Record noisy audio**
(mic, unprocessed source) or **pick a clip**, then A/B **Noisy** vs **Enhanced**.

Model: `litert-community/CMGAN-LiteRT` (Hugging Face). Upstream:
[ruizhecao96/CMGAN](https://github.com/ruizhecao96/CMGAN) (MIT), trained on VoiceBank-DEMAND.
