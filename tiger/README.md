# TIGER-DnR — On-device cinematic sound separation (LiteRT GPU, fully-GPU)

[TIGER](https://github.com/JusperLee/TIGER) (Tsinghua, ICASSP 2025, MIT/Apache-2.0) **audio source
separation** running **fully on the LiteRT CompiledModel GPU**: split any clip (movie scene, game,
vlog) into **Dialogue / Sound effects / Music** stems, on the phone. Three sibling ~1.4 M-param
band-split TIGER graphs (dialog / effect / music, [JusperLee/TIGER-DnR](https://huggingface.co/JusperLee/TIGER-DnR),
trained on the openly-built DnR dataset) each process a 12.06 s 44.1 kHz chunk; per DnR convention
each graph contributes one stem. The **STFT runs inside the GPU graph** (windowed DFT as one
`Conv1d`, PANNs recipe); the host does only reflect-padding, iSTFT and overlap-add.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **23 974 / 23 974** LITERT_CL (full residency, 1 partition — largest graph in this zoo) |
| inference | **~4.5 s** per 12.06 s chunk per stem-graph (bandwidth-bound) |
| fp16 size | 16.1 MB × 3 graphs |
| accuracy | device-vs-PyTorch waveform corr **0.99987** (dialog 0.9999 / effect 0.9989 / music 0.9998) |

```
wav[1,534016] (44.1 kHz chunk, host reflect-pad) →[GPU: DFT-conv STFT → 57-band split →
8 weight-tied (freq-path + frame-path) UConv+MHSA iterations → complex masks]→
(real, imag)[1,3,1025,1040] →[host iSTFT + OLA]→ stem waveform
```

## How it converts (litert-torch) — all numerically-equivalent

No RNN, no gather, no dense warp — but plenty of Mali-hostile constructs, each re-authored exactly:

1. **`torch.stft` → hann-windowed DFT as one `Conv1d`** (stride=hop); reflect pad moves to the host.
   `torch.istft` moves to the host (Kotlin radix-2 IFFT + OLA) — the graph outputs (real, imag).
2. **Folded-batch `Conv1d`** (`(B·T, N, band)` / `(B·band, N, T)`) → 4D `[1, N, T, band]` `Conv2d`
   with `(1, k)` kernels; the per-sample `GlobLN` becomes a per-position norm (chained single-axis
   means — multi-axis reduces mis-execute on Mali).
3. **`adaptive_avg_pool1d` / nearest `interpolate`**: chunk length chosen so T=1040 is divisible by
   16 → every pool window is uniform (`AVERAGE_POOL_2D`) and every nearest resize is an exact
   integer repeat (on-stride `RESIZE_NEAREST`). The non-uniform band axis (57→29→15→8→4) uses
   exact constant averaging/one-hot matrices as `FULLY_CONNECTED`.
4. **MHSA**: batch-concatenated heads → per-head batch-1 3D BMM (`[1,tokens,d]`), 1/√d folded into
   Q before the matmul; heads re-joined by channel concat.
5. **PReLU → `relu(x) − w·relu(−x)`**; **6-D mask-head view → static channel slices** (≤4D).
6. **Device-only fix ⭐ eps underflow**: GlobLN eps 1e-8 / LN4D eps 1e-5 underflow to 0 in fp16 →
   silent bands hit `0/0 = NaN` on Mali and the NaNs spread across time (desktop is fine). eps=1e-4
   is exact-equivalent (silence still → 0) and NaN-free.
7. **Device-only fix ⭐ no dim-1 broadcasts in the mask head**: `spec[1,1,bw,T] × mask[1,3,bw,T]`
   broadcast MUL made the Mali delegate return source 0's output for **all three** sources
   (desktop CPU correct). Rewritten as per-source same-shape `[1,bw,T]` arithmetic + final concat.

Result: banned ops NONE, all tensors ≤4D, tflite-vs-torch corr 0.99991, device corr 0.99987.

## Build & run

```bash
python scripts/build_tiger.py dialog     # + effect, music -> tiger_{stem}_fp16.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-tiger_*_fp16.tflite>
```

The first launch fails with "Model not found" until the models are pushed. **Pick audio / video
clip** (any format; decoded + resampled on-device) or **Record 15 s**, then play the separated
Dialogue / Sound effects / Music stems. Clips are processed in 12.06 s windows with 10 s hop
(overlaps averaged), first 32 s.

Model: `litert-community/TIGER-DnR-LiteRT` (Hugging Face). Upstream:
[JusperLee/TIGER-DnR](https://huggingface.co/JusperLee/TIGER-DnR) (Apache-2.0),
[JusperLee/TIGER](https://github.com/JusperLee/TIGER) (MIT).
