# Basic Pitch — On-device music transcription / audio-to-MIDI (LiteRT GPU, fully-GPU)

[Basic Pitch](https://github.com/spotify/basic-pitch) (Spotify, ICASSP 2022, Apache-2.0) music
transcription running **fully on the LiteRT CompiledModel GPU — including the conv-based CQT
front-end**. Play an instrument (or sing, or pick a clip) and see the notes on a piano roll.

## On-device (Pixel 8a, Tensor G3 — verified)

| | |
|---|---|
| nodes on GPU | **241 / 241** LITERT_CL (full residency, 1 partition) |
| inference | **~4.4 ms** per 2 s window (RTF ≈ 0.002) |
| size | 0.84 MB FP32 (weights are tiny; fp16 quantization costs ~0.005 corr — ship fp32) |
| accuracy | device-vs-reference: note corr 0.998 / contour 0.982; note-event F1@0.5 **0.98**, per-frame argmax agreement 98% |

```
wav[1,43844] (22.05 kHz) →[GPU: 9-octave conv-CQT (shared 36×256 kernel banks + lowpass
downsample chain) → NormalizedLog → harmonic stacking → 3-branch CNN]→ contour[172,264],
note[172,88], onset[172,88] →[host]→ note events → piano roll
```

## How it converts — re-authored from the official ONNX (no TensorFlow needed)

All 102 constants extracted from `nmp.onnx`; torch re-implementation is **bit-exact vs the
official model (corr 1.000000, max|d| 0.0)**. Key rewrites: reflect-pad → anti-diagonal-constant
`FULLY_CONNECTED` (no GATHER); PACK/stacking → concat + static slices; `divide_no_nan` → clamped
div. Two device-only fp16 fixes: ⭐ **post-log clamp** `clamp(10·log10(p+1e-10), min=-100)` —
a desktop no-op that recovers `log(0)=-inf` after the fp16-flushed `1e-10` floor (raising the
floor *before* the log is wrong: it shifts the min/max normalization globally); ⭐ **per-bin CQT
norm (~200×) folded into per-octave kernel copies** (magnitude is linear in kernel scale = exact)
— lifts conv outputs out of the fp16-precision-poor ~1e-3 range (device contour 0.845 → 0.982).

## Build & run

```bash
python scripts/build_bp.py            # parity vs official ONNX -> basicpitch.tflite
./gradlew :app:installDebug
./scripts/install_to_device.sh <dir-with-basicpitch.tflite>
```

**Record & transcribe** (mic) or pick a clip → piano roll + note list. 2 s windows with the
official 30-frame overlap stitching; onset-triggered note decoding on the host.

Model: `litert-community/Basic-Pitch-LiteRT` (Hugging Face). Upstream:
[spotify/basic-pitch](https://github.com/spotify/basic-pitch) (Apache-2.0).
