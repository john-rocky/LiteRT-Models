"""Host mel front-end tables + a numpy-only mel mirror, gated against the transformers processor.

Writes (float32 little-endian):
  exports/frontend_mel128_257.bin  librosa slaney mel filter [128,257] (the processor's own table)
  exports/hann400.bin              torch.hann_window(400, periodic=False)
Mirror (numpy only), on the continuous stream:
  preemphasis y[n] = x[n] - 0.97 x[n-1] (y[0] = x[0]) -> frame i = y[160i-256 : 160i+256), negative
  indices zero -> hann(400) centered in the 512 frame (offset 56) -> rfft (fp32 path) -> |X|^2 via
  sqrt then square (the processor's rounding) -> mel @ -> log(x + 2^-24)
Gate: every frame the processor emits in the low_latency schedule (center=True first chunk,
center=False later chunks, look-ahead frames included) and the offline pass: max|d| <= 1e-4 (log).

Usage: .venv/bin/python extract_frontend.py --run-dir <run> --model-dir <dir> --fixtures a.wav b.wav
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor

N_FFT = 512
HOP = 160
WIN = 400
PREEMPH = np.float32(0.97)
LOG_GUARD = np.float32(2.0**-24)
MAX_ABS = 1e-4


def mel_mirror(audio, mel_fb, hann, num_frames):
  """numpy log-mel of frames 0..num_frames-1 of the continuous stream `audio` (float32 [N])."""
  x = audio.astype(np.float32)
  y = np.empty_like(x)
  y[0] = x[0]
  y[1:] = x[1:] - PREEMPH * x[:-1]
  pad = N_FFT // 2
  last = (num_frames - 1) * HOP + N_FFT
  yp = np.zeros(pad + max(len(y), last) + N_FFT, np.float32)
  yp[pad : pad + len(y)] = y
  idx = np.arange(num_frames)[:, None] * HOP + np.arange(N_FFT)[None, :]
  frames = yp[idx]  # frame i = y[160i-256 : 160i+256)
  window = np.zeros(N_FFT, np.float32)
  off = (N_FFT - WIN) // 2
  window[off : off + WIN] = hann
  spec = np.fft.rfft(frames * window, axis=-1, norm="forward") * np.float32(N_FFT)
  assert spec.dtype == np.complex64, spec.dtype
  mag = np.sqrt(spec.real * spec.real + spec.imag * spec.imag)
  power = mag * mag
  mel = (mel_fb @ power.T).T  # [T,128]
  return np.log(mel + LOG_GUARD).astype(np.float32)


def streaming_chunks(processor, audio):
  """(global first frame, per-chunk features) in the README low_latency schedule."""
  out = []
  f = processor(audio[: processor.num_samples_first_audio_chunk], sampling_rate=16000,
                is_streaming=True, is_first_audio_chunk=True)
  out.append((0, f["input_features"][0].numpy()))
  mel_frame_idx = processor.num_mel_frames_per_step
  start = processor.audio_chunk_start(mel_frame_idx)
  while (end := start + processor.num_samples_per_audio_chunk) <= audio.shape[0]:
    f = processor(audio[start:end], sampling_rate=16000, is_streaming=True, is_first_audio_chunk=False)
    out.append((mel_frame_idx, f["input_features"][0].numpy()))
    mel_frame_idx += processor.num_mel_frames_per_step
    start = processor.audio_chunk_start(mel_frame_idx)
  f = processor(audio[start:], sampling_rate=16000, is_streaming=True, is_first_audio_chunk=False,
                is_last_audio_chunk=True)
  out.append((mel_frame_idx, f["input_features"][0].numpy()))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--model-dir", required=True)
  ap.add_argument("--fixtures", nargs="+", required=True)
  args = ap.parse_args()
  exports = os.path.join(args.run_dir, "exports")
  results = os.path.join(args.run_dir, "results")
  os.makedirs(exports, exist_ok=True)

  processor = AutoProcessor.from_pretrained(args.model_dir)
  fe = processor.feature_extractor
  assert (fe.n_fft, fe.hop_length, fe.win_length, fe.feature_size) == (N_FFT, HOP, WIN, 128)
  assert np.float32(fe.preemphasis) == PREEMPH
  mel_fb = fe.mel_filters.numpy().astype(np.float32)
  assert mel_fb.shape == (128, 257), mel_fb.shape
  hann = torch.hann_window(WIN, periodic=False).numpy().astype(np.float32)
  mel_fb.astype("<f4").tofile(os.path.join(exports, "frontend_mel128_257.bin"))
  hann.astype("<f4").tofile(os.path.join(exports, "hann400.bin"))
  # the written files round-trip bit-exactly
  assert np.array_equal(np.fromfile(os.path.join(exports, "frontend_mel128_257.bin"), "<f4").reshape(128, 257), mel_fb)
  assert np.array_equal(np.fromfile(os.path.join(exports, "hann400.bin"), "<f4"), hann)

  report = {"tables": {"mel": [128, 257], "hann": [WIN], "periodic": False, "dtype": "float32 LE"},
            "threshold_max_abs": MAX_ABS, "fixtures": {}}
  all_pass = True
  for wav in args.fixtures:
    name = os.path.splitext(os.path.basename(wav))[0]
    audio, sr = sf.read(wav, dtype="float32")
    assert sr == 16000

    chunks = streaming_chunks(processor, audio)
    total = max(g0 + c.shape[0] for g0, c in chunks)
    mirror = mel_mirror(audio, mel_fb, hann, total)
    worst, worst_at, n_frames, per_chunk = 0.0, None, 0, []
    for k, (g0, c) in enumerate(chunks):
      d = np.abs(mirror[g0 : g0 + c.shape[0]] - c)
      m = float(d.max()) if d.size else 0.0
      per_chunk.append(m)
      n_frames += c.shape[0]
      if m > worst:
        r = np.unravel_index(int(d.argmax()), d.shape)
        worst, worst_at = m, dict(chunk=k, frame=int(g0 + r[0]), mel_bin=int(r[1]),
                                  first_chunk=k == 0)
    # the same mirror against the offline (center=True, whole recording) features
    off = processor(audio, sampling_rate=16000)
    off_f = off["input_features"][0].numpy()
    valid = int(off["attention_mask"][0].sum())
    mirror_off = mel_mirror(audio, mel_fb, hann, valid)
    off_max = float(np.abs(mirror_off - off_f[:valid]).max())
    # continuity of the processor itself: streaming frames vs the offline frames at the same index
    cont = max(float(np.abs(off_f[g0 : g0 + c.shape[0]][: max(0, valid - g0)] -
                            c[: max(0, min(c.shape[0], valid - g0))]).max(initial=0.0))
               for g0, c in chunks)
    fx = dict(chunks=len(chunks), frames_compared=n_frames, frames_distinct=total,
              streaming_max_abs=worst, worst=worst_at,
              first_chunk_max_abs=per_chunk[0], later_chunks_max_abs=max(per_chunk[1:]),
              offline_frames=valid, offline_max_abs=off_max,
              processor_streaming_vs_offline_max_abs=cont,
              PASS=bool(worst <= MAX_ABS and off_max <= MAX_ABS))
    all_pass &= fx["PASS"]
    report["fixtures"][name] = fx
    print(json.dumps({name: fx}, indent=1), flush=True)

  report["PASS"] = bool(all_pass)
  with open(os.path.join(results, "mel_mirror.json"), "w") as f:
    json.dump(report, f, indent=1)
  print("MEL_MIRROR", "PASS" if all_pass else "FAIL", flush=True)


if __name__ == "__main__":
  main()
