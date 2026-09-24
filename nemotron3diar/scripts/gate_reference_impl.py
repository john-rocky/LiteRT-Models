"""Gate: the shipped files through nemotron3_diar_litert.py (the reference host) vs transformers, Mac CPU.

Runs nemotron3_diar_litert.Nemotron3Diarizer on the fixtures in both modes, with the files as shipped (--model-dir =
the staging dir: nemotron3_diar_*.tflite + assets/), and compares with transformers fp32:
  low_latency  1600-sample pushes + finish() vs results/ref_<fixture>_low_latency.npz (the streaming logits) and
               results/chunk_io_<fixture>.npz (per step: L, packed rows, cache state)
  offline      run_file() vs results/ref_<fixture>_offline.npz (the offline forward, attention mask) and
               results/chunk_io_<fixture>_offline.npz
Metrics: max|dlogit|, max|dp|, agreement@0.5 and flips over every frame x speaker, segments (the module's
speaker_segments vs the reference's extract_speaker_dict output), steps whose L or packed rows differ (> 1e-2 = a
different cache selection), cache-state mismatches, max packed-row difference.
Results: results/gate_reference_impl.json.

Usage: .venv/bin/python gate_reference_impl.py --run-dir <run> --model-dir <run>/hf
           [--fixtures diarization_example_16k test_multispk_16k] [--modes low_latency offline] [--accelerator cpu]
"""

import argparse
import json
import os
import time

import numpy as np

import nemotron3_diar_litert as n3l

SELECTION_CHANGED = 1e-2


class Recorder(n3l.Nemotron3Diarizer):
  """Records every step's encoder input rows and the cache state around it."""

  def reset(self):
    super().reset()
    self.log = []

  def _encode(self, k, g0, frames, embeds, num_chunk, valid, times):
    cached = self.cache.rows()
    pre = (self.cache.embeds.shape[0], self.cache.fifo.shape[0], int(self.cache.is_compressed))
    step = super()._encode(k, g0, frames, embeds, num_chunk, valid, times)
    post = (self.cache.embeds.shape[0], self.cache.fifo.shape[0], int(self.cache.is_compressed))
    self.log.append(dict(rows=np.concatenate([cached, embeds], 0), pre=pre, post=post))
    return step


def sigmoid64(x):
  return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def compare(logits, ref_logits, log, cio, segs, ref_segs):
  res = dict(frames=int(logits.shape[0]), ref_frames=int(ref_logits.shape[0]), steps=len(log),
             ref_steps=int(cio["length"].shape[0]))
  if logits.shape != ref_logits.shape or len(log) != res["ref_steps"]:
    res["shape_mismatch"] = True
    return res
  p, pr = sigmoid64(logits), sigmoid64(ref_logits)
  flips = (p > 0.5) != (pr > 0.5)
  res.update(max_abs_logit=float(np.abs(logits - ref_logits).max()), max_dp=float(np.abs(p - pr).max()),
             agreement=float(1.0 - flips.mean()), flips=int(flips.sum()), cells=int(flips.size))
  packed_max, changed, state_bad = 0.0, [], []
  for i, st in enumerate(log):
    L = int(cio["length"][i])
    if st["rows"].shape[0] != L:
      changed.append(i)
    else:
      d = float(np.abs(st["rows"] - cio["packed_embeds"][i, :L]).max())
      packed_max = max(packed_max, d)
      if d > SELECTION_CHANGED:
        changed.append(i)
    if st["pre"] != tuple(int(x) for x in cio["cache_pre"][i]) or \
        st["post"] != tuple(int(x) for x in cio["cache_post"][i]):
      state_bad.append(i)
  res.update(packed_rows_max_abs=packed_max, steps_selection_changed=changed, steps_cache_state_mismatch=state_bad,
             compressions=sum(1 for s in log if s["post"][2] and s["pre"][0] + s["pre"][1] > 0 and
                              s["post"][0] == n3l.CACHE_LENGTH and s["rows"].shape[0] > 0))
  res["segments"] = dict(count=len(segs), ref_count=len(ref_segs), identical=segs == ref_segs)
  return res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--model-dir", required=True)
  ap.add_argument("--fixtures", nargs="+", default=["diarization_example_16k", "test_multispk_16k"])
  ap.add_argument("--modes", nargs="+", default=["low_latency", "offline"])
  ap.add_argument("--accelerator", default="cpu")
  ap.add_argument("--precision", default="fp32")
  args = ap.parse_args()
  results = os.path.join(args.run_dir, "results")
  out_path = os.path.join(results, "gate_reference_impl.json")
  report = json.load(open(out_path)) if os.path.exists(out_path) else {"runs": {}}
  for mode in args.modes:
    d = Recorder(args.model_dir, mode, args.accelerator, args.precision)
    for fx in args.fixtures:
      audio = n3l.load_wav(os.path.join(args.run_dir, "fixtures", fx + ".wav"))
      ref = np.load(os.path.join(results, f"ref_{fx}_{mode}.npz"))
      cio = np.load(os.path.join(results, f"chunk_io_{fx}.npz" if mode == "low_latency" else
                                 f"chunk_io_{fx}_offline.npz"))
      cio = {k: cio[k] for k in ("length", "packed_embeds", "cache_pre", "cache_post")}
      d.reset()
      t0 = time.perf_counter()
      if mode == "offline":
        steps = d.run_file(audio)
        valid = audio.shape[0] // n3l.HOP
        assert valid == int(ref["attention_mask"].sum()), (valid, int(ref["attention_mask"].sum()))
      else:
        steps = []
        for i in range(0, audio.shape[0], 1600):
          steps += d.push(audio[i : i + 1600])
        steps += d.finish()
        valid = None
      sec = time.perf_counter() - t0
      logits = np.concatenate([s.logits for s in steps], 0)
      segs = n3l.speaker_segments(logits, 0.5, valid)
      ref_segs = sorted(({"Start": round(float(a), 2), "End": round(float(b), 2), "Speaker": int(s)}
                         for a, b, s in zip(ref["seg_start"], ref["seg_end"], ref["seg_speaker"])),
                        key=lambda x: (x["Start"], x["Speaker"]))
      res = compare(logits, ref["logits"], d.log, cio, segs, ref_segs)
      res.update(seconds=round(sec, 2), audio_seconds=round(audio.shape[0] / n3l.SAMPLE_RATE, 2),
                 accelerator=args.accelerator, precision=args.precision,
                 compile_ms=dict(frontend=round(d.frontend.compile_ms, 1), encoder=round(d.encoder.compile_ms, 1)),
                 encoder_ms_median=float(np.median([s.ms["encoder"] for s in steps])))
      res["PASS"] = bool(not res.get("shape_mismatch") and res["agreement"] == 1.0 and res["segments"]["identical"]
                         and not res["steps_selection_changed"] and not res["steps_cache_state_mismatch"])
      key = f"{mode}/{fx}/{args.accelerator}_{args.precision}"
      report["runs"][key] = res
      print(key, json.dumps({k: res.get(k) for k in ("PASS", "max_abs_logit", "max_dp", "agreement", "flips",
                                                     "packed_rows_max_abs", "steps_selection_changed",
                                                     "steps_cache_state_mismatch", "segments", "seconds")}),
            flush=True)
      with open(out_path, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
  main()
