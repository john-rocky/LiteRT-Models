"""Closed-loop streaming host for Nemotron-3-Diarization on LiteRT (numpy; Mac CPU).

audio 16 kHz -> numpy log-mel (extract_frontend.mel_mirror, the continuous stream) -> per chunk:
  graph A  mel [1,104,128] (zero-padded) -> chunk_embeds [1,13,512]
  packed = [speaker cache | FIFO | chunk + look-ahead] (L rows, zero tail to T=541)
  graph B  packed_embeds, attn_bias, rope_cos, rope_sin -> logits [1,4328,8] (first L*8 rows are real)
  host     sigmoid -> avg_pool(8) -> speaker cache / FIFO update: a numpy port of transformers'
           Nemotron3DiarizationSpeakerCache (update, _get_frame_scores, _boost_scores, _compress:
           score -> boost -> top-k -> sort -> gather, silence slots from silence_embeds)
The chunk schedule is the processor's low_latency one (chunk 9 + look-ahead 4 encoder frames; the first chunk
is mel frames [0,104), chunk k >= 1 is [72k, 72k+104) while its audio fits, the last chunk takes the rest with no
look-ahead). Constants come from config.json (streaming_config) and processor_config.json.

Backends: torch_fp32 (the re-authored torch model, plain LayerNorm), tflite_fp32 (n3d_encoder_ll.tflite),
tflite_fp16 (n3d_encoder_ll_fp16.tflite), tflite_safe_fp16 (n3d_encoder_ll_safe_fp16.tflite); graph A is the
torch frontend for torch_fp32 and n3d_frontend.tflite otherwise. tflite runs on the Mac CPU (CompiledModel).

Compared with transformers (results/ref_<fixture>_low_latency.npz, results/chunk_io_<fixture>.npz):
  final streaming logits (max|d|, corr, max|dp|, agreement@0.5), every step's packed rows and cache state
  (num_cache_frames, num_fifo_frames, is_compressed), and the segments (extract_speaker_dict port,
  threshold 0.5): boundary moves in 10 ms frames, vanished / added segments. A step whose packed rows differ
  from transformers by more than 1e-2 (or whose L differs) counts as a changed compress selection.
Gate (torch_fp32): logits <= 1e-4, packed rows <= 1e-5, cache state equal at every step.

--mel-source chunk_io feeds the transformers processor's per-chunk mel (captured in chunk_io) instead of the
numpy mirror (a diagnostic that isolates the mirror's contribution).
--score-impl numpy (default) is the portable port. --score-impl torch computes the pooled probabilities, frame
scores, boosts and top-k with the same torch ops as transformers (a diagnostic: numpy and torch round log / sum
differently by an ulp, which decides exact-tie boundaries of the top-k).

Usage: .venv/bin/python host_loop.py --run-dir <run> --model-dir <dir> \
           [--fixtures diarization_example_16k test_multispk_16k] [--backends torch_fp32 tflite_safe_fp16 ...]
"""

import argparse
import json
import math
import os
import time

import numpy as np
import soundfile as sf

import nemotron3diar_model as n3d
from extract_frontend import mel_mirror

T = n3d.T_LOW_LATENCY
MEL_FRAMES = n3d.MEL_FRAMES_LOW_LATENCY
LOGITS_MAX = 1e-4
PACKED_MAX = 1e-5
COMPRESS_CHANGED = 1e-2
B_FILES = {"tflite_fp32": "n3d_encoder_ll.tflite", "tflite_fp16": "n3d_encoder_ll_fp16.tflite",
           "tflite_safe_fp16": "n3d_encoder_ll_safe_fp16.tflite"}
F32 = np.float32


# ---------------------------------------------------------------------------- speaker cache (numpy port)


class SpeakerCache:
  """numpy port of transformers' Nemotron3DiarizationSpeakerCache (batch 1, streaming sizes)."""

  def __init__(self, cfg, silence_embeds, score_impl="numpy"):
    self.cfg = cfg
    self.score_impl = score_impl
    self.fifo_length = cfg["fifo_length"]
    self.update_period = cfg["speaker_cache_update_period"]
    self.cache_length = cfg["speaker_cache_length"]
    self.num_silence = cfg["speaker_cache_silence_frames_per_speaker"]
    self.threshold = F32(cfg["prediction_score_threshold"])
    self.latest_boost = F32(cfg["latest_frames_score_boost"])
    self.num_speakers = cfg["num_speakers"]
    self.factor = cfg["subsampling_factor"]
    budget = self.cache_length // self.num_speakers - self.num_silence
    self.min_positive_scores = math.floor(budget * cfg["min_positive_scores_rate"])
    self.num_strong = math.floor(budget * cfg["strong_boost_rate"])
    self.num_weak = math.floor(budget * cfg["weak_boost_rate"])
    self.silence = silence_embeds.astype(F32)
    hidden = self.silence.shape[0]
    self.embeds = np.zeros((0, hidden), F32)
    self.probs = np.zeros((0, self.num_speakers), F32)
    self.fifo = np.zeros((0, hidden), F32)
    self.is_compressed = False

  @property
  def num_cache_frames(self):
    return self.embeds.shape[0]

  @property
  def num_fifo_frames(self):
    return self.fifo.shape[0]

  def state(self):
    return (self.num_cache_frames, self.num_fifo_frames, int(self.is_compressed))

  def get_embeds(self):
    return np.concatenate([self.embeds, self.fifo], 0)

  def pool_probs(self, logits):
    """sigmoid, then the mean of every `factor` rows: [L*8, S] -> [L, S]."""
    if self.score_impl == "torch":
      import torch
      x = torch.from_numpy(np.ascontiguousarray(logits, F32))[None]
      pooled = torch.nn.functional.avg_pool1d(x.sigmoid().transpose(1, 2), self.factor, self.factor)
      return pooled.transpose(1, 2)[0].numpy()
    p = (F32(1) / (F32(1) + np.exp(-logits.astype(F32)))).astype(F32)
    return p.reshape(-1, self.factor, p.shape[-1]).mean(axis=1, dtype=F32)

  def num_popped_frames(self, n):
    if n <= self.fifo_length:
      return 0
    return min(max(self.update_period, n - self.fifo_length), n)

  def update(self, chunk_input_embeds, chunk_logits, num_chunk_frames):
    num_cache, num_fifo = self.num_cache_frames, self.num_fifo_frames
    probs = self.pool_probs(chunk_logits)
    chunk_start = num_cache + num_fifo
    chunk_embeds = chunk_input_embeds[chunk_start : chunk_start + num_chunk_frames]
    fifo_embeds = np.concatenate([self.fifo, chunk_embeds], 0)
    popped = self.num_popped_frames(fifo_embeds.shape[0])
    if popped:
      fifo_probs = probs[num_cache : num_cache + fifo_embeds.shape[0]]
      stored = self.probs if self.is_compressed else probs[:num_cache]
      cache_embeds = np.concatenate([self.embeds, fifo_embeds[:popped]], 0)
      cache_probs = np.concatenate([stored, fifo_probs[:popped]], 0)
      fifo_embeds = fifo_embeds[popped:]
      if cache_embeds.shape[0] > self.cache_length:
        cache_embeds, cache_probs = self.compress(cache_embeds, cache_probs)
        self.is_compressed = True
      self.embeds, self.probs = cache_embeds, cache_probs
    self.fifo = fifo_embeds

  def frame_scores(self, probs):
    thr = self.threshold
    log_p = np.log(np.maximum(probs, thr))
    log_c = np.log(np.maximum(F32(1) - probs, thr))
    scores = log_p - log_c + log_c.sum(axis=-1, keepdims=True, dtype=F32) - F32(math.log(0.5))
    is_speech = probs > 0.5
    scores = np.where(is_speech, scores, F32(-np.inf))
    is_positive = scores > 0
    enough = is_positive.sum(axis=0, keepdims=True) >= self.min_positive_scores
    return np.where(~is_positive & is_speech & enough, F32(-np.inf), scores).astype(F32)

  @staticmethod
  def topk_indices(values, k):
    """Indices of the k largest values (a stable order among ties: the lower index first)."""
    return np.argsort(-values, kind="stable")[:k]

  def boost(self, scores, num_boosted, boost):
    out = scores.copy()
    for s in range(scores.shape[1]):
      idx = self.topk_indices(scores[:, s], num_boosted)
      out[idx, s] = out[idx, s] + F32(boost)
    return out

  def compress_torch(self, embeds, probs):
    """The same selection computed with transformers' torch ops (diagnostic)."""
    import torch
    cfg = self.cfg
    p = torch.from_numpy(np.ascontiguousarray(probs))[None]
    thr = cfg["prediction_score_threshold"]
    log_p = torch.log(p.clamp(min=thr))
    log_c = torch.log((1.0 - p).clamp(min=thr))
    scores = log_p - log_c + log_c.sum(dim=-1, keepdim=True) - math.log(0.5)
    is_speech = p > 0.5
    scores = scores.masked_fill(~is_speech, float("-inf"))
    is_positive = scores > 0
    enough = is_positive.sum(dim=1, keepdim=True) >= self.min_positive_scores
    scores = scores.masked_fill(~is_positive & is_speech & enough, float("-inf"))
    scores[:, self.cache_length :] += cfg["latest_frames_score_boost"]
    for num, boost in ((self.num_strong, -2.0 * math.log(0.5)), (self.num_weak, -math.log(0.5))):
      _, idx = torch.topk(scores, num, dim=1, sorted=False)
      scores = scores.scatter_add(1, idx, scores.new_full(idx.shape, boost))
    num_frames, num_speakers = probs.shape
    scores = torch.nn.functional.pad(scores, (0, 0, 0, self.num_silence), value=float("inf"))
    num_scored = num_frames + self.num_silence
    sentinel = num_scored * num_speakers
    flat = scores.transpose(1, 2).reshape(1, -1)
    top_scores, top = torch.topk(flat, self.cache_length, dim=1, sorted=False)
    top = top.masked_fill(top_scores == float("-inf"), sentinel)
    top, _ = torch.sort(top, dim=1)
    frame = torch.where(top == sentinel, num_frames, (top % num_scored).clamp(max=num_frames))[0].numpy()
    embeds = np.concatenate([embeds, self.silence[None]], 0)
    probs = np.concatenate([probs, np.zeros((1, num_speakers), F32)], 0)
    return embeds[frame], probs[frame]

  def compress(self, embeds, probs):
    if self.score_impl == "torch":
      return self.compress_torch(embeds, probs)
    num_frames, num_speakers = probs.shape
    scores = self.frame_scores(probs)
    scores[self.cache_length :] += self.latest_boost
    scores = self.boost(scores, self.num_strong, -2.0 * math.log(0.5))
    scores = self.boost(scores, self.num_weak, -math.log(0.5))
    scores = np.concatenate([scores, np.full((self.num_silence, num_speakers), np.inf, F32)], 0)
    embeds = np.concatenate([embeds, self.silence[None]], 0)
    probs = np.concatenate([probs, np.zeros((1, num_speakers), F32)], 0)
    num_scored = num_frames + self.num_silence
    sentinel = num_scored * num_speakers
    flat = scores.T.reshape(-1)  # speaker-major: index = speaker * num_scored + frame
    top = self.topk_indices(flat, self.cache_length)
    top = np.where(flat[top] == -np.inf, sentinel, top)
    top = np.sort(top)
    frame = np.where(top == sentinel, num_frames, np.minimum(top % num_scored, num_frames))
    return embeds[frame], probs[frame]


# ---------------------------------------------------------------------------- schedule, segments


def schedule(num_samples, proc):
  """[(first mel frame, mel frames, look-ahead encoder frames, is_last)] of the processor's streaming mode."""
  fe = proc["feature_extractor"]
  hop, win, n_fft = fe["hop_length"], fe["win_length"], fe["n_fft"]
  chunk, right = proc["streaming_modes"][proc["streaming_mode"]]
  factor = proc["subsampling_factor"]
  per_chunk = (chunk + right) * factor
  per_step = chunk * factor
  first_samples = (per_chunk - 1) * hop + win // 2
  chunk_samples = per_chunk * hop + win
  assert num_samples >= first_samples, "audio shorter than the first streaming chunk"
  out = [(0, per_chunk, right, False)]
  mel_idx = per_step
  start = mel_idx * hop - n_fft // 2
  while start + chunk_samples <= num_samples:
    out.append((mel_idx, per_chunk, right, False))
    mel_idx += per_step
    start = mel_idx * hop - n_fft // 2
  last = (num_samples - start - n_fft) // hop + 1  # center=False frame count of the remaining audio
  assert last >= 1, last
  out.append((mel_idx, last, 0, True))
  return out


def segments(logits, threshold=0.5):
  """extract_speaker_dict port: [(start_frame, end_frame, speaker)] sorted by (start, speaker)."""
  p = (F32(1) / (F32(1) + np.exp(-logits.astype(F32)))).astype(F32)
  active = (p > threshold).astype(np.int8)
  pad = np.zeros((1, active.shape[1]), np.int8)
  changes = np.diff(np.concatenate([pad, active, pad], 0), axis=0)
  segs = []
  for s in range(active.shape[1]):
    starts = np.nonzero(changes[:, s] == 1)[0]
    ends = np.nonzero(changes[:, s] == -1)[0]
    segs.extend((int(a), int(b), s) for a, b in zip(starts, ends))
  segs.sort(key=lambda x: (x[0], x[2]))
  return segs


def compare_segments(ref, test):
  """Per speaker, one-to-one matches by overlap; boundary moves in frames (10 ms), vanished / added."""
  moved_start = moved_end = matched = vanished = added = split_merge = 0
  shifts = []
  for s in sorted({x[2] for x in ref} | {x[2] for x in test}):
    r = [(a, b) for a, b, k in ref if k == s]
    t = [(a, b) for a, b, k in test if k == s]
    ov = [[j for j, (c, d) in enumerate(t) if min(b, d) > max(a, c)] for a, b in r]
    back = [[i for i, (a, b) in enumerate(r) if min(b, d) > max(a, c)] for c, d in t]
    for i, js in enumerate(ov):
      if not js:
        vanished += 1
      elif len(js) == 1 and len(back[js[0]]) == 1:
        (a, b), (c, d) = r[i], t[js[0]]
        matched += 1
        moved_start += int(a != c)
        moved_end += int(b != d)
        shifts += [abs(a - c), abs(b - d)]
      else:
        split_merge += 1
    added += sum(1 for js in back if not js)
  nz = [x for x in shifts if x]
  return dict(ref_segments=len(ref), test_segments=len(test), matched=matched,
              boundary_moves=moved_start + moved_end, moved_starts=moved_start, moved_ends=moved_end,
              max_shift_frames=max(shifts) if shifts else 0,
              shift_hist_frames={str(k): nz.count(k) for k in sorted(set(nz))},
              vanished=vanished, added=added, split_or_merge=split_merge, identical=bool(ref == test))


# ---------------------------------------------------------------------------- backends


class TorchBackend:

  def __init__(self, model_dir):
    import torch
    self.torch = torch
    self.frontend_model, self.encoder_model, _, _ = n3d.load_models(os.path.join(model_dir, "model.safetensors"))

  def frontend(self, mel):
    with self.torch.no_grad():
      return self.frontend_model(self.torch.from_numpy(mel[None])).numpy()[0]

  def encoder(self, packed, bias, cos, sin):
    t = self.torch
    with t.no_grad():
      return self.encoder_model(t.from_numpy(packed), t.from_numpy(bias), t.from_numpy(cos), t.from_numpy(sin)).numpy()[0]


class TfliteBackend:

  def __init__(self, exports, b_file, threads):
    from gate_tflite_cpu import Runner
    self.a = Runner(os.path.join(exports, "n3d_frontend.tflite"), threads)
    self.b = Runner(os.path.join(exports, b_file), threads)

  def frontend(self, mel):
    return self.a(mel=mel[None])["chunk_embeds"][0]

  def encoder(self, packed, bias, cos, sin):
    return self.b(packed_embeds=packed, attn_bias=bias, rope_cos=cos, rope_sin=sin)["logits"][0]


# ---------------------------------------------------------------------------- the loop


def run_stream(audio, backend, stream_cfg, proc, silence, mel_fb, hann, score_impl="numpy", chunk_mel=None):
  sched = schedule(audio.shape[0], proc)
  total = max(g0 + nf for g0, nf, _, _ in sched)
  mel = mel_mirror(audio, mel_fb, hann, total) if chunk_mel is None else None
  cos, sin = (x.numpy() for x in n3d.rope_tables())
  factor = proc["subsampling_factor"]
  cache = SpeakerCache(stream_cfg, silence, score_impl)
  outs, steps = [], []
  for k, (g0, nf, right, is_last) in enumerate(sched):
    m = np.zeros((MEL_FRAMES, 128), F32)
    m[:nf] = mel[g0 : g0 + nf] if chunk_mel is None else chunk_mel[k][:nf]
    emb = backend.frontend(m)
    num_embeds = -(-nf // factor)
    num_chunk = num_embeds - right
    cached = cache.get_embeds()
    rows = np.concatenate([cached, emb[:num_embeds]], 0)
    L = rows.shape[0]
    assert L <= T, L
    logits = backend.encoder(n3d.pack(rows), n3d.attn_bias_for(L), cos, sin)[: L * factor]
    pre = cache.state()
    cache.update(rows, logits, num_chunk)
    c = cached.shape[0]
    outs.append(logits[c * factor : (c + num_chunk) * factor][:nf])
    steps.append(dict(L=L, rows=rows, pre=pre, post=cache.state(), first_frame=g0, frames=nf))
  return np.concatenate(outs, 0), steps


def corr(a, b):
  a = a.astype(np.float64).ravel() - a.mean()
  b = b.astype(np.float64).ravel() - b.mean()
  return float((a @ b) / math.sqrt((a @ a) * (b @ b)))


def sigmoid64(x):
  return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def evaluate(logits, steps, ref, cio):
  ref_logits = ref["logits"]
  n = len(steps)
  res = dict(frames=int(logits.shape[0]), ref_frames=int(ref_logits.shape[0]), steps=n,
             ref_steps=int(cio["length"].shape[0]))
  if logits.shape != ref_logits.shape or n != res["ref_steps"]:
    res["shape_mismatch"] = True
    return res
  p, pr = sigmoid64(logits), sigmoid64(ref_logits)
  flips = (p > 0.5) != (pr > 0.5)
  res.update(max_abs=float(np.abs(logits - ref_logits).max()), corr=corr(logits, ref_logits),
             max_dp=float(np.abs(p - pr).max()), agreement=float(1.0 - flips.mean()), flips=int(flips.sum()),
             cells=int(flips.size))
  packed_max, changed, state_bad, L_bad, per_step = 0.0, [], [], [], []
  for i, st in enumerate(steps):
    L_ref = int(cio["length"][i])
    pre_ref = tuple(int(x) for x in cio["cache_pre"][i])
    post_ref = tuple(int(x) for x in cio["cache_post"][i])
    d = None
    if st["L"] == L_ref:
      d = float(np.abs(st["rows"] - cio["packed_embeds"][i, :L_ref]).max())
      packed_max = max(packed_max, d)
    else:
      L_bad.append(i)
    if d is None or d > COMPRESS_CHANGED:
      changed.append(i)
    if tuple(st["pre"]) != pre_ref or tuple(st["post"]) != post_ref:
      state_bad.append(i)
    per_step.append(dict(step=i, L=st["L"], L_ref=L_ref, packed_max_abs=d, pre=list(st["pre"]),
                         post=list(st["post"]), post_ref=list(post_ref)))
  res.update(packed_rows_max_abs=packed_max, steps_L_mismatch=L_bad, steps_compress_changed=changed,
             num_steps_compress_changed=len(changed), steps_cache_state_mismatch=state_bad,
             first_compress_step=next((i for i, s in enumerate(steps) if s["post"][2] and not s["pre"][2]), None),
             first_compress_step_ref=next((i for i in range(n) if cio["cache_post"][i, 2] and not cio["cache_pre"][i, 2]),
                                          None),
             per_step=per_step)
  ref_segs = segments(ref_logits)
  stored = sorted(zip(np.round(ref["seg_start"], 2), np.round(ref["seg_end"], 2), ref["seg_speaker"]),
                  key=lambda x: (x[0], x[2]))
  port_ok = [(round(a * 0.01, 2), round(b * 0.01, 2), s) for a, b, s in ref_segs] == \
      [(float(a), float(b), int(s)) for a, b, s in stored]
  res["segments"] = compare_segments(ref_segs, segments(logits))
  res["segments"]["port_reproduces_transformers_segments"] = bool(port_ok)
  return res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--model-dir", required=True)
  ap.add_argument("--fixtures", nargs="+", default=["diarization_example_16k", "test_multispk_16k"])
  ap.add_argument("--backends", nargs="+", default=["torch_fp32", "tflite_fp32", "tflite_fp16", "tflite_safe_fp16"])
  ap.add_argument("--threads", type=int, default=8)
  ap.add_argument("--score-impl", choices=["numpy", "torch"], default="numpy")
  ap.add_argument("--mel-source", choices=["mirror", "chunk_io"], default="mirror")
  args = ap.parse_args()
  results = os.path.join(args.run_dir, "results")
  exports = os.path.join(args.run_dir, "exports")
  cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
  proc = json.load(open(os.path.join(args.model_dir, "processor_config.json")))
  stream_cfg = cfg["streaming_config"]
  mel_fb = np.fromfile(os.path.join(exports, "frontend_mel128_257.bin"), "<f4").reshape(128, 257)
  hann = np.fromfile(os.path.join(exports, "hann400.bin"), "<f4")
  _, _, silence, _ = n3d.load_models(os.path.join(args.model_dir, "model.safetensors"))
  silence = silence.numpy()

  out_path = os.path.join(results, "host_loop.json")
  report = json.load(open(out_path)) if os.path.exists(out_path) else {}
  report["thresholds"] = dict(torch_fp32_logits_max_abs=LOGITS_MAX, torch_fp32_packed_max_abs=PACKED_MAX,
                              compress_changed_packed_max_abs=COMPRESS_CHANGED)
  report["streaming_config"] = stream_cfg
  report.setdefault("runs", {})
  for b in args.backends:
    backend = TorchBackend(args.model_dir) if b == "torch_fp32" else TfliteBackend(exports, B_FILES[b], args.threads)
    for fx in args.fixtures:
      audio, sr = sf.read(os.path.join(args.run_dir, "fixtures", fx + ".wav"), dtype="float32")
      assert sr == 16000
      ref = np.load(os.path.join(results, f"ref_{fx}_low_latency.npz"))
      cio = np.load(os.path.join(results, f"chunk_io_{fx}.npz"))
      t0 = time.time()
      chunk_mel = cio["input_features"] if args.mel_source == "chunk_io" else None
      logits, steps = run_stream(audio, backend, stream_cfg, proc, silence, mel_fb, hann, args.score_impl, chunk_mel)
      res = evaluate(logits, steps, ref, cio)
      res["seconds"] = round(time.time() - t0, 1)
      if b == "torch_fp32" and not res.get("shape_mismatch"):
        res["PASS"] = bool(res["max_abs"] <= LOGITS_MAX and res["packed_rows_max_abs"] <= PACKED_MAX
                           and not res["steps_cache_state_mismatch"] and not res["steps_L_mismatch"])
      res["score_impl"] = args.score_impl
      res["mel_source"] = args.mel_source
      key = f"{b}/{fx}" + ("" if args.score_impl == "numpy" else "/score_impl_torch") + \
          ("" if args.mel_source == "mirror" else "/mel_chunk_io")
      report["runs"][key] = res
      brief = {k: res.get(k) for k in ("PASS", "max_abs", "max_dp", "agreement", "flips", "packed_rows_max_abs",
                                       "num_steps_compress_changed", "steps_cache_state_mismatch",
                                       "first_compress_step", "first_compress_step_ref", "seconds")}
      brief["segments"] = {k: res.get("segments", {}).get(k) for k in (
          "identical", "boundary_moves", "max_shift_frames", "vanished", "added", "split_or_merge",
          "port_reproduces_transformers_segments")}
      print(b, fx, args.score_impl, args.mel_source, json.dumps(brief), flush=True)
      with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
  print("HOST_LOOP done", flush=True)


if __name__ == "__main__":
  main()
