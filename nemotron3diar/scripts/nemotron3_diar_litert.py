"""Nemotron-3-Diarization on LiteRT: the streaming host as an importable reference (numpy + ai-edge-litert).

Who spoke when, up to 8 speakers, one decision per 10 ms frame. The two graphs run on the LiteRT CompiledModel;
everything else is host code in this file:

  16 kHz mono audio -> log-mel (128 bins, 10 ms hop) -> per chunk:
    graph A  nemotron3_diar_frontend.tflite   mel [1,104,128] -> chunk_embeds [1,13,512]
             (8-frame stacking + projection = one 80 ms encoder frame per row)
    pack     [speaker cache | FIFO | chunk + look-ahead] = L rows, zero rows up to the graph's fixed T
    graph B  nemotron3_diar_encoder_<mode>_fp16.tflite
             packed_embeds [1,T,512], attn_bias [1,1,1,T], rope_cos / rope_sin [1,1,T,64] -> logits [1,8T,8]
    host     sigmoid -> mean of every 8 rows -> speaker cache / FIFO update (the Arrival-Order Speaker Cache,
             compressed back to 264 frames when it overflows) -> the chunk's own logit rows

Modes (encoder frames of 80 ms):
  low_latency  streaming: T=541, chunk 9 + look-ahead 4 (the first decision needs 1.04 s of audio, then one
               step per 0.72 s), FIFO 264, cache update period 222.
  offline      whole file: T=684, chunk 340 + look-ahead 40, FIFO 40, cache update period 300.

attn_bias is added to every attention score: 0 on the L real rows, -3e4 on the zero rows (low_latency); the
offline graph takes three levels, 0 real, -16384 real row whose key is masked (the frame after the last full hop
of a file), -32768 zero row. The RoPE tables are inputs (positions 0..T-1, the same every step), so no large
constant is baked into the graph.

The host follows transformers' Nemotron3DiarizationProcessor (the low_latency chunk schedule, the log-mel) and
Nemotron3DiarizationSpeakerCache (update / compress), batch 1. Speaker probabilities are summed in torch's CPU
order so that exact-tie cache selections resolve as in the reference.

Usage:
  python nemotron3_diar_litert.py meeting_16k.wav [--mode low_latency|offline] [--model-dir DIR]
      [--accelerator cpu|gpu] [--precision fp32|fp16] [--threads N] [--json out.json]

  from nemotron3_diar_litert import Nemotron3Diarizer, diarize, load_wav, speaker_segments
  diarize("meeting_16k.wav")  # [{'Start': 0.0, 'End': 3.2, 'Speaker': 0}, ...] like extract_speaker_dict
"""

import argparse
import json
import math
import os
import time
from typing import NamedTuple

import numpy as np

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 160
WIN = 400
N_MELS = 128
N_BINS = N_FFT // 2 + 1
PREEMPH = np.float32(0.97)
LOG_GUARD = np.float32(2.0**-24)
SUBSAMPLING = 8  # mel frames per encoder frame
FRONTEND_FRAMES = 104  # graph A input: 13 encoder frames
HIDDEN = 512
HEAD_DIM = 64
NUM_SPEAKERS = 8
ROPE_THETA = 10000.0
FRAME_SECONDS = HOP / SAMPLE_RATE

# config.json streaming_config: the speaker-cache rules (both modes)
CACHE_LENGTH = 264
NUM_SILENCE = 1  # silence slots per speaker in a compressed cache
SCORE_THRESHOLD = np.float32(0.25)
MIN_POSITIVE_SCORES_RATE = 0.5
STRONG_BOOST_RATE = 0.75
WEAK_BOOST_RATE = 1.5
LATEST_BOOST = np.float32(0.05)
LOG_HALF = np.float32(math.log(0.5))
STRONG_BOOST = np.float32(-2.0 * math.log(0.5))
WEAK_BOOST = np.float32(-math.log(0.5))

F32 = np.float32


class Mode(NamedTuple):
  name: str
  encoder_file: str
  chunk: int  # encoder frames per step
  right_context: int  # look-ahead encoder frames
  fifo_length: int
  update_period: int
  max_rows: int  # graph B's fixed T = cache 264 + FIFO + chunk + look-ahead
  pad_bias: float
  masked_bias: float


MODES = {
    "low_latency": Mode("low_latency", "nemotron3_diar_encoder_low_latency_fp16.tflite", 9, 4, 264, 222, 541,
                        -3.0e4, -3.0e4),
    "offline": Mode("offline", "nemotron3_diar_encoder_offline_fp16.tflite", 340, 40, 40, 300, 684,
                    -32768.0, -16384.0),
}
FRONTEND_FILE = "nemotron3_diar_frontend.tflite"


class Step(NamedTuple):
  index: int
  first_frame: int  # first 10 ms frame of `logits`
  length: int  # encoder input rows L (cache + FIFO + chunk + look-ahead)
  logits: np.ndarray  # [frames, 8] of the frames this step emits; sigmoid gives the speaker activity
  ms: dict  # mel / frontend / encoder / cache / total milliseconds


# ------------------------------------------------------------------------------------------------ audio, tables


def load_wav(path):
  """16 kHz audio as float32 mono (channels averaged)."""
  import soundfile as sf
  audio, sr = sf.read(path, dtype="float32", always_2d=True)
  if sr != SAMPLE_RATE:
    raise ValueError(f"{path}: {sr} Hz; resample to 16 kHz first "
                     "(ffmpeg -i in.wav -ac 1 -ar 16000 out_16k.wav)")
  return audio.mean(axis=1, dtype=F32) if audio.shape[1] > 1 else audio[:, 0]


def _floats(path):
  return np.fromfile(path, "<f4")


def rope_tables(t):
  """cos / sin [1,1,T,64] for positions 0..T-1 (fp32, as transformers builds them)."""
  inv_freq = F32(1.0) / (F32(ROPE_THETA) ** (np.arange(0, HEAD_DIM, 2).astype(F32) / F32(HEAD_DIM)))
  freqs = np.arange(t).astype(F32)[:, None] * inv_freq[None, :]
  emb = np.concatenate([freqs, freqs], axis=-1)
  return np.cos(emb)[None, None].astype(F32), np.sin(emb)[None, None].astype(F32)


def speaker_segments(logits, threshold=0.5, valid_frames=None):
  """Per-frame logits [N, 8] -> [{'Start', 'End', 'Speaker'}] (seconds), as the processor's extract_speaker_dict."""
  active = sigmoid(logits) > threshold
  if valid_frames is not None:
    active[valid_frames:] = False
  pad = np.zeros((1, active.shape[1]), np.int8)
  changes = np.diff(np.concatenate([pad, active.astype(np.int8), pad], 0), axis=0)
  segs = []
  for s in range(active.shape[1]):
    starts = np.nonzero(changes[:, s] == 1)[0]
    ends = np.nonzero(changes[:, s] == -1)[0]
    segs += [{"Start": round(int(a) * FRAME_SECONDS, 2), "End": round(int(b) * FRAME_SECONDS, 2), "Speaker": s}
             for a, b in zip(starts, ends)]
  segs.sort(key=lambda x: (x["Start"], x["Speaker"]))
  return segs


def sigmoid(x):
  return F32(1) / (F32(1) + np.exp(-np.asarray(x, F32)))


# ------------------------------------------------------------------------------------------------ log-mel


class MelFrontend:
  """Log-mel of a continuous 16 kHz stream: preemphasis 0.97 (first sample kept) -> frame i = samples
  [160i - 256, 160i + 256), zero outside the stream -> hann(400) centered in the 512 window -> fp32 real FFT
  -> |X|^2 -> slaney mel [128, 257] -> log(x + 2^-24). No normalization."""

  def __init__(self, mel_filters, hann):
    self.mel_filters = np.asarray(mel_filters, F32).reshape(N_MELS, N_BINS)
    self.window = np.zeros(N_FFT, F32)
    off = (N_FFT - WIN) // 2
    self.window[off : off + WIN] = np.asarray(hann, F32)
    self.reset()

  def reset(self):
    self.y = np.zeros(0, F32)
    self.last = F32(0)

  @property
  def num_samples(self):
    return self.y.shape[0]

  def append(self, x):
    x = np.asarray(x, F32)
    if x.size == 0:
      return
    y = np.empty_like(x)
    y[0] = x[0] if self.y.size == 0 else x[0] - PREEMPH * self.last
    y[1:] = x[1:] - PREEMPH * x[:-1]
    self.last = x[-1]
    self.y = np.concatenate([self.y, y])

  def frames(self, first, count):
    """Frames first .. first + count - 1 [count, 128]; samples past the end of the stream read as zero."""
    start = first * HOP - N_FFT // 2
    seg = np.zeros((count - 1) * HOP + N_FFT, F32)
    lo, hi = max(start, 0), min(start + seg.size, self.y.size)
    if hi > lo:
      seg[lo - start : hi - start] = self.y[lo:hi]
    idx = np.arange(count)[:, None] * HOP + np.arange(N_FFT)[None, :]
    spec = np.fft.rfft(seg[idx] * self.window, axis=-1, norm="forward") * F32(N_FFT)  # fp32 FFT path
    mag = np.sqrt(spec.real * spec.real + spec.imag * spec.imag)
    mel = (self.mel_filters @ (mag * mag).T).T
    return np.log(mel + LOG_GUARD).astype(F32)


# ------------------------------------------------------------------------------------------------ speaker cache


class SpeakerCache:
  """Arrival-Order Speaker Cache + FIFO of transformers' Nemotron3DiarizationSpeakerCache (batch 1)."""

  def __init__(self, fifo_length, update_period, silence):
    self.fifo_length = fifo_length
    self.update_period = update_period
    self.silence = np.asarray(silence, F32)
    budget = CACHE_LENGTH // NUM_SPEAKERS - NUM_SILENCE
    self.min_positive = math.floor(budget * MIN_POSITIVE_SCORES_RATE)
    self.num_strong = math.floor(budget * STRONG_BOOST_RATE)
    self.num_weak = math.floor(budget * WEAK_BOOST_RATE)
    self.embeds = np.zeros((0, HIDDEN), F32)
    self.probs = np.zeros((0, NUM_SPEAKERS), F32)
    self.fifo = np.zeros((0, HIDDEN), F32)
    self.is_compressed = False

  def rows(self):
    """Cached rows then FIFO rows: the first rows of the next encoder input."""
    return np.concatenate([self.embeds, self.fifo], 0)

  @staticmethod
  def pool(logits, valid=None):
    """sigmoid, then the mean of every 8 rows (summed in order): [L*8, 8] -> [L, 8]; invalid rows get 0."""
    p = sigmoid(logits).reshape(-1, SUBSAMPLING, NUM_SPEAKERS)
    acc = p[:, 0]
    for j in range(1, SUBSAMPLING):
      acc = acc + p[:, j]
    out = (acc / F32(SUBSAMPLING)).astype(F32)
    if valid is not None:
      out[~valid] = 0
    return out

  def update(self, rows, logits, num_chunk, valid=None):
    """rows [L, 512] this step's encoder input, logits [L*8, 8], num_chunk chunk rows that join the FIFO."""
    nc, nf = self.embeds.shape[0], self.fifo.shape[0]
    probs = self.pool(logits, valid)
    fifo = np.concatenate([self.fifo, rows[nc + nf : nc + nf + num_chunk]], 0)
    n = fifo.shape[0]
    popped = 0 if n <= self.fifo_length else min(max(self.update_period, n - self.fifo_length), n)
    if popped:
      # an uncompressed cache holds plain chunk frames whose probabilities this step re-estimates
      stored = self.probs if self.is_compressed else probs[:nc]
      embeds = np.concatenate([self.embeds, fifo[:popped]], 0)
      cand = np.concatenate([stored, probs[nc : nc + popped]], 0)
      if embeds.shape[0] > CACHE_LENGTH:
        keep = self.compress(cand)
        embeds = np.concatenate([embeds, self.silence[None]], 0)[keep]
        cand = np.concatenate([cand, np.zeros((1, NUM_SPEAKERS), F32)], 0)[keep]
        self.is_compressed = True
      self.embeds, self.probs = embeds, cand
    self.fifo = fifo[popped:]

  def frame_scores(self, p):
    lp = np.log(np.maximum(p, SCORE_THRESHOLD))
    lc = np.log(np.maximum(F32(1) - p, SCORE_THRESHOLD))
    total = (((lc[:, 0] + lc[:, 4]) + (lc[:, 1] + lc[:, 5])) + (lc[:, 2] + lc[:, 6])) + (lc[:, 3] + lc[:, 7])
    speech = p > 0.5
    scores = np.where(speech, ((lp - lc) + total[:, None]) - LOG_HALF, F32(-np.inf)).astype(F32)
    positive = scores > 0
    enough = positive.sum(axis=0, keepdims=True) >= self.min_positive
    return np.where(~positive & speech & enough, F32(-np.inf), scores).astype(F32)

  @staticmethod
  def top_k(values, k):
    """Indices of the k largest values; equal values keep the lower index first."""
    return np.argsort(-values, kind="stable")[:k]

  def compress(self, probs):
    """Candidate index per kept slot (the candidate count = a silence slot), grouped by speaker."""
    m = probs.shape[0]
    scores = self.frame_scores(probs)
    scores[CACHE_LENGTH:] += LATEST_BOOST  # the frames just popped from the FIFO
    for k, boost in ((self.num_strong, STRONG_BOOST), (self.num_weak, WEAK_BOOST)):
      for s in range(NUM_SPEAKERS):
        idx = self.top_k(scores[:, s], k)
        scores[idx, s] += boost
    scored = m + NUM_SILENCE
    flat = np.concatenate([scores, np.full((NUM_SILENCE, NUM_SPEAKERS), np.inf, F32)], 0).T.reshape(-1)
    top = self.top_k(flat, CACHE_LENGTH)
    sentinel = flat.size
    top = np.sort(np.where(flat[top] == -np.inf, sentinel, top))
    return np.where(top == sentinel, m, np.minimum(top % scored, m))


# ------------------------------------------------------------------------------------------------ LiteRT graphs


class Graph:
  """One .tflite on the LiteRT CompiledModel, inputs and outputs by signature name."""

  def __init__(self, path, accelerator="cpu", precision="fp32", threads=4):
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator
    from ai_edge_litert.options import CpuOptions, GpuOptions, Options
    if accelerator == "cpu":
      options = Options(hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads))
    elif accelerator == "gpu":
      options = Options(hardware_accelerators=HardwareAccelerator.GPU,
                        gpu_options=GpuOptions(enforce_f32=precision == "fp32"))
    else:
      raise ValueError(accelerator)
    t0 = time.perf_counter()
    self.model = CompiledModel.from_file(path, options=options)
    self.compile_ms = (time.perf_counter() - t0) * 1e3
    sigs = self.model.get_signature_list()
    self.key = next(iter(sigs))
    self.inputs = {n: self.model.create_input_buffer_by_name(self.key, n) for n in sigs[self.key]["inputs"]}
    self.outputs = {n: self.model.create_output_buffer_by_name(self.key, n) for n in sigs[self.key]["outputs"]}
    details = self.model.get_output_tensor_details(self.key)
    self.shapes = {n: tuple(details[n]["shape"]) for n in self.outputs}

  def __call__(self, **feeds):
    for name, x in feeds.items():
      self.inputs[name].write(np.ascontiguousarray(x, F32))
    self.model.run_by_name(self.key, self.inputs, self.outputs)
    return {n: b.read(int(np.prod(self.shapes[n])), F32).reshape(self.shapes[n]) for n, b in self.outputs.items()}


# ------------------------------------------------------------------------------------------------ the host loop


class Nemotron3Diarizer:
  """Streaming ([push] / [finish], low_latency) or whole-file ([run_file], offline) diarization."""

  def __init__(self, model_dir=None, mode="low_latency", accelerator="cpu", precision="fp32", threads=4):
    model_dir = model_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    self.mode = MODES[mode]
    assets = os.path.join(model_dir, "assets")
    self.mel = MelFrontend(_floats(os.path.join(assets, "frontend_mel128_257.bin")),
                           _floats(os.path.join(assets, "hann400.bin")))
    self.silence = _floats(os.path.join(assets, "silence_embeds.bin"))
    # graph A always in fp32: its rows stay in the speaker cache and FIFO for the whole session
    self.frontend = Graph(os.path.join(model_dir, FRONTEND_FILE), accelerator, "fp32", threads)
    self.encoder = Graph(os.path.join(model_dir, self.mode.encoder_file), accelerator, precision, threads)
    self.rope_cos, self.rope_sin = rope_tables(self.mode.max_rows)
    self.reset()

  def reset(self):
    """Starts a new stream / file."""
    self.mel.reset()
    self.cache = SpeakerCache(self.mode.fifo_length, self.mode.update_period, self.silence)
    self.next_chunk = 0
    self.finished = False

  # -------------------------------------------------------------------------------------------- streaming

  def _chunk_start(self, k):
    return k * self.mode.chunk * SUBSAMPLING * HOP - N_FFT // 2

  def push(self, samples):
    """Appends 16 kHz samples; runs and returns every step they complete (low_latency)."""
    assert self.mode.name == "low_latency" and not self.finished
    self.mel.append(samples)
    per_chunk = (self.mode.chunk + self.mode.right_context) * SUBSAMPLING
    first_samples = (per_chunk - 1) * HOP + WIN // 2  # 16680: the first chunk uses centered windows
    chunk_samples = per_chunk * HOP + WIN  # 17040
    steps = []
    while True:
      k = self.next_chunk
      n = self.mel.num_samples
      if not (n >= first_samples if k == 0 else self._chunk_start(k) + chunk_samples <= n):
        return steps
      steps.append(self._run_chunk(k, per_chunk, self.mode.right_context))

  def finish(self):
    """Ends the stream: the remaining audio is the last chunk, every frame scored (no look-ahead)."""
    assert self.mode.name == "low_latency" and not self.finished
    self.finished = True
    k, n = self.next_chunk, self.mel.num_samples
    frames = n // HOP if k == 0 else (n - self._chunk_start(k) - N_FFT) // HOP + 1
    return [self._run_chunk(k, frames, 0)] if frames >= 1 else []

  def _run_chunk(self, k, frames, lookahead):
    t0 = time.perf_counter()
    g0 = k * self.mode.chunk * SUBSAMPLING
    mel = np.zeros((FRONTEND_FRAMES, N_MELS), F32)
    mel[:frames] = self.mel.frames(g0, frames)
    t1 = time.perf_counter()
    emb = self.frontend(mel=mel[None])["chunk_embeds"][0]
    t2 = time.perf_counter()
    num_embeds = -(-frames // SUBSAMPLING)
    step = self._encode(k, g0, frames, emb[:num_embeds], num_embeds - lookahead, None, (t0, t1, t2))
    self.next_chunk = k + 1
    return step

  def _encode(self, k, g0, frames, embeds, num_chunk, valid, times):
    """[cache rows | embeds] -> graph B -> cache update -> the logits of the num_chunk chunk rows."""
    t0, t1, t2 = times
    cached = self.cache.rows()
    c = cached.shape[0]
    length = c + embeds.shape[0]
    t = self.mode.max_rows
    assert length <= t, (k, length, t)
    packed = np.zeros((1, t, HIDDEN), F32)
    packed[0, :c] = cached
    packed[0, c:length] = embeds
    bias = np.full((1, 1, 1, t), self.mode.pad_bias, F32)
    bias[..., :length] = 0
    row_valid = None
    if valid is not None:
      row_valid = np.concatenate([np.ones(c, bool), valid])
      bias[0, 0, 0, :length][~row_valid] = self.mode.masked_bias
    t3 = time.perf_counter()
    logits = self.encoder(packed_embeds=packed, attn_bias=bias, rope_cos=self.rope_cos,
                          rope_sin=self.rope_sin)["logits"][0, : length * SUBSAMPLING]
    t4 = time.perf_counter()
    self.cache.update(packed[0, :length], logits, num_chunk, row_valid)
    out = logits[c * SUBSAMPLING : c * SUBSAMPLING + min(num_chunk * SUBSAMPLING, frames)]
    t5 = time.perf_counter()
    ms = dict(mel=(t1 - t0) * 1e3, frontend=(t2 - t1) * 1e3, encoder=(t4 - t3) * 1e3,
              cache=((t3 - t2) + (t5 - t4)) * 1e3, total=(t5 - t0) * 1e3)
    return Step(k, g0, length, out, ms)

  # -------------------------------------------------------------------------------------------- file mode

  def run_file(self, samples):
    """Offline pass over a whole file: centered frames 0 .. N/160 (the last one masked and zero, as the processor
    returns it), graph A over 104-frame blocks, chunks of 340 encoder frames + up to 40 look-ahead frames."""
    assert self.mode.name == "offline" and self.next_chunk == 0 and not self.finished
    self.finished = True
    t0 = time.perf_counter()
    self.mel.append(samples)
    valid = self.mel.num_samples // HOP
    frames = valid + 1
    num_embeds = -(-frames // SUBSAMPLING)
    embeds = np.zeros((num_embeds, HIDDEN), F32)
    embed_valid = np.arange(num_embeds) * SUBSAMPLING < valid
    mel_all = self.mel.frames(0, valid)
    t1 = time.perf_counter()
    block = FRONTEND_FRAMES // SUBSAMPLING
    for b in range(-(-num_embeds // block)):
      mel = np.zeros((FRONTEND_FRAMES, N_MELS), F32)
      part = mel_all[b * FRONTEND_FRAMES : (b + 1) * FRONTEND_FRAMES]  # the masked frame stays zero
      mel[: part.shape[0]] = part
      e = self.frontend(mel=mel[None])["chunk_embeds"][0]
      ne = min(block, num_embeds - b * block)
      embeds[b * block : b * block + ne] = e[:ne]
    t2 = time.perf_counter()
    steps, start, k = [], 0, 0
    while start < num_embeds:
      end = min(start + self.mode.chunk, num_embeds)
      hi = min(end + self.mode.right_context, num_embeds)
      g0 = start * SUBSAMPLING
      times = (t0, t1, t2) if k == 0 else (time.perf_counter(),) * 3
      steps.append(self._encode(k, g0, min(frames - g0, (hi - start) * SUBSAMPLING), embeds[start:hi],
                                end - start, embed_valid[start:hi], times))
      start, k = end, k + 1
    self.next_chunk = k
    return steps


def diarize(wav_path, mode="low_latency", precision="fp32", accelerator="cpu", model_dir=None, threshold=0.5,
            diarizer=None):
  """Speech segments of a 16 kHz file: [{'Start': s, 'End': s, 'Speaker': k}], speakers numbered by arrival."""
  audio = load_wav(wav_path)
  d = diarizer or Nemotron3Diarizer(model_dir, mode, accelerator, precision)
  d.reset()
  if d.mode.name == "offline":
    steps = d.run_file(audio)
    valid = audio.shape[0] // HOP  # the processor masks the frame after the last full hop
  else:
    steps = []
    for i in range(0, audio.shape[0], 1600):  # 0.1 s pushes, as a microphone delivers them
      steps += d.push(audio[i : i + 1600])
    steps += d.finish()
    valid = None
  logits = np.concatenate([s.logits for s in steps], 0)
  return speaker_segments(logits, threshold, valid)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("wav", help="16 kHz audio file")
  ap.add_argument("--mode", choices=sorted(MODES), default="low_latency")
  ap.add_argument("--model-dir", default=None, help="repo root with the .tflite files and assets/")
  ap.add_argument("--accelerator", choices=["cpu", "gpu"], default="cpu")
  ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32", help="graph B GPU precision")
  ap.add_argument("--threads", type=int, default=4, help="CPU threads")
  ap.add_argument("--json", default=None, help="write the segments here")
  args = ap.parse_args()
  d = Nemotron3Diarizer(args.model_dir, args.mode, args.accelerator, args.precision, args.threads)
  t0 = time.perf_counter()
  segs = diarize(args.wav, diarizer=d)
  sec = time.perf_counter() - t0
  for s in segs:
    print(f"speaker_{s['Speaker']}: {s['Start']:.2f}s - {s['End']:.2f}s")
  audio_s = d.mel.num_samples / SAMPLE_RATE
  print(f"# {len(segs)} segments, {len({s['Speaker'] for s in segs})} speakers, {audio_s:.1f} s audio in "
        f"{sec:.1f} s ({args.mode}, {args.accelerator})")
  if args.json:
    with open(args.json, "w") as f:
      json.dump(segs, f, indent=1)


if __name__ == "__main__":
  main()
