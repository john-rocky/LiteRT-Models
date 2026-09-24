"""Transformers fp32 reference for nvidia/Nemotron-3-Diarization (the round-1 gate source).

Runs the transformers implementation (git main) in fp32 on CPU over each fixture wav:
  (a) offline      - the whole recording in one forward (chunk 340 + look-ahead 40, FIFO 40)
  (b) low_latency  - the README `inputs_generator` chunking (chunk 9 + look-ahead 4), one forward
                     per chunk, the last chunk with `is_last_audio_chunk=True`
  (c) per-step capture of (b) through forward hooks: the packed encoder input (`chunk_input_embeds`),
      its valid length L, `num_lookahead_frames`, the chunk logits [1, L*8, 8] and the
      speaker-cache state before and after the step.

Outputs in <run>/results/:
  ref_<fixture>_offline.npz, ref_<fixture>_low_latency.npz, chunk_io_<fixture>.npz, make_reference.json

--offline-capture (round 3) instead captures (a) per chunk of the offline forward's own loop (chunk 340 + look-ahead
40, FIFO 40, update period 300): the packed encoder input, its valid rows (the step mask: the offline pass masks
the frame after the last full hop), the chunk logits, the cache state before / after and the chunk frame count;
plus the whole-recording embeddings and features -> chunk_io_<fixture>_offline.npz, make_reference_offline.json.

Usage:
  .venv/bin/python make_reference.py --run-dir <run> --model-dir <dir with config.json,
      processor_config.json, model.safetensors> [--fixtures a.wav b.wav]
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
import transformers
from transformers import AutoProcessor, Nemotron3DiarizationForAudioFrameClassification

T_MAX = 541  # 264 speaker cache + 264 FIFO + 9 chunk + 4 look-ahead


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
      h.update(block)
  return h.hexdigest()


def transformers_commit():
  dist = os.path.join(
      os.path.dirname(os.path.dirname(transformers.__file__)),
      f"transformers-{transformers.__version__}.dist-info",
      "direct_url.json",
  )
  try:
    with open(dist) as f:
      return json.load(f)["vcs_info"]["commit_id"]
  except (OSError, KeyError):
    return "unknown"


def load_wav(path):
  audio, sr = sf.read(path, dtype="float32")
  assert sr == 16000 and audio.ndim == 1, (path, sr, audio.shape)
  return audio


def segments_to_arrays(segments):
  return (
      np.array([s["Start"] for s in segments], np.float64),
      np.array([s["End"] for s in segments], np.float64),
      np.array([s["Speaker"] for s in segments], np.int32),
  )


def run_offline(model, processor, audio):
  inputs = processor(audio, sampling_rate=16000)
  inputs = inputs.to("cpu", dtype=torch.float32)
  with torch.inference_mode():
    logits = model(**inputs).logits
  segments = processor.extract_speaker_dict(logits, inputs.attention_mask, threshold=0.5)[0]
  return inputs, logits, segments


def inputs_generator(processor, audio):
  """The README streaming generator, verbatim."""
  yield processor(
      audio[: processor.num_samples_first_audio_chunk],
      sampling_rate=16000,
      is_streaming=True,
      is_first_audio_chunk=True,
  ), True, False

  mel_frame_idx = processor.num_mel_frames_per_step
  start_idx = processor.audio_chunk_start(mel_frame_idx)
  while (end_idx := start_idx + processor.num_samples_per_audio_chunk) <= audio.shape[0]:
    yield processor(
        audio[start_idx:end_idx],
        sampling_rate=16000,
        is_streaming=True,
        is_first_audio_chunk=False,
    ), False, False
    mel_frame_idx += processor.num_mel_frames_per_step
    start_idx = processor.audio_chunk_start(mel_frame_idx)

  yield processor(
      audio[start_idx:],
      sampling_rate=16000,
      is_streaming=True,
      is_first_audio_chunk=False,
      is_last_audio_chunk=True,
  ), False, True


def cache_state(cache):
  if cache is None:
    return 0, 0, False
  return int(cache.num_cache_frames), int(cache.num_fifo_frames), bool(cache.is_compressed)


def run_streaming(model, processor, audio, capture):
  """Streams `audio` chunk by chunk; with `capture`, records every step's encoder I/O."""
  step = {}
  hooks = []
  if capture:

    def encoder_pre_hook(module, args, kwargs):
      step["packed"] = kwargs["inputs_embeds"].detach().clone()
      step["position_ids"] = kwargs["position_ids"].detach().clone()
      mask = kwargs.get("attention_mask")
      step["mask_all_valid"] = bool(mask is None or bool(mask.all()))
      step["encoder_calls"] = step.get("encoder_calls", 0) + 1

    def classifier_hook(module, args, output):
      step["chunk_logits"] = output.detach().clone()

    hooks.append(model.model.register_forward_pre_hook(encoder_pre_hook, with_kwargs=True))
    hooks.append(model.classifier.register_forward_hook(classifier_hook))

  records = []
  speaker_cache, logits = None, []
  with torch.inference_mode():
    for inputs, is_first, is_last in inputs_generator(processor, audio):
      inputs = inputs.to("cpu", dtype=torch.float32)
      pre = cache_state(speaker_cache)
      step.clear()
      outputs = model(**inputs, speaker_cache=speaker_cache)
      logits.append(outputs.logits)
      speaker_cache = outputs.speaker_cache
      if capture:
        assert step["encoder_calls"] == 1, step["encoder_calls"]
        packed = step["packed"]
        length = packed.shape[1]
        position_ids = step["position_ids"][0].numpy()
        assert np.array_equal(position_ids, np.arange(length)), "positions must restart at 0"
        assert step["chunk_logits"].shape == (1, length * 8, 8), step["chunk_logits"].shape
        records.append(
            dict(
                input_features=inputs["input_features"][0].numpy(),
                num_lookahead_frames=int(inputs.get("num_lookahead_frames", 0) or 0),
                is_first=is_first,
                is_last=is_last,
                pre=pre,
                post=cache_state(speaker_cache),
                packed=packed[0].numpy(),
                length=length,
                mask_all_valid=step["mask_all_valid"],
                chunk_logits=step["chunk_logits"][0].numpy(),
                out_logits=outputs.logits[0].numpy(),
            )
        )
  for h in hooks:
    h.remove()
  logits = torch.cat(logits, dim=1)
  segments = processor.extract_speaker_dict(logits, threshold=0.5)[0]
  return logits, segments, records


def run_offline_capture(model, processor, audio):
  """The offline forward with every internal chunk captured (encoder input, step mask, chunk logits, cache state)."""
  from transformers.models.nemotron3_diarization import modeling_nemotron3_diarization as mod
  cur, records, full = {}, [], {}

  def embedder_hook(module, args, output):
    full.setdefault("embeds", output.detach().clone())

  def encoder_pre_hook(module, args, kwargs):
    cur["packed"] = kwargs["inputs_embeds"].detach().clone()
    mask = kwargs.get("attention_mask")
    cur["mask"] = None if mask is None else mask.detach().clone()
    cur["position_ids"] = kwargs["position_ids"].detach().clone()

  def classifier_hook(module, args, output):
    cur["chunk_logits"] = output.detach().clone()

  original_update = mod.Nemotron3DiarizationSpeakerCache.update

  def update(self, chunk_input_embeds, chunk_logits, silence_embeds, num_chunk_frames, mask=None):
    pre = (int(self.num_cache_frames), int(self.num_fifo_frames), bool(self.is_compressed))
    original_update(self, chunk_input_embeds, chunk_logits, silence_embeds, num_chunk_frames, mask=mask)
    post = (int(self.num_cache_frames), int(self.num_fifo_frames), bool(self.is_compressed))
    records.append(dict(cur, pre=pre, post=post, num_chunk_frames=int(num_chunk_frames),
                        fifo_length=self.fifo_length, update_period=self.speaker_cache_update_period))

  hooks = [model.model.audio_tower.embedder.register_forward_hook(embedder_hook),
           model.model.register_forward_pre_hook(encoder_pre_hook, with_kwargs=True),
           model.classifier.register_forward_hook(classifier_hook)]
  mod.Nemotron3DiarizationSpeakerCache.update = update
  try:
    inputs = processor(audio, sampling_rate=16000).to("cpu", dtype=torch.float32)
    with torch.inference_mode():
      logits = model(**inputs).logits
  finally:
    mod.Nemotron3DiarizationSpeakerCache.update = original_update
    for h in hooks:
      h.remove()
  return inputs, logits, full["embeds"], records


def save_chunk_io_offline(path, inputs, logits, embeds, records, silence_embeds, t_max=684):
  n = len(records)
  packed = np.zeros((n, t_max, 512), np.float32)
  chunk_logits = np.zeros((n, t_max * 8, 8), np.float32)
  row_valid = np.zeros((n, t_max), np.bool_)
  length = np.zeros(n, np.int32)
  num_chunk = np.zeros(n, np.int32)
  pre = np.zeros((n, 3), np.int32)
  post = np.zeros((n, 3), np.int32)
  outs = []
  for i, r in enumerate(records):
    L = r["packed"].shape[1]
    assert L <= t_max, L
    assert np.array_equal(r["position_ids"][0].numpy(), np.arange(L)), "positions must restart at 0"
    packed[i, :L] = r["packed"][0].numpy()
    chunk_logits[i, : L * 8] = r["chunk_logits"][0].numpy()
    row_valid[i, :L] = True if r["mask"] is None else r["mask"][0].numpy().astype(bool)
    length[i], num_chunk[i] = L, r["num_chunk_frames"]
    pre[i], post[i] = r["pre"], r["post"]
    c = r["pre"][0] + r["pre"][1]
    outs.append(r["chunk_logits"][0, c * 8 : (c + r["num_chunk_frames"]) * 8].numpy())
  frames = inputs["input_features"].shape[1]
  concat = np.concatenate(outs, 0)[:frames]
  self_check = float(np.abs(concat - logits[0].numpy()).max())
  np.savez(
      path,
      packed_embeds=packed, length=length, row_valid=row_valid, chunk_logits=chunk_logits,
      num_chunk_frames=num_chunk, cache_pre=pre, cache_post=post,
      cache_names=np.array(["num_cache_frames", "num_fifo_frames", "is_compressed"]),
      embeds=embeds[0].numpy(), input_features=inputs["input_features"][0].numpy(),
      attention_mask=inputs["attention_mask"][0].numpy(), logits=logits[0].numpy(), silence_embeds=silence_embeds,
      fifo_length=np.int32(records[0]["fifo_length"]), update_period=np.int32(records[0]["update_period"]),
  )
  return dict(chunks=n, lengths=length.tolist(), masked_rows=[int((~row_valid[i, : length[i]]).sum()) for i in range(n)],
              num_chunk_frames=num_chunk.tolist(), cache_pre=pre.tolist(), cache_post=post.tolist(),
              frames=int(frames), embeds=int(embeds.shape[1]), concat_vs_logits_max_abs=self_check,
              fifo_length=int(records[0]["fifo_length"]), update_period=int(records[0]["update_period"]))


def save_chunk_io(path, records, silence_embeds):
  n = len(records)
  packed = np.zeros((n, T_MAX, 512), np.float32)
  chunk_logits = np.zeros((n, T_MAX * 8, 8), np.float32)
  mel = np.zeros((n, 104, 128), np.float32)
  out_logits = np.zeros((n, 104, 8), np.float32)
  length = np.zeros(n, np.int32)
  mel_frames = np.zeros(n, np.int32)
  out_frames = np.zeros(n, np.int32)
  lookahead = np.zeros(n, np.int32)
  flags = np.zeros((n, 3), np.bool_)  # is_first, is_last, mask_all_valid
  pre = np.zeros((n, 3), np.int32)  # num_cache_frames, num_fifo_frames, is_compressed
  post = np.zeros((n, 3), np.int32)
  for i, r in enumerate(records):
    L = r["length"]
    assert L <= T_MAX, L
    packed[i, :L] = r["packed"]
    chunk_logits[i, : L * 8] = r["chunk_logits"]
    f = r["input_features"].shape[0]
    assert f <= 104, f
    mel[i, :f] = r["input_features"]
    o = r["out_logits"].shape[0]
    out_logits[i, :o] = r["out_logits"]
    length[i], mel_frames[i], out_frames[i] = L, f, o
    lookahead[i] = r["num_lookahead_frames"]
    flags[i] = (r["is_first"], r["is_last"], r["mask_all_valid"])
    pre[i] = r["pre"]
    post[i] = r["post"]
  np.savez(
      path,
      packed_embeds=packed,
      length=length,
      chunk_logits=chunk_logits,
      input_features=mel,
      mel_frames=mel_frames,
      out_logits=out_logits,
      out_frames=out_frames,
      num_lookahead_frames=lookahead,
      flags=flags,
      flags_names=np.array(["is_first", "is_last", "mask_all_valid"]),
      cache_pre=pre,
      cache_post=post,
      cache_names=np.array(["num_cache_frames", "num_fifo_frames", "is_compressed"]),
      silence_embeds=silence_embeds,
  )


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--model-dir", required=True)
  ap.add_argument("--fixtures", nargs="+", required=True)
  ap.add_argument("--eager-floor", action="store_true",
                  help="also stream with attn_implementation='eager' and record max|dlogit| vs default")
  ap.add_argument("--offline-capture", action="store_true",
                  help="only capture the offline forward's internal chunks (chunk_io_<fixture>_offline.npz)")
  args = ap.parse_args()
  results = os.path.join(args.run_dir, "results")
  os.makedirs(results, exist_ok=True)

  torch.manual_seed(0)
  meta = dict(
      transformers_version=transformers.__version__,
      transformers_commit=transformers_commit(),
      torch_version=torch.__version__,
      model_dir=os.path.realpath(args.model_dir),
      model_safetensors_sha256=sha256(os.path.join(args.model_dir, "model.safetensors")),
  )
  processor = AutoProcessor.from_pretrained(args.model_dir)
  model = Nemotron3DiarizationForAudioFrameClassification.from_pretrained(
      args.model_dir, dtype=torch.float32
  ).eval()
  meta["attn_implementation"] = model.config._attn_implementation
  meta["streaming_mode"] = processor.streaming_mode
  meta["num_samples_first_audio_chunk"] = processor.num_samples_first_audio_chunk
  meta["num_samples_per_audio_chunk"] = processor.num_samples_per_audio_chunk
  silence = model.silence_embeds.detach().numpy().copy()
  print(json.dumps(meta, indent=1), flush=True)

  if args.offline_capture:
    meta["fixtures"] = {}
    for wav in args.fixtures:
      name = os.path.splitext(os.path.basename(wav))[0]
      audio = load_wav(wav)
      inputs, logits, embeds, records = run_offline_capture(model, processor, audio)
      ref = np.load(os.path.join(results, f"ref_{name}_offline.npz"))
      fx = save_chunk_io_offline(os.path.join(results, f"chunk_io_{name}_offline.npz"), inputs, logits, embeds,
                                 records, silence)
      fx["logits_vs_ref_offline_npz_max_abs"] = float(np.abs(logits[0].numpy() - ref["logits"]).max())
      meta["fixtures"][name] = fx
      print(json.dumps({name: fx}), flush=True)
    with open(os.path.join(results, "make_reference_offline.json"), "w") as f:
      json.dump(meta, f, indent=1)
    print("DONE", flush=True)
    return

  eager = None
  if args.eager_floor:
    eager = Nemotron3DiarizationForAudioFrameClassification.from_pretrained(
        args.model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()

  meta["fixtures"] = {}
  for wav in args.fixtures:
    name = os.path.splitext(os.path.basename(wav))[0]
    audio = load_wav(wav)
    fx = dict(path=os.path.realpath(wav), sha256=sha256(wav), samples=int(audio.shape[0]),
              seconds=audio.shape[0] / 16000)

    t0 = time.time()
    inputs, logits, segments = run_offline(model, processor, audio)
    fx["offline_seconds"] = round(time.time() - t0, 2)
    s, e, k = segments_to_arrays(segments)
    np.savez(
        os.path.join(results, f"ref_{name}_offline.npz"),
        logits=logits[0].numpy(),
        probabilities=torch.sigmoid(logits)[0].numpy(),
        input_features=inputs["input_features"][0].numpy(),
        attention_mask=inputs["attention_mask"][0].numpy(),
        seg_start=s, seg_end=e, seg_speaker=k,
        transformers_commit=np.array(meta["transformers_commit"]),
    )
    fx["offline_frames"] = int(logits.shape[1])
    fx["offline_segments"] = len(segments)

    t0 = time.time()
    s_logits, s_segments, records = run_streaming(model, processor, audio, capture=True)
    fx["low_latency_seconds"] = round(time.time() - t0, 2)
    s, e, k = segments_to_arrays(s_segments)
    mel_chunks = [r["input_features"] for r in records]
    np.savez(
        os.path.join(results, f"ref_{name}_low_latency.npz"),
        logits=s_logits[0].numpy(),
        probabilities=torch.sigmoid(s_logits)[0].numpy(),
        mel_chunks=np.concatenate(mel_chunks, 0),
        mel_chunk_frames=np.array([m.shape[0] for m in mel_chunks], np.int32),
        seg_start=s, seg_end=e, seg_speaker=k,
        transformers_commit=np.array(meta["transformers_commit"]),
    )
    save_chunk_io(os.path.join(results, f"chunk_io_{name}.npz"), records, silence)
    lengths = [r["length"] for r in records]
    compressed_steps = [i for i, r in enumerate(records) if r["post"][2]]
    first_compress = next((i for i, r in enumerate(records) if r["post"][2] and not r["pre"][2]), None)
    fx.update(
        low_latency_frames=int(s_logits.shape[1]),
        low_latency_segments=len(s_segments),
        steps=len(records),
        max_length=int(max(lengths)),
        steps_with_length_541=[i for i, L in enumerate(lengths) if L == T_MAX],
        first_compress_step=first_compress,
        num_steps_compressed_after=len(compressed_steps),
        all_masks_valid=all(r["mask_all_valid"] for r in records),
    )
    if eager is not None:
      e_logits, _, _ = run_streaming(eager, processor, audio, capture=False)
      fx["eager_vs_default_low_latency_max_abs_dlogit"] = float((e_logits - s_logits).abs().max())
    meta["fixtures"][name] = fx
    print(json.dumps({name: fx}, indent=1), flush=True)

  with open(os.path.join(results, "make_reference.json"), "w") as f:
    json.dump(meta, f, indent=1)
  print("DONE", flush=True)


if __name__ == "__main__":
  main()
