"""Gate: the app's Kotlin host (MelFrontend / SpeakerCache / Nemotron3Diarizer) on the Mac JVM vs transformers.

The Kotlin sources under app/src/main/java/com/nemotron3diar are compiled with kotlinc together with
scripts/jvm/KotlinParity.kt and run with a replay engine: graph A returns the transformers chunk rows of the step and
graph B the transformers chunk_logits (results/chunk_io_<fixture>.npz), so everything the host computes itself --
the log-mel, the chunk schedule, the packing, the speaker cache / FIFO / compression and the emitted rows -- is
compared with what transformers did, step by step. The wav is pushed 1600 samples at a time, like a microphone.

  (a) mel      Kotlin log-mel of every chunk vs the processor's input_features:  max|d| <= 1e-4 (log)
  (b) replay   the packed input of every step vs transformers' chunk_input_embeds:  max|d| <= 1e-5, and the cache
               state (num_cache_frames, num_fifo_frames, is_compressed) before / after every step equal;
               compressions whose kept frames differ are listed with the top-k boundary margin (a 1-ulp tie is
               recorded, not failed)
  also         the chunk schedule vs host_loop.schedule, the emitted logits vs the streaming reference, the RoPE
               tables vs nemotron3diar_model.rope_tables (T = 541 and 684), the bundled assets vs their sources.
  offline      the file mode (Nemotron3Diarizer.runFile, T=684) on results/chunk_io_<fixture>_offline.npz: the
               Kotlin centered mel of the whole file (masked last frame zero) vs the processor, the packed input
               and two-level attn_bias of every chunk, the cache state, and the emitted logits.
Writes results/kotlin_parity.json.

Usage: .venv/bin/python gate_kotlin_parity.py --run-dir <run> --model-dir <dir>
           [--fixtures diarization_example_16k test_multispk_16k] [--kotlinc /opt/homebrew/bin/kotlinc]
"""

import argparse
import glob
import json
import os
import subprocess
import time

import numpy as np

import nemotron3diar_model as n3d
from host_loop import schedule

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.dirname(HERE)
# the app sits next to scripts/ in the sample module and under android/ in the model repo (conversion/)
APP = next((p for p in (os.path.join(MODULE, "app"), os.path.join(MODULE, "android", "app")) if os.path.isdir(p)),
           os.path.join(MODULE, "app"))
APP_SRC = os.path.join(APP, "src", "main", "java", "com", "nemotron3diar")
ASSETS = os.path.join(APP, "src", "main", "assets")
HOST_SOURCES = ["MelFrontend.kt", "SpeakerCache.kt", "Nemotron3Diarizer.kt", "WavReader.kt", "StepLog.kt"]
MEL_MAX = 1e-4
PACKED_MAX = 1e-5
T = 541


def load_npz(path):
  """Every array read once (an NpzFile re-reads and decompresses the member on each [] access)."""
  with np.load(path) as z:
    return {k: z[k] for k in z.files}


def f32(path, arr):
  np.ascontiguousarray(arr, dtype="<f4").tofile(path)


def install_assets(run_dir, cio):
  """The three host tables the app bundles; written from their sources and checked bit-exact."""
  os.makedirs(ASSETS, exist_ok=True)
  exports = os.path.join(run_dir, "exports")
  out = {}
  for name, src in (("frontend_mel128_257.bin", np.fromfile(os.path.join(exports, "frontend_mel128_257.bin"), "<f4")),
                    ("hann400.bin", np.fromfile(os.path.join(exports, "hann400.bin"), "<f4")),
                    ("silence_embeds.bin", cio["silence_embeds"].astype(np.float32))):
    path = os.path.join(ASSETS, name)
    if not (os.path.exists(path) and np.array_equal(np.fromfile(path, "<f4"), src)):
      f32(path, src)
    out[name] = dict(floats=int(src.size), bit_exact=bool(np.array_equal(np.fromfile(path, "<f4"), src)))
  return out


def build_jar(run_dir, kotlinc):
  jvm = os.path.join(run_dir, "jvm")
  os.makedirs(jvm, exist_ok=True)
  jar = os.path.join(jvm, "parity.jar")
  sources = [os.path.join(APP_SRC, s) for s in HOST_SOURCES] + [os.path.join(HERE, "jvm", "KotlinParity.kt")]
  if not os.path.exists(jar) or os.path.getmtime(jar) < max(os.path.getmtime(s) for s in sources):
    t0 = time.time()
    r = subprocess.run([kotlinc, *sources, "-include-runtime", "-d", jar], capture_output=True, text=True)
    if r.returncode != 0:
      raise SystemExit("kotlinc failed:\n" + r.stderr[-4000:])
    print(f"kotlinc {len(sources)} files -> {jar} in {time.time() - t0:.0f}s", flush=True)
  return jar


def java(jar, *args):
  r = subprocess.run(["java", "-Xmx8g", "-cp", jar, "com.nemotron3diar.KotlinParityKt", *args],
                     capture_output=True, text=True)
  if r.returncode != 0:
    raise SystemExit("java failed:\n" + r.stdout[-2000:] + r.stderr[-4000:])
  return r.stdout.strip()


def export_reference(cio, out_dir):
  """ref_rows.bin [n,13,512]: each step's chunk rows (graph A output) as transformers packed them;
  ref_logits.bin [n,4328,8]: transformers chunk_logits (zero tail)."""
  os.makedirs(out_dir, exist_ok=True)
  n = cio["length"].shape[0]
  rows = np.zeros((n, 13, 512), np.float32)
  for i in range(n):
    L = int(cio["length"][i])
    k = -(-int(cio["mel_frames"][i]) // 8)
    rows[i, :k] = cio["packed_embeds"][i, L - k : L]
  f32(os.path.join(out_dir, "ref_rows.bin"), rows)
  f32(os.path.join(out_dir, "ref_logits.bin"), cio["chunk_logits"])


def reference_row_ids(cio):
  """Frame ids of the rows transformers fed at every step (cache + FIFO), recovered by exact row matching against
  the chunk rows of earlier steps (gather copies rows bit-exactly). A row several frames share (identical mel,
  e.g. digital silence) maps to all of them. The silence embedding is id -1."""
  n = cio["length"].shape[0]
  owner = {cio["silence_embeds"].astype(np.float32).tobytes(): {-1}}
  ids = []
  for i in range(n):
    nc, nf = int(cio["cache_pre"][i, 0]), int(cio["cache_pre"][i, 1])
    step_ids = []
    for r in range(nc + nf):
      key = cio["packed_embeds"][i, r].tobytes()
      step_ids.append(sorted(owner[key]) if key in owner else None)
    ids.append(step_ids)
    c = nc + nf
    k = -(-int(cio["mel_frames"][i]) // 8) - int(cio["num_lookahead_frames"][i])
    for j in range(k):
      owner.setdefault(cio["packed_embeds"][i, c + j].tobytes(), set()).add(9 * i + j)
  return ids


def compare_ids(ref_ids, got_ids):
  """Steps whose fed rows carry different frame ids (a row matches if its id is among the reference row's ids)."""
  bad = []
  for i, (r, g) in enumerate(zip(ref_ids, got_ids)):
    if len(r) != len(g):
      bad.append(dict(step=i, ref_rows=len(r), got_rows=len(g)))
      continue
    diff = [j for j, (a, b) in enumerate(zip(r, g)) if a is None or b not in a]
    if diff:
      bad.append(dict(step=i, rows=len(r), differing_rows=len(diff),
                      examples=[dict(row=j, ref=r[j], got=g[j]) for j in diff[:8]]))
  return bad


def ulps(a, b):
  """Distance in fp32 ulps between two finite floats (inf if either is not finite)."""
  a, b = np.float32(a), np.float32(b)
  if not (np.isfinite(a) and np.isfinite(b)):
    return float("inf")
  ia, ib = (int(np.array(x).view(np.int32)) for x in (a, b))
  ia = ia if ia >= 0 else -(ia & 0x7FFFFFFF)
  ib = ib if ib >= 0 else -(ib & 0x7FFFFFFF)
  return abs(ia - ib)


def compression_summary(comps):
  out = []
  for c in comps:
    sel = c["select_boundary"]
    bb = [b for b in c["boost_boundaries"] if b is not None]
    out.append(dict(step=c["step"], candidates=c["candidates"],
                    kept_silence_slots=int(sum(1 for x in c["kept_ids"] if x == -1)),
                    select_boundary=sel, select_gap_ulps=ulps(*sel),
                    min_boost_gap_ulps=min((ulps(*b) for b in bb), default=None)))
  return out


def replay_fixture(args, jar, fx, proc):
  results = os.path.join(args.run_dir, "results")
  cio = load_npz(os.path.join(results, f"chunk_io_{fx}.npz"))
  ref = load_npz(os.path.join(results, f"ref_{fx}_low_latency.npz"))
  n = cio["length"].shape[0]
  base = os.path.join(args.run_dir, "jvm", fx)
  export_reference(cio, os.path.join(base, "ref"))
  out_dir = os.path.join(base, "out")
  t0 = time.time()
  print(java(jar, "replay", ASSETS, os.path.join(args.run_dir, "fixtures", fx + ".wav"), os.path.join(base, "ref"),
             out_dir, "1600"), flush=True)
  secs = time.time() - t0
  run = json.load(open(os.path.join(out_dir, "steps.json")))
  steps = run["steps"]
  res = dict(steps=len(steps), ref_steps=n, jvm_seconds=round(secs, 1))
  assert len(steps) == n, (len(steps), n)

  # schedule
  audio_n = run["meta"]["samples"]
  sched = schedule(audio_n, proc)
  got = [(s["g0"], s["nf"], s["la"]) for s in steps]
  exp = [(g, f, r) for g, f, r, _ in sched]
  cio_sched = [(None, int(cio["mel_frames"][i]), int(cio["num_lookahead_frames"][i])) for i in range(n)]
  res["schedule"] = dict(equal_host_loop=got == exp,
                         equal_chunk_io=[(f, r) for _, f, r in got] == [(f, r) for _, f, r in cio_sched],
                         first=got[:2], last=got[-1])

  # (a) mel
  mel = np.fromfile(os.path.join(out_dir, "mel.bin"), "<f4").reshape(n, 104, 128)
  worst, worst_at, per_step, frames = 0.0, None, [], 0
  for i in range(n):
    f = int(cio["mel_frames"][i])
    d = np.abs(mel[i, :f] - cio["input_features"][i, :f])
    m = float(d.max())
    per_step.append(m)
    frames += f
    pad_zero = bool(np.all(mel[i, f:] == 0))
    if m > worst:
      r = np.unravel_index(int(d.argmax()), d.shape)
      worst, worst_at = m, dict(step=i, frame=int(steps[i]["g0"] + r[0]), mel_bin=int(r[1]),
                                ref=float(cio["input_features"][i, r[0], r[1]]), got=float(mel[i, r[0], r[1]]))
    if not pad_zero:
      res.setdefault("mel_pad_not_zero_steps", []).append(i)
  res["mel"] = dict(chunks=n, frames_compared=frames, max_abs=worst, worst=worst_at,
                    p99_step_max=float(np.percentile(per_step, 99)), PASS=bool(worst <= MEL_MAX))

  # (b) replay: packed rows, state, emitted logits
  packed = np.memmap(os.path.join(out_dir, "packed.bin"), dtype="<f4", mode="r", shape=(n, T, 512))
  packed_max, L_bad, state_bad, tail_bad, per = 0.0, [], [], [], []
  for i, s in enumerate(steps):
    L = int(cio["length"][i])
    if s["L"] != L:
      L_bad.append(i)
      continue
    d = float(np.abs(packed[i, :L] - cio["packed_embeds"][i, :L]).max())
    packed_max = max(packed_max, d)
    if np.any(packed[i, L:] != 0):
      tail_bad.append(i)
    if s["pre"] != [int(x) for x in cio["cache_pre"][i]] or s["post"] != [int(x) for x in cio["cache_post"][i]]:
      state_bad.append(i)
    per.append(dict(step=i, L=L, packed_max_abs=d, pre=s["pre"], post=s["post"]))
  out = np.fromfile(os.path.join(out_dir, "out.bin"), "<f4").reshape(-1, 8)
  logits_equal = out.shape == ref["logits"].shape and float(np.abs(out - ref["logits"]).max()) == 0.0
  ref_ids = reference_row_ids(cio)
  id_bad = compare_ids(ref_ids, [s["row_ids"] for s in steps])
  unmatched = sum(1 for r in ref_ids for x in r if x is None)
  comps = compression_summary(run["compressions"])
  res["replay"] = dict(
      packed_rows_max_abs=packed_max, steps_L_mismatch=L_bad, steps_state_mismatch=state_bad,
      steps_nonzero_tail=tail_bad, steps_row_ids_mismatch=[b["step"] for b in id_bad], row_id_mismatch_detail=id_bad[:5],
      reference_rows_unmatched=unmatched, emitted_logits_bit_equal_reference=bool(logits_equal),
      emitted_frames=int(out.shape[0]), ref_frames=int(ref["logits"].shape[0]),
      compressions=comps, first_compress_step=next((c["step"] for c in comps), None),
      PASS=bool(packed_max <= PACKED_MAX and not L_bad and not state_bad and not tail_bad),
      per_step=per)
  return res


def replay_offline_fixture(args, jar, fx):
  results = os.path.join(args.run_dir, "results")
  d = load_npz(os.path.join(results, f"chunk_io_{fx}_offline.npz"))
  t = n3d.T_OFFLINE
  base = os.path.join(args.run_dir, "jvm", fx + "_offline")
  ref_dir = os.path.join(base, "ref")
  os.makedirs(ref_dir, exist_ok=True)
  f32(os.path.join(ref_dir, "embeds.bin"), d["embeds"])
  f32(os.path.join(ref_dir, "chunk_logits.bin"), d["chunk_logits"])
  out_dir = os.path.join(base, "out")
  print(java(jar, "replay_offline", ASSETS, os.path.join(args.run_dir, "fixtures", fx + ".wav"), ref_dir, out_dir),
        flush=True)
  run = json.load(open(os.path.join(out_dir, "steps.json")))
  steps = run["steps"]
  n = d["length"].shape[0]
  feats = d["input_features"]
  frames = feats.shape[0]
  mel = np.fromfile(os.path.join(out_dir, "mel.bin"), "<f4").reshape(-1, 128)[:frames]
  valid = int(d["attention_mask"].sum())
  packed = np.fromfile(os.path.join(out_dir, "packed.bin"), "<f4").reshape(-1, t, 512)
  bias = np.fromfile(os.path.join(out_dir, "bias.bin"), "<f4").reshape(-1, t)
  out = np.fromfile(os.path.join(out_dir, "out.bin"), "<f4").reshape(-1, 8)
  packed_max, bias_bad, state_bad, L_bad = 0.0, [], [], []
  for i in range(min(n, len(steps))):
    L = int(d["length"][i])
    if steps[i]["L"] != L:
      L_bad.append(i)
      continue
    packed_max = max(packed_max, float(np.abs(packed[i, :L] - d["packed_embeds"][i, :L]).max()))
    if not np.array_equal(bias[i], n3d.attn_bias_for(L, t, valid=d["row_valid"][i, :L], two_level=True)[0, 0, 0]):
      bias_bad.append(i)
    if steps[i]["pre"] != [int(x) for x in d["cache_pre"][i]] or steps[i]["post"] != [int(x) for x in d["cache_post"][i]]:
      state_bad.append(i)
  res = dict(
      chunks=len(steps), ref_chunks=n, lengths=[s["L"] for s in steps],
      mel_frames=int(mel.shape[0]), ref_frames=int(frames), valid_frames=valid,
      mel_max_abs=float(np.abs(mel - feats).max()), masked_frame_zero=bool(np.all(mel[valid:] == 0)),
      packed_rows_max_abs=packed_max, steps_bias_mismatch=bias_bad, steps_state_mismatch=state_bad,
      steps_L_mismatch=L_bad,
      emitted_logits_bit_equal_reference=bool(out.shape == d["logits"].shape and np.array_equal(out, d["logits"])),
      compressions=len(run["compressions"]))
  res["PASS"] = bool(len(steps) == n and res["mel_max_abs"] <= MEL_MAX and packed_max <= PACKED_MAX and not bias_bad
                     and not state_bad and not L_bad and res["emitted_logits_bit_equal_reference"])
  return res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--model-dir", required=True)
  ap.add_argument("--fixtures", nargs="+", default=["diarization_example_16k", "test_multispk_16k"])
  ap.add_argument("--kotlinc", default="/opt/homebrew/bin/kotlinc")
  ap.add_argument("--offline-only", action="store_true", help="only the offline file-mode replay")
  args = ap.parse_args()
  results = os.path.join(args.run_dir, "results")
  proc = json.load(open(os.path.join(args.model_dir, "processor_config.json")))
  report = dict(thresholds=dict(mel_max_abs=MEL_MAX, packed_rows_max_abs=PACKED_MAX), sources=HOST_SOURCES)
  report["assets"] = install_assets(args.run_dir, load_npz(os.path.join(results, f"chunk_io_{args.fixtures[0]}.npz")))
  jar = build_jar(args.run_dir, args.kotlinc)
  report["java"] = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.splitlines()[0]

  rope_dir = os.path.join(args.run_dir, "jvm", "rope")
  report["rope"] = {}
  for t in (541, 684):
    java(jar, "rope", str(t), rope_dir)
    c_ref, s_ref = (x.numpy()[0, 0] for x in n3d.rope_tables(t))
    c = np.fromfile(os.path.join(rope_dir, f"rope_cos_{t}.bin"), "<f4").reshape(t, 64)
    s = np.fromfile(os.path.join(rope_dir, f"rope_sin_{t}.bin"), "<f4").reshape(t, 64)
    report["rope"][str(t)] = dict(cos_max_abs=float(np.abs(c - c_ref).max()), sin_max_abs=float(np.abs(s - s_ref).max()),
                                  cos_differing=int((c != c_ref).sum()), sin_differing=int((s != s_ref).sum()),
                                  elements=int(c.size))

  report_path = os.path.join(results, "kotlin_parity.json")
  if args.offline_only:
    report = json.load(open(report_path))
    jar = build_jar(args.run_dir, args.kotlinc)
  report["offline"] = {}
  for fx in args.fixtures:
    if os.path.exists(os.path.join(results, f"chunk_io_{fx}_offline.npz")):
      report["offline"][fx] = replay_offline_fixture(args, jar, fx)
      print(json.dumps({fx + "_offline": report["offline"][fx]}), flush=True)
  if args.offline_only:
    with open(report_path, "w") as f:
      json.dump(report, f, indent=1)
    print("KOTLIN_PARITY_OFFLINE", "PASS" if all(v["PASS"] for v in report["offline"].values()) else "FAIL", flush=True)
    return
  report["fixtures"] = {}
  ok = all(v["PASS"] for v in report["offline"].values())
  for fx in args.fixtures:
    res = replay_fixture(args, jar, fx, proc)
    report["fixtures"][fx] = res
    ok &= res["mel"]["PASS"] and res["replay"]["PASS"] and res["schedule"]["equal_host_loop"]
    brief = dict(mel=res["mel"], replay={k: v for k, v in res["replay"].items() if k not in ("per_step", "compressions")},
                 compressions=res["replay"]["compressions"], schedule=res["schedule"])
    print(json.dumps({fx: brief}, indent=1), flush=True)
  report["PASS"] = bool(ok)
  with open(os.path.join(results, "kotlin_parity.json"), "w") as f:
    json.dump(report, f, indent=1)
  print("KOTLIN_PARITY", "PASS" if ok else "FAIL", json.dumps(report["rope"]), flush=True)


if __name__ == "__main__":
  main()
