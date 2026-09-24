"""Device gate: the whole streaming host (Kotlin mel + graph A + graph B + speaker cache) on the S26 LiteRT GPU,
closed loop, vs the transformers streaming reference.

For each run (<b_precision>:<fixture>, one app launch each): pushes the wav to the app's files/wav (once), pushes
files/selftest.json {"mode": "closed_loop", ...}, checks that nothing else owns the device, `am start`s
com.nemotron3diar, captures `adb logcat --pid`, polls status.txt until DONE, pulls
getExternalFilesDir/n3d/closed_loop_<precision>_<fixture>/ and compares with results/ref_<fixture>_low_latency.npz
and results/chunk_io_<fixture>.npz:
  logits      the emitted rows of every step concatenated (all frames): agreement@0.5, flips, max|dp|, max|dlogit|
  segments    extract_speaker_dict port (threshold 0.5): boundary moves (10 ms frames), vanished / added
  state       (num_cache_frames, num_fifo_frames, is_compressed) before / after every step vs transformers
  compress    the frame ids of the rows every step was fed vs the ids transformers fed (recovered by exact row
              matching); a compression whose kept frames differ is listed with the device's top-k boundary gap
  timing      per-step median of mel / graph A / graph B / cache / total ms, RTF, first-chunk ms, compile ms,
              device conditions
Writes results/gate_closed_loop.json and results/closed_loop_timing.md. Only com.nemotron3diar is touched.

Usage: .venv/bin/python gate_closed_loop.py --run-dir <run> --runs fp32:diarization_example_16k
           default:diarization_example_16k accum:diarization_example_16k fp32:test_multispk_16k
"""

import argparse
import json
import math
import os
import subprocess
import time

import numpy as np

from gate_device import Adb, PKG, ACTIVITY, LOGCAT_KEEP, device_conditions, pidof, wait_device_free
from gate_kotlin_parity import compare_ids, load_npz, reference_row_ids, ulps
from host_loop import compare_segments, segments

EXT = f"/sdcard/Android/data/{PKG}/files/n3d"
TIMEOUT_S = 15 * 60
STALL_S = 300


def push_file(adb, local, remote_rel):
  name = os.path.basename(local)
  adb("push", local, f"/data/local/tmp/{name}")
  adb.shell(f"chmod 644 /data/local/tmp/{name}")
  adb.shell(f"run-as {PKG} mkdir -p files/{os.path.dirname(remote_rel)}")
  adb.shell(f"run-as {PKG} cp /data/local/tmp/{name} files/{remote_rel}")
  adb.shell(f"rm /data/local/tmp/{name}")


def run_on_device(adb, request, tag, run_dir):
  dev_dir = os.path.join(run_dir, "device", f"closed_loop_{tag}")
  os.makedirs(dev_dir, exist_ok=True)
  req_path = os.path.join(dev_dir, "selftest.json")
  with open(req_path, "w") as f:
    json.dump(request, f, indent=1)
  adb.shell(f"am force-stop {PKG}")
  push_file(adb, req_path, "selftest.json")
  why = wait_device_free(adb, 15 * 60)
  cond_before = device_conditions(adb)
  print(f"[{tag}] device free: {why}", flush=True)
  remote = f"{EXT}/closed_loop_{tag}"
  adb.shell(f"rm -rf {remote}", check=False)
  adb.shell(f"rm -f {EXT}/status.txt", check=False)
  t0 = time.time()
  adb.shell(f"am start -W -n {ACTIVITY}")
  pid = ""
  for _ in range(50):
    pid = pidof(adb)
    if pid:
      break
    time.sleep(0.2)
  if not pid:
    raise RuntimeError("app process did not start")
  raw_path = os.path.join(dev_dir, "logcat_pid.txt")
  raw = open(raw_path, "w")
  cat = subprocess.Popen(["adb", "-s", adb.serial, "logcat", "-v", "threadtime", f"--pid={pid}"], stdout=raw,
                         stderr=subprocess.STDOUT)
  status, state, last_change = "", "timeout", time.time()
  try:
    while time.time() - t0 < TIMEOUT_S:
      time.sleep(3)
      prev = status
      status = adb.shell(f"cat {remote}/status.txt", check=False)
      aborted = adb.shell(f"cat {EXT}/status.txt", check=False)
      if status != prev:
        last_change = time.time()
      last = status.strip().splitlines()[-1] if status.strip() else ""
      if "FAILED" in aborted:
        state = "failed"
        status += aborted
        break
      if last.startswith("DONE"):
        state = "done"
        break
      if time.time() - last_change > STALL_S:
        state = "stalled"
        break
      if pidof(adb) != pid:
        state = "process_died"
        break
  finally:
    time.sleep(1)
    cat.terminate()
    raw.close()
  wall = time.time() - t0
  crash = adb("logcat", "-d", "-b", "crash", check=False)
  crash_lines = [l for l in crash.splitlines() if f" {pid} " in l]
  pulled = os.path.join(dev_dir, "out")
  if os.path.isdir(pulled):
    for n in os.listdir(pulled):
      os.remove(os.path.join(pulled, n))
  os.makedirs(pulled, exist_ok=True)
  adb("pull", remote + "/.", pulled, check=False)
  cond_after = device_conditions(adb)
  adb.shell(f"am force-stop {PKG}")
  adb.shell(f"run-as {PKG} rm -f files/selftest.done files/selftest.json", check=False)
  with open(raw_path) as f:
    lines = f.read().splitlines()
  kept = [l for l in lines if LOGCAT_KEEP.search(l)] + crash_lines
  with open(os.path.join(run_dir, "results", f"device_logcat_closed_loop_{tag}.txt"), "w") as f:
    f.write(f"# pid {pid}, {len(lines)} pid-filtered lines, {len(kept)} kept; raw: {raw_path}\n")
    f.write("\n".join(kept) + "\n")
  return dict(state=state, pid=pid, wall_s=round(wall, 1), status=status.strip().splitlines(), out_dir=pulled,
              logcat_lines=len(lines), crash_lines=crash_lines, conditions_before=cond_before,
              conditions_after=cond_after)


def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def evaluate(out_dir, ref, cio, ref_ids, jvm_comps):
  run = json.load(open(os.path.join(out_dir, "steps.json")))
  steps = run["steps"]
  n = cio["length"].shape[0]
  res = dict(steps=len(steps), ref_steps=n)
  rows = [np.fromfile(os.path.join(out_dir, f"out_rows_step{s['k']}.bin"), "<f4").reshape(-1, 8) for s in steps]
  logits = np.concatenate(rows, 0)
  rl = ref["logits"]
  res.update(frames=int(logits.shape[0]), ref_frames=int(rl.shape[0]))
  if logits.shape != rl.shape or len(steps) != n:
    res["shape_mismatch"] = True
    return res
  p, pr = sigmoid(logits), sigmoid(rl)
  flips = (p > 0.5) != (pr > 0.5)
  where = [dict(frame=int(f), speaker=int(s), p_ref=float(pr[f, s]), p_dev=float(p[f, s]))
           for f, s in zip(*np.nonzero(flips))]
  res.update(agreement=float(1.0 - flips.mean()), flips=int(flips.sum()), cells=int(flips.size),
             max_dp=float(np.abs(p - pr).max()), max_abs_logit=float(np.abs(logits - rl).max()),
             nonfinite=int((~np.isfinite(logits)).sum()), flip_detail=where[:50])
  rs, ds = segments(rl), segments(logits)
  res["segments"] = compare_segments(rs, ds)
  res["segments"]["reference_only"] = [list(x) for x in rs if x not in ds]  # (start frame, end frame, speaker)
  res["segments"]["device_only"] = [list(x) for x in ds if x not in rs]

  state_bad = [i for i, s in enumerate(steps)
               if s["pre"] != [int(x) for x in cio["cache_pre"][i]] or s["post"] != [int(x) for x in cio["cache_post"][i]]]
  L_bad = [i for i, s in enumerate(steps) if s["L"] != int(cio["length"][i])]
  id_bad = compare_ids(ref_ids, [s["row_ids"] for s in steps])
  res.update(steps_state_mismatch=state_bad, steps_L_mismatch=L_bad,
             steps_row_ids_mismatch=[b["step"] for b in id_bad], row_id_mismatch_detail=id_bad[:5])
  # compressions: kept ids of the device vs the reference (the cache rows transformers fed at the next step)
  comps = []
  jvm = {c["step"]: c for c in jvm_comps}
  for c in run["compressions"]:
    k = c["step"]
    ref_next = ref_ids[k + 1][: int(cio["cache_post"][k, 0])] if k + 1 < n else None
    same = None
    diff_frames = None
    if ref_next is not None:
      dev = c["kept_ids"]
      same = len(dev) == len(ref_next) and all(r is not None and d in r for d, r in zip(dev, ref_next))
      if not same:
        ref_set = {x for r in ref_next if r for x in r}
        dev_set = set(dev)
        diff_frames = dict(num_device_only=len(dev_set - ref_set), num_reference_only=len(ref_set - dev_set),
                           device_only=sorted(dev_set - ref_set)[:20], reference_only=sorted(ref_set - dev_set)[:20])
    sel = c["select_boundary"]
    bb = [b for b in c["boost_boundaries"] if b is not None]
    comps.append(dict(step=k, candidates=c["candidates"], same_as_reference=same, differing_frames=diff_frames,
                      device_select_gap_ulps=ulps(*sel), device_min_boost_gap_ulps=min((ulps(*b) for b in bb), default=None),
                      reference_select_gap_ulps=jvm.get(k, {}).get("select_gap_ulps"),
                      reference_min_boost_gap_ulps=jvm.get(k, {}).get("min_boost_gap_ulps")))
  res["compressions"] = comps
  res["compressions_differing"] = [c["step"] for c in comps if c["same_as_reference"] is False]
  res["first_compress_step"] = comps[0]["step"] if comps else None
  res["first_compress_step_ref"] = next((i for i in range(n) if cio["cache_post"][i, 2] and not cio["cache_pre"][i, 2]),
                                        None)
  res["PASS_fp32_criteria"] = bool(res["agreement"] == 1.0 and res["segments"]["identical"] and not state_bad
                                   and not res["compressions_differing"])
  return res


def timing_summary(out_dir):
  t = json.load(open(os.path.join(out_dir, "timing.json")))
  keys = ("b_precision", "litert_version", "frontend_compile_ms", "encoder_compile_ms", "warmup", "steps",
          "audio_seconds", "step_ms_median", "step_ms_max", "sum_step_ms", "rtf_steps", "wall_ms", "rtf_wall",
          "first_chunk_ms", "first_chunk_audio_ms", "compressions", "conditions_start", "conditions_end")
  out = {k: t.get(k) for k in keys}
  lat = [x["latency_ms"] for x in t.get("step_latency", [])]
  if lat and t.get("realtime"):
    out["realtime"] = True
    out["latency_ms"] = dict(median=float(np.median(lat)), p95=float(np.percentile(lat, 95)), max=float(np.max(lat)),
                             first=lat[0])
  return out


def timing_md(report):
  rows = ["| run | B precision | compile A / B ms | warm-up A / B ms | step median ms: mel / A / B / cache / total | "
          "max step ms | RTF (steps) | first chunk ms | realtime latency ms median / p95 / max | thermal start→end | "
          "screen / lock / power (start) |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
  for tag, r in report["runs"].items():
    t = r.get("timing")
    if not t:
      continue
    m = t["step_ms_median"]
    w = (t.get("warmup") or [{}])[0]
    c0, c1 = t.get("conditions_start") or {}, t.get("conditions_end") or {}
    rows.append(
        f"| {tag} | {t['b_precision']} | {t['frontend_compile_ms']:.0f} / {t['encoder_compile_ms']:.0f} | "
        f"{w.get('frontend_ms', float('nan')):.1f} / {w.get('encoder_ms', float('nan')):.0f} | "
        f"{m['mel']:.2f} / {m['frontend']:.2f} / {m['encoder']:.1f} / {m['cache']:.2f} / {m['total']:.1f} | "
        f"{t['step_ms_max']['total']:.1f} | {t['rtf_steps']:.4f} | {t['first_chunk_ms']:.1f} | "
        + (f"{t['latency_ms']['median']:.1f} / {t['latency_ms']['p95']:.1f} / {t['latency_ms']['max']:.1f} | "
           if t.get("latency_ms") else "back to back | ") +
        f"{c0.get('thermal_status')}→{c1.get('thermal_status')} | "
        f"{'on' if c0.get('screen_interactive') else 'off'} / {'locked' if c0.get('keyguard_locked') else 'unlocked'}"
        f" / {c0.get('plugged')} {c0.get('battery_level')} % {c0.get('battery_temp_c')} °C |")
  return ("Closed loop on the device (LiteRT CompiledModel GPU, graph A FP32, one app launch per run; each step = "
          "mel + graph A write/run/read + graph B write/run/read + cache update; the wav is pushed 1600 samples at "
          "a time, back to back (as fast as the steps run) or paced at the audio rate (_rt runs, latency = from the "
          "arrival of the push that completes a chunk to its logits); RTF = sum of step ms / audio seconds; first "
          "chunk = step 0 after one warm-up inference, plus 1042.5 ms of audio it waits for)\n\n" + "\n".join(rows) + "\n")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--runs", nargs="+", required=True, help="<fp32|default|accum>:<fixture>")
  ap.add_argument("--serial", default="RFGL80R6A6H")
  ap.add_argument("--cooldown", type=int, default=20, help="seconds between launches")
  ap.add_argument("--skip-push-wav", action="store_true")
  ap.add_argument("--evaluate-only", action="store_true",
                  help="re-evaluate runs already pulled under <run>/device/closed_loop_<tag>/out (no device)")
  ap.add_argument("--realtime", action="store_true",
                  help="pace the pushes at the audio rate (a microphone) and record each step's latency")
  args = ap.parse_args()
  adb = Adb(args.serial)
  results = os.path.join(args.run_dir, "results")
  parity = json.load(open(os.path.join(results, "kotlin_parity.json")))
  report_path = os.path.join(results, "gate_closed_loop.json")
  report = json.load(open(report_path)) if os.path.exists(report_path) else {"serial": args.serial, "runs": {}}
  cache = {}
  pushed = set()
  for i, spec in enumerate(args.runs):
    precision, fx = spec.split(":")
    if fx not in cache:
      cio = load_npz(os.path.join(results, f"chunk_io_{fx}.npz"))
      cache[fx] = (load_npz(os.path.join(results, f"ref_{fx}_low_latency.npz")), cio, reference_row_ids(cio))
    ref, cio, ref_ids = cache[fx]
    if fx not in pushed and not args.skip_push_wav and not args.evaluate_only:
      push_file(adb, os.path.join(args.run_dir, "fixtures", fx + ".wav"), f"wav/{fx}.wav")
      pushed.add(fx)
    tag = f"{precision}_{fx}" + ("_rt" if args.realtime else "")
    if i and args.cooldown and not args.evaluate_only:
      time.sleep(args.cooldown)
    request = dict(mode="closed_loop", wav=f"wav/{fx}.wav", b_precision=precision, push=1600, warmup=1,
                   realtime=bool(args.realtime))
    print(f"[{tag}] start", flush=True)
    if args.evaluate_only:
      dev = dict(report["runs"][tag]["device"])
    else:
      dev = run_on_device(adb, request, tag, args.run_dir)
    print(f"[{tag}] {dev['state']} in {dev['wall_s']} s; status tail: {dev['status'][-3:]}", flush=True)
    rec = dict(precision=precision, fixture=fx, realtime=bool(args.realtime), device=dev)
    if dev["state"] == "done":
      jvm_comps = parity["fixtures"].get(fx, {}).get("replay", {}).get("compressions", [])
      rec["compare"] = evaluate(dev["out_dir"], ref, cio, ref_ids, jvm_comps)
      rec["timing"] = timing_summary(dev["out_dir"])
      brief = {k: rec["compare"].get(k) for k in ("agreement", "flips", "max_dp", "max_abs_logit", "nonfinite",
                                                 "steps_state_mismatch", "compressions_differing",
                                                 "steps_row_ids_mismatch", "PASS_fp32_criteria")}
      brief["segments"] = {k: rec["compare"]["segments"].get(k) for k in ("identical", "boundary_moves",
                                                                         "max_shift_frames", "vanished", "added")}
      brief["timing"] = {k: rec["timing"].get(k) for k in ("step_ms_median", "rtf_steps", "first_chunk_ms",
                                                          "encoder_compile_ms")}
      print(json.dumps({tag: brief}, indent=1), flush=True)
    report["runs"][tag] = rec
    with open(report_path, "w") as f:
      json.dump(report, f, indent=1)
    with open(os.path.join(results, "closed_loop_timing.md"), "w") as f:
      f.write(timing_md(report))
  print("GATE_CLOSED_LOOP done", flush=True)


if __name__ == "__main__":
  main()
