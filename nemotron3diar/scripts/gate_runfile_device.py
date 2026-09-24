"""Device run: the offline file mode (Nemotron3Diarizer.runFile, T=684) on the S26 GPU, plus a GPU-clock trace.

Runs (one app launch each, the ClosedLoopTest harness):
  file:<precision>    {"mode": "closed_loop", "file_mode": true, "repeats": N} on the 97.6 s fixture: the whole
                      file through the offline graph N times back to back (fresh diarizer each pass); the last
                      pass's logits vs transformers' offline forward (results/ref_<fixture>_offline.npz):
                      agreement@0.5, flips, max|dp|, segments (the processor's attention mask applied)
  stream:<precision>  the low_latency closed loop back to back (as round 3), for the same-session ratio
While each run is going, a device-side shell loop samples every --period s: /sys/class/kgsl/kgsl-3d0/clock_mhz
(current GPU clock; gpuclk itself is not readable by the shell user), max_clock_mhz, gpu_busy_percentage, temp,
thermal_pwrlevel, throttling, and cpu7's scaling_cur_freq; gpu_clock_stats (time per GPU level) is read before
and after. The trace is aligned with the app's step times through timing.json's epoch anchors.
Only com.nemotron3diar is touched (the models must be in files/models under their shipped names).

Writes results/runfile_device.json and results/runfile_device.md.

Usage: .venv/bin/python gate_runfile_device.py --run-dir <run> [--runs file:fp32 file:default stream:fp32]
           [--repeats 3] [--period 0.5] [--cooldown 60]
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np

from gate_device import Adb, ACTIVITY, EXT, LOGCAT_KEEP, PKG, device_conditions, pidof, wait_device_free
from host_loop import compare_segments, segments

FIXTURE = "diarization_example_16k"
TIMEOUT_S = 15 * 60
KGSL = "/sys/class/kgsl/kgsl-3d0"
TRACE_NODES = ["clock_mhz", "max_clock_mhz", "gpu_busy_percentage", "temp", "thermal_pwrlevel", "throttling"]
CPU7 = "/sys/devices/system/cpu/cpu7/cpufreq/scaling_cur_freq"


def start_trace(adb, period, path):
  reads = " ".join(f"$(cat {KGSL}/{n} 2>/dev/null | tr -d ' %')" for n in TRACE_NODES)
  loop = f"while true; do echo \"$(date +%s.%N) {reads} $(cat {CPU7} 2>/dev/null)\"; sleep {period}; done"
  out = open(path, "w")
  proc = subprocess.Popen(["adb", "-s", adb.serial, "shell", loop], stdout=out, stderr=subprocess.STDOUT)
  return proc, out


def read_trace(path):
  rows = []
  for line in open(path):
    parts = line.split()
    if len(parts) != 2 + len(TRACE_NODES):
      continue
    try:
      rows.append([float(p) for p in parts])
    except ValueError:
      continue
  names = ["epoch_s"] + TRACE_NODES + ["cpu7_khz"]
  return [dict(zip(names, r)) for r in rows]


def clock_stats(adb):
  return [int(x) for x in adb.shell(f"cat {KGSL}/gpu_clock_stats", check=False).split()]


def run_on_device(adb, request, tag, remote_name, run_dir, period):
  dev_dir = os.path.join(run_dir, "device", f"r4_{tag}")
  os.makedirs(dev_dir, exist_ok=True)
  req_path = os.path.join(dev_dir, "selftest.json")
  with open(req_path, "w") as f:
    json.dump(request, f, indent=1)
  adb.shell(f"am force-stop {PKG}")
  adb("push", req_path, "/data/local/tmp/selftest.json")
  adb.shell("chmod 644 /data/local/tmp/selftest.json")
  adb.shell(f"run-as {PKG} cp /data/local/tmp/selftest.json files/selftest.json")
  adb.shell("rm /data/local/tmp/selftest.json")
  why = wait_device_free(adb, 15 * 60)
  cond_before = device_conditions(adb)
  print(f"[{tag}] device free: {why}", flush=True)
  remote = f"{EXT}/{remote_name}"
  adb.shell(f"rm -rf {remote}", check=False)
  adb.shell(f"rm -f {EXT}/status.txt", check=False)
  freq = [int(x) for x in adb.shell(f"cat {KGSL}/freq_table_mhz", check=False).split()]
  stats_before = clock_stats(adb)
  trace_path = os.path.join(dev_dir, "gpu_trace.txt")
  trace, trace_out = start_trace(adb, period, trace_path)
  t0 = time.time()
  adb.shell(f"am start -W -n {ACTIVITY}")
  pid = ""
  for _ in range(50):
    pid = pidof(adb)
    if pid:
      break
    time.sleep(0.2)
  if not pid:
    trace.terminate()
    raise RuntimeError("app process did not start")
  raw_path = os.path.join(dev_dir, "logcat_pid.txt")
  raw = open(raw_path, "w")
  cat = subprocess.Popen(["adb", "-s", adb.serial, "logcat", "-v", "threadtime", f"--pid={pid}"], stdout=raw,
                         stderr=subprocess.STDOUT)
  state, status = "timeout", ""
  try:
    while time.time() - t0 < TIMEOUT_S:
      time.sleep(2)
      status = adb.shell(f"cat {remote}/status.txt", check=False)
      aborted = adb.shell(f"cat {EXT}/status.txt", check=False)
      last = status.strip().splitlines()[-1] if status.strip() else ""
      if "FAILED" in aborted:
        state, status = "failed", status + aborted
        break
      if last.startswith("DONE"):
        state = "done"
        break
      if pidof(adb) != pid:
        state = "process_died"
        break
  finally:
    time.sleep(1)
    trace.terminate()
    trace_out.close()
    cat.terminate()
    raw.close()
  stats_after = clock_stats(adb)
  wall = time.time() - t0
  pulled = os.path.join(dev_dir, "out")
  os.makedirs(pulled, exist_ok=True)
  for n in os.listdir(pulled):
    os.remove(os.path.join(pulled, n))
  adb("pull", remote + "/.", pulled, check=False)
  cond_after = device_conditions(adb)
  adb.shell(f"am force-stop {PKG}")
  adb.shell(f"run-as {PKG} rm -f files/selftest.done files/selftest.json", check=False)
  with open(raw_path) as f:
    lines = f.read().splitlines()
  kept = [l for l in lines if LOGCAT_KEEP.search(l)]
  with open(os.path.join(run_dir, "results", f"device_logcat_r4_{tag}.txt"), "w") as f:
    f.write(f"# pid {pid}, {len(lines)} pid-filtered lines, {len(kept)} kept; raw: {raw_path}\n")
    f.write("\n".join(kept) + "\n")
  time_in_level = None
  if len(stats_before) == len(stats_after) == len(freq):
    time_in_level = {str(f): b - a for f, a, b in zip(freq, stats_before, stats_after)}
  return dict(state=state, pid=pid, wall_s=round(wall, 1), status=status.strip().splitlines(), out_dir=pulled,
              trace=trace_path, gpu_time_in_level_us=time_in_level, conditions_before=cond_before,
              conditions_after=cond_after)


def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def evaluate_file(out_dir, ref):
  run = json.load(open(os.path.join(out_dir, "steps.json")))
  rows = [np.fromfile(os.path.join(out_dir, f"out_rows_step{s['k']}.bin"), "<f4").reshape(-1, 8) for s in run["steps"]]
  logits = np.concatenate(rows, 0)
  rl = ref["logits"]
  res = dict(chunks=len(rows), frames=int(logits.shape[0]), ref_frames=int(rl.shape[0]),
             L=[s["L"] for s in run["steps"]])
  if logits.shape != rl.shape:
    res["shape_mismatch"] = True
    return res
  p, pr = sigmoid(logits), sigmoid(rl)
  flips = (p > 0.5) != (pr > 0.5)
  valid = int(ref["attention_mask"].sum())
  seg_dev = segments(logits[:valid])
  seg_ref = segments(rl[:valid])
  res.update(max_abs_logit=float(np.abs(logits - rl).max()), max_dp=float(np.abs(p - pr).max()),
             agreement=float(1.0 - flips.mean()), flips=int(flips.sum()), cells=int(flips.size),
             nonfinite=int((~np.isfinite(logits)).sum()), segments=compare_segments(seg_ref, seg_dev))
  return res


def align_trace(trace, epoch0_ms, marks):
  """Trace rows relative to epoch0 (ms) with the app marks (label, start_ms, end_ms) that cover them."""
  out = []
  for r in trace:
    t = r["epoch_s"] * 1000.0 - epoch0_ms
    lab = next((m[0] for m in marks if m[1] <= t <= m[2]), "")
    out.append(dict(t_ms=round(t, 1), mark=lab, **{k: v for k, v in r.items() if k != "epoch_s"}))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--serial", default="RFGL80R6A6H")
  ap.add_argument("--runs", nargs="+", default=["file:fp32", "file:default", "stream:fp32"])
  ap.add_argument("--repeats", type=int, default=3)
  ap.add_argument("--period", type=float, default=0.5)
  ap.add_argument("--cooldown", type=int, default=60)
  ap.add_argument("--suffix", default="", help="appended to the run tags (e.g. _unlocked)")
  args = ap.parse_args()
  adb = Adb(args.serial)
  results = os.path.join(args.run_dir, "results")
  ref = np.load(os.path.join(results, f"ref_{FIXTURE}_offline.npz"))
  report_path = os.path.join(results, "runfile_device.json")
  report = json.load(open(report_path)) if os.path.exists(report_path) else {"runs": {}}
  for i, spec in enumerate(args.runs):
    kind, precision = spec.split(":")
    if i and args.cooldown:
      time.sleep(args.cooldown)
    request = dict(mode="closed_loop", wav=f"wav/{FIXTURE}.wav", b_precision=precision, push=1600, warmup=1)
    if kind == "file":
      request.update(file_mode=True, repeats=args.repeats)
      remote_name = f"file_{precision}_{FIXTURE}"
    else:
      remote_name = f"closed_loop_{precision}_{FIXTURE}"
    tag = f"{kind}_{precision}{args.suffix}"
    dev = run_on_device(adb, request, tag, remote_name, args.run_dir, args.period)
    print(f"[{tag}] {dev['state']} in {dev['wall_s']} s; status tail: {dev['status'][-4:]}", flush=True)
    rec = dict(kind=kind, precision=precision, request=request, device=dev)
    if dev["state"] == "done":
      timing = json.load(open(os.path.join(dev["out_dir"], "timing.json")))
      rec["timing"] = timing
      trace = read_trace(dev["trace"])
      if kind == "file":
        rec["compare"] = evaluate_file(dev["out_dir"], ref)
        epoch0 = timing["passes"][0]["start_epoch_ms"]
        marks = [(f"pass{p['pass']}", p["start_epoch_ms"] - epoch0, p["start_epoch_ms"] - epoch0 + p["wall_ms"])
                 for p in timing["passes"]]
      else:
        epoch0 = timing["wall0_epoch_ms"]
        marks = [(f"step{s['k']}", s["arrival_ms"], s["end_ms"]) for s in timing["step_latency"]]
      rec["trace"] = align_trace(trace, epoch0, marks)
      brief = rec.get("compare", {})
      print(f"[{tag}] " + json.dumps({k: brief.get(k) for k in ("agreement", "flips", "max_dp", "max_abs_logit")}
                                      | {"segments_identical": brief.get("segments", {}).get("identical")}),
            flush=True)
    report["runs"][tag] = rec
    with open(report_path, "w") as f:
      json.dump(report, f, indent=1)
  print("RUNFILE_DEVICE done", flush=True)


if __name__ == "__main__":
  main()
