"""Device gate: the Nemotron-3-Diarization graphs on the S26 LiteRT CompiledModel GPU vs transformers.

For each run (<variant>:<precision>, one app launch each so a native crash only loses that run):
  1. writes the chunk_io steps (default 0,30,55,60,100,128,135 of diarization_example_16k) as float32 LE bins
     step<i>_{mel,packed,bias,cos,sin}.bin and pushes them to the app's files/steps (once per invocation)
  2. pushes files/selftest.json, checks that nothing else owns the device, `am start`s com.nemotron3diar,
     captures `adb logcat --pid=<pid>`, polls status.txt until DONE (timeout 15 min), pulls the outputs
  3. compares with transformers (chunk_io chunk_logits, the packed chunk rows for graph A):
       graph A   max|d| vs the transformers chunk rows
       graph B   per step and overall over the L*8 logit rows and over the chunk's own output rows:
                 max|dlogit|, corr, max|dp| (sigmoid), agreement@0.5, flips; plus the same file on the Mac CPU
                 (CompiledModel XNNPACK) to separate GPU arithmetic from the graph itself
  4. merges into results/gate_device.json and rewrites results/device_timing.md; the pid-filtered logcat
     goes to <run>/device/<tag>/logcat_pid.txt, the compile/delegate/partition/fallback/error lines to
     results/device_logcat_<tag>.txt
Models must already be in files/models (scripts/install_to_device.sh). Only com.nemotron3diar is touched:
no logcat -c, no pm clear, no other package.

Usage: .venv/bin/python gate_device.py --run-dir <run> --runs safe_fp16:default plain_fp16:default
           [--serial RFGL80R6A6H] [--suffix _litert216] [--steps 0,30,...] [--skip-push-steps]
           [--frontend-precision fp32|default]
Precisions: default | fp32 | fp16 | fp16acc32 (GpuOptions.Precision.FP16_WITH_FP32_ACCUM).
Runs before 16:20 (tags without _Afp32 / fp16acc32) ran graph A at the variant's own precision.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

import nemotron3diar_model as n3d

PKG = "com.nemotron3diar"
ACTIVITY = f"{PKG}/.MainActivity"
EXT = f"/sdcard/Android/data/{PKG}/files/n3d"
FIXTURE = "diarization_example_16k"
STEPS = [0, 30, 55, 60, 100, 128, 135]
MODELS = {
    "safe_fp16": "n3d_encoder_ll_safe_fp16.tflite",
    "plain_fp16": "n3d_encoder_ll_fp16.tflite",
    "safe_fp32": "n3d_encoder_ll_safe.tflite",
    "plain_fp32": "n3d_encoder_ll.tflite",
    "off_safe_fp16": "n3d_encoder_off_safe_fp16.tflite",
    "off_safe_fp32": "n3d_encoder_off_safe.tflite",
}
OFFLINE_STEPS = [0, 1, 3]  # L 380, 684 (first compressed input), 505 (last chunk, one masked key row)
FRONTEND = "n3d_frontend.tflite"
OTHER_MEASUREMENTS = re.compile(r"am instrument|phase.*chain|gated_measure")
LOGCAT_KEEP = re.compile(
    r"N3D|compil|delegat|partition|fallback|error|fail|fatal|abort|signal|backtrace|opencl|accelerat|gpu|"
    r"litert|tflite|ml_drift|replacing|unsupported|not supported", re.I)
TIMEOUT_S = 15 * 60
STALL_FIRST_S = 120  # no status.txt at all after this long -> the self-test never started
STALL_S = 300  # status.txt unchanged this long -> stalled (a first GPU compile can take a while)


class Adb:

  def __init__(self, serial):
    self.serial = serial

  def __call__(self, *args, check=True, timeout=600):
    r = subprocess.run(["adb", "-s", self.serial, *args], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
      raise RuntimeError(f"adb {' '.join(args)} -> {r.returncode}: {r.stderr.strip()} {r.stdout.strip()}")
    return r.stdout

  def shell(self, cmd, check=True, timeout=600):
    return self("shell", cmd, check=check, timeout=timeout)


def other_measurements():
  out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
  return [l for l in out.splitlines() if OTHER_MEASUREMENTS.search(l) and "gate_device.py" not in l
          and "grep" not in l]


def foreground(adb):
  out = adb.shell("dumpsys activity activities | grep -E 'topResumedActivity|ResumedActivity'", check=False)
  return [l.strip() for l in out.splitlines() if l.strip()]


def device_conditions(adb):
  """Screen / lock / foreground / power state, recorded with every run."""
  power = adb.shell("dumpsys power | grep -E 'mWakefulness=|mHoldingDisplaySuspendBlocker='", check=False)
  keyguard = adb.shell("dumpsys window | grep -E 'isKeyguardShowing|mDreamingLockscreen'", check=False)
  battery = adb.shell("dumpsys battery | grep -E 'AC powered|USB powered|level|temperature|status'", check=False)
  return dict(power=[l.strip() for l in power.splitlines() if l.strip()],
              keyguard=[l.strip() for l in keyguard.splitlines() if l.strip()],
              foreground=foreground(adb), battery=[l.strip() for l in battery.splitlines() if l.strip()])


ALLOWED_FOREGROUND = []  # extra foreground substrings the supervisor cleared for this session (--allow-foreground)


def wait_device_free(adb, wait_s):
  """Another measurement on the Mac -> wait (up to wait_s). A foreground app other than a launcher or this
  package -> stop at once (the device belongs to someone else; the supervisor decides)."""
  t0 = time.time()
  while True:
    fg = foreground(adb)
    bad = [l for l in fg if "launcher" not in l.lower() and PKG not in l
           and not any(a in l for a in ALLOWED_FOREGROUND)]
    if bad:
      raise SystemExit("DEVICE BUSY: foreground is not the launcher / " + PKG + ": " + " | ".join(bad))
    others = other_measurements()
    if not others:
      return " | ".join(fg)
    if time.time() - t0 > wait_s:
      raise SystemExit(f"DEVICE BUSY (waited {wait_s}s): other measurement running: " +
                       " | ".join(o[:160] for o in others))
    print("other measurement running on the Mac, waiting: " + " | ".join(o[:160] for o in others), flush=True)
    time.sleep(30)


def f32(path, arr):
  np.ascontiguousarray(arr, dtype="<f4").tofile(path)


def write_steps(d, steps, out_dir):
  os.makedirs(out_dir, exist_ok=True)
  cos, sin = (x.numpy() for x in n3d.rope_tables())
  for i in steps:
    L = int(d["length"][i])
    f32(os.path.join(out_dir, f"step{i}_mel.bin"), d["input_features"][i])  # [104,128], zero-padded
    f32(os.path.join(out_dir, f"step{i}_packed.bin"), d["packed_embeds"][i])  # [541,512], zero tail
    f32(os.path.join(out_dir, f"step{i}_bias.bin"), n3d.attn_bias_for(L))
    f32(os.path.join(out_dir, f"step{i}_cos.bin"), cos)
    f32(os.path.join(out_dir, f"step{i}_sin.bin"), sin)


def write_steps_offline(d, steps, out_dir):
  """Offline chunks (T=684): packed, two-level attn_bias (the masked key row -16384, pad -32768), RoPE for 684."""
  os.makedirs(out_dir, exist_ok=True)
  t = n3d.T_OFFLINE
  cos, sin = (x.numpy() for x in n3d.rope_tables(t))
  for i in steps:
    L = int(d["length"][i])
    f32(os.path.join(out_dir, f"step{i}_packed.bin"), d["packed_embeds"][i])
    f32(os.path.join(out_dir, f"step{i}_bias.bin"), n3d.attn_bias_for(L, t, valid=d["row_valid"][i, :L], two_level=True))
    f32(os.path.join(out_dir, f"step{i}_cos.bin"), cos)
    f32(os.path.join(out_dir, f"step{i}_sin.bin"), sin)


def compare_offline(d, steps, out_dir, name, precision):
  per = []
  for i in steps:
    L = int(d["length"][i])
    fb = os.path.join(out_dir, f"out_{name}_{precision}_step{i}.bin")
    if not os.path.exists(fb):
      per.append(dict(step=i, L=L, missing=True))
      continue
    full = np.fromfile(fb, "<f4").reshape(n3d.T_OFFLINE * 8, 8)
    g, ref = full[: L * 8], d["chunk_logits"][i, : L * 8]
    nc, nf = int(d["cache_pre"][i, 0]), int(d["cache_pre"][i, 1])
    c0, c1 = (nc + nf) * 8, (nc + nf + int(d["num_chunk_frames"][i])) * 8
    p, pr = sigmoid(g), sigmoid(ref)
    per.append(dict(step=i, L=L, masked_rows=int((~d["row_valid"][i, :L]).sum()),
                    max_abs=float(np.abs(g - ref).max()), corr=corr(g, ref), max_dp=float(np.abs(p - pr).max()),
                    flips=int(((p > 0.5) != (pr > 0.5)).sum()), cells=int(ref.size),
                    out_flips=int(((p[c0:c1] > 0.5) != (pr[c0:c1] > 0.5)).sum()), out_cells=int(c1 - c0) * 8,
                    out_max_dp=float(np.abs(p[c0:c1] - pr[c0:c1]).max()),
                    nonfinite=int((~np.isfinite(full)).sum())))
  ok = [r for r in per if not r.get("missing")]
  summary = {}
  if ok:
    summary = dict(steps=len(ok), max_abs=max(r["max_abs"] for r in ok), min_corr=min(r["corr"] for r in ok),
                   max_dp=max(r["max_dp"] for r in ok), flips=sum(r["flips"] for r in ok),
                   agree_all_rows=1.0 - sum(r["flips"] for r in ok) / sum(r["cells"] for r in ok),
                   out_flips=sum(r["out_flips"] for r in ok), out_max_dp=max(r["out_max_dp"] for r in ok),
                   nonfinite=sum(r["nonfinite"] for r in ok))
  return dict(summary=summary, graph_A={}, graph_A_steps=[], steps=per)


def push_steps(adb, local_dir, remote="steps"):
  names = sorted(os.listdir(local_dir))
  adb.shell("rm -rf /data/local/tmp/n3d_steps && mkdir -p /data/local/tmp/n3d_steps")
  adb("push", local_dir + "/.", "/data/local/tmp/n3d_steps/")
  adb.shell("chmod 644 /data/local/tmp/n3d_steps/*")
  adb.shell(f"run-as {PKG} mkdir -p files/{remote}")
  for n in names:
    adb.shell(f"run-as {PKG} cp /data/local/tmp/n3d_steps/{n} files/{remote}/{n}")
  adb.shell("rm -rf /data/local/tmp/n3d_steps")
  listing = adb.shell(f"run-as {PKG} ls files/{remote}")
  assert set(names) <= set(listing.split()), "steps missing on the device"
  return len(names)


def push_request(adb, spec, local_path):
  with open(local_path, "w") as f:
    json.dump(spec, f, indent=1)
  adb("push", local_path, "/data/local/tmp/n3d_selftest.json")
  adb.shell("chmod 644 /data/local/tmp/n3d_selftest.json")
  adb.shell(f"run-as {PKG} cp /data/local/tmp/n3d_selftest.json files/selftest.json")
  adb.shell("rm /data/local/tmp/n3d_selftest.json")


def pidof(adb):
  return adb.shell(f"pidof {PKG}", check=False).strip()


def run_on_device(adb, tag, spec, run_dir):
  dev_dir = os.path.join(run_dir, "device", tag)
  os.makedirs(dev_dir, exist_ok=True)
  push_request(adb, spec, os.path.join(dev_dir, "selftest.json"))
  adb.shell(f"am force-stop {PKG}")
  why = wait_device_free(adb, 15 * 60)
  cond_before = device_conditions(adb)
  print(f"[{tag}] device free: {why}", flush=True)
  t_start = time.time()
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
  cat = subprocess.Popen(["adb", "-s", adb.serial, "logcat", "-v", "threadtime", f"--pid={pid}"],
                         stdout=raw, stderr=subprocess.STDOUT)
  status, state = "", "timeout"
  last_change = time.time()
  try:
    while time.time() - t_start < TIMEOUT_S:
      time.sleep(3)
      prev = status
      status = adb.shell(f"cat {EXT}/status.txt", check=False)
      if status != prev:
        last_change = time.time()
      last = status.strip().splitlines()[-1] if status.strip() else ""
      if not status.strip() and time.time() - t_start > STALL_FIRST_S:
        state = "stalled_no_status"
        break
      if time.time() - last_change > STALL_S:
        state = "stalled"
        break
      if last.startswith("DONE"):
        state = "done"
        break
      if last.startswith("FAILED") or last.startswith("ABORTED"):
        state = "failed"
        break
      if pidof(adb) != pid:
        state = "process_died"
        break
  finally:
    time.sleep(1)
    cat.terminate()
    raw.close()
  wall = time.time() - t_start
  crash = adb("logcat", "-d", "-b", "crash", check=False)
  crash_lines = [l for l in crash.splitlines() if f" {pid} " in l]
  with open(os.path.join(dev_dir, "status.txt"), "w") as f:
    f.write(status)
  pulled = os.path.join(dev_dir, "n3d")
  if os.path.isdir(pulled):
    for n in os.listdir(pulled):
      os.remove(os.path.join(pulled, n))
  adb("pull", EXT, dev_dir, check=False)
  cond_after = device_conditions(adb)
  adb.shell(f"am force-stop {PKG}")
  with open(raw_path) as f:
    lines = f.read().splitlines()
  kept = [l for l in lines if LOGCAT_KEEP.search(l)] + crash_lines
  with open(os.path.join(run_dir, "results", f"device_logcat_{tag}.txt"), "w") as f:
    f.write(f"# pid {pid}, {len(lines)} pid-filtered lines, {len(kept)} kept "
            f"(compile/delegate/partition/fallback/error/GPU); raw: {raw_path}\n")
    f.write("\n".join(kept) + "\n")
  return dict(state=state, pid=pid, wall_s=round(wall, 1), status=status.strip().splitlines(),
              logcat_lines=len(lines), crash_lines=crash_lines, out_dir=pulled,
              conditions_before=cond_before, conditions_after=cond_after)


def corr(a, b):
  a = a.astype(np.float64).ravel()
  b = b.astype(np.float64).ravel()
  a -= a.mean()
  b -= b.mean()
  den = math.sqrt((a @ a) * (b @ b))
  return float((a @ b) / den) if den > 0 else float("nan")


def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def compare(d, steps, out_dir, name, precision, cpu_runner):
  cos, sin = (x.numpy() for x in n3d.rope_tables())
  per_step = []
  a_rows = []
  for i in steps:
    L = int(d["length"][i])
    k = -(-int(d["mel_frames"][i]) // 8)
    fa = os.path.join(out_dir, f"outA_step{i}.bin")
    if os.path.exists(fa):
      emb = np.fromfile(fa, "<f4").reshape(13, 512)
      ref_rows = d["packed_embeds"][i, L - k : L]
      a_rows.append(dict(step=i, max_abs=float(np.abs(emb[:k] - ref_rows).max()),
                         ref_absmax=float(np.abs(ref_rows).max()), corr=corr(emb[:k], ref_rows),
                         finite=bool(np.isfinite(emb).all())))
    fb = os.path.join(out_dir, f"out_{name}_{precision}_step{i}.bin")
    if not os.path.exists(fb):
      per_step.append(dict(step=i, L=L, missing=True))
      continue
    full = np.fromfile(fb, "<f4").reshape(541 * 8, 8)
    g = full[: L * 8]
    ref = d["chunk_logits"][i, : L * 8]
    p_ref, p_dev = sigmoid(ref), sigmoid(g)
    nc, nf = int(d["cache_pre"][i, 0]), int(d["cache_pre"][i, 1])
    c0 = (nc + nf) * 8
    c1 = c0 + int(d["out_frames"][i])
    act_ref, act_dev = p_ref > 0.5, p_dev > 0.5
    flips = []
    for row, spk in zip(*np.nonzero(act_ref != act_dev)):
      region = ("cache" if row < nc * 8 else "fifo" if row < c0 else "chunk_output" if row < c1
                else "after_output")
      flips.append(dict(logit_row=int(row), region=region, speaker=int(spk),
                        p_ref=float(p_ref[row, spk]), p_dev=float(p_dev[row, spk])))
    rec = dict(
        step=i, L=L, compressed=bool(d["cache_pre"][i, 2]), out_rows=[c0, c1],
        nonfinite_all_rows=int((~np.isfinite(full)).sum()),
        max_abs=float(np.abs(g - ref).max()), corr=corr(g, ref), max_dp=float(np.abs(p_dev - p_ref).max()),
        agree=float((act_ref == act_dev).mean()), flips=int((act_ref != act_dev).sum()), cells=int(ref.size),
        out_max_abs=float(np.abs(g[c0:c1] - ref[c0:c1]).max()),
        out_max_dp=float(np.abs(p_dev[c0:c1] - p_ref[c0:c1]).max()),
        out_agree=float((act_ref[c0:c1] == act_dev[c0:c1]).mean()),
        out_flips=int((act_ref[c0:c1] != act_dev[c0:c1]).sum()), out_cells=int(ref[c0:c1].size),
        flip_detail=flips[:50],
    )
    if cpu_runner is not None:
      feeds = dict(packed_embeds=d["packed_embeds"][i][None], attn_bias=n3d.attn_bias_for(L),
                   rope_cos=cos, rope_sin=sin)
      c = cpu_runner(**feeds)["logits"][0, : L * 8]
      rec["vs_mac_cpu_same_file_max_abs"] = float(np.abs(g - c).max())
      rec["mac_cpu_vs_ref_max_abs"] = float(np.abs(c - ref).max())
    per_step.append(rec)
  ok = [r for r in per_step if not r.get("missing")]
  summary = {}
  if ok:
    cells = sum(r["cells"] for r in ok)
    out_cells = sum(r["out_cells"] for r in ok)
    summary = dict(
        steps=len(ok), missing_steps=[r["step"] for r in per_step if r.get("missing")],
        max_abs=max(r["max_abs"] for r in ok), min_corr=min(r["corr"] for r in ok),
        max_dp=max(r["max_dp"] for r in ok),
        agree_all_rows=1.0 - sum(r["flips"] for r in ok) / cells, flips=sum(r["flips"] for r in ok), cells=cells,
        agree_out_rows=1.0 - sum(r["out_flips"] for r in ok) / out_cells,
        out_flips=sum(r["out_flips"] for r in ok), out_cells=out_cells,
        out_max_abs=max(r["out_max_abs"] for r in ok), out_max_dp=max(r["out_max_dp"] for r in ok),
        min_step_agree_out_rows=min(r["out_agree"] for r in ok),
        nonfinite=sum(r["nonfinite_all_rows"] for r in ok),
    )
    if cpu_runner is not None:
      summary["vs_mac_cpu_same_file_max_abs"] = max(r["vs_mac_cpu_same_file_max_abs"] for r in ok)
    summary["PASS_out_rows_100"] = bool(summary["out_flips"] == 0 and summary["steps"] == len(steps))
    summary["PASS_all_rows_999"] = bool(summary["agree_all_rows"] >= 0.999)
  a_summary = {}
  if a_rows:
    a_summary = dict(steps=len(a_rows), max_abs=max(r["max_abs"] for r in a_rows),
                     min_corr=min(r["corr"] for r in a_rows), ref_absmax=max(r["ref_absmax"] for r in a_rows),
                     all_finite=all(r["finite"] for r in a_rows))
  return dict(summary=summary, graph_A=a_summary, graph_A_steps=a_rows, steps=per_step)


def timing_table(report):
  rows = ["| launch | graph | file | precision | LiteRT | load+compile ms | median ms | min | max | thermal before→after |"
          " screen / lock at start |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
  seen = set()
  for tag, r in report["runs"].items():
    t = r.get("timing") or {}
    if t.get("tag") in seen:
      continue
    seen.add(t.get("tag"))
    c = t.get("conditions_start") or {}
    cond = (f"{'on' if c.get('screen_interactive') else 'off'} / {'locked' if c.get('keyguard_locked') else 'unlocked'}"
            if c else "")
    tag = t.get("tag", tag)
    for g in t.get("graphs", []):
      if g.get("skipped"):
        continue
      if "error" in g:
        rows.append(f"| {tag} | {g['graph']} | {g['file']} | {g['precision']} | {t.get('litert_version')} | "
                    f"ERROR: {g['error'][:80]} | | | | | {cond} |")
        continue
      rows.append(
          f"| {tag} | {g['graph']} | {g['file']} | {g['precision']} | {t.get('litert_version')} | "
          f"{g['load_compile_ms']:.0f} | {g['median_ms']:.2f} | {g['min_ms']:.2f} | {g['max_ms']:.2f} | "
          f"{g['thermal_before']}→{g['thermal_after']} | {cond} |")
  return "\n".join(rows) + "\n"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--runs", nargs="+", required=True, help="<variant>:<precision>, e.g. safe_fp16:default")
  ap.add_argument("--serial", default="RFGL80R6A6H")
  ap.add_argument("--suffix", default="", help="appended to the run tag (e.g. _litert216)")
  ap.add_argument("--steps", default=",".join(map(str, STEPS)))
  ap.add_argument("--skip-push-steps", action="store_true")
  ap.add_argument("--no-cpu-compare", action="store_true")
  ap.add_argument("--same-launch", action="store_true",
                  help="run every --runs entry in ONE app launch (same process and screen state, for timing A/B)")
  ap.add_argument("--frontend-precision", default="fp32",
                  help="GPU precision of graph A (fp32 by default: 0.2 ms, and its rows go into the cache)")
  ap.add_argument("--allow-foreground", action="append", default=[],
                  help="extra foreground activity substring cleared by the supervisor (e.g. a system dialog)")
  ap.add_argument("--mode", choices=["low_latency", "offline"], default="low_latency",
                  help="offline: the T=684 graph on chunk_io_<fixture>_offline.npz chunks (default steps 0,1,3)")
  args = ap.parse_args()
  ALLOWED_FOREGROUND.extend(args.allow_foreground)
  offline = args.mode == "offline"
  if offline and args.steps == ",".join(map(str, STEPS)):
    args.steps = ",".join(map(str, OFFLINE_STEPS))
  steps = [int(s) for s in args.steps.split(",")]
  results = os.path.join(args.run_dir, "results")
  adb = Adb(args.serial)
  with np.load(os.path.join(results, f"chunk_io_{FIXTURE}" + ("_offline" if offline else "") + ".npz")) as z:
    d = {k: z[k] for k in z.files}

  if not args.skip_push_steps:
    local = os.path.join(args.run_dir, "device", "steps_off" if offline else "steps")
    (write_steps_offline if offline else write_steps)(d, steps, local)
    n = push_steps(adb, local, "steps_off" if offline else "steps")
    print(f"pushed {n} step files", flush=True)

  report_path = os.path.join(results, "gate_device_offline.json" if offline else "gate_device.json")
  report = json.load(open(report_path)) if os.path.exists(report_path) else {
      "fixture": FIXTURE, "steps": steps, "serial": args.serial, "runs": {}}
  runs = [r.split(":") for r in args.runs]
  launches = [runs] if args.same_launch else [[r] for r in runs]
  for launch in launches:
    launch_tag = ("samelaunch_" + "+".join(f"{n}_{p}" for n, p in launch) if args.same_launch
                  else f"{launch[0][0]}_{launch[0][1]}") + args.suffix
    spec = dict(tag=launch_tag, frontend=FRONTEND, frontend_precision=args.frontend_precision, steps=steps,
                timing_step=128 if 128 in steps else steps[0], warmup=5, iters=20,
                variants=[dict(name=n, file=MODELS[n], precision=p) for n, p in launch])
    if offline:
      spec.update(T=n3d.T_OFFLINE, steps_dir="steps_off", skip_frontend=True, timing_step=1 if 1 in steps else steps[0])
    print(f"[{launch_tag}] start", flush=True)
    dev = run_on_device(adb, launch_tag, spec, args.run_dir)
    print(f"[{launch_tag}] {dev['state']} in {dev['wall_s']} s; status tail: {dev['status'][-3:]}", flush=True)
    timing_path = os.path.join(dev["out_dir"], "timing.json")
    timing = json.load(open(timing_path)) if os.path.exists(timing_path) else None
    for name, precision in launch:
      tag = f"{name}_{precision}{args.suffix}" + ("@samelaunch" if args.same_launch else "")
      cpu = None
      if not args.no_cpu_compare and not offline:
        from gate_tflite_cpu import Runner
        cpu = Runner(os.path.join(args.run_dir, "exports", MODELS[name]), 8)
      cmp_ = (compare_offline(d, steps, dev["out_dir"], name, precision) if offline
              else compare(d, steps, dev["out_dir"], name, precision, cpu))
      report["runs"][tag] = dict(variant=name, file=MODELS[name], precision=precision, device=dev,
                                 timing=timing, launch=launch_tag, allowed_foreground=args.allow_foreground, **cmp_)
      print(json.dumps({tag: dict(summary=cmp_["summary"], graph_A=cmp_["graph_A"])}, indent=1), flush=True)
    with open(report_path, "w") as f:
      json.dump(report, f, indent=1)
    with open(os.path.join(results, "device_timing_offline.md" if offline else "device_timing.md"), "w") as f:
      f.write(timing_table(report))

  adb.shell(f"am force-stop {PKG}")
  adb.shell(f"run-as {PKG} rm -f files/selftest.done files/selftest.json", check=False)
  print("GATE_DEVICE done", flush=True)


if __name__ == "__main__":
  main()
