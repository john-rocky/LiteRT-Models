#!/usr/bin/env python3
"""Package everything the Kotlin tracker autotest needs into models/device_tracker/:

  consts/<name>.bin      host constants from models/tracker_host/consts.npz, raw LE f32
  consts/manifest.json   name -> shape
  clip/<i>.jpg           the frames to track (models/clip8)
  expected/f<i>_{ids,probs,masks}.bin + manifest.json
                         per-frame expected outputs from the fp16 host-loop run
                         (ids <i4, probs <f4, masks packed 1 bit/px LSB-first)

The expected outputs are produced by scripts/tracker_host_loop.py --fp16 --dump-device
(the Mali-reality mode), invoked below.

Usage: dump_tracker_device_assets.py [--clip models/clip8] [--skip-expected]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEST = os.path.join(ROOT, "models", "device_tracker")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=os.path.join(ROOT, "models", "clip8"))
    ap.add_argument("--ref", default=os.path.join(ROOT, "models", "tracker_ref", "ref_person.npy"))
    ap.add_argument("--skip-expected", action="store_true")
    a = ap.parse_args()

    cdir = os.path.join(DEST, "consts")
    os.makedirs(cdir, exist_ok=True)
    consts = dict(np.load(os.path.join(ROOT, "models", "tracker_host", "consts.npz")))
    manifest = {}
    for name, arr in sorted(consts.items()):
        fname = name.replace(".", "_") + ".bin"
        arr.astype("<f4").tofile(os.path.join(cdir, fname))
        manifest[name] = {"file": fname, "shape": list(arr.shape)}
    json.dump(manifest, open(os.path.join(cdir, "manifest.json"), "w"), indent=1)
    print(f"[consts] {len(manifest)} tensors -> {cdir}")

    fdir = os.path.join(DEST, "clip")
    os.makedirs(fdir, exist_ok=True)
    names = sorted((n for n in os.listdir(a.clip) if n.lower().endswith((".jpg", ".jpeg", ".png"))),
                   key=lambda n: int(os.path.splitext(n)[0]))
    for n in names:
        shutil.copy(os.path.join(a.clip, n), os.path.join(fdir, n))
    print(f"[clip] {len(names)} frames -> {fdir}")

    flags_src = os.path.join(ROOT, "models", "tracker_host", "flags.json")
    shutil.copy(flags_src, os.path.join(DEST, "flags.json"))

    if not a.skip_expected:
        edir = os.path.join(DEST, "expected")
        cmd = [sys.executable, os.path.join(HERE, "tracker_host_loop.py"),
               "--clip", a.clip, "--ref", a.ref, "--fp16", "--dump-device", edir]
        print("[expected] running:", " ".join(cmd))
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print("[expected] WARNING: host loop exited nonzero (fp16 gate);"
                  " fixtures were still written if the run completed")
    print("done")


if __name__ == "__main__":
    main()
