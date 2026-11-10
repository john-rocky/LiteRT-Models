# Mac CPU reference outputs for the scorer: every graph on every fixture through the ai_edge_litert Interpreter (CPU,
# the XNNPACK default, num_threads given), every output in the subgraph's output order converted to float32 and
# concatenated -- the same layout as the phone runner's dumps (ftfl_bench --dump full).
#   python -I tools/mac_ref.py <work dir> [--threads 4]
# Writes fixtures/mac/<graph>/model0/<id3>.f32 and fixtures/mac/<graph>/mac_ref.json (versions, input and output
# details, per-fixture sha256). <graph> = det, pose or seg.
import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import ai_edge_litert
from ai_edge_litert.interpreter import Interpreter

GRAPHS = {"det": "det_efficientdet_lite0_int8.tflite", "pose": "pose_movenet_lightning_int8_v4.tflite",
          "seg": "seg_selfie_segmenter_f16.tflite"}
EXT_DTYPE = {".u8": np.uint8, ".f32": np.float32}

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
root = os.path.abspath(a.root)
for g, gfile in GRAPHS.items():
    man = json.load(open(os.path.join(root, "fixtures", f"fx_{g}.manifest.json")))
    gpath = os.path.join(root, "graphs", gfile)
    it = Interpreter(model_path=gpath, num_threads=a.threads)
    it.allocate_tensors()
    ins, outs = it.get_input_details(), it.get_output_details()
    ext = man["order"]["inputs_subgraph_order"][0]["ext"]
    out_dir = os.path.join(root, "fixtures", "mac", g, "model0")
    os.makedirs(out_dir, exist_ok=True)
    rec = {"graph": g, "file": f"graphs/{gfile}", "graph_sha256": hashlib.sha256(open(gpath, "rb").read()).hexdigest(),
           "runtime": f"ai_edge_litert {getattr(ai_edge_litert, '__version__', 'unknown')} Interpreter (CPU)",
           "num_threads": a.threads, "numpy": np.__version__, "python": sys.version.split()[0],
           "machine": f"{platform.system()} {platform.release()} {platform.machine()}",
           "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "inputs": [{"name": d["name"], "shape": [int(x) for x in d["shape"]], "dtype": np.dtype(d["dtype"]).name} for d in ins],
           "outputs": [{"name": d["name"], "shape": [int(x) for x in d["shape"]], "dtype": np.dtype(d["dtype"]).name} for d in outs],
           "fixtures": {}}
    for k in range(int(man["count"])):
        id3 = f"{k:03d}"
        src = os.path.join(root, man["dir"], f"{id3}_in0{ext}")
        x = np.fromfile(src, dtype=EXT_DTYPE[ext]).reshape(ins[0]["shape"])
        it.set_tensor(ins[0]["index"], x)
        it.invoke()
        flat = np.concatenate([it.get_tensor(d["index"]).astype(np.float32).reshape(-1) for d in outs])
        if flat.size != int(man["out_elems"]):
            sys.exit(f"{g} {id3}: {flat.size} output values, the manifest says {man['out_elems']}")
        dst = os.path.join(out_dir, f"{id3}.f32")
        flat.astype("<f4").tofile(dst)
        rec["fixtures"][id3] = {"sha256": hashlib.sha256(open(dst, "rb").read()).hexdigest(),
                                "nonfinite": int((~np.isfinite(flat)).sum()), "first4": [float(v) for v in flat[:4]]}
    json.dump(rec, open(os.path.join(root, "fixtures", "mac", g, "mac_ref.json"), "w"), indent=1)
    print(g, "outputs", [o["shape"] for o in rec["outputs"]], "fixtures", len(rec["fixtures"]), "->", out_dir)
