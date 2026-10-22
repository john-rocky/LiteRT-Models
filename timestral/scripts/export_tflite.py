"""Export the re-authored Timestral graph with litert-torch, plus fp16 / dynamic-int8 weight variants.

  python export_tflite.py --bucket 96x32 --rows 1 [--norm exact|pow2|amax] [--fp16] [--int8]

Writes out/tflite/timestral209m_r{R}_l{L}_h{H}_{norm}_{fp32|wfp16|drq8}.tflite, the op scan of each
file (out/tflite/*.opscan.json), and once per run the host tables the graph needs:
out/host/forecast_query.f32 (768 float32) and out/host/rope_{cos,sin}_p{min}_{max}.f32 ([P, 64] float32).
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import litert_helpers as lh  # noqa: E402
import timestral_graph as tg  # noqa: E402

ROPE_P_MIN, ROPE_P_MAX = -1023, 1055  # covers every bucket with L <= 1024 and L + H <= 1056


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_host_tables(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fq = model.representation.forecast_query.detach().numpy().astype("<f4")
    fq.tofile(os.path.join(out_dir, "forecast_query.f32"))
    cos, sin = tg.rope_table(model, ROPE_P_MIN, ROPE_P_MAX)
    cos.astype("<f4").tofile(os.path.join(out_dir, f"rope_cos_p{ROPE_P_MIN}_{ROPE_P_MAX}.f32"))
    sin.astype("<f4").tofile(os.path.join(out_dir, f"rope_sin_p{ROPE_P_MIN}_{ROPE_P_MAX}.f32"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, help="LxH in patches, e.g. 96x32")
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--norm", default="exact", choices=["exact", "pow2", "amax"])
    ap.add_argument("--norm-k", default=None, help="json file: {layer: {attn0|attn1|mlp: k}, 'out': {'out': k}}")
    ap.add_argument("--resid-k", type=int, default=0, help="run the residual stream 2^-k smaller (exact)")
    ap.add_argument("--attn-k", type=int, default=0, help="logits 2^-k smaller, max subtracted before scaling back")
    ap.add_argument("--key-center", default=None, help="npz L00..L23 -> [12, 32] keys' unrotated-dim centre")
    ap.add_argument("--attn-clamp", action="store_true", help="floor the softmax input at -3e4 (exact)")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--skip-fp32-export", action="store_true", help="reuse an existing fp32 file")
    args = ap.parse_args()
    ll, hh = (int(v) for v in args.bucket.split("x"))
    torch.set_grad_enabled(False)
    out_dir = os.path.join(common.OUT, "tflite")
    os.makedirs(out_dir, exist_ok=True)
    model = common.load_official()
    write_host_tables(model, os.path.join(common.OUT, "host"))
    norm_k = None
    if args.norm_k:
        with open(args.norm_k) as f:
            raw = json.load(f)
        norm_k = {(int(k) if k.isdigit() else k): v for k, v in raw.items()}
    key_center = None
    if args.key_center:
        kc = np.load(args.key_center)
        key_center = {int(k[1:]): kc[k] for k in kc.files}
    graph = tg.TimestralGraph(model, ll, hh, norm_mode=args.norm, norm_k=norm_k, resid_k=args.resid_k,
                              attn_k=args.attn_k, key_center=key_center, attn_clamp=args.attn_clamp).eval()
    norm_tag = args.norm + (f"k{os.path.splitext(os.path.basename(args.norm_k))[0]}" if args.norm_k else "")
    stem = (f"timestral209m_r{args.rows}_l{ll}_h{hh}_{norm_tag}" + (f"_res{args.resid_k}" if args.resid_k else "")
            + (f"_att{args.attn_k}" if args.attn_k else "") + ("c" if args.attn_clamp else "")
            + ("_kc" if key_center else ""))
    fp32 = os.path.join(out_dir, stem + "_fp32.tflite")
    log = {"bucket": [ll, hh], "rows": args.rows, "norm": args.norm, "files": {}}
    if not (args.skip_fp32_export and os.path.exists(fp32)):
        import litert_torch
        sample = dict(zip(tg.INPUT_NAMES, tg.sample_inputs(args.rows, ll, hh, graph.d_model)))
        t0 = time.time()
        litert_torch.convert(graph, sample_args=(), sample_kwargs=sample).export(fp32)
        log["export_seconds"] = time.time() - t0
    variants = {"fp32": fp32}
    if args.fp16:
        variants["wfp16"] = lh.to_fp16(fp32, os.path.join(out_dir, stem + "_wfp16.tflite"))
    if args.int8:
        variants["drq8"] = lh.to_drq_int8(fp32, os.path.join(out_dir, stem + "_drq8.tflite"))
    for kind, path in variants.items():
        scan = lh.op_scan(path)
        common.write_json(path.replace(".tflite", ".opscan.json"), scan)
        log["files"][kind] = {"path": path, "bytes": os.path.getsize(path), "sha256": sha256(path),
                              "num_ops": scan["num_ops"], "max_rank": scan["max_rank"], "banned": scan["banned"],
                              "io": scan["io"]}
        print(kind, os.path.basename(path), os.path.getsize(path), "B")
        print(lh.fmt_op_scan(scan))
    common.write_json(os.path.join(out_dir, stem + ".export.json"), log)


if __name__ == "__main__":
    main()
