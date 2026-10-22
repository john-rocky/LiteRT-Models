"""Export the r4 forms (r4_graph.TimestralGraphR4) with litert-torch, plus the fp16-weight file.

  python r4_export.py --bucket 64x8 --rows 1 --norm pow2 --norm-k out/normk/normk_fixture.json --attn-k 4 \
      --attn-clamp --dec-only-last --fp16

Same file naming as export_tflite.py (r1) with `_dl` after the attention tag when --dec-only-last is set:
out/tflite/timestral209m_r{R}_l{L}_h{H}_{norm}{_attK[c]}_dl_{fp32|wfp16}.tflite, the op scan of each file and
<stem>.export.json. The host tables (forecast_query, RoPE) are r1's (out/host), unchanged by these forms.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import export_tflite as ex  # noqa: E402  (r1: sha256)
import litert_helpers as lh  # noqa: E402
import r4_graph  # noqa: E402
import timestral_graph as tg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, help="LxH in patches, e.g. 64x8")
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--norm", default="exact", choices=["exact", "pow2", "amax"])
    ap.add_argument("--norm-k", default=None, help="json file: {layer: {attn0|attn1|mlp: k}, 'out': {'out': k}}")
    ap.add_argument("--attn-k", type=int, default=0)
    ap.add_argument("--attn-clamp", action="store_true")
    ap.add_argument("--dec-only-last", action="store_true", help="last block: queries / MLP on the decoder tokens only")
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    ll, hh = (int(v) for v in args.bucket.split("x"))
    torch.set_grad_enabled(False)
    out_dir = os.path.join(common.OUT, "tflite")
    os.makedirs(out_dir, exist_ok=True)
    model = common.load_official()
    norm_k = None
    if args.norm_k:
        with open(args.norm_k) as f:
            raw = json.load(f)
        norm_k = {(int(k) if k.isdigit() else k): v for k, v in raw.items()}
    graph = r4_graph.TimestralGraphR4(model, ll, hh, dec_only_last=args.dec_only_last, norm_mode=args.norm,
                                      norm_k=norm_k, attn_k=args.attn_k, attn_clamp=args.attn_clamp).eval()
    norm_tag = args.norm + (f"k{os.path.splitext(os.path.basename(args.norm_k))[0]}" if args.norm_k else "")
    stem = (f"timestral209m_r{args.rows}_l{ll}_h{hh}_{norm_tag}" + (f"_att{args.attn_k}" if args.attn_k else "")
            + ("c" if args.attn_clamp else "") + ("_dl" if args.dec_only_last else ""))
    fp32 = os.path.join(out_dir, stem + "_fp32.tflite")
    log = {"bucket": [ll, hh], "rows": args.rows, "norm": args.norm, "dec_only_last": args.dec_only_last,
           "script": "scripts/r4_export.py", "files": {}}
    import litert_torch
    sample = dict(zip(tg.INPUT_NAMES, tg.sample_inputs(args.rows, ll, hh, graph.d_model)))
    # torch-side check before the export: the r4 graph against the r1 graph on a random input (same weights)
    if args.dec_only_last:
        base = tg.TimestralGraph(model, ll, hh, norm_mode=args.norm, norm_k=norm_k, attn_k=args.attn_k,
                                 attn_clamp=args.attn_clamp).eval()
        g = torch.Generator().manual_seed(0)
        s = ll + hh
        x = [torch.randn(args.rows, ll, 64, generator=g), sample["dec"] + torch.randn(args.rows, hh, graph.d_model, generator=g),
             torch.zeros(1, 1, s, s), torch.cos(torch.randn(1, 1, s, 64, generator=g)), torch.sin(torch.randn(1, 1, s, 64, generator=g))]
        x[0][..., 32:] = 0.0
        a, b = graph(*x), base(*x)
        log["torch_vs_r1_graph_max_abs"] = float((a - b).abs().max())
        log["torch_vs_r1_graph_bit_equal"] = bool(torch.equal(a, b))
        print(f"torch dec_only_last vs r1 graph: max|d| {log['torch_vs_r1_graph_max_abs']:.3e} "
              f"bit-equal {log['torch_vs_r1_graph_bit_equal']}")
        del base
    t0 = time.time()
    litert_torch.convert(graph, sample_args=(), sample_kwargs=sample).export(fp32)
    log["export_seconds"] = time.time() - t0
    variants = {"fp32": fp32}
    if args.fp16:
        variants["wfp16"] = lh.to_fp16(fp32, os.path.join(out_dir, stem + "_wfp16.tflite"))
    for kind, path in variants.items():
        scan = lh.op_scan(path)
        common.write_json(path.replace(".tflite", ".opscan.json"), scan)
        log["files"][kind] = {"path": path, "bytes": os.path.getsize(path), "sha256": ex.sha256(path),
                              "num_ops": scan["num_ops"], "max_rank": scan["max_rank"], "banned": scan["banned"],
                              "io": scan["io"]}
        print(kind, os.path.basename(path), os.path.getsize(path), "B", scan["num_ops"], "ops")
        print(lh.fmt_op_scan(scan))
    common.write_json(os.path.join(out_dir, stem + ".export.json"), log)
    np.random.seed(0)


if __name__ == "__main__":
    main()
