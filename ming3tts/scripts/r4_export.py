"""Export the node-reduced flow and aggregator (conversion venv, litert-torch 0.9.4).

  python r4_export.py --name flow_r4b_s7|flow_r4a_s7|velocity_r4b|agg_r4b [--fp32-keep]

Names: *_r4b* = the r4_modules rewrite; flow_r4a_s<N> = ming_modules.Flow at N EPSS steps with
r4_modules.forward_epsdiv. One signature per file ("flow" / "velocity" / "agg", inputs as in
export_graphs.py). fp32 -> ai_edge_quantizer FLOAT_CASTING -> wfp16 (export_graphs.py's path). Op scan per file in
out/r4/opscan_<file>.json, row in out/r4/manifest.json. Keep this script and r4_modules.py as run at
export time: module/class names and op order are part of the bytes.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import export_graphs as E  # noqa: E402
import litert_helpers as LH  # noqa: E402
import ming_modules as M  # noqa: E402
import r4_modules as R  # noqa: E402

OUT = C.OUT / "r4"
Z = torch.zeros


def build(name):
    if name.startswith("flow_r4b_s"):
        n = int(name.split("_s")[-1])
        core1, _ = R.r1_core()
        return [("flow", R.R4Flow(R.R4Core(core1).eval(), n).eval(), E.sig_flow(n))]
    if name == "velocity_r4b":
        core1, _ = R.r1_core()
        return [("velocity", R.R4Velocity(R.R4Core(core1).eval()).eval(), E.sig_velocity())]
    if name == "agg_r4b":
        agg1, _ = R.r1_agg()
        return [("agg", R.R4Aggregator(agg1).eval(), (Z(1, 1, 64),))]
    if name.startswith("flow_r4a_s"):
        n = int(name.split("_s")[-1])
        core1, _ = R.r1_core()
        return [("flow", M.Flow(None, n, core=core1).eval(), E.sig_flow(n))]
    raise ValueError(name)


def manifest_add(name, entry):
    import json
    man = OUT / "manifest.json"
    d = json.load(open(man)) if man.exists() else {}
    d[name] = entry
    C.write_json(man, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fp32-keep", action="store_true", help="keep the fp32 file (default: only wfp16 kept)")
    ap.add_argument("--fp32-only", action="store_true",
                    help="write only the fp32 file to out/r4/fp32/ (the device one-call reference); wfp16 untouched")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import json
    versions = C.env_versions(["litert-torch", "ai-edge-litert", "ai-edge-quantizer", "torch", "jax"])
    orig = M.SafeRMSNorm.forward
    if "_r4a_" in args.name:
        M.SafeRMSNorm.forward = R.forward_epsdiv
    t0 = time.time()
    try:
        sigs = build(args.name)
        import litert_torch
        conv = None
        for key, mod, sample in sigs:
            conv = litert_torch.signature(key, mod, sample) if conv is None else conv.signature(key, mod, sample)
        fp32 = (OUT / "fp32" if args.fp32_only else OUT) / f"{args.name}_fp32.tflite"
        fp32.parent.mkdir(parents=True, exist_ok=True)
        conv.convert().export(str(fp32))
    finally:
        M.SafeRMSNorm.forward = orig
    dt = time.time() - t0
    if args.fp32_only:
        print("EXPORTED_FP32_ONLY", fp32, os.path.getsize(fp32), "bytes sha256", C.sha256(fp32), round(dt, 1), "s", flush=True)
        return
    scan = E.scan_all(fp32)
    C.write_json(OUT / f"opscan_{fp32.stem}.json", scan)
    manifest_add(f"{args.name}_fp32", {"file": fp32.name, "form": "fp32", "bytes": os.path.getsize(fp32),
                                       "sha256": C.sha256(fp32), "export_seconds": dt, "signatures": list(scan),
                                       "versions": versions, "kept": bool(args.fp32_keep)})
    print("EXPORTED", fp32.name, os.path.getsize(fp32), "bytes", round(dt, 1), "s",
          json.dumps({k: (v["num_ops"], v["max_rank"], v["banned"], v["batch_matmul_input_ranks"]) for k, v in scan.items()}),
          flush=True)
    t1 = time.time()
    f16 = OUT / f"{args.name}_wfp16.tflite"
    LH.to_fp16(str(fp32), str(f16))
    scan16 = E.scan_all(f16)
    C.write_json(OUT / f"opscan_{f16.stem}.json", scan16)
    st = LH.structural_checks(f16)
    manifest_add(f"{args.name}_wfp16", {"file": f16.name, "form": "wfp16", "bytes": os.path.getsize(f16),
                                        "sha256": C.sha256(f16), "export_seconds": time.time() - t1, "from": fp32.name,
                                        "signatures": list(scan16), "versions": versions, "structural": st})
    print("FP16", f16.name, os.path.getsize(f16), "bytes", json.dumps({k: (v["num_ops"], v["ops"]) for k, v in scan16.items()}),
          flush=True)
    print("STRUCT", f16.name, json.dumps({k: v for k, v in st.items() if k in (
        "const_only_ops", "outputs_with_consumers", "large_const_operands", "mixed_rank_elementwise", "batch_matmul_forms")}),
        flush=True)
    if not args.fp32_keep:
        fp32.unlink()


if __name__ == "__main__":
    main()
