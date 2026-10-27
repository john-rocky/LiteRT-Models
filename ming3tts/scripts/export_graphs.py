"""Export the Ming-3.0-tts-0.5B LiteRT graphs with litert-torch (conversion venv).

  python export_graphs.py --graph agg|flow|lm|dec|frame [--fp16] [--chunk 8] [--steps 10]

fp32 graphs: litert-torch from the ming_modules modules (traced on zeros; trace values do not
enter the file). --fp16: ai_edge_quantizer FLOAT_CASTING of the fp32 file (fp16 weights,
fp32 compute = "wfp16"). Every file gets an all-subgraph op scan in out/opscan_<file>.json and
a row in out/export/manifest.json. Keep this script and ming_modules.py as run at export time:
module/class names and op order are part of the bytes.
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import litert_helpers as LH  # noqa: E402
import ming_modules as M  # noqa: E402

Z = torch.zeros


def sig_lm_prefill(n=64):
    return (Z(1, n, M.HID), Z(1, 1, n, 64), Z(1, 1, n, 64), Z(1, 1, n, M.P_LM + n),
            Z(1, 48, M.P_LM, 64), Z(1, 48, M.P_LM, 64))


def sig_lm_step():
    return (Z(1, 1, M.HID), Z(1, 1, 1, 64), Z(1, 1, 1, 64), Z(1, 1, 1, M.P_LM + 1),
            Z(1, 48, M.P_LM, 64), Z(1, 48, M.P_LM, 64))


def sig_flow(steps):
    return (Z(1, 1, M.HID), Z(1, 1, 64), Z(1, 1, 64), Z(1, steps, M.DIT_HID))


def sig_velocity():
    return (Z(1, 1, M.DIT_HID), Z(1, 1, M.HID), Z(1, 1, 64), Z(1, 1, 64))


def sig_dec(c):
    return (Z(1, c + 2, 64), Z(1, 1, 4 * c, 64), Z(1, 1, 4 * c, 64), Z(1, 1, 4 * c, M.P_DEC + 4 * c),
            Z(1, 48, M.P_DEC, 64), Z(1, 48, M.P_DEC, 64))


def sig_frame(steps):
    return (Z(1, 1, M.HID), Z(1, 1, 64), Z(1, 1, 64), Z(1, steps, M.DIT_HID)) + sig_lm_step()[1:]


def build(graph, chunk, steps):
    if graph == "agg":
        w = M.load_weights(["linear_proj_audio."])
        return [("agg", M.Aggregator(w).eval(), (Z(1, 1, 64),))]
    if graph == "flow":
        w = M.load_weights(["flowloss."])
        core = M.DiTCore(w).eval()
        return [("flow", M.Flow(w, steps, core=core).eval(), sig_flow(steps)),
                ("velocity", M.Velocity(w, core=core).eval(), sig_velocity())]
    if graph == "lm":
        w = M.load_weights(["model.model.layers.", "model.model.norm.", "stop_head."])
        lm = M.QwenStack(w, "model.model.").eval()
        return [("prefill", M.LMChunk(w, lm=lm).eval(), sig_lm_prefill()),
                ("step", M.LMStep(w, lm=lm).eval(), sig_lm_step())]
    if graph == "lm2":
        w = M.load_weights(["model.model.layers.", "model.model.norm."])
        lm = M.QwenStack(w, "model.model.", rs=0.25).eval()
        return [("prefill", M.LMChunk(w, lm=lm).eval(), sig_lm_prefill()),
                ("step", M.LMStepZ(w, lm=lm).eval(), sig_lm_step())]
    if graph == "frame2":
        w = M.load_weights(["model.model.layers.", "model.model.norm.", "stop_head.", "flowloss.",
                            "linear_proj_audio."])
        lm = M.QwenStack(w, "model.model.", rs=0.25).eval()
        return [("prefill", M.LMChunk(w, lm=lm).eval(), sig_lm_prefill()),
                ("frame", M.Frame(w, steps, lm=lm).eval(), sig_frame(steps))]
    if graph == "dec":
        w = M.load_weights(["audio.decoder."])
        return [("dec", M.DecChunk(w, chunk=chunk).eval(), sig_dec(chunk))]
    if graph == "frame":
        w = M.load_weights(["model.model.layers.", "model.model.norm.", "stop_head.", "flowloss.",
                            "linear_proj_audio."])
        lm = M.QwenStack(w, "model.model.").eval()
        return [("prefill", M.LMChunk(w, lm=lm).eval(), sig_lm_prefill()),
                ("frame", M.Frame(w, steps, lm=lm).eval(), sig_frame(steps))]
    raise ValueError(graph)


def scan_all(path):
    """Op histogram / ranks / dtypes for every subgraph (signature) of a flatbuffer."""
    schema, model = LH._load_model_t(path)
    name_of = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    types = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    names = [name_of.get(max(c.builtinCode, c.deprecatedBuiltinCode), "?") for c in model.operatorCodes]
    sig_of = {s.subgraphIndex: s.signatureKey.decode() for s in (model.signatureDefs or [])}
    out = {}
    for gi, g in enumerate(model.subgraphs):
        hist = collections.Counter(names[op.opcodeIndex] for op in g.operators)
        ranks = [len(t.shape) if t.shape is not None else 0 for t in g.tensors]
        dtypes = collections.Counter(types[t.type] for t in g.tensors)
        bmm = collections.Counter()
        fc_in = collections.Counter()
        for op in g.operators:
            n = names[op.opcodeIndex]
            ins = [i for i in op.inputs if i >= 0]
            if n == "BATCH_MATMUL":
                bmm["x".join(str(len(g.tensors[i].shape)) for i in ins)] += 1
            if n == "FULLY_CONNECTED":
                fc_in[str(len(g.tensors[ins[0]].shape))] += 1
        out[sig_of.get(gi, f"subgraph{gi}")] = {
            "num_ops": int(sum(hist.values())), "ops": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
            "max_rank": int(max(ranks)), "tensors_over_4d": int(sum(r > 4 for r in ranks)),
            "banned": {k: v for k, v in hist.items() if k in LH.BANNED_OPS},
            "tensor_dtypes": dict(dtypes), "batch_matmul_input_ranks": dict(bmm),
            "fully_connected_input_ranks": dict(fc_in)}
    return out


def manifest_add(name, entry):
    man = C.EXPORT / "manifest.json"
    d = json.load(open(man)) if man.exists() else {}
    d[name] = entry
    C.write_json(man, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--fp16", action="store_true", help="also write the wfp16 file")
    ap.add_argument("--fp16-only", action="store_true", help="fp16 from an existing fp32 file")
    args = ap.parse_args()
    C.ensure_dirs()
    tag = args.graph
    if args.graph == "dec":
        tag = f"dec_c{args.chunk}"
    if args.graph in ("flow", "frame", "frame2") and args.steps != 10:
        tag = f"{tag}_s{args.steps}"
    fp32 = C.EXPORT / f"{tag}_fp32.tflite"
    versions = C.env_versions(["litert-torch", "ai-edge-litert", "ai-edge-quantizer", "torch", "jax"])
    if not args.fp16_only:
        t0 = time.time()
        sigs = build(args.graph, args.chunk, args.steps)
        import litert_torch
        conv = None
        for key, mod, sample in sigs:
            conv = litert_torch.signature(key, mod, sample) if conv is None else conv.signature(key, mod, sample)
        edge = conv.convert()
        edge.export(str(fp32))
        dt = time.time() - t0
        scan = scan_all(fp32)
        C.write_json(C.OUT / f"opscan_{fp32.stem}.json", scan)
        manifest_add(f"{tag}_fp32", {"file": fp32.name, "form": "fp32", "bytes": os.path.getsize(fp32),
                                     "sha256": C.sha256(fp32), "export_seconds": dt,
                                     "signatures": list(scan), "versions": versions,
                                     "chunk": args.chunk if args.graph == "dec" else None,
                                     "steps": args.steps if args.graph in ("flow", "frame", "frame2") else None})
        print("EXPORTED", fp32.name, os.path.getsize(fp32), "bytes", round(dt, 1), "s",
              json.dumps({k: (v["num_ops"], v["max_rank"], v["banned"]) for k, v in scan.items()}), flush=True)
    if args.fp16 or args.fp16_only:
        t0 = time.time()
        f16 = C.EXPORT / f"{tag}_wfp16.tflite"
        LH.to_fp16(str(fp32), str(f16))
        scan = scan_all(f16)
        C.write_json(C.OUT / f"opscan_{f16.stem}.json", scan)
        manifest_add(f"{tag}_wfp16", {"file": f16.name, "form": "wfp16", "bytes": os.path.getsize(f16),
                                      "sha256": C.sha256(f16), "export_seconds": time.time() - t0,
                                      "from": fp32.name, "signatures": list(scan), "versions": versions})
        print("FP16", f16.name, os.path.getsize(f16), "bytes", round(time.time() - t0, 1), "s",
              json.dumps({k: (v["num_ops"], v["max_rank"], v["banned"]) for k, v in scan.items()}), flush=True)


if __name__ == "__main__":
    main()
