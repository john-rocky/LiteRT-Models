#!/usr/bin/env python3
"""r3 micro graphs: ONE Mamba scan (one form, one axis, one chunk length) as its own .tflite, to rank the scan forms on
the Mac before the full graphs. Inputs delta, du, B, C (runtime, fp32), A = the real layer's (constant); output y.

  python -I scan_micro_r3.py export OUTDIR FORM AXIS SECS     AXIS = time | freq; SECS = 0.5 | 1 | 2
      -> OUTDIR/micro_<FORM>_<axis>_<ms>ms.tflite + .json (op table, analytic bytes, CPU check vs the sequential ref)
  python -I scan_micro_r3.py time OUT.json TFLITE...            Metal (default precision) and CPU (XNNPACK, NTHREADS)
      warm medians (N timed calls after 2 warm-ups), inside a quiet_hold window
Analytic bytes (graph_bytes): over every op, the bytes of its runtime input tensors + its output tensors at 2 B per
value (fp16, Metal's default precision; no fusion assumed) = "rw_fp16"; outputs only = "w_fp16".
"""
import collections
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import litert_helpers as LH  # noqa: E402
import scan_forms_r3 as SF  # noqa: E402

K = Path(__file__).resolve().parent.parent
LAYER = {"time": "TSMamba.0.time_mamba.forward_blocks", "freq": "TSMamba.14.freq_mamba.backward_blocks"}


def enc_len(secs, sr=16000, hop=80):
    T = 1 + int(round(secs * sr)) // hop
    return (T + 2 - 1) // 4 + 1


def shapes(axis, secs):
    Tp, Fp = enc_len(secs), 161
    return (Tp, Fp) if axis == "time" else (Fp, Tp)          # (L, M)


def graph_bytes(path):
    """Analytic bytes of a .tflite (see the module docstring) and the op count."""
    schema, m = LH._load_model_t(path)
    g = m.subgraphs[0]

    def is_const(ti):
        b = m.buffers[g.tensors[ti].buffer]
        return (b.data is not None and len(b.data) > 0) or (b.offset is not None and b.offset > 1 and b.size)

    def nel(ti):
        s = g.tensors[ti].shape
        return int(np.prod(s)) if s is not None and len(s) else 1

    rw = w = 0
    for op in g.operators:
        for i in (op.inputs if op.inputs is not None else []):
            if i >= 0 and not is_const(i):
                rw += nel(i) * 2
        for o in op.outputs:
            rw += nel(o) * 2
            w += nel(o) * 2
    return dict(rw_fp16=rw, w_fp16=w, ops=len(g.operators))


class ScanModule(torch.nn.Module):
    def __init__(self, form, A, reverse):
        super().__init__()
        self.fn, self.kw = SF.get(form)
        self.reverse = reverse
        D, N = A.shape
        self.register_buffer("A", A.clone())
        self.register_buffer("A4", A.reshape(1, 1, D, N).clone())
        self.register_buffer("AT", A.t().reshape(1, N, 1, D).contiguous())
        self.register_buffer("AP", A.t().reshape(1, 1, N, D).contiguous())

    def forward(self, delta, du, Bm, Cm):
        return self.fn(delta, du, Bm, Cm, self.A, self.reverse, A4=self.A4, AT=self.AT, AP=self.AP, **self.kw)


def inputs(m, L, M, seed=0):
    st = json.loads((K / "r1" / "ref_stats_mic_test2.json").read_text())
    s = next(r for r in st["rows"] if r["kind"] == "scan")
    gen = torch.Generator().manual_seed(seed)
    u = torch.randn(1, L, M, 256, generator=gen) * s["x_std"]
    dt = torch.randn(1, L, M, 256, generator=gen) * s["dt_std"] + m.dt_proj.bias.detach()
    delta = torch.nn.functional.softplus(dt)
    B = torch.randn(1, L, M, 16, generator=gen) * s["B_std"]
    C = torch.randn(1, L, M, 16, generator=gen) * s["C_std"]
    return delta, delta * u, B, C


def seq_ref(delta, du, B, C, A, reverse):
    _, L, M, D = delta.shape
    h = torch.zeros(1, M, D, A.shape[1])
    ys = [None] * L
    for l in (range(L) if not reverse else range(L - 1, -1, -1)):
        h = torch.exp(delta[:, l, :, :, None] * A) * h + du[:, l, :, :, None] * B[:, l, :, None, :]
        ys[l] = (h * C[:, l, :, None, :]).sum(-1)
    return torch.stack(ys, 1)


def cmd_export(od, form, axis, secs):
    import reuse_ref as R
    od.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("NTHREADS", "1")))
    model, _ = R.load_model()
    m = dict(model.named_modules())[LAYER[axis]]
    A = (-torch.exp(m.A_log.detach().float()))
    reverse = axis == "freq"
    L, M = shapes(axis, secs)
    mod = ScanModule(form, A, reverse).eval()
    x = inputs(m, L, M)
    with torch.no_grad():
        ref = seq_ref(*x, A, reverse)
        y = mod(*x)
    d_torch = float((y - ref).abs().max())
    fan = os.environ.get("R3_CAT_FANIN")
    name = f"micro_{form}_{axis}_{int(round(secs * 1000))}ms" + (f"_cat{fan}" if fan else "")
    path = od / f"{name}.tflite"
    t0 = time.time()
    import litert_torch
    litert_torch.convert(mod, x).export(str(path))
    conv_s = time.time() - t0
    scan = LH.op_scan(str(path))
    gb = graph_bytes(str(path))
    r = LH.TFLiteRunner(str(path), num_threads=1, accelerator="cpu")
    out = r.run_shaped([t.numpy() for t in x], [list(ref.shape)])[0]
    d_cpu = float(np.abs(out - ref.numpy()).max())
    info = dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), form=form, axis=axis,
                secs=secs, L=L, M=M, reverse=reverse, convert_s=round(conv_s, 1), num_ops=scan["num_ops"],
                ops=scan["ops"], max_rank=scan["max_rank"], tensors_over_4d=scan["tensors_over_4d"],
                banned=scan["banned"], structural=scan["structural"], graph_bytes=gb,
                state_units=SF.state_tensors(form if not form.startswith("F1q") else form, L),
                torch_vs_seqref=d_torch, cpu_vs_seqref=d_cpu, ref_amax=float(ref.abs().max()),
                cpu_fully_accelerated=r.fully_accelerated, cat_fanin=SF.CAT_FANIN)
    path.with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n")
    print(name, json.dumps({k: info[k] for k in ("num_ops", "max_rank", "banned", "convert_s", "torch_vs_seqref",
                                                  "cpu_vs_seqref", "state_units", "cpu_fully_accelerated")}),
          json.dumps(gb), flush=True)


def cmd_time(out, paths):
    n = int(os.environ.get("N", "9"))
    nth = int(os.environ.get("NTHREADS", "4"))
    bes = os.environ.get("BACKENDS", "metal,cpu").split(",")
    res = json.loads(out.read_text()) if out.exists() else {}
    for p in paths:
        meta = json.loads(Path(p).with_suffix(".json").read_text())
        L, M = meta["L"], meta["M"]
        x = [np.random.RandomState(0).randn(*s).astype(np.float32) * 0.1
             for s in ((1, L, M, 256), (1, L, M, 256), (1, L, M, 16), (1, L, M, 16))]
        x[0] = np.abs(x[0])
        for be in bes:
            t0 = time.perf_counter()
            if be == "cpu":
                run = LH.TFLiteRunner(p, num_threads=nth, accelerator="cpu")
            else:
                run = LH.TFLiteRunner(p, accelerator="gpu", gpu_fp32=(be == "metal32"))
            cs = time.perf_counter() - t0
            ts = []
            for i in range(n + 2):
                t1 = time.perf_counter()
                run.run_shaped(x, [[1, L, M, 256]])
                ts.append((time.perf_counter() - t1) * 1000)
            key = f"{Path(p).stem}:{be}"
            res[key] = dict(compile_s=round(cs, 2), ms_median=round(statistics.median(ts[2:]), 2),
                            ms_min=round(min(ts[2:]), 2), warm_ms=[round(t, 2) for t in ts[2:]], n=n,
                            threads=nth if be == "cpu" else None, fully_accelerated=run.fully_accelerated,
                            num_ops=meta["num_ops"], written=time.strftime("%H:%M:%S"))
            out.write_text(json.dumps(res, indent=1) + "\n")
            print(key, json.dumps({k: res[key][k] for k in ("compile_s", "ms_median", "ms_min", "fully_accelerated")}),
                  flush=True)
            del run


if __name__ == "__main__":
    if sys.argv[1] == "export":
        cmd_export(Path(sys.argv[2]), sys.argv[3], sys.argv[4], float(sys.argv[5]))
    else:
        cmd_time(Path(sys.argv[2]), sys.argv[3:])
