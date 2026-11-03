#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1 (exports nvidia/NVIDIA-NemotronLabs-RE-USE weights)
"""Export the r6 GPU form (reuse_gpu_r6.R6ReUSE = r3's form with the dense encoder fixed for TFLite's mobile im2col
check) to fp32 .tflite with litert-torch, one fixed chunk length per file. A copy of scripts/export_reuse_r3.py; the
only changes are the model class, the @FIX part of CONFIG, the file name and the CONV_2D table in the .json.

  python -I export_reuse_r6.py OUTDIR SECONDS CONFIG [SR]      CONFIG = <time_form>/<freq_form>[+npu]@<fix>
     fix = tile (A) | split (B) | none
Writes OUTDIR/reuse_<sr/1000>k_<ms>ms_<tag>_fp32.tflite (tag = config with / + @ as _) and a .json next to it (as r3:
op histogram, op count, >4D tensors, banned ops, the largest tensors, the analytic bytes, and unless NO_CPU=1 a
CompiledModel CPU check on one real chunk against the torch form; plus conv2d = every CONV_2D's im2col).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

K = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(K / "scripts"))
sys.path.insert(0, str(K / "r6"))
import reuse_ref as R  # noqa: E402
import reuse_gpu as G  # noqa: E402
import reuse_gpu_r3 as G3  # noqa: E402
import reuse_gpu_r6 as G6  # noqa: E402
import host as H  # noqa: E402
import litert_helpers as LH  # noqa: E402
import scan_micro_r3 as SM  # noqa: E402
import conv_table_r6 as CT  # noqa: E402
from export_reuse import tensor_table, BANNED_HERE  # noqa: E402

CLIPS = Path.home() / "models" / "reuse" / "clips"
DIVLIKE = {"DIV", "FLOOR_DIV", "FLOOR_MOD", "RECIPROCAL"}


class _CounterUUID:
    """Env JAX_NAMES=counter. litert-torch 0.9.4 names every JAX-lowered MLIR function
    f"{fn.__name__}_{uuid.uuid4().hex[:8]}" (backend/jax_bridge/_wrap.py 225) = 32 random bits, and every aten add / sub
    / mul / div node gets a function of its own named jax_lowering_<8 hex> (lowerings/_jax_lowerings/lowerings.py
    486-700). With tens of thousands of such nodes two names collide by chance and the MLIR verify fails with
    "redefinition of symbol named 'jax_lowering_...'" (r6 05:36, the 2 s F2x/F1pq8 export). A counter in those 8 hex
    digits keeps every name unique. The names are private symbols that do not reach the .tflite (0 occurrences of
    "jax_lowering" in r3's and r6's graphs), so the graph is the same."""

    def __init__(self):
        self.n = 0

    def uuid4(self):
        import uuid
        self.n += 1
        return uuid.UUID(int=self.n << 96)


def main():
    od, secs, config = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    sr = int(sys.argv[4]) if len(sys.argv) > 4 else 16000
    od.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
    form, fix = config.split("@")
    enc_fix = None if fix == "none" else fix
    npu = form.endswith("+npu")
    tf, ff = (form[:-4] if npu else form).split("/")
    model, cfg = R.load_model()
    n_fft, hop, win = R.stft_params(cfg, sr)
    N = int(round(secs * sr))
    T, Fq = 1 + N // hop, n_fft // 2 + 1
    tag = config.replace("/", "_").replace("+", "_").replace("@", "_")
    name = f"reuse_{sr // 1000}k_{int(round(secs * 1000))}ms_{tag}_fp32"
    path = od / f"{name}.tflite"
    wav, wsr = R.read_wav(CLIPS / "vbd_p232_021.wav")
    assert wsr == sr and wav.shape[1] >= N
    spec = H.spec_in(wav[:, :N], sr, cfg)
    assert tuple(spec.shape) == (1, 2, T, Fq), spec.shape
    G3.EVAL_M_CHUNK = None
    G.EVAL_M_CHUNK = None
    G.EVAL_CONV_TILE = None
    gm = G6.R6ReUSE(model, time_form=tf, freq_form=ff, npu=npu, enc_fix=enc_fix).eval()
    with torch.no_grad():
        ref = gm(spec).numpy()
    t0 = time.time()
    import litert_torch
    jax_names = os.environ.get("JAX_NAMES", "uuid")
    if jax_names == "counter":
        import litert_torch.backend.jax_bridge._wrap as jax_wrap
        jax_wrap.uuid = _CounterUUID()
    litert_torch.convert(gm, (spec,)).export(str(path))
    conv_s = time.time() - t0
    scan = LH.op_scan(str(path))
    tt = tensor_table(str(path))
    gb = SM.graph_bytes(str(path))
    ct = CT.scan(str(path))
    banned = {k: v for k, v in scan["ops"].items() if k in BANNED_HERE or k in LH.BANNED_OPS}
    divlike = {k: v for k, v in scan["ops"].items() if k in DIVLIKE}
    info = dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
                config=config, time_form=tf, freq_form=ff, npu=npu, enc_fix=enc_fix, im2col_budget=G6.IM2COL_BUDGET,
                chunk_s=secs, sr=sr, n_fft=n_fft, hop=hop,
                win=win, input=[1, 2, T, Fq], output=list(ref.shape), convert_seconds=round(conv_s, 1),
                num_ops=scan["num_ops"], ops=scan["ops"], max_rank=scan["max_rank"],
                tensors_over_4d=scan["tensors_over_4d"], banned=banned, div_like=divlike,
                structural=scan["structural"], graph_bytes=gb,
                largest_tensors=[dict(name=n, shape=s, mb=round(b / 1e6, 1)) for b, n, s in tt[:8]],
                conv2d=[{k: c[k] for k in ("node", "input", "filter", "output", "stride", "dilation", "padding",
                                           "im2col_bytes", "im2col_gib", "margin_to_line", "fails_mobile_prepare")}
                        for c in ct["conv2d"]],
                max_im2col_gib=max(c["im2col_gib"] for c in ct["conv2d"]),
                mobile_prepare_fails=[c["node"] for c in ct["conv2d"] if c["fails_mobile_prepare"]],
                jax_symbol_names=jax_names,
                versions=dict(torch=torch.__version__, litert_torch=getattr(litert_torch, "__version__", "0.9.4")))
    if os.environ.get("NO_CPU") != "1":
        r = LH.TFLiteRunner(str(path), num_threads=int(os.environ.get("NTHREADS", "4")), accelerator="cpu")
        t1 = time.time()
        out = r.run_shaped([spec.numpy()], [list(ref.shape)])[0]
        info["cpu_check"] = dict(max_abs_vs_torch=float(np.abs(out - ref).max()), ref_amax=float(np.abs(ref).max()),
                                 first_run_s=round(time.time() - t1, 2), fully_accelerated=r.fully_accelerated)
        del r
    (od / f"{name}.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: info.get(k) for k in ("file", "bytes", "num_ops", "max_rank", "tensors_over_4d", "banned",
                                                "div_like", "graph_bytes", "cpu_check", "convert_seconds",
                                                "max_im2col_gib", "mobile_prepare_fails")}))
    print("ops", json.dumps(scan["ops"]))


if __name__ == "__main__":
    main()
