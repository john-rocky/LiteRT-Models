#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1 (exports nvidia/NVIDIA-NemotronLabs-RE-USE weights)
"""Export the RE-USE GPU form (reuse_gpu.GPUReUSE) to fp32 .tflite with litert-torch, one fixed chunk length per file.

  python -I export_reuse.py OUTDIR SECONDS [SR]      e.g. 0.5 / 1 / 2 at 16000

input  spec [1, 2, T, F]  (T = 1 + N / hop frames for N = SECONDS * SR samples, F = n_fft / 2 + 1)
output out  [1, 3, T, F]  (compressed magnitude, x_r, x_i; host post-processing in host.py)
Writes OUTDIR/reuse_<sr/1000>k_<ms>ms_fp32.tflite and a .json next to it: op histogram, op count, >4D tensors,
banned ops, the largest tensors (fp32 bytes), tensors >= 100 MB, and a CompiledModel CPU check on one real chunk
against the torch GPU form (max |d|).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reuse_ref as R  # noqa: E402
import reuse_gpu as G  # noqa: E402
import host as H  # noqa: E402
import litert_helpers as LH  # noqa: E402

CLIPS = Path.home() / "models" / "reuse" / "clips"
BANNED_HERE = {"GATHER_ND", "GATHER", "SELECT", "SELECT_V2", "REVERSE_V2", "ATAN2"}


def tensor_table(path):
    schema, m = LH._load_model_t(path)
    g = m.subgraphs[0]
    rows = []
    for t in g.tensors:
        shp = [int(d) for d in t.shape] if t.shape is not None else []
        n = int(np.prod(shp)) if shp else 1
        rows.append((n * 4, LH._tname(t), shp))
    rows.sort(key=lambda r: -r[0])
    return rows


def main():
    od, secs = Path(sys.argv[1]), float(sys.argv[2])
    sr = int(sys.argv[3]) if len(sys.argv) > 3 else 16000
    od.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
    model, cfg = R.load_model()
    n_fft, hop, win = R.stft_params(cfg, sr)
    N = int(round(secs * sr))
    T, Fq = 1 + N // hop, n_fft // 2 + 1
    name = f"reuse_{sr // 1000}k_{int(round(secs * 1000))}ms_fp32"
    path = od / f"{name}.tflite"
    # one real chunk (the first N samples of the longest VoiceBank clip) as the sample / check input
    wav, wsr = R.read_wav(CLIPS / "vbd_p232_021.wav")
    assert wsr == sr and wav.shape[1] >= N
    spec = H.spec_in(wav[:, :N], sr, cfg)
    assert tuple(spec.shape) == (1, 2, T, Fq), spec.shape
    G.EVAL_M_CHUNK = None
    gm = G.GPUReUSE(model, row_scale=True).eval()
    with torch.no_grad():
        ref = gm(spec).numpy()
    t0 = time.time()
    import litert_torch
    em = litert_torch.convert(gm, (spec,))
    em.export(str(path))
    conv_s = time.time() - t0
    scan = LH.op_scan(str(path))
    tt = tensor_table(str(path))
    big = [dict(name=n, shape=s, mb=round(b / 1e6, 1)) for b, n, s in tt if b >= 100e6]
    banned = {k: v for k, v in scan["ops"].items() if k in BANNED_HERE or k in LH.BANNED_OPS}
    # CompiledModel CPU check
    r = LH.TFLiteRunner(str(path), num_threads=8, accelerator="cpu")
    t1 = time.time()
    out = r.run_shaped([spec.numpy()], [list(ref.shape)])[0]
    cpu_s = time.time() - t1
    d = float(np.abs(out - ref).max())
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    info = dict(file=str(path), sha256=h, bytes=path.stat().st_size, chunk_s=secs, sr=sr, n_fft=n_fft, hop=hop, win=win,
                input=[1, 2, T, Fq], output=list(ref.shape), T_enc=(T + 2 - 1) // 4 + 1, F_enc=(Fq + 2 - 3) // 2 + 1,
                convert_seconds=round(conv_s, 1), num_ops=scan["num_ops"], ops=scan["ops"], max_rank=scan["max_rank"],
                tensors_over_4d=scan["tensors_over_4d"], banned=banned, structural=scan["structural"],
                largest_tensors=[dict(name=n, shape=s, mb=round(b / 1e6, 1)) for b, n, s in tt[:8]],
                tensors_ge_100MB=len(big), cpu_check=dict(max_abs_vs_torch=d, ref_amax=float(np.abs(ref).max()),
                                                           first_run_s=round(cpu_s, 2),
                                                           fully_accelerated=r.fully_accelerated),
                versions=dict(torch=torch.__version__, litert_torch=getattr(litert_torch, "__version__", "0.9.4")))
    (od / f"{name}.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({k: info[k] for k in ("file", "bytes", "num_ops", "max_rank", "tensors_over_4d", "banned",
                                            "tensors_ge_100MB", "cpu_check", "convert_seconds")}))
    print("ops", json.dumps(scan["ops"]))
    print("largest", json.dumps(info["largest_tensors"][:4]))


if __name__ == "__main__":
    main()
