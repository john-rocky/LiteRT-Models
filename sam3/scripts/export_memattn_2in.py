#!/usr/bin/env python3
"""Re-export the memory-attention graph with its flat input SPLIT into two tensors.

The iOS Metal runtime rejects tensor buffers above ~112 MiB (0x7000000 bytes):
managed host-memory buffers fail with kLiteRtStatusErrorRuntimeFailure and the
required MetalBufferPacked type errors out too. The n7 memattn input is 117.4 MB —
just over. Splitting the flat input at the mem_img_pos/maskmem boundary yields
79.6 MB + 37.8 MB tensors, both comfortably under, with bit-identical numerics
(the two inputs are concatenated back inside the graph). The concatenation order
matches the packed host array, so a multi-input-aware runner can feed the same
flat array sliced sequentially — no host-side changes.

Usage: export_memattn_2in.py [--slots 7] [--chunks 9]
Writes models/tracker_precheck/trk_memattn_n{N}_2in{,_fp32}.tflite
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import precheck_sam3 as P  # noqa: E402
import tracker_precheck as TP  # noqa: E402
from tracker_patches import patch_memattn  # noqa: E402


class MemAttn2In(TP.MemAttn):
    """Same graph, flat input split at offs[3] (pix|mem_img|mem_img_pos // rest)."""

    def forward(self, a, b):  # noqa: D102
        return super().forward(torch.cat([a, b], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=7)
    ap.add_argument("--ptr-frames", type=int, default=16)
    ap.add_argument("--chunks", type=int, default=9)
    ap.add_argument("--out", default=os.path.join(P.ROOT, "models", "tracker_precheck"))
    a = ap.parse_args()
    name = f"trk_memattn_n{a.slots}_2in"

    trk = TP.build_tracker(os.path.join(P.ROOT, "models", "sam3.1_multiplex.pt"))
    m = MemAttn2In(trk, a.slots, a.ptr_frames)
    torch.manual_seed(0)
    x = torch.randn(1, m.offs[-1]) * 0.5
    x[:, m.offs[6]:] = 1.0
    split = m.offs[3]
    xa, xb = x[:, :split].contiguous(), x[:, split:].contiguous()
    print(f"[{name}] inputs {tuple(xa.shape)} ({xa.numel()*4/1e6:.1f} MB) + "
          f"{tuple(xb.shape)} ({xb.numel()*4/1e6:.1f} MB)")

    print("[patch] memattn RoPE attentions:", patch_memattn(trk, q_chunks=a.chunks, n_slots=a.slots))
    with torch.inference_mode():
        ref = m(xa, xb)

    import litert_torch
    import time
    fp32 = os.path.join(a.out, f"{name}_fp32.tflite")
    t0 = time.time()
    ep = litert_torch.convert(m.eval(), (xa, xb))
    ep.export(fp32)
    print(f"[convert {name}] ok {time.time()-t0:.0f}s  fp32={os.path.getsize(fp32)/1e6:.1f} MB")
    P.opcheck(fp32, name)
    dst = os.path.join(a.out, f"{name}.tflite")
    print(f"[fp16 {name}] {P.fp16(fp32, dst):.1f} MB")

    # CPU parity of the fp32 file vs the patched torch reference, inputs matched by size
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=fp32, num_threads=8)
    it.allocate_tensors()
    dets = it.get_input_details()
    by_size = {int(np.prod(d["shape"])): d["index"] for d in dets}
    assert len(by_size) == 2, dets
    it.set_tensor(by_size[xa.numel()], xa.numpy())
    it.set_tensor(by_size[xb.numel()], xb.numpy())
    it.invoke()
    y = it.get_tensor(it.get_output_details()[0]["index"]).reshape(-1)
    r = ref.numpy().reshape(-1)
    d = np.abs(y - r).max()
    corr = float(np.corrcoef(y, r)[0, 1])
    print(f"[tflite-cpu {name}] corr={corr:.7f} max|diff|={d:.3g}")
    assert corr > 0.99999 and d < 1e-3, "parity gate failed"
    print("[done]")


if __name__ == "__main__":
    main()
