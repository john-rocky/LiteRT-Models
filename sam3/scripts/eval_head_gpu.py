#!/usr/bin/env python3
"""Run models/out/sam3_head.tflite on the Mac GPU with the saved fixture; split metrics."""
import os, sys, time, numpy as np
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.options import Options
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fx = os.path.join(ROOT, "models/out/fixtures")
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models/out/sam3_head.tflite")
vis = np.load(os.path.join(fx, "vision_ref.npy")); txt = np.load(os.path.join(fx, "text_ref.npy")); pad = np.load(os.path.join(fx, "text_pad.npy"))
ref = np.load(os.path.join(fx, "head_ref.npy")).reshape(-1)
xin = np.concatenate([vis, txt, pad], 1).astype(np.float32)
def split(y):
    out = []
    for tag, sl in [("logits", slice(0,200)), ("boxes", slice(200,1000)), ("presence", slice(1000,1001)), ("masks", slice(1001,None))]:
        a, b = y[sl], ref[sl]
        c = np.corrcoef(a, b)[0,1] if a.size > 1 else float("nan")
        out.append(f"{tag}: corr={c:.5f} max|d|={np.abs(a-b).max():.3g}")
    prob = 1/(1+np.exp(-y[:200])) * 1/(1+np.exp(-y[1000])); pref = 1/(1+np.exp(-ref[:200])) * 1/(1+np.exp(-ref[1000]))
    kg, kr = np.where(prob>0.5)[0], np.where(pref>0.5)[0]
    out.append(f"kept: gpu={kg.tolist()} ref={kr.tolist()} max|dprob|={np.abs(prob-pref).max():.4f}")
    if len(kr):
        my = y[1001:].reshape(200, 288, 288) > 0
        mr = ref[1001:].reshape(200, 288, 288) > 0
        ious = []
        for q in kr:
            inter = (my[q] & mr[q]).sum(); union = max((my[q] | mr[q]).sum(), 1)
            ious.append(inter / union)
        out.append(f"kept-mask IoU min/mean={min(ious):.4f}/{np.mean(ious):.4f}")
        bg = y[200:1000].reshape(200, 4); br = ref[200:1000].reshape(200, 4)
        def xyxy(t):
            return np.stack([t[:,0]-t[:,2]/2, t[:,1]-t[:,3]/2, t[:,0]+t[:,2]/2, t[:,1]+t[:,3]/2], 1)
        A, B = xyxy(bg[kr]), xyxy(br[kr])
        lt = np.maximum(A[:, :2], B[:, :2]); rb = np.minimum(A[:, 2:], B[:, 2:])
        wh = np.clip(rb - lt, 0, None); inter = wh[:,0]*wh[:,1]
        area = lambda t: (t[:,2]-t[:,0])*(t[:,3]-t[:,1])
        iou = inter / (area(A) + area(B) - inter + 1e-9)
        out.append(f"kept-box IoU min/mean={iou.min():.4f}/{iou.mean():.4f}")
    return "  ".join(out)
for f32 in (False, True):
    if f32:
        o = Options.create(); o.hardware_accelerators = HardwareAccelerator.GPU; o.gpu_options.enforce_f32 = True
        m = CompiledModel.from_file(path, options=o)
    else:
        m = CompiledModel.from_file(path, HardwareAccelerator.GPU)
    ib = m.create_input_buffers(0); ob = m.create_output_buffers(0); ib[0].write(xin.ravel())
    ts = []
    for _ in range(3):
        t0 = time.time(); m.run_by_index(0, ib, ob); y = np.array(ob[0].read(ref.size, np.float32)); ts.append(time.time() - t0)
    m.close()
    print(f"[head {'f32' if f32 else 'fp16'}] {min(ts)*1000:.0f} ms  " + split(y), flush=True)
