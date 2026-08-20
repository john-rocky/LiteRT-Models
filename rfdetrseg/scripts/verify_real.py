# Real-image end-to-end verify (detector ship criterion: IoU/class/mask agreement, NOT raw corr).
#   reference = official RFDETRSegNano.predict() (PyTorch, threshold 0.5)
#   candidate = device chain: rfsA(GPU on Pixel 8a) -> host select -> rfsB(GPU) -> decode
# usage: verify_real.py <image> [thr]
import sys, os, subprocess
import numpy as np, torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_rfdetrseg_split as S

IMG = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/meeting/rfdetr-work/demo.jpg")
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
MEANS = np.array([0.485, 0.456, 0.406], np.float32)
STDS = np.array([0.229, 0.224, 0.225], np.float32)

net, inner, clspos, pospatch = S.build_net()
proposals = S.build_proposals(S.GH, S.GW)
rp = net.refpoint_embed.weight[:S.NQ].unsqueeze(0).detach().clone()
qf0 = net.query_feat.weight[:S.NQ].unsqueeze(0).detach().clone()

pil = Image.open(IMG).convert("RGB")
W0, H0 = pil.size
im = np.asarray(pil.resize((S.R, S.R), Image.BILINEAR), np.float32) / 255.0
x = torch.from_numpy(((im - MEANS) / STDS).transpose(2, 0, 1)[None].copy())

# ---- official reference (eager torch, full model path) ----
with torch.no_grad():
    ref_coord, ref_cls, ref_masks = net.forward_export(x)
def decode(coord, cls, masks, thr):
    sc = torch.sigmoid(cls)
    score, label = sc.max(-1)
    keep = (score[0] > thr).nonzero().flatten()
    cx, cy, w, h = coord[0, keep].T
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)
    boxes = boxes * torch.tensor([W0, H0, W0, H0], dtype=torch.float32)
    m = torch.nn.functional.interpolate(masks[:, keep], size=(H0, W0), mode="bilinear",
                                        align_corners=False)[0] > 0
    return boxes, label[0, keep], score[0, keep], m, keep
rb, rl, rs, rm, rkeep = decode(ref_coord, ref_cls, ref_masks, THR)
print(f"reference: {len(rb)} detections over thr={THR}")

# ---- device chain ----
def dev_run(model, base, arrays, nout):
    for i, a in enumerate(arrays):
        a.astype(np.float32).tofile(f"{HERE}/{base}.bin.{i}")
        subprocess.run(["adb", "push", f"{HERE}/{base}.bin.{i}", f"/data/local/tmp/{base}.bin.{i}"],
                       capture_output=True)
    r = subprocess.run(["adb", "shell",
                        f"cd /data/local/tmp && LD_LIBRARY_PATH=. ./gpu_test_bin {model} {nout} {base}.bin {base}_o.bin"],
                       capture_output=True, text=True)
    ok = [l for l in (r.stdout + r.stderr).splitlines() if "RUN OK" in l]
    assert ok, r.stdout + r.stderr
    outs = []
    for i in range(nout):
        lp = f"{HERE}/{base}_dev{i}.bin"
        subprocess.run(["adb", "pull", f"/data/local/tmp/{base}_o.bin.{i}", lp], capture_output=True)
        outs.append(np.fromfile(lp, np.float32))
    return outs

oa = dev_run("rfsA.tflite", "vr_a", [x.numpy(), clspos.numpy(), pospatch.numpy()], 3)
d_ec = torch.from_numpy(oa[0].reshape(1, S.GH * S.GW, S.NCLS))
d_ed = torch.from_numpy(oa[1].reshape(1, S.GH * S.GW, 4))
d_mem = torch.from_numpy((oa[2] * 0.5).reshape(1, S.GH * S.GW, S.HID))
refpoint, ts, idx = S.host_select(d_ec, d_ed, proposals, rp)
ob = dev_run("rfsB.tflite", "vr_b", [d_mem.numpy(), refpoint.numpy(), qf0.numpy()], 3)
d_boxes = torch.from_numpy(ob[0].reshape(1, S.NQ, 4))
d_logits = torch.from_numpy(ob[1].reshape(1, S.NQ, S.NCLS))
d_masks = torch.from_numpy(ob[2].reshape(1, S.NQ, S.MH, S.MW))
db, dl, ds, dm, dkeep = decode(d_boxes, d_logits, d_masks, THR)
print(f"device   : {len(db)} detections over thr={THR}")

# ---- match & report ----
def iou(a, b):
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb_ = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb_ - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    ar_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ar_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (ar_a[:, None] + ar_b[None, :] - inter + 1e-9)

if len(rb) and len(db):
    M = iou(rb, db)
    used = set()
    from rfdetr.assets import coco_classes as CC
    names = getattr(CC, "COCO_CLASSES", None) or getattr(CC, "coco_classes", None)
    for i in range(len(rb)):
        j = int(M[i].argmax())
        bi = float(M[i, j])
        mi = float((rm[i] & dm[j]).sum()) / max(float((rm[i] | dm[j]).sum()), 1.0)
        cls_ok = "OK" if int(rl[i]) == int(dl[j]) else f"MISMATCH({int(rl[i])} vs {int(dl[j])})"
        nm = names[int(rl[i])] if names is not None and int(rl[i]) < len(names) else int(rl[i])
        dup = " (dup)" if j in used else ""
        used.add(j)
        print(f"  det {i} [{nm}] score {rs[i]:.3f}: box IoU {bi:.4f}  mask IoU {mi:.4f}  class {cls_ok}{dup}")
sys.stdout.flush()
