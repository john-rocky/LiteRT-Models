#!/usr/bin/env python3
"""End-to-end quality gate for the SAM 3.1 image side on the host-Mac GPU.

Runs the converted vision graph (models/precheck/sam3_vision.tflite, 4-D + SafeLN
re-authoring) on the Mac GPU via CompiledModel -- fp16 (delegate default) and
enforce_f32 -- feeds the resulting FPN features into the PyTorch text encoder +
detector head, and compares detections against the all-PyTorch fp32 pipeline at
the level that matters: box IoU, score deltas, mask IoU of the kept detections.
Also reports the stock-model detections for the prompt (sanity: the checkpoint /
prompt actually detect something).

Usage: e2e_gate.py [--image PATH] [--prompt TEXT] [--thresh 0.5]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precheck_sam3 as P  # noqa: E402


def preprocess(path):
    im = Image.open(path).convert("RGB").resize((1008, 1008), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    return (x - 0.5) / 0.5, im.size


def gpu_run(path, x, n_out, f32):
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator
    from ai_edge_litert.options import Options
    if f32:
        o = Options.create()
        o.hardware_accelerators = HardwareAccelerator.GPU
        o.gpu_options.enforce_f32 = True
        m = CompiledModel.from_file(path, options=o)
    else:
        m = CompiledModel.from_file(path, HardwareAccelerator.GPU)
    ib = m.create_input_buffers(0)
    ob = m.create_output_buffers(0)
    ib[0].write(x.numpy().ravel())
    ts = []
    for _ in range(3):
        t0 = time.time()
        m.run_by_index(0, ib, ob)
        y = np.array(ob[0].read(n_out, np.float32))
        ts.append(time.time() - t0)
    m.close()
    return torch.from_numpy(y.reshape(1, -1)), min(ts)


def detect(head, vis_flat, txt_flat, thresh):
    with torch.inference_mode():
        y = head(torch.cat([vis_flat, txt_flat], 1))
    logits = y[0, :200]
    boxes = y[0, 200:1000].reshape(200, 4)
    presence = y[0, 1000]
    masks = y[0, 1001:].reshape(200, 288, 288)
    prob = torch.sigmoid(logits) * torch.sigmoid(presence)
    keep = prob > thresh
    return dict(prob=prob, boxes=boxes, masks=masks, keep=keep, presence=torch.sigmoid(presence))


def box_iou(a, b):  # cxcywh normalized
    def xyxy(t):
        return torch.stack([t[:, 0] - t[:, 2] / 2, t[:, 1] - t[:, 3] / 2,
                            t[:, 0] + t[:, 2] / 2, t[:, 1] + t[:, 3] / 2], 1)
    A, B = xyxy(a), xyxy(b)
    lt = torch.max(A[:, None, :2], B[None, :, :2])
    rb = torch.min(A[:, None, 2:], B[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area = lambda t: (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])  # noqa: E731
    return inter / (area(A)[:, None] + area(B)[None, :] - inter + 1e-9)


def compare(ref, alt, tag):
    kr = ref["keep"].nonzero().flatten()
    ka = alt["keep"].nonzero().flatten()
    print(f"[{tag}] kept: ref={len(kr)} alt={len(ka)}  presence ref={ref['presence']:.3f} "
          f"alt={alt['presence']:.3f}  max|dprob| over all 200 queries="
          f"{(ref['prob'] - alt['prob']).abs().max():.4f}")
    if len(kr) == 0:
        return
    # same-query comparison (DETR queries are index-aligned)
    iou_b = box_iou(ref["boxes"][kr], alt["boxes"][kr]).diag()
    mr = ref["masks"][kr] > 0
    ma = alt["masks"][kr] > 0
    inter = (mr & ma).flatten(1).sum(1).float()
    union = (mr | ma).flatten(1).sum(1).float().clamp(min=1)
    miou = inter / union
    dprob = (ref["prob"][kr] - alt["prob"][kr]).abs()
    print(f"        same-query box IoU min/mean={iou_b.min():.4f}/{iou_b.mean():.4f}  "
          f"mask IoU min/mean={miou.min():.4f}/{miou.mean():.4f}  |dprob| max={dprob.max():.4f}")
    missed = [int(i) for i in kr if not alt["keep"][i]]
    extra = [int(i) for i in ka if not ref["keep"][i]]
    print(f"        kept-set: missed={missed} extra={extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(P.ROOT, "vendor_sam3/assets/images/truck.jpg"))
    ap.add_argument("--prompt", default="wheel")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--vision", default=os.path.join(P.ROOT, "models/precheck/sam3_vision.tflite"))
    a = ap.parse_args()
    det = P.build_detector(os.path.join(P.ROOT, "models", "sam3.1_multiplex.pt"))
    sizes = [(288, 288), (144, 144), (72, 72)]
    vis = P.VisionFlat(det)
    txt = P.TextFlat(det)
    head = P.HeadFlat(det, sizes)
    x, _ = preprocess(a.image)
    tok = det.backbone.language_backbone.tokenizer([a.prompt], context_length=P.CONTEXT)
    with torch.inference_mode():
        t0 = time.time()
        vis_ref = vis(x)
        print(f"[torch] vision fp32 {time.time()-t0:.2f}s")
        txt_ref = txt(tok)
    ref = detect(head, vis_ref, txt_ref, a.thresh)
    kr = ref["keep"].nonzero().flatten()
    print(f"[ref] prompt='{a.prompt}' image={os.path.basename(a.image)} kept={len(kr)} "
          f"probs={[round(float(p), 3) for p in ref['prob'][kr]]} presence={ref['presence']:.3f}")
    n_out = int(vis_ref.numel())
    for f32 in (False, True):
        y, t = gpu_run(a.vision, x, n_out, f32)
        d = (y - vis_ref).abs()
        print(f"[gpu {'f32 ' if f32 else 'fp16'}] vision {t*1000:.0f} ms  feature corr="
              f"{np.corrcoef(y.numpy().ravel(), vis_ref.numpy().ravel())[0, 1]:.6f} "
              f"max|diff|={d.max():.4g} rel-rms={(d.pow(2).mean().sqrt() / vis_ref.pow(2).mean().sqrt()):.4g}")
        alt = detect(head, y, txt_ref, a.thresh)
        compare(ref, alt, "gpu-f32 vs torch" if f32 else "gpu-fp16 vs torch")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
