#!/usr/bin/env python3
"""SAM 3.1 shared-trunk vision graph for detection + tracking (`sam3_vision_tri.tflite`).

One ViT-L pass + the three tri-neck heads; the tracker-side high-res projections
(conv_s0: 256->32 @288², conv_s1: 256->64 @144², from the interactive / multiplex mask
decoders) are folded in so the per-frame feature handoff is small:

  image (1,3,1008,1008) ->
    [ sam3 fpn288(256) | fpn144(256) | fpn72(256)          (detector head input, 27.9M)
    | inter h0(32,288²) | h1(64,144²) | f2(256,72²)        (interactive decoder, 5.3M)
    | prop  h0(32,288²) | h1(64,144²) | f2(256,72²) ]      (propagation decoder, 5.3M)

Exact torch parity vs the stock tri-neck + stock convs is asserted before export.
Usage: build_sam3_tri.py [--gpu-mac] [--chunks 9] [--out models/out]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import precheck_sam3 as P  # noqa: E402
from build_sam3 import load_image, parity, patch_neck  # noqa: E402
from vit4d import patch_vit_4d  # noqa: E402

LAYOUT = [("sam3_fpn288", 256, 288), ("sam3_fpn144", 256, 144), ("sam3_fpn72", 256, 72),
          ("inter_h0", 32, 288), ("inter_h1", 64, 144), ("inter_f2", 256, 72),
          ("prop_h0", 32, 288), ("prop_h1", 64, 144), ("prop_f2", 256, 72)]


def layout_offsets():
    offs, o = {}, 0
    for name, c, hw in LAYOUT:
        n = c * hw * hw
        offs[name] = (o, o + n, c, hw)
        o += n
    return offs, o


class VisionTriFlat(nn.Module):
    def __init__(self, det, convs):
        super().__init__()
        self.neck = det.backbone.vision_backbone
        self.inter_s0, self.inter_s1, self.prop_s0, self.prop_s1 = convs

    def forward(self, x):
        sam3_out, _, inter_out, _, prop_out, _ = self.neck(
            x, need_sam3_out=True, need_interactive_out=True, need_propagation_out=True)
        g = lambda f: getattr(f, "tensors", f)  # noqa: E731
        parts = [g(f).flatten(1) for f in sam3_out]
        parts += [self.inter_s0(g(inter_out[0])).flatten(1), self.inter_s1(g(inter_out[1])).flatten(1),
                  g(inter_out[2]).flatten(1)]
        parts += [self.prop_s0(g(prop_out[0])).flatten(1), self.prop_s1(g(prop_out[1])).flatten(1),
                  g(prop_out[2]).flatten(1)]
        return torch.cat(parts, 1)


def load_convs(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    out = []
    for mod in ("interactive_sam_mask_decoder", "sam_mask_decoder"):
        for cname, cin, cout in (("conv_s0", 256, 32), ("conv_s1", 256, 64)):
            w = ckpt[f"tracker.model.{mod}.{cname}.weight"]
            b = ckpt[f"tracker.model.{mod}.{cname}.bias"]
            conv = nn.Conv2d(cin, cout, kernel_size=w.shape[-1], stride=1, padding=w.shape[-1] // 2)
            conv.weight.data.copy_(w)
            conv.bias.data.copy_(b)
            conv.eval()
            out.append(conv)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(P.ROOT, "models", "out"))
    ap.add_argument("--image", default=os.path.join(P.ROOT, "vendor_sam3/assets/images/truck.jpg"))
    ap.add_argument("--chunks", type=int, default=9)
    ap.add_argument("--gpu-mac", action="store_true")
    ap.add_argument("--no-convert", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ckpt = os.path.join(P.ROOT, "models", "sam3.1_multiplex.pt")
    det = P.build_detector(ckpt)
    convs = load_convs(ckpt)
    x = load_image(a.image)
    m = VisionTriFlat(det, convs)
    with torch.inference_mode():
        ref = m(x)
    offs, total = layout_offsets()
    print(f"[tri] output floats={total} ({total*4/1e6:.0f} MB fp32)  layout={ {k: v[:2] for k, v in offs.items()} }")
    patch_vit_4d(det.backbone.vision_backbone.trunk, safe_ln=True, global_chunks=a.chunks)
    patch_neck(det.backbone.vision_backbone)
    with torch.inference_mode():
        parity("vision_tri torch 4d vs stock", m(x), ref)
    if a.no_convert:
        return
    p = P.convert(m, x, "sam3_vision_tri", a.out)
    if p:
        P.tflite_parity(p, x, ref, "vision_tri")
        fp16 = os.path.join(a.out, "sam3_vision_tri.tflite")
        if a.gpu_mac:
            P.gpu_mac(fp16, x, ref, "vision_tri")
            P.gpu_mac(fp16, x, ref, "vision_tri", f32=True)
    np.save(os.path.join(a.out, "fixtures", "vision_tri_ref.npy"), ref.numpy())
    sys.stdout.flush()


if __name__ == "__main__":
    main()
