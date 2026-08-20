#!/usr/bin/env python3
"""Full SAM 3.1 image pipeline on the host-Mac GPU via LiteRT CompiledModel.

Host side = CLIP BPE tokenizer + fp16 token-embedding lookup + thresholding only
(exactly what the Android/iOS app will do). Graph side = the three converted graphs:
  sam3_vision.tflite  GPU fp16 (feature error is detection-invariant)
  sam3_text.tflite    GPU enforce_f32 (fp16 execution of the |x|~1.2e3 CLIP residual
                      stream breaks multi-meaning prompts: 'window'/'paper bag' lose all
                      detections while 'wheel' survives -- text embeddings are the whole
                      conditioning signal, so run this 10 ms graph in f32)
  sam3_head.tflite    GPU enforce_f32 (fp16 costs mask IoU ~0.989 -> f32 is exact and
                      only ~60 ms slower on M4 Max; Mali has no such knob -> phase-2 topic)

For each (image, prompt): compares detections against the all-PyTorch stock model
(fresh unpatched build) and writes an overlay PNG per pair to models/out/overlays/.

Usage: run_pipeline_mac.py [--pairs img:prompt ...] [--thresh 0.5] [--no-ref]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import precheck_sam3 as P  # noqa: E402
from build_sam3 import load_image  # noqa: E402

OUT = os.path.join(P.ROOT, "models", "out")
CTX = 32


class GpuGraph:
    def __init__(self, path, f32=False, n_out=None):
        from ai_edge_litert.compiled_model import CompiledModel
        from ai_edge_litert.hardware_accelerator import HardwareAccelerator
        from ai_edge_litert.options import Options
        t0 = time.time()
        if f32:
            o = Options.create()
            o.hardware_accelerators = HardwareAccelerator.GPU
            o.gpu_options.enforce_f32 = True
            self.m = CompiledModel.from_file(path, options=o)
        else:
            self.m = CompiledModel.from_file(path, HardwareAccelerator.GPU)
        self.compile_s = time.time() - t0
        self.ib = self.m.create_input_buffers(0)
        self.ob = self.m.create_output_buffers(0)
        self.n_out = n_out
        self.name = os.path.basename(path)
        self.fully = self.m.is_fully_accelerated()

    def __call__(self, x):
        self.ib[0].write(np.ascontiguousarray(x, dtype=np.float32).ravel())
        t0 = time.time()
        self.m.run_by_index(0, self.ib, self.ob)
        y = np.array(self.ob[0].read(self.n_out, np.float32))
        self.last_ms = (time.time() - t0) * 1000
        return y


def decode(y, thresh):
    logits = y[:200]
    boxes = y[200:1000].reshape(200, 4)
    presence = y[1000]
    masks = y[1001:].reshape(200, 288, 288)
    prob = 1 / (1 + np.exp(-logits)) * 1 / (1 + np.exp(-presence))
    keep = np.where(prob > thresh)[0]
    return prob, boxes, masks, keep


def overlay(img_path, boxes, masks, keep, prob, out_path):
    from PIL import Image, ImageDraw
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    colors = [(255, 64, 64), (64, 200, 96), (80, 128, 255), (255, 200, 32),
              (200, 80, 255), (32, 220, 220)]
    ov = np.array(im).astype(np.float32)
    for j, q in enumerate(keep):
        c = np.array(colors[j % len(colors)], dtype=np.float32)
        m = np.array(Image.fromarray((1 / (1 + np.exp(-masks[q])) > 0.5).astype(np.uint8) * 255)
                     .resize((W, H), Image.BILINEAR)) > 127
        ov[m] = ov[m] * 0.5 + c * 0.5
    im = Image.fromarray(ov.astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for j, q in enumerate(keep):
        cx, cy, w, h = boxes[q]
        x0, y0, x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
        dr.rectangle([x0, y0, x1, y1], outline=colors[j % len(colors)], width=3)
        dr.text((x0 + 3, y0 + 3), f"{prob[q]:.2f}", fill=colors[j % len(colors)])
    im.save(out_path)


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = max(np.logical_or(a, b).sum(), 1)
    return inter / union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=[
        "truck.jpg:wheel", "truck.jpg:window", "groceries.jpg:paper bag",
        "groceries.jpg:tail light", "test_image.jpg:shoe"])
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--vision-f32", action="store_true",
                    help="run the vision graph with enforce_f32 too (exact, ~200 ms slower)")
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "overlays"), exist_ok=True)

    # host tokenizer + embedding table
    from sam3.model.tokenizer_ve import SimpleTokenizer
    import pkg_resources
    bpe = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
    tokenizer = SimpleTokenizer(bpe_path=bpe)
    table = np.fromfile(os.path.join(OUT, "sam3_token_embed.bin"), dtype=np.float16)
    table = table.reshape(-1, 1024)
    print(f"[host] embed table {table.shape}")

    n_vis = 256 * (288 * 288 + 144 * 144 + 72 * 72)
    vision = GpuGraph(os.path.join(OUT, "sam3_vision.tflite"), f32=a.vision_f32, n_out=n_vis)
    text = GpuGraph(os.path.join(OUT, "sam3_text.tflite"), f32=True, n_out=CTX * 256)
    head = GpuGraph(os.path.join(OUT, "sam3_head.tflite"), f32=True, n_out=1001 + 200 * 288 * 288)
    for g in (vision, text, head):
        print(f"[compile] {g.name}: {g.compile_s:.1f}s fully_accelerated={g.fully}")

    ref_det = None
    if not a.no_ref:
        ref_det = P.build_detector(os.path.join(P.ROOT, "models", "sam3.1_multiplex.pt"))
        sizes = [(288, 288), (144, 144), (72, 72)]
        ref_vis = P.VisionFlat(ref_det)
        ref_txt = P.TextFlat(ref_det)
        ref_head = P.HeadFlat(ref_det, sizes)

    img_dir = os.path.join(P.ROOT, "vendor_sam3/assets/images")
    for pair in a.pairs:
        img_name, prompt = pair.split(":", 1)
        img_path = os.path.join(img_dir, img_name)
        x = load_image(img_path).numpy()
        tok = tokenizer([prompt], context_length=CTX)[0].numpy()
        emb = table[tok].astype(np.float32)[None]              # (1,32,1024)
        pad = (tok == 0).astype(np.float32)[None]              # (1,32)

        t0 = time.time()
        vis_y = vision(x)
        txt_y = text(emb)
        head_in = np.concatenate([vis_y, txt_y, pad.ravel()])[None]
        y = head(head_in)
        total = (time.time() - t0) * 1000
        prob, boxes, masks, keep = decode(y, a.thresh)
        line = (f"[gpu] {img_name} '{prompt}': {total:.0f} ms "
                f"(vis {vision.last_ms:.0f} + txt {text.last_ms:.0f} + head {head.last_ms:.0f}) "
                f"kept={keep.tolist()} probs={[round(float(prob[q]), 3) for q in keep]}")
        if ref_det is not None:
            with torch.inference_mode():
                rv = ref_vis(torch.from_numpy(x))
                rt = ref_txt(torch.from_numpy(tok[None]).long())
                ry = ref_head(torch.cat([rv, rt], 1))[0].numpy()
            rprob, rboxes, rmasks, rkeep = decode(ry, a.thresh)
            match = set(keep.tolist()) == set(rkeep.tolist())
            ious = [mask_iou(1 / (1 + np.exp(-masks[q])) > 0.5,
                             1 / (1 + np.exp(-rmasks[q])) > 0.5) for q in rkeep]
            line += (f" | ref kept={rkeep.tolist()} same-set={match} "
                     f"mask IoU={'/'.join(f'{v:.3f}' for v in ious) if ious else '-'}")
        print(line, flush=True)
        overlay(img_path, boxes, masks, keep, prob,
                os.path.join(OUT, "overlays", f"{os.path.splitext(img_name)[0]}_{prompt.replace(' ', '_')}.png"))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
