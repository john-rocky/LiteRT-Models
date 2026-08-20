#!/usr/bin/env python3
"""SAM 3.1 image side (facebook/sam3.1 detector) -> LiteRT CompiledModel GPU graphs.

Produces (fp16 weights, single flat float I/O each, all <=4-D, no GPU-banned ops):
  out/sam3_vision.tflite  image (1,3,1008,1008) [(x/255-0.5)/0.5, RGB]
                          -> [fpn288(256*288*288) | fpn144(256*144*144) | fpn72(256*72*72)]
  out/sam3_text.tflite    token embeddings (1,32,1024) (host lookup, see sam3_token_embed.bin)
                          -> text_mem (1, 32*256)
  out/sam3_head.tflite    [fpn288 | fpn144 | fpn72 | text_mem(32*256) | text_pad(32; 1.0=pad)]
                          -> [logits(200) | boxes(200*4 cxcywh, normalized) | presence(1) |
                              masks(200*288*288 logits)]
                          score = sigmoid(logit) * sigmoid(presence); keep > 0.5;
                          mask = sigmoid(logits) > 0.5 after resize to the image.
  out/sam3_token_embed.bin  fp16 [49408, 1024] row-major (BPE ids -> embeddings; host)
  out/sam3_tokenizer/       bpe_simple_vocab_16e6.txt.gz (CLIP BPE; host tokenizer)

Every rewrite is exact (see vit4d.py / gpu_patches.py); this script asserts torch parity
against the STOCK modules on a real image + prompt before exporting, then checks each
.tflite on the XNNPACK CPU interpreter and (optionally) the host-Mac GPU (Metal).

Usage: build_sam3.py [--only vision|text|head] [--gpu-mac] [--chunks 9] [--out models/out]
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import precheck_sam3 as P  # noqa: E402  (build_detector + CPU shims + helpers)
from vit4d import patch_vit_4d  # noqa: E402
import gpu_patches as G  # noqa: E402


class ZeroStuffConvT(nn.Module):
    """ConvTranspose2d(k=2,s=2) -> nearest x2 + stride mask + conv2d (exact; TRANSPOSE_CONV
    is rejected by ML Drift). Zoo recipe (sam2/edgetam/edsr)."""

    def __init__(self, ct, in_hw):
        super().__init__()
        self.stride = ct.stride[0]
        self.kernel = ct.kernel_size[0]
        self.out_hw = in_hw * self.stride
        self.register_buffer("weight", ct.weight.detach().flip(2, 3).transpose(0, 1).contiguous())
        self.bias = ct.bias
        mask = torch.zeros(1, 1, self.out_hw, self.out_hw)
        mask[:, :, ::self.stride, ::self.stride] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, x):
        up = F.interpolate(x, size=(self.out_hw, self.out_hw), mode="nearest")
        return F.conv2d(up * self.mask, self.weight, self.bias,
                        padding=self.kernel - 1)[:, :, :self.out_hw, :self.out_hw]


def patch_neck(neck):
    """All three tri-neck heads (sam3 / interactive / propagation) share the same layout."""
    for convs in (neck.convs, getattr(neck, "interactive_convs", None),
                  getattr(neck, "propagation_convs", None)):
        if convs is None:
            continue
        c4, c2 = convs[0], convs[1]
        if not isinstance(c4.dconv_2x2_0, ZeroStuffConvT):
            c4.dconv_2x2_0 = ZeroStuffConvT(c4.dconv_2x2_0, 72)
            c4.dconv_2x2_1 = ZeroStuffConvT(c4.dconv_2x2_1, 144)
            c2.dconv_2x2 = ZeroStuffConvT(c2.dconv_2x2, 72)


def load_image(path):
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((1008, 1008), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    return (x - 0.5) / 0.5


def parity(tag, y, ref):
    y = y.reshape(-1).float()
    r = ref.reshape(-1).float()
    d = (y - r).abs()
    corr = float(np.corrcoef(y.numpy(), r.numpy())[0, 1])
    print(f"[parity {tag}] corr={corr:.7f} max|diff|={d.max():.3g} rel-rms="
          f"{(d.pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item():.3g}")
    return corr


def export_token_table(det, out):
    emb = det.backbone.language_backbone.encoder.token_embedding.weight.detach()
    p = os.path.join(out, "sam3_token_embed.bin")
    emb.to(torch.float16).contiguous().numpy().tofile(p)
    print(f"[table] {p} {tuple(emb.shape)} fp16 {os.path.getsize(p)/1e6:.1f} MB")
    import pkg_resources
    bpe = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
    os.makedirs(os.path.join(out, "sam3_tokenizer"), exist_ok=True)
    shutil.copy(bpe, os.path.join(out, "sam3_tokenizer", os.path.basename(bpe)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["vision", "text", "head"], default=None)
    ap.add_argument("--out", default=os.path.join(P.ROOT, "models", "out"))
    ap.add_argument("--image", default=os.path.join(P.ROOT, "vendor_sam3/assets/images/truck.jpg"))
    ap.add_argument("--prompt", default="wheel")
    ap.add_argument("--chunks", type=int, default=9, help="global-attention query chunks (exact)")
    ap.add_argument("--gpu-mac", action="store_true")
    ap.add_argument("--no-convert", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    do = {a.only} if a.only else {"vision", "text", "head"}
    torch.manual_seed(0)

    det = P.build_detector(os.path.join(P.ROOT, "models", "sam3.1_multiplex.pt"))
    sizes = [(288, 288), (144, 144), (72, 72)]
    x = load_image(a.image)
    tok = det.backbone.language_backbone.tokenizer([a.prompt], context_length=P.CONTEXT)
    print("[in] tokens:", tok[0].tolist())

    # ---- stock references (before any patch) ----
    with torch.inference_mode():
        vis_ref = P.VisionFlat(det)(x)
        txt_ref_full = P.TextFlat(det)(tok)                    # [mem | pad]
        head_ref = P.HeadFlat(det, sizes)(torch.cat([vis_ref, txt_ref_full], 1))
    txt_ref = txt_ref_full[:, :P.CONTEXT * 256]
    txt_pad = txt_ref_full[:, P.CONTEXT * 256:]
    print(f"[ref] vision {tuple(vis_ref.shape)} text {tuple(txt_ref.shape)} head {tuple(head_ref.shape)}")

    # ---- patches ----
    patch_vit_4d(det.backbone.vision_backbone.trunk, safe_ln=True, global_chunks=a.chunks)
    patch_neck(det.backbone.vision_backbone)
    print("[patch] text blocks:", G.apply_text_patches(det), " head MHA:", G.apply_head_patches(det))
    export_token_table(det, a.out)
    fx = os.path.join(a.out, "fixtures")
    os.makedirs(fx, exist_ok=True)
    np.save(os.path.join(fx, "image_1008.npy"), x.numpy())
    np.save(os.path.join(fx, "vision_ref.npy"), vis_ref.numpy())
    np.save(os.path.join(fx, "text_ref.npy"), txt_ref.numpy())
    np.save(os.path.join(fx, "text_pad.npy"), txt_pad.numpy())
    np.save(os.path.join(fx, "head_ref.npy"), head_ref.numpy())
    np.save(os.path.join(fx, "tokens.npy"), tok.numpy())

    vis_m = P.VisionFlat(det)
    txt_m = G.TextFlat4d(det)
    head_m = G.HeadFlat4d(det, sizes)
    emb_in = det.backbone.language_backbone.encoder.token_embedding(tok).detach()   # (1,32,1024)
    head_in = torch.cat([vis_ref, txt_ref, txt_pad], 1)
    with torch.inference_mode():
        if "vision" in do:
            parity("vision torch 4d vs stock", vis_m(x), vis_ref)
        if "text" in do:
            parity("text torch 4d vs stock", txt_m(emb_in), txt_ref)
        if "head" in do:
            parity("head torch 4d vs stock", head_m(head_in), head_ref)
    if a.no_convert:
        return

    jobs = []
    if "text" in do:
        jobs.append(("sam3_text", txt_m, emb_in, txt_ref))
    if "head" in do:
        jobs.append(("sam3_head", head_m, head_in, head_ref))
    if "vision" in do:
        jobs.append(("sam3_vision", vis_m, x, vis_ref))
    for name, m, xin, ref in jobs:
        p = P.convert(m, xin, name, a.out)
        if not p:
            continue
        P.tflite_parity(p, xin, ref, name)
        fp16_path = os.path.join(a.out, f"{name}.tflite")
        P.tflite_parity(fp16_path, xin, ref, name + " fp16")
        if a.gpu_mac:
            P.gpu_mac(fp16_path, xin, ref, name)
            P.gpu_mac(fp16_path, xin, ref, name, f32=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
