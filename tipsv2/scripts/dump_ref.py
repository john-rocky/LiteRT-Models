"""Dump official TIPSv2-B14-DPT outputs (depth / normals / seg) for parity checks."""
import os, sys, numpy as np, torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
S = 448
im = Image.open(os.path.join(HERE, "test.jpg")).convert("RGB").resize((S, S), Image.BILINEAR)
x = torch.from_numpy((np.asarray(im, np.float32) / 255.0).transpose(2, 0, 1)[None])
from transformers import AutoModel
m = AutoModel.from_pretrained("google/tipsv2-b14-dpt", trust_remote_code=True).eval()
with torch.no_grad():
    out = m(x)
    # also the raw (cls, patch) taps for debugging
    taps = m._extract_intermediate(x)
np.save(os.path.join(HERE, "ref_depth.npy"), out.depth.numpy())
np.save(os.path.join(HERE, "ref_normals.npy"), out.normals.numpy())
np.save(os.path.join(HERE, "ref_seg.npy"), out.segmentation.numpy())
np.save(os.path.join(HERE, "ref_tap3_cls.npy"), taps[-1][0].numpy())
np.save(os.path.join(HERE, "ref_tap3_patch.npy"), taps[-1][1].numpy())
print("depth", out.depth.shape, float(out.depth.min()), float(out.depth.max()))
print("normals", out.normals.shape, "seg", out.segmentation.shape)
print("seg argmax classes:", np.unique(out.segmentation.argmax(1).numpy())[:20])
print("tap3 patch", taps[-1][1].shape, "cls", taps[-1][0].shape)
