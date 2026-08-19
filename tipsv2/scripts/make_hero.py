"""Render the card hero from ON-DEVICE fp16 GPU outputs: input | depth | normals | ADE20K seg."""
import os, subprocess, sys, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
import build_tipsv2 as B
HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1]
im = Image.open(src).convert("RGB")
w, h = im.size; s = min(w, h)
im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((B.IMG, B.IMG), Image.BILINEAR)
x = (np.asarray(im, np.float32) / 255.0).transpose(2, 0, 1)[None]
x.astype(np.float32).tofile(os.path.join(HERE, "hin.bin"))
dev = "/data/local/tmp"
subprocess.run(["adb", "push", os.path.join(HERE, "hin.bin"), dev + "/hin.bin"], capture_output=True)
r = subprocess.run(["adb", "shell", "cd %s && LD_LIBRARY_PATH=. ./gpu_test_bin tipsv2_b14_dpt_fp16.tflite 1 hin.bin hout.bin" % dev],
                   capture_output=True, text=True)
print([l for l in r.stderr.splitlines() if "Replacing" in l or "RUN OK" in l])
outs = []
for i in range(3):
    subprocess.run(["adb", "pull", "%s/hout.bin.%d" % (dev, i), os.path.join(HERE, "hout.bin.%d" % i)], capture_output=True)
    outs.append(np.fromfile(os.path.join(HERE, "hout.bin.%d" % i), np.float32))
depth = outs[0].reshape(B.IMG, B.IMG); normals = outs[1].reshape(3, B.IMG, B.IMG); seg = outs[2].reshape(B.N_CLS, 256, 256)
# depth: inverse depth, 2..98 percentile, Spectral (matplotlib)
from matplotlib import cm
disp = 1.0 / np.clip(depth, 1e-3, None)
lo, hi = np.percentile(disp, 2), np.percentile(disp, 98)
n = 1.0 - np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
depth_rgb = (cm.get_cmap("Spectral")(n)[..., :3] * 255).astype(np.uint8)
normals_rgb = ((normals.transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
lab = seg.argmax(0)
# ADE20K palette + names: read from the Kotlin predictor so the hero matches the app exactly
import re
kt = open(os.path.join(HERE, "..", "app", "src", "main", "java", "com", "tipsv2", "TipsPredictor.kt")).read()
cols = re.findall(r"0xFF([0-9A-F]{6})\.toInt\(\)", kt)[:150]
pal = np.array([[int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)] for c in cols], np.uint8)
seg_rgb = np.array(Image.fromarray(pal[lab]).resize((B.IMG, B.IMG), Image.NEAREST))
names = re.search(r"val ADE_CLASSES = listOf\((.*?)\)\n", kt, re.S).group(1)
names = re.findall(r'"([^"]*)"', names)
hist = np.bincount(lab.ravel(), minlength=150); top = np.argsort(-hist)[:5]
print("depth range %.2f..%.2f m; top classes: %s" % (depth.min(), depth.max(), [(names[i], round(100*hist[i]/lab.size,1)) for i in top]))
tiles = [np.asarray(im), depth_rgb, normals_rgb, seg_rgb]
pad = 8; W = B.IMG * 4 + pad * 5; H = B.IMG + pad * 2 + 28
canvas = Image.new("RGB", (W, H), (18, 18, 18)); d = ImageDraw.Draw(canvas)
try: font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
except Exception: font = ImageFont.load_default()
for i, (t, cap) in enumerate(zip(tiles, ["input 448²", "metric depth (%.1f–%.1f m)" % (depth.min(), depth.max()), "surface normals", "ADE20K segmentation"])):
    xo = pad + i * (B.IMG + pad)
    canvas.paste(Image.fromarray(t), (xo, pad))
    d.text((xo + 4, B.IMG + pad + 4), cap, fill=(230, 230, 230), font=font)
out = os.path.join(HERE, "hero.png"); canvas.save(out); print("wrote", out)
