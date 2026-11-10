# Make the runner's fixture files from the five source photos: letterbox (black) to a square, bilinear resize to the
# model's input size, then uint8 RGB (det, pose) or float32 RGB / 255 (seg). One file per fixture and model:
# fixtures/fx_<task>/<id3>_in0.<ext>, plus a manifest fixtures/fx_<task>.manifest.json.
# usage: python -I make_fixtures.py <work dir>
import hashlib
import json
import os
import sys
import time

import numpy as np
import PIL
from PIL import Image

ROOT = os.path.abspath(sys.argv[1])
SRC = os.path.join(ROOT, "fixtures", "src")
ORDER = ["portrait", "cyclist", "dog", "jogger", "park"]  # ids 000..004
TASKS = {
    # task: graph file, input size, dtype, output elements (all outputs, subgraph order, as float32)
    "det": ("det_efficientdet_lite0_int8.tflite", 320, "uint8", 25 * 4 + 25 + 25 + 1),
    "pose": ("pose_movenet_lightning_int8_v4.tflite", 192, "uint8", 17 * 3),
    "seg": ("seg_selfie_segmenter_f16.tflite", 256, "float32", 256 * 256),
}


def letterbox(img, size):
    img = img.convert("RGB")
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.BILINEAR)


sources = {r["name"]: r for r in json.load(open(os.path.join(SRC, "SOURCES.json")))}
for task, (graph, size, dtype, out_elems) in TASKS.items():
    fx = "fx_" + task
    out_dir = os.path.join(ROOT, "fixtures", fx)
    os.makedirs(out_dir, exist_ok=True)
    ext = ".u8" if dtype == "uint8" else ".f32"
    names = []
    for i, name in enumerate(ORDER):
        src = os.path.join(SRC, name + ".jpg")
        if hashlib.sha256(open(src, "rb").read()).hexdigest() != sources[name]["sha256"]:
            sys.exit("source %s does not match SOURCES.json" % src)
        arr = np.asarray(letterbox(Image.open(src), size), dtype=np.uint8)
        if dtype == "float32":
            arr = (arr.astype(np.float32) / 255.0).astype(np.float32)
        fn = "%03d_in0%s" % (i, ext)
        arr.reshape(1, size, size, 3).tofile(os.path.join(out_dir, fn))
        names.append(fn)
    names = sorted(names)  # byte order (`ls | LC_ALL=C sort`)
    h = hashlib.sha256()
    total = 0
    for fn in names:
        b = open(os.path.join(out_dir, fn), "rb").read()
        h.update(b)
        total += len(b)
    manifest = {
        "step": "fixtures: five CC0 photos (fixtures/src/SOURCES.json), letterbox + bilinear (tools/make_fixtures.py)",
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "pillow": PIL.__version__, "numpy": np.__version__,
        "L": size, "dir": "fixtures/" + fx, "files": len(names), "bytes": total,
        "concat_sha256_bytesorted": h.hexdigest(), "count": len(names), "out_elems": out_elems,
        "ids": {"%03d" % i: n for i, n in enumerate(ORDER)},
        "order": {"file": "graphs/" + graph, "signature": "(subgraph 0)",
                  "inputs_subgraph_order": [{"index": 0, "name": "in0", "shape": [1, size, size, 3],
                                             "dtype": dtype.upper(), "ext": ext}]},
    }
    json.dump(manifest, open(os.path.join(ROOT, "fixtures", fx + ".manifest.json"), "w"), indent=1)
    print(fx, len(names), "files", total, "B", h.hexdigest()[:16])
