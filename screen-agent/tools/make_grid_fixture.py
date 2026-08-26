#!/usr/bin/env python3
"""Synthetic fixtures for testing what a runtime actually shows the model.

Two things get written, both 512x512 — exactly the 32x32-patch grid the exported
LFM2.5-VL vision graph is frozen at:

- `<out>.png` / `<out>.json` — a 2x2 grid of huge labelled buttons in known
  quadrants, plus ground truth in the ground_probe format. Anything that can see
  at all should point at the right quadrant; a miss here is spatial handling,
  not legibility.
- `rows16.png`, `rows8.png`, `cols8.png` (with `--rulers`) — visibility rulers.
  Numbered horizontal bands and vertical columns; ask the model to list every
  number it can see and the answer says exactly how much of the image reached
  it. This is the 12-second check that caught the truncation described in
  ../README.md.
"""
import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

CELLS = [
    ("Settings", (40, 70, 200, 90), (52, 152, 219)),
    ("Camera", (30, 200, 60, 130), (231, 76, 60)),
    ("Messages", (60, 30, 90, 220), (46, 204, 113)),
    ("Photos", (200, 40, 130, 60), (241, 196, 15)),
]


def font_at(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def make_rulers(outdir, S=512):
    """Numbered bands/columns: the answer reports the visible extent directly."""
    written = []
    for n, name in ((16, "rows16"), (8, "rows8")):
        img = Image.new("RGB", (S, S), (255, 255, 255))
        d = ImageDraw.Draw(img)
        font = font_at(24 if n == 16 else 46)
        # Repeat the number across the band so a partial read is still legible.
        xs = (10, S // 2 - 14, S - 46) if n == 16 else (14, S - 70)
        for i in range(n):
            y0, y1 = i * S // n, (i + 1) * S // n
            if i % 2:
                d.rectangle([0, y0, S, y1], fill=(232, 232, 232))
            for x in xs:
                d.text((x, y0 + 3), f"{i + 1}", fill=(0, 0, 0), font=font)
        path = os.path.join(outdir, f"{name}.png")
        img.save(path)
        written.append(path)

    # Horizontal counterpart: proves width is not being lost.
    img = Image.new("RGB", (S, S), (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = font_at(46)
    for i in range(8):
        x0, x1 = i * S // 8, (i + 1) * S // 8
        if i % 2:
            d.rectangle([x0, 0, x1, S], fill=(235, 235, 235))
        d.text((x0 + 12, 10), f"{i + 1}", fill=(0, 0, 0), font=font)
        d.text((x0 + 12, S - 60), f"{i + 1}", fill=(0, 0, 0), font=font)
    path = os.path.join(outdir, "cols8.png")
    img.save(path)
    written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fixtures/grid2x2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--rulers", action="store_true",
                    help="also write rows16/rows8/cols8 visibility rulers "
                         "next to <out>")
    args = ap.parse_args()

    S = args.size
    img = Image.new("RGB", (S, S), (28, 30, 34))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", int(S * 0.062))
    except OSError:
        font = ImageFont.load_default()

    truth = []
    pad = int(S * 0.035)
    for i, (label, _, colour) in enumerate(CELLS):
        row, col = divmod(i, 2)
        x0 = col * S // 2 + pad
        y0 = row * S // 2 + pad
        x1 = (col + 1) * S // 2 - pad
        y1 = (row + 1) * S // 2 - pad
        d.rounded_rectangle([x0, y0, x1, y1], radius=int(S * 0.04), fill=colour)
        tb = d.textbbox((0, 0), label, font=font)
        d.text(((x0 + x1 - tb[2]) / 2, (y0 + y1 - tb[3]) / 2), label,
               fill=(20, 20, 20), font=font)
        truth.append({
            "target": label,
            "prompt": f'Point to the "{label}" button.',
            "box": [round(x0 / S * 1000), round(y0 / S * 1000),
                    round(x1 / S * 1000), round(y1 / S * 1000)],
        })

    outdir = os.path.dirname(args.out) or "."
    os.makedirs(outdir, exist_ok=True)
    img.save(args.out + ".png")
    json.dump(truth, open(args.out + ".json", "w"), indent=2)
    print(f"{args.out}.png ({S}x{S})  {args.out}.json ({len(truth)} targets)")

    if args.rulers:
        for path in make_rulers(outdir, S):
            print(f"{path} ({S}x{S})")


if __name__ == "__main__":
    main()
