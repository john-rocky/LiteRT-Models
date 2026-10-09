#!/usr/bin/env python3
"""The end card of the X videos: a 1080 x 1920 PNG on the app's background, three centred lines, nothing else.

    venv-demo/bin/python -B scripts/make_end_card.py <out.png>

Lines (white bold title, grey link, small grey credit): "d1-3B on LiteRT", "huggingface.co/litert-community/
d1-3B-LiteRT", "Liquid AI's d1-3B · LFM Open License v1.0 · converted to LiteRT". Font: the system's SF Pro
(/System/Library/Fonts/SFNS.ttf, its named instances). Each line takes the largest size up to its cap that keeps it
within 80 % of the width (864 px). Prints the sizes, each line's box, and the PNG's sha256; refuses to write a line
that does not fit or a box outside the card.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
BG, WHITE, GREY = (0x0E, 0x11, 0x16), (0xFF, 0xFF, 0xFF), (0xB8, 0xBE, 0xC6)
MAX_W = int(W * 0.80)
FONT = "/System/Library/Fonts/SFNS.ttf"
LINES = [  # text, colour, weight instance, size cap, gap below (px)
    ("d1-3B on LiteRT", WHITE, b"Bold", 112, 56),
    ("huggingface.co/litert-community/d1-3B-LiteRT", GREY, b"Regular", 44, 72),
    ("Liquid AI's d1-3B · LFM Open License v1.0 · converted to LiteRT", GREY, b"Regular", 30, 0),
]


def font(weight: bytes, size: int) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(FONT, size)
    names = f.get_variation_names()
    if weight not in names:
        raise SystemExit(f"{FONT} has no {weight!r} instance: {names}")
    f.set_variation_by_name(weight)
    return f


def fit(text: str, weight: bytes, cap: int) -> tuple[ImageFont.FreeTypeFont, tuple[int, int, int, int]]:
    for size in range(cap, 9, -1):
        f = font(weight, size)
        box = f.getbbox(text)
        if box[2] - box[0] <= MAX_W:
            return f, box
    raise SystemExit(f"{text!r} does not fit {MAX_W} px")


def main() -> int:
    out = Path(sys.argv[1])
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    fitted = [(text, colour, *fit(text, weight, cap), gap) for text, colour, weight, cap, gap in LINES]
    heights = [box[3] - box[1] for _, _, _, box, _ in fitted]
    total = sum(heights) + sum(gap for *_, gap in fitted)
    y = (H - total) // 2
    report = []
    for (text, colour, f, box, gap), h in zip(fitted, heights):
        w = box[2] - box[0]
        x = (W - w) // 2 - box[0]
        d.text((x, y - box[1]), text, font=f, fill=colour)
        drawn = (x + box[0], y, x + box[2], y + h)
        if drawn[0] < 0 or drawn[2] > W or drawn[1] < 0 or drawn[3] > H:
            raise SystemExit(f"{text!r} drawn at {drawn}, outside the card")
        report.append(f"{f.size:>3} px  {w:>4} px wide ({w / W:.0%})  box {drawn}  {text}")
        y += h + gap
    img.save(out, optimize=True)
    print("\n".join(report))
    print(f"{out.name}: {W}x{H}, background #{''.join(f'{c:02X}' for c in BG)}, sha256 "
          f"{hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
