"""Render text overlay PNGs for the showcase video.

The local ffmpeg build has no drawtext filter (no libfreetype), so labels
and cards are rendered here with Pillow and burned in with ffmpeg's core
`overlay` filter instead.

Usage:
    python3 showcase_overlay.py card <out.png> <line1> <line2>
    python3 showcase_overlay.py banner <out.png> <label> <tagline>
"""

import sys

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1080
HEIGHT = 1920
FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"
ACCENT = (138, 180, 248, 255)  # light blue, readable on black
WHITE = (255, 255, 255, 255)
BOX = (0, 0, 0, 140)  # semi-transparent black behind overlay text


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


MARGIN = 60  # minimum horizontal margin on each side


def _centered_text(draw, y, text, size, fill, boxed=False):
    """Draw horizontally centered text, shrinking to fit the frame width."""
    while True:
        font = _font(size)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        w, h = right - left, bottom - top
        if w <= WIDTH - 2 * MARGIN or size <= 20:
            break
        size -= 2
    x = (WIDTH - w) // 2
    if boxed:
        pad = 22
        draw.rounded_rectangle(
            (x - pad, y - pad, x + w + pad, y + h + pad),
            radius=16,
            fill=BOX,
        )
    draw.text((x - left, y - top), text, font=font, fill=fill)


def card(out_path: str, line1: str, line2: str) -> None:
    """Full-frame black card (title / end)."""
    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    _centered_text(draw, HEIGHT // 2 - 140, line1, 62, WHITE)
    _centered_text(draw, HEIGHT // 2 + 10, line2, 46, ACCENT)
    img.convert("RGB").save(out_path)


def banner(out_path: str, label: str, tagline: str) -> None:
    """Transparent full-frame overlay: label on top, tagline at bottom."""
    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    _centered_text(draw, 96, label, 54, WHITE, boxed=True)
    _centered_text(draw, HEIGHT - 150, tagline, 38, WHITE, boxed=True)
    img.save(out_path)


def main() -> None:
    if len(sys.argv) != 5 or sys.argv[1] not in ("card", "banner"):
        sys.exit(__doc__)
    mode, out_path, a, b = sys.argv[1:5]
    if mode == "card":
        card(out_path, a, b)
    else:
        banner(out_path, a, b)


if __name__ == "__main__":
    main()
