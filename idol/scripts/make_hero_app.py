"""Hero image of the card: the photo next to the splat the LiteRT pipeline made from it, turned to the sides the
photo never showed. The views are rendered with the contract renderer (render_views.py, gsplat's PyTorch functions).

  python make_hero_app.py --image photo.jpg --ply out/gaussian.ply --out hero.png \
      --views "90:Front,0:Side,270:Back" --credit "Photo: ... (Pexels License)"

Azimuths are in render_views' camera rig (degrees about the PLY's vertical axis); 90 faces the object's front for
idol's outputs.
"""
import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import render_views as RV  # noqa: E402

FONT_PATHS = ('/System/Library/Fonts/Supplemental/Arial.ttf', '/System/Library/Fonts/Helvetica.ttc',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
BG, FG, DIM = (17, 17, 17), (240, 240, 240), (160, 160, 160)


def font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def square(img, size):
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((size, size), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--ply', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--views', default='90:Front,0:Side,270:Back')
    ap.add_argument('--size', type=int, default=640, help='tile side in pixels (a multiple of 16)')
    ap.add_argument('--fov', type=float, default=28.0, help='vertical field of view of the views (the contract uses 40)')
    ap.add_argument('--photo-label', default='The photo')
    ap.add_argument('--credit', default='')
    a = ap.parse_args()
    views = [(float(v.split(':', 1)[0]), v.split(':', 1)[1]) for v in a.views.split(',')]
    s = a.size
    RV.SIZE, RV.AZIMUTHS, RV.FOV_Y = s, tuple(az for az, _ in views), a.fov
    tiles = [Image.fromarray(RV.to_u8(im)) for im in RV.render(a.ply)]
    photo = square(Image.open(a.image).convert('RGB'), s)

    pad, label_h, credit_h = 24, 60, 44 if a.credit else 0
    n = 1 + len(tiles)
    canvas = Image.new('RGB', (pad + n * (s + pad), pad + label_h + s + pad + credit_h), BG)
    draw = ImageDraw.Draw(canvas)
    f_label, f_credit = font(30), font(22)
    for i, (img, label) in enumerate([(photo, a.photo_label)] + list(zip(tiles, [lab for _, lab in views]))):
        x = pad + i * (s + pad)
        canvas.paste(img, (x, pad + label_h))
        tw = draw.textlength(label, font=f_label)
        draw.text((x + (s - tw) / 2, pad + 10), label, fill=FG, font=f_label)
    if a.credit:
        draw.text((pad, pad + label_h + s + pad // 2), a.credit, fill=DIM, font=f_credit)
    canvas.save(a.out)
    print(a.out, canvas.size)


if __name__ == '__main__':
    main()
