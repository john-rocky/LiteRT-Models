"""Calibrate the substitute matting's alpha floor on held-out images: the floor (uint8 alpha below it -> 0, applied to
the DIS alpha before the author's preprocessing) whose crop box best matches the box the author's pipeline gets with
BiRefNet. The contract inputs are excluded; the images are the author's demo gallery (idol-sdk `demo/examples`),
every third file in name order.

Why: DIS leaves faint responses BiRefNet does not (alpha 1-5 of 255 at the image corners and along thin wisps). The
author's rule crops to the bounding box of the nonzero alpha after a 3x3 minimum filter, so a faint corner or wisp
stretches the crop to the image edge and the object comes out smaller than in the author's pipeline.

  python calibrate_matting_floor.py --dis dis.tflite --out contract/matting_floor.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageFilter
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser('~/code/idol_work/src/idol-sdk'))
import f0_export as F0  # noqa: E402
import idol_glue as GL  # noqa: E402
from idol.models import BiRefNet  # noqa: E402
from lrt import Graph  # noqa: E402

GALLERY = os.path.expanduser('~/code/idol_work/src/idol-sdk/demo/examples')
CONTRACT_INPUTS = {'nb-armchair.webp', 'fig-axolotl.webp'}
FLOORS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def resized_rgb(path):
    image = Image.open(path)
    w, h = image.size
    s = GL.IMAGE_SIZE / min(w, h)
    image = image.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.LANCZOS)
    return image, GL.has_real_alpha(image)


def box_of(alpha, margin):
    eroded = np.array(Image.fromarray(alpha).filter(ImageFilter.MinFilter(3)))
    ys, xs = np.nonzero(eroded)
    if len(xs) == 0:
        return None
    left, top, right, bottom = xs.min(), ys.min(), xs.max(), ys.max()
    cx, cy = (left + right) / 2, (top + bottom) / 2
    half = max(right - left, bottom - top) / 2 * margin
    return [int(cx - half), int(cy - half), int(cx + half), int(cy + half)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dis', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    root = os.path.dirname(F0.snap('speridlabs/idol', 'config.json'))
    bir = BiRefNet()
    bir.load_state_dict(load_file(os.path.join(root, 'birefnet.safetensors')), strict=True)
    bir = bir.eval().requires_grad_(False).float()
    dis = Graph(a.dis, 'gpu')
    names = sorted(f for f in os.listdir(GALLERY) if f.endswith('.webp') and f not in CONTRACT_INPUTS)[::3]
    rows = []
    for n in names:
        rgb, has_alpha = resized_rgb(os.path.join(GALLERY, n))
        if has_alpha:
            print(f'{n}: has its own alpha, skipped', flush=True)
            continue
        rgb = rgb.convert('RGB')
        a_bir = np.array(bir.remove_background(rgb).getchannel(3))
        a_dis = GL.dis_alpha(rgb, lambda x: list(dis.run([x]).values())[0])
        ref = box_of(a_bir, 1.2)
        side = ref[2] - ref[0]
        row = {'image': n, 'box_birefnet': ref, 'iou128': float(((a_bir >= 128) & (a_dis >= 128)).sum() /
                                                               max(1, ((a_bir >= 128) | (a_dis >= 128)).sum()))}
        for f in FLOORS:
            b = box_of(np.where(a_dis >= f, a_dis, 0).astype(np.uint8), 1.2)
            row[f'err_{f}'] = None if b is None else max(abs(x - y) for x, y in zip(b, ref)) / side
        rows.append(row)
        print(n, ref, ' '.join(f'{f}:{row[f"err_{f}"]:.3f}' for f in FLOORS if row[f'err_{f}'] is not None),
              flush=True)
    summary = {}
    for f in FLOORS:
        e = np.array([r[f'err_{f}'] for r in rows if r[f'err_{f}'] is not None])
        summary[f] = {'mean': float(e.mean()), 'median': float(np.median(e)), 'max': float(e.max()),
                      'over_2pct': int((e > 0.02).sum()), 'n': int(len(e))}
        print(f'floor {f:3d}: mean {e.mean():.4f} median {np.median(e):.4f} max {e.max():.4f} '
              f'images over 2 % of the side {int((e > 0.02).sum())}/{len(e)}', flush=True)
    best = min(FLOORS, key=lambda f: (summary[f]['mean'], f))
    json.dump({'images': rows, 'summary': summary, 'best_floor_by_mean': best,
               'metric': 'max corner difference of the crop box / the BiRefNet box side'}, open(a.out, 'w'), indent=1)
    print('best floor by mean', best)


if __name__ == '__main__':
    main()
