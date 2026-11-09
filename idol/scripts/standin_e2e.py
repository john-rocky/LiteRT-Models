"""CONTRACT.md line (b) on the Mac stand-in: the whole phone pipeline (matting: the author's BiRefNet graph on CPU, or
DIS on the GPU -> preprocessing -> the shipped graphs on LiteRT GPU = Metal fp16 -> host glue -> .ply) from the raw input image with the reference run's noise, rendered from
the 4 contract cameras next to the author's .ply (ref/<tag>/gaussian.ply), PSNR >= 25 dB per input.

  python standin_e2e.py --models models/v2 --ref ref/armchair --image IMG --matting birefnet --out contract/standin/armchair
  python standin_e2e.py --models models/v2 --ref ref/armchair --image IMG --matting dis --dis dis.tflite --out ...
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from idol_runner import IdolLiteRT  # noqa: E402

LINE_DB = 25.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--matting', default='birefnet', choices=('birefnet', 'dis'))
    ap.add_argument('--dis', default=None, help='dis.tflite (with --matting dis)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--floor', type=int, default=None, help='DIS alpha floor (default: idol_glue.DIS_ALPHA_FLOOR)')
    a = ap.parse_args()
    import idol_glue as GL
    if a.floor is not None:
        GL.DIS_ALPHA_FLOOR = a.floor
    torch.set_grad_enabled(False)
    meta = json.load(open(os.path.join(a.ref, 'meta.json')))
    noise = dict(np.load(os.path.join(a.ref, 'noise.npz')))
    tag = os.path.basename(os.path.normpath(a.ref))
    p = IdolLiteRT(a.models, 'gpu', dis_path=a.dis, matting=a.matting)
    res = p.run(a.image, meta['mode'], noise, a.out)
    res['box_reference'] = meta['box']
    print('pipeline', json.dumps(res), flush=True)
    r = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'render_views.py'),
                        '--ply', os.path.join(a.ref, 'gaussian.ply'), '--ply', os.path.join(a.out, 'gaussian.ply'),
                        '--out', os.path.join(a.out, 'render'), '--tag', tag], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:])
        sys.exit(1)
    rr = json.load(open(os.path.join(a.out, 'render', 'render.json')))
    res.update(tag=tag, psnr_db=rr['psnr_db'], psnr_per_view_db=rr['psnr_per_view_db'], line_db=LINE_DB,
               matting=a.matting, dis_alpha_floor=GL.DIS_ALPHA_FLOOR if a.matting == 'dis' else None,
               pass_=rr['psnr_db'] >= LINE_DB, precision='metal default (fp16)', models=a.models)
    json.dump(res, open(os.path.join(a.out, 'b.json'), 'w'), indent=1)
    print(f'{tag}: PSNR {rr["psnr_db"]:.2f} dB (views {[round(v, 2) for v in rr["psnr_per_view_db"]]}) '
          f'{"PASS" if res["pass_"] else "FAIL"}', flush=True)


if __name__ == '__main__':
    main()
