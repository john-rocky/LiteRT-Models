"""Idol on LiteRT: one image -> a 3D Gaussian splat (.ply), with LiteRT CompiledModel.

  pip install -r requirements.txt
  python run_idol.py photo.jpg --out out            # downloads both repos with huggingface_hub (the default form)
  python run_idol.py photo.jpg --out out --models <Idol folder> --dino <DINOv3 folder>

litert-community/Idol holds two forms: the default in the repo's root (65,536 Gaussians, the faster one) and the
first release's in `262k/` (262,144 Gaussians, the author's default size, closer to the author's output). Without
--models the runner downloads the default form only; for the other, download the repo and pass
`--models <Idol folder>/262k`.

The DINOv3 encoder repo is gated: accept the DINOv3 License on its Hub page and log in (`hf auth login`) first.
Background removal (the author's BiRefNet) runs on the CPU, everything else on the GPU (Metal on a Mac).
--mode: `real` for photos (crop margin 1.7), `synthetic` for renders on a plain background (1.2).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import idol_glue as GL  # noqa: E402
from idol_runner import IdolLiteRT  # noqa: E402

MAIN_REPO = 'litert-community/Idol'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--out', default='out')
    ap.add_argument('--mode', choices=('real', 'synthetic'), default='real')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--models', default=None, help='a local copy of litert-community/Idol')
    ap.add_argument('--dino', default=None, help='a local copy of the DINOv3 encoder repo')
    a = ap.parse_args()
    models, dino = a.models, a.dino
    if models is None or dino is None:
        from huggingface_hub import snapshot_download
        if models is None:
            models = snapshot_download(MAIN_REPO, ignore_patterns=['262k/*'])
        if dino is None:
            dino_repo = json.load(open(os.path.join(models, 'manifest.json')))['repos']['dino']
            dino = snapshot_download(dino_repo)
    idol = IdolLiteRT(models, 'gpu', dino_dir=dino, matting='birefnet')
    res = idol.run(a.image, a.mode, GL.make_noise(a.seed), a.out)
    print(json.dumps(res['times']))
    print(os.path.join(a.out, 'gaussian.ply'))


if __name__ == '__main__':
    main()
