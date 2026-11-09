"""CONTRACT.md line (b) for a phone run: the .ply the app wrote from the raw input image with the reference run's noise,
rendered from the 4 contract cameras next to the author's .ply (ref/<tag>/gaussian.ply), PSNR >= 25 dB per input.

  python score_phone_b.py --ref ref/armchair --ply device/app/<run>/armchair/gaussian.ply --where "Galaxy S26, ..."
      --out contract/phone/s26/armchair --record contract/b_armchair_s26.json [--run-json <the app's run record>]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LINE_DB = 25.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--ply', required=True)
    ap.add_argument('--where', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--record', required=True)
    ap.add_argument('--run-json', default=None)
    a = ap.parse_args()
    tag = os.path.basename(os.path.normpath(a.ref))
    r = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'render_views.py'),
                        '--ply', os.path.join(a.ref, 'gaussian.ply'), '--ply', a.ply,
                        '--out', os.path.join(a.out, 'render'), '--tag', tag], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-4000:])
        sys.exit(1)
    rr = json.load(open(os.path.join(a.out, 'render', 'render.json')))
    rec = {'tag': tag, 'where': a.where, 'psnr_db': rr['psnr_db'], 'psnr_per_view_db': rr['psnr_per_view_db'],
           'line_db': LINE_DB, 'pass_': rr['psnr_db'] >= LINE_DB, 'ply': a.ply, 'ply_sha256': sha256(a.ply),
           'ply_bytes': os.path.getsize(a.ply)}
    if a.run_json:
        rec['app_run'] = json.load(open(a.run_json))
    os.makedirs(os.path.dirname(os.path.abspath(a.record)), exist_ok=True)
    json.dump(rec, open(a.record, 'w'), indent=1)
    print(f'{tag} ({a.where}): PSNR {rr["psnr_db"]:.2f} dB (views {[round(v, 2) for v in rr["psnr_per_view_db"]]}) '
          f'{"PASS" if rec["pass_"] else "FAIL"}', flush=True)


if __name__ == '__main__':
    main()
