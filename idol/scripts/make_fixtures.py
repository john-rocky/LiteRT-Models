"""device/fx/<tag>/: the inputs the app's headless runs read (CONTRACT.md lines (a), (b), (c) on the phone).

  fixture.json  {"tag", "mode", "seed", "image", "arrays": {name: {"file", "shape"}}}
  input image   the raw input file, copied as is
  noise         vae_noise, latent_noise, camera_noise, octree_pool, leaf_pool (the reference run's shared randomness)
  parity        pixels, feature1, feature2, t_grid, x_latent_i, x_camera_i (i < 5), oct_cells_l (l = 1..8), latent,
                anchors: each part's own reference input for line (a)
Arrays are little-endian float32, C order.

  python make_fixtures.py --ref ref/armchair --image IMG --out device/fx/armchair
"""
import argparse
import json
import os
import shutil

import numpy as np

NOISE = ('vae_noise', 'latent_noise', 'camera_noise', 'octree_pool', 'leaf_pool')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.ref, 'meta.json')))
    ref = dict(np.load(os.path.join(a.ref, 'ref.npz')))
    noise = dict(np.load(os.path.join(a.ref, 'noise.npz')))
    os.makedirs(a.out, exist_ok=True)
    image = 'input' + os.path.splitext(a.image)[1].lower()
    shutil.copyfile(a.image, os.path.join(a.out, image))
    arrays = {}

    def put(name, arr):
        arr = np.ascontiguousarray(np.asarray(arr, dtype='<f4'))
        arr.tofile(os.path.join(a.out, f'{name}.f32'))
        arrays[name] = {'file': f'{name}.f32', 'shape': list(arr.shape)}

    for n in NOISE:
        put(n, noise[n])
    put('pixels', ref['pixels'])
    put('feature1', ref['feature1'])
    put('feature2', ref['feature2'])
    put('t_grid', ref['t_grid'])
    for i in range(len(ref['t_grid']) - 1):
        put(f'x_latent_{i}', ref[f'x_latent_{i}'])
        put(f'x_camera_{i}', ref[f'x_camera_{i}'])
    for lv in range(1, 9):
        put(f'oct_cells_{lv}', ref[f'oct_cells_{lv}'])
    put('latent', ref['latent'])
    put('anchors', ref['anchors'])
    fx = {'tag': os.path.basename(os.path.normpath(a.out)), 'mode': meta['mode'], 'seed': meta['seed'], 'image': image,
          'arrays': arrays}
    json.dump(fx, open(os.path.join(a.out, 'fixture.json'), 'w'), indent=1)
    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f'{a.out}: {len(arrays)} arrays, {total / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
