"""models/<v>/manifest.json: every graph's signatures (input / output names and shapes, read from the flatbuffer), the
accelerator and precision it runs at, which matting graph the app uses, and the raw side files (little-endian float32) with their shapes, for the Kotlin app.
Also writes gs_perturbation.f32 next to gs_perturbation.npy.

  python make_manifest.py models/v1 [--dis PATH_TO_dis.tflite]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrt import sig_io  # noqa: E402

FP32_GRAPHS = set()   # graphs whose fp16 form fails its CONTRACT.md line (none since the key-chunked attention)
CPU_GRAPHS = {'bir'}  # BiRefNet: its deformable convolutions lower to GATHER_ND, which the LiteRT GPU does not run
# A CPU graph's working memory comes with its call, after the compile: how far MemAvailable fell from just before the
# compile to the low point of one call on the Galaxy S26 (LiteRT 2.3.0 CPU; device/g2/bir3.samples.txt: 6,492,924 kB
# before, 3,760,320 kB lowest = 2,732,604 kB). The app's memory check counts it before BiRefNet runs.
RUN_PEAK_BYTES = {'bir': 2_732_604 * 1024}
ORDER = (['bir', 'dis'] + [f'dino_{i}' for i in range(4)] + ['vae_0', 'vae_1', 'vae_2', 'vae_3', 'vae_attn', 'vae_4'] +
         ['den_ctx', 'den_in'] + [f'den_joint_{k:02d}' for k in range(24)] + ['oct'] + [f'gs_{i}' for i in range(8)])
SIDE = {'den_pos_emb.f32': [1, 8192, 1024], 'dino_rope_cos.f32': [1, 1, 4101, 64], 'dino_rope_sin.f32': [1, 1, 4101, 64],
        'gs_perturbation.f32': None}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('models')
    ap.add_argument('--dis', default=None)
    ap.add_argument('--no-sha', action='store_true')
    a = ap.parse_args()
    d = a.models
    pert = np.load(os.path.join(d, 'gs_perturbation.npy')).astype('<f4')
    pert.tofile(os.path.join(d, 'gs_perturbation.f32'))
    SIDE['gs_perturbation.f32'] = list(pert.shape)
    graphs = {}
    for name in ORDER:
        path = a.dis if name == 'dis' else os.path.join(d, f'{name}.tflite')
        if not path or not os.path.exists(path):
            print('missing', name)
            continue
        graphs[name] = {'file': 'dis.tflite' if name == 'dis' else f'{name}.tflite', 'bytes': os.path.getsize(path),
                        'accelerator': 'cpu' if name in CPU_GRAPHS else 'gpu',
                        'precision': 'fp32' if name in FP32_GRAPHS | CPU_GRAPHS else 'fp16_with_fp32_accum',
                        'signatures': sig_io(path)}
        if name in RUN_PEAK_BYTES:
            graphs[name]['run_peak_bytes'] = RUN_PEAK_BYTES[name]
        if not a.no_sha:
            graphs[name]['sha256'] = sha256(path)
    side = {}
    for f, shape in SIDE.items():
        p = os.path.join(d, f)
        n = os.path.getsize(p) // 4
        assert n == int(np.prod(shape)), (f, n, shape)
        side[f] = {'shape': shape, 'dtype': 'float32_le'}
    man = {'matting': {'default': 'bir', 'alternative': 'dis'}, 'graphs': graphs, 'side_files': side, 'json_files': ['vae_bn.json', 'gs_rep.json'],
           'total_graph_bytes': sum(g['bytes'] for g in graphs.values())}
    tmp = os.path.join(d, 'manifest.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(man, f, indent=1)
    os.replace(tmp, os.path.join(d, 'manifest.json'))   # atomic: runs reading the manifest never see half a file
    print(f'{len(graphs)} graphs, {man["total_graph_bytes"] / 1e9:.2f} GB')


if __name__ == '__main__':
    main()
