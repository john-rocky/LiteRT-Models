"""models/<v>/manifest.json: every graph's signatures (input / output names and shapes, read from the flatbuffer), the
accelerator and precision it runs at, which matting graph the app uses, and the raw side files (little-endian float32) with their shapes, for the Kotlin app.
Also writes gs_perturbation.f32 next to gs_perturbation.npy.

  python make_manifest.py models/v1 [--dis PATH_TO_dis.tflite]

With den_joint_mid.tflite and den_joint_w.f32 in the folder (export_shared_joint.py) and no den_joint_01..22, the
manifest lists den_joint_mid in their place with `shared_blocks` [1..22], `block_weights` and `first_weight_input`
(the inputs before the weights), and the weights file as a side file [22, floats per block]. gs_mid.tflite and
gs_w.f32 (export_shared_gauss.py) with no gs_1..6 likewise stand for gs_1..6.

--precision fp16 runs the GPU graphs at FP16 without fp32 accumulation (the default keeps FP16_WITH_FP32_ACCUM).
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
# shared graph -> (the json its export script writes, its weights file, the graph name of block k)
SHARED = {'den_joint_mid': ('shared_joint.json', 'den_joint_w.f32', lambda k: f'den_joint_{k:02d}'),
          'gs_mid': ('shared_gs.json', 'gs_w.f32', lambda k: f'gs_{k}')}
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
    ap.add_argument('--precision', choices=['fp16_with_fp32_accum', 'fp16'], default='fp16_with_fp32_accum',
                    help='precision of the GPU graphs')
    a = ap.parse_args()
    d = a.models
    pert = np.load(os.path.join(d, 'gs_perturbation.npy')).astype('<f4')
    pf = os.path.join(d, 'gs_perturbation.f32')
    pert.tofile(pf + '.tmp')
    os.replace(pf + '.tmp', pf)   # a new file: model folders share files by hard link
    SIDE['gs_perturbation.f32'] = list(pert.shape)
    order = list(ORDER)
    shared = {}
    for name, (js, wfile, block_name) in SHARED.items():
        if not os.path.exists(os.path.join(d, f'{name}.tflite')):
            continue
        sj = json.load(open(os.path.join(d, js)))
        mids = [block_name(k) for k in sj['blocks']]
        if os.path.exists(os.path.join(d, f'{mids[0]}.tflite')):
            continue   # the per-block graphs are there: the folder runs them
        i = order.index(mids[0])
        order = order[:i] + [name] + [n for n in order[i:] if n not in mids]
        SIDE[wfile] = [len(sj['blocks']), sj['per_block_floats']]
        shared[name] = {'shared_blocks': sj['blocks'], 'block_weights': wfile,
                        'first_weight_input': sj.get('first_weight_input', 3)}
    graphs = {}
    for name in order:
        path = a.dis if name == 'dis' else os.path.join(d, f'{name}.tflite')
        if not path or not os.path.exists(path):
            print('missing', name)
            continue
        graphs[name] = {'file': 'dis.tflite' if name == 'dis' else f'{name}.tflite', 'bytes': os.path.getsize(path),
                        'accelerator': 'cpu' if name in CPU_GRAPHS else 'gpu',
                        'precision': 'fp32' if name in FP32_GRAPHS | CPU_GRAPHS else a.precision,
                        'signatures': sig_io(path)}
        if name in RUN_PEAK_BYTES:
            graphs[name]['run_peak_bytes'] = RUN_PEAK_BYTES[name]
        graphs[name].update(shared.get(name, {}))
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
