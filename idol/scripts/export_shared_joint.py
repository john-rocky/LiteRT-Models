"""The state lever's graph form (round 2): one graph for the denoiser's 22 middle joint blocks, each block's weights as
runtime inputs, so the phone compiles it once per photo instead of once per block (the S26 compiled the joint blocks
96 times per photo, 0.39 s each, because only one or two fit in memory at a time).

Writes <out>/den_joint_mid.tflite (no weights; inputs h, t_mod, gate, then the 30 weight inputs of
idol_graphs.SHARED_WEIGHT_NAMES) and <out>/den_joint_w.f32 (little-endian float32, blocks 1..22 one after another,
each block's inputs in that order, flattened), and checks the shared module against every block's own module in
PyTorch on one random input.

  IDOL_SDK=<idol-sdk> python export_shared_joint.py --out models/state
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import f0_export as F0  # noqa: E402
import idol_graphs as IG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--chunk', type=int, default=512)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_grad_enabled(False)
    den, _ = F0.load_denoiser()
    den = den.float()
    lora = load_file(F0.snap('speridlabs/idol-turbo', 'idol_turbo_lora.safetensors'))
    blocks = list(range(1, len(den.blocks) - 1))
    mids = [IG.DenJoint(den, k, lora, a.chunk).eval() for k in blocks]
    shared = IG.DenJointShared(mids[0]).eval()
    ref = mids[0].block
    shapes = [list(t.shape) for t in IG.shared_block_weights(mids[0])]
    L = IG.DEN_TOKENS + IG.COND_TOKENS + 1
    torch.manual_seed(0)
    h = torch.randn(1, L, 1024)
    t_mod = torch.randn(1, 6144) * 0.1
    gate = IG.lora_gate('slow')
    worst = 0.0
    wpath = os.path.join(a.out, 'den_joint_w.f32')
    with open(wpath, 'wb') as f:
        for k, jb in zip(blocks, mids):
            b = jb.block
            assert b.modulation and (b.c, b.h, b.d, b.eps1, b.eps2, b.repo.eps) == \
                (ref.c, ref.h, ref.d, ref.eps1, ref.eps2, ref.repo.eps), k
            w = IG.shared_block_weights(jb)
            assert [list(t.shape) for t in w] == shapes, k
            got, want = shared(h, t_mod, gate, *w), jb(h, t_mod, gate)
            rel = float((got - want).norm() / want.norm())
            worst = max(worst, rel)
            print(f'block {k:2d}: shared vs own module rel L2 {rel:.2e}', flush=True)
            for t in w:
                f.write(t.numpy().astype('<f4').tobytes())
    assert worst < 1e-5, worst
    per_block = sum(int(np.prod(s)) for s in shapes)
    assert os.path.getsize(wpath) == per_block * len(blocks) * 4
    t0 = time.time()
    import litert_torch
    path = os.path.join(a.out, 'den_joint_mid.tflite')
    litert_torch.convert(shared, (h, t_mod, gate, *IG.shared_block_weights(mids[0]))).export(path)
    info = F0.scan_tflite(path)
    json.dump({'blocks': blocks, 'names': IG.SHARED_WEIGHT_NAMES, 'shapes': shapes, 'per_block_floats': per_block,
               'worst_rel_l2': worst, 'export_s': round(time.time() - t0, 1), 'scan': info},
              open(os.path.join(a.out, 'shared_joint.json'), 'w'), indent=1, default=str)
    print(f'den_joint_mid.tflite {os.path.getsize(path):,} B; den_joint_w.f32 {os.path.getsize(wpath):,} B '
          f'({len(blocks)} blocks x {per_block:,} floats); worst rel L2 {worst:.2e}', flush=True)


if __name__ == '__main__':
    main()
