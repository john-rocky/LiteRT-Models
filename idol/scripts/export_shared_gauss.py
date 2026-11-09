"""The state lever for the Gaussian decoder (round 2): one graph for its six middle graphs gs_1 .. gs_6 (two blocks
each, the same shapes), each graph's weights as runtime inputs, so the phone compiles it once per photo instead of six
times (in the app the Pixel 11 Pro compiled each gs graph in 8.2-8.7 s, the Galaxy S26 in 1.3 s).

Writes <out>/gs_mid.tflite (no weights; inputs h, latent, then the two blocks' weight inputs, each block in
idol_graphs.GAUSS_BLOCK_WEIGHT_NAMES order), <out>/gs_w.f32 (little-endian float32, graphs 1..6 one after another,
each graph's weight inputs in input order, flattened) and <out>/shared_gs.json, and checks the shared module against
every middle graph's own module in PyTorch on one random input.

  IDOL_SDK=<idol-sdk> python export_shared_gauss.py --out models/state --anchors 2048
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import export_graphs as EG  # noqa: E402  (also puts the idol-sdk on the path)
import f0_export as F0  # noqa: E402
import idol_graphs as IG  # noqa: E402

LATENT_TOKENS, LATENT_CHANNELS = 8192, 16
FIRST_WEIGHT_INPUT = 2   # gs_mid's inputs before its weights: h, latent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--anchors', type=int, default=8192)
    ap.add_argument('--chunk', type=int, default=512)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_grad_enabled(False)
    dgs = EG.load_decoder('decoder_gs')
    graphs = list(range(1, len(EG.GS_SPLIT) - 1))
    mids = [IG.Gauss(dgs, *EG.GS_SPLIT[i], a.chunk).eval() for i in graphs]
    shared = IG.GaussShared(mids[0]).eval()
    shapes = [list(t.shape) for t in IG.gauss_shared_weights(mids[0])]
    torch.manual_seed(0)
    h = torch.randn(1, a.anchors, 1024)
    latent = torch.randn(1, LATENT_TOKENS, LATENT_CHANNELS)
    worst = 0.0
    wpath = os.path.join(a.out, 'gs_w.f32')
    with open(wpath, 'wb') as f:
        for i, g in zip(graphs, mids):
            assert not g.first and not g.last and len(g.blocks) == shared.n, i
            for b in g.blocks:
                assert (b.e1, b.e2, b.e3) == shared.eps, i
                assert (b.self_attn.h, b.self_attn.c, b.self_attn.d, b.self_attn.chunk) == shared.sa, i
                assert (b.cross_attn.h, b.cross_attn.c, b.cross_attn.d, b.cross_attn.chunk) == shared.ca, i
            w = IG.gauss_shared_weights(g)
            assert [list(t.shape) for t in w] == shapes, i
            got, want = shared(h, latent, *w), g(h, latent)
            rel = float((got - want).norm() / want.norm())
            worst = max(worst, rel)
            print(f'gs_{i}: shared vs own module rel L2 {rel:.2e}', flush=True)
            for t in w:
                f.write(t.numpy().astype('<f4').tobytes())
    assert worst < 1e-5, worst
    per_graph = sum(int(np.prod(s)) for s in shapes)
    assert os.path.getsize(wpath) == per_graph * len(graphs) * 4
    t0 = time.time()
    import litert_torch
    path = os.path.join(a.out, 'gs_mid.tflite')
    litert_torch.convert(shared, (h, latent, *IG.gauss_shared_weights(mids[0]))).export(path)
    info = F0.scan_tflite(path)
    names = [f'b{k}_{n}' for k in range(shared.n) for n in IG.GAUSS_BLOCK_WEIGHT_NAMES]
    json.dump({'blocks': graphs, 'first_weight_input': FIRST_WEIGHT_INPUT, 'names': names, 'shapes': shapes,
               'per_block_floats': per_graph, 'anchors': a.anchors, 'chunk': a.chunk, 'worst_rel_l2': worst,
               'export_s': round(time.time() - t0, 1), 'scan': info},
              open(os.path.join(a.out, 'shared_gs.json'), 'w'), indent=1, default=str)
    print(f'gs_mid.tflite {os.path.getsize(path):,} B; gs_w.f32 {os.path.getsize(wpath):,} B '
          f'({len(graphs)} graphs x {per_graph:,} floats); worst rel L2 {worst:.2e}', flush=True)


if __name__ == '__main__':
    main()
