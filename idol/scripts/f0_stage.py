"""F0 device staging: one fixture dir + manifest per F0 graph for devicebench npu_chain.sh (the C-API runner's
--fixtures contract: <id>_<signature input name>.f32 in the subgraph's input order).

  python f0_stage.py --graphs <dir of f0_*.tflite> --lane <K>
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser('~/code/idol_work/src/idol-sdk'))


def signature(path):
    from ai_edge_litert import schema_py_generated as schema
    m = schema.ModelT.InitFromPackedBuf(open(path, 'rb').read(), 0)
    g = m.subgraphs[0]
    sig = m.signatureDefs[0]
    name_of = {t.tensorIndex: t.name.decode() if isinstance(t.name, bytes) else t.name for t in sig.inputs}
    out_name = {t.tensorIndex: t.name.decode() if isinstance(t.name, bytes) else t.name for t in sig.outputs}
    ins = [{'name': name_of[i], 'ext': '.f32', 'dtype': 'FLOAT32', 'shape': [int(x) for x in g.tensors[i].shape]}
           for i in g.inputs]
    o = g.outputs[0]
    return ins, {'name': out_name[o], 'dtype': 'FLOAT32', 'shape': [int(x) for x in g.tensors[o].shape]}


def inputs_for(graph, shapes):
    import f0_export as F
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    if graph.startswith('f0_den_block'):
        m, _ = F.load_denoiser()
        mod = m.adaLN_modulation(m.t_embedder(torch.tensor([700.0]))).float().numpy()
        return [rng.standard_normal(shapes[0]).astype(np.float32) * 3, mod.reshape(shapes[1])]
    if graph.startswith('f0_dino_block'):
        cfg = F.dino_config()
        grid = int(round((shapes[0][1] - 5) ** 0.5))
        _, _, cos_all, sin_all = F.dino_rope(cfg, grid)
        return [rng.standard_normal(shapes[0]).astype(np.float32) * 2, cos_all.numpy(), sin_all.numpy()]
    if graph.startswith('f0_vae_enc'):
        return [(rng.random(shapes[0]) * 2 - 1).astype(np.float32)]
    if graph.startswith('f0_gs_block'):
        return [rng.standard_normal(shapes[0]).astype(np.float32), rng.standard_normal(shapes[1]).astype(np.float32)]
    if graph.startswith('f0_oct_block'):
        from idol.models.decoder import OctreeProbabilityFixedlenDecoder
        from safetensors.torch import load_file
        cfg = json.load(open(F.snap('speridlabs/idol', 'config.json')))
        d = OctreeProbabilityFixedlenDecoder(**cfg['decoder']['args'])
        d.load_state_dict(load_file(F.snap('speridlabs/idol', 'decoder.safetensors')), strict=True)
        d = d.float().eval()
        mod = d.adaLN_modulation(d.l_embedder(torch.tensor([256.0]))).float().numpy()
        return [rng.standard_normal(shapes[0]).astype(np.float32), mod.reshape(shapes[1]),
                rng.standard_normal(shapes[2]).astype(np.float32)]
    raise SystemExit(f'no inputs rule for {graph}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graphs', required=True)
    ap.add_argument('--lane', required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    fx_root = os.path.join(a.lane, 'device', 'fx')
    os.makedirs(fx_root, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(a.graphs, 'f0_*.tflite'))):
        graph = os.path.basename(path)[:-len('.tflite')]
        fx = graph.replace('f0_', 'fx_')
        d = os.path.join(fx_root, fx)
        os.makedirs(d, exist_ok=True)
        ins, out0 = signature(path)
        arrays = inputs_for(graph, [x['shape'] for x in ins])
        files = []
        for spec, arr in zip(ins, arrays):
            assert list(arr.shape) == spec['shape'], (graph, spec, arr.shape)
            fn = f"000_{spec['name']}.f32"
            np.ascontiguousarray(arr, dtype=np.float32).tofile(os.path.join(d, fn))
            files.append(fn)
        names = sorted(os.listdir(d), key=lambda s: s.encode())
        h = hashlib.sha256()
        total = 0
        for fn in names:
            b = open(os.path.join(d, fn), 'rb').read()
            h.update(b)
            total += len(b)
        out_elems = int(np.prod(out0['shape']))
        man = {'dir': os.path.relpath(d, a.lane), 'files': len(names), 'bytes': total,
               'concat_sha256_bytesorted': h.hexdigest(), 'count': 1, 'L': int(ins[0]['shape'][1]),
               'out_elems': out_elems, 'output0': out0, 'order': {'inputs_subgraph_order': ins},
               'graph': f'models/f0/{graph}.tflite (serving_default)',
               'pattern': 'seeded random inputs (numpy default_rng(0)); real modulation / rope inputs from the weights'}
        with open(os.path.join(fx_root, f'{fx}.manifest.json'), 'w') as f:
            json.dump(man, f, indent=1)
        print(f'{fx}: {len(names)} files {total:,} B, out_elems {out_elems:,}, inputs '
              + ', '.join(f"{x['name']}{x['shape']}" for x in ins), flush=True)


if __name__ == '__main__':
    main()
