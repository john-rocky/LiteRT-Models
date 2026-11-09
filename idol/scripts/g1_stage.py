"""Round g1 staging: every shipped graph (and every signature of the octree graph) as one devicebench leg, with the
inputs that graph really gets in a run of the whole pipeline on the Mac (LiteRT, Metal), for the S26 sweep of
delegation, per-call time and per-graph agreement (npu_chain.sh, the C-API runner).

  python g1_stage.py --models models/v2 --ref ref/armchair --image IMG --lane <K> [--plan device/plan_g1.tsv]

Writes device/fx/g1_<graph>[_<signature>]/000_<input>.f32 + device/fx/<fx>.manifest.json (the runner's --fixtures
contract, as f0_stage.py), and the plan: fp16acc32 on the GPU (BiRefNet: CPU), one parity fixture, warm-up 1,
3 timed rounds.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import idol_runner as R  # noqa: E402


def signature_io(path, key):
    """(inputs in the subgraph's order with their signature names, output 0) of the signature `key`."""
    from ai_edge_litert import schema_py_generated as schema
    m = schema.ModelT.InitFromPackedBuf(open(path, 'rb').read(), 0)
    dec = lambda b: b.decode() if isinstance(b, bytes) else b  # noqa: E731
    sig = next(s for s in m.signatureDefs if dec(s.signatureKey) == key)
    g = m.subgraphs[sig.subgraphIndex]
    name_of = {t.tensorIndex: dec(t.name) for t in sig.inputs}
    outs = sorted(((dec(t.name), t.tensorIndex) for t in sig.outputs), key=lambda x: int(x[0].split('_')[-1]))
    ins = [{'name': name_of[i], 'ext': '.f32', 'dtype': 'FLOAT32', 'shape': [int(x) for x in g.tensors[i].shape]}
           for i in g.inputs]
    o_name, o_idx = outs[0]
    return ins, {'name': o_name, 'dtype': 'FLOAT32', 'shape': [int(x) for x in g.tensors[o_idx].shape]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--lane', required=True)
    ap.add_argument('--plan', default=None)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    meta = json.load(open(os.path.join(a.ref, 'meta.json')))
    noise = dict(np.load(os.path.join(a.ref, 'noise.npz')))
    captured = {}   # (graph, signature key) -> list of input arrays, the first call only
    orig = R.IdolLiteRT.run1

    def run1(self, name, inputs, key=None):
        k = (name, key or self.graph(name).keys[0])
        if k not in captured:
            arrs = inputs if not isinstance(inputs, dict) else [inputs[n] for n, _ in self.graph(name).io[k[1]]['inputs']]
            captured[k] = [np.ascontiguousarray(np.asarray(x, dtype=np.float32)).copy() for x in arrs]
        return orig(self, name, inputs, key)

    R.IdolLiteRT.run1 = run1
    p = R.IdolLiteRT(a.models, 'gpu', matting='birefnet')
    out_dir = os.path.join(a.lane, 'device', 'g1_run')
    res = p.run(a.image, meta['mode'], noise, out_dir)
    print('pipeline', json.dumps(res['times']), flush=True)

    man = json.load(open(os.path.join(a.models, 'manifest.json')))
    fx_root = os.path.join(a.lane, 'device', 'fx')
    rows = []
    for (name, key), arrays in captured.items():
        g = man['graphs'][name]
        path = os.path.join(a.models, g['file'])
        ins, out0 = signature_io(path, key)
        multi = len(g['signatures']) > 1
        fx = f'g1_{name}_{key}' if multi else f'g1_{name}'
        d = os.path.join(fx_root, fx)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
        for spec, arr in zip(ins, arrays):
            assert list(arr.shape) == spec['shape'], (name, key, spec, arr.shape)
            arr.tofile(os.path.join(d, f"000_{spec['name']}.f32"))
        names = sorted(os.listdir(d), key=lambda s: s.encode())
        h, total = hashlib.sha256(), 0
        for fn in names:
            b = open(os.path.join(d, fn), 'rb').read()
            h.update(b)
            total += len(b)
        fman = {'dir': os.path.relpath(d, a.lane), 'files': len(names), 'bytes': total,
                'concat_sha256_bytesorted': h.hexdigest(), 'count': 1, 'L': int(ins[0]['shape'][1]),
                'out_elems': int(np.prod(out0['shape'])), 'output0': out0, 'order': {'inputs_subgraph_order': ins},
                'graph': f'{os.path.relpath(path, a.lane)} ({key})',
                'pattern': f'the inputs this graph got in the Mac pipeline run of {os.path.basename(a.ref)} (LiteRT Metal)'}
        with open(os.path.join(fx_root, f'{fx}.manifest.json'), 'w') as f:
            json.dump(fman, f, indent=1)
        cpu = g['accelerator'] == 'cpu'
        tag = fx[3:].replace('den_joint_', 'dj').replace('_', '')[:20]
        extras = 'rest=5' + (f' args=--signature,{key}' if multi else '')   # extras are space-separated key=value
        rows.append(f"{tag:<22} {'cpu' if cpu else 'gpu':<5} {'-' if cpu else 'fp16acc32':<10} {name:<14} {fx:<20} "
                    f"1 1 3 000 - 300 gate {extras}")
        print(f'{fx}: {len(names)} files {total:,} B, output0 {out0["shape"]}', flush=True)
    if a.plan:
        with open(a.plan, 'w') as f:
            f.write('# g1: every shipped graph once on the S26 (fp16acc32 GPU; BiRefNet CPU), real inputs from the Mac run of\n'
                    f'# {os.path.basename(a.ref)}; parity fixture 000 dumped, warm-up 1, 3 timed rounds. GRAPH_DIR = models/v2.\n'
                    '# tag accel prec graph fx count warmup rounds ab cache compile_s ready extras\n')
            f.write('\n'.join(rows) + '\n')
        print('plan', a.plan, len(rows), 'legs')


if __name__ == '__main__':
    main()
