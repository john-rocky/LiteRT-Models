"""Score round g1: each graph's output on the phone (the runner's dump of fixture 000) against the same graph run by
LiteRT on the Mac CPU in float32 with the same inputs, plus the phone's per-call time from the leg record.

  python g1_score.py --lane <K> --models models/v2 --results device/g1 [--plan device/plan_g1.tsv] [--out contract/g1_s26.json]
(--results = the chain's round dir: <tag>.dump/000.f32 and <tag>.leg.json per leg)
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lrt import Graph  # noqa: E402


def cos_rel(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return {'cos': float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)),
            'rel_l2': float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30)),
            'max_abs_diff': float(np.abs(a - b).max()), 'finite': bool(np.isfinite(a).all())}


def legs(plan):
    for line in open(plan):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        f = line.split()
        sig = None
        for x in f[12:]:
            if x.startswith('args=--signature,'):
                sig = x.split(',', 1)[1]
        yield {'tag': f[0], 'accel': f[1], 'prec': f[2], 'graph': f[3], 'fx': f[4], 'signature': sig}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lane', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--results', required=True)
    ap.add_argument('--plan', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    plan = a.plan or os.path.join(a.lane, 'device', 'plan_g1.tsv')
    rows = []
    for leg in legs(plan):
        fman = json.load(open(os.path.join(a.lane, 'device', 'fx', f"{leg['fx']}.manifest.json")))
        fx_dir = os.path.join(a.lane, fman['dir'])
        ins = [np.fromfile(os.path.join(fx_dir, f"000_{s['name']}.f32"), np.float32).reshape(s['shape'])
               for s in fman['order']['inputs_subgraph_order']]
        rec = {'leg': leg['tag'], 'graph': leg['graph'], 'signature': leg['signature'] or 'serving_default',
               'accel': leg['accel'], 'precision': leg['prec']}
        lj = os.path.join(a.results, f"{leg['tag']}.leg.json")
        if os.path.exists(lj):
            rec['leg_record'] = json.load(open(lj))
        so = os.path.join(a.results, f"{leg['tag']}.stdout.txt")
        if os.path.exists(so):
            txt = open(so, errors='replace').read()
            m = re.search(r'^summary .* median=([\d.]+) min=([\d.]+) max=([\d.]+) .*compile_ms=([\d.]+) '
                          r'fully_accelerated=(\w+)', txt, re.M)
            if m:   # per call: write + run + readback, median of the timed rounds
                rec.update(median_ms=float(m.group(1)), min_ms=float(m.group(2)), max_ms=float(m.group(3)),
                           compile_ms=float(m.group(4)), fully_accelerated=m.group(5) == 'true')
        dl = os.path.join(a.results, f"{leg['tag']}.delegate.txt")
        if os.path.exists(dl):
            m = re.search(r'Replacing (\d+) out of (\d+) node\(s\) with delegate \((\w+)\) node, yielding (\d+) partitions',
                          open(dl, errors='replace').read())
            if m:
                rec['delegate'] = {'replaced': int(m.group(1)), 'nodes': int(m.group(2)), 'kind': m.group(3),
                                   'partitions': int(m.group(4))}
        dumps = sorted(glob.glob(os.path.join(a.results, f"{leg['tag']}.dump", '000.f32')))   # the chain's layout
        if not dumps:
            rec['status'] = 'no dump'
            rows.append(rec)
            print(f"{leg['tag']:<22} no dump", flush=True)
            continue
        phone = np.fromfile(dumps[0], np.float32)
        g = Graph(os.path.join(a.models, f"{leg['graph']}.tflite"), 'cpu')
        key = leg['signature'] or g.keys[0]
        ref = g.run(dict(zip([n for n, _ in g.io[key]['inputs']], ins)), key)
        name0 = sorted(ref, key=lambda s: int(s.split('_')[-1]))[0]
        r = ref[name0].ravel()
        rec.update(cos_rel(phone[:r.size], r), status='scored', out_elems=int(r.size), phone_elems=int(phone.size))
        rows.append(rec)
        print(f"{leg['tag']:<22} cos {rec['cos']:.6f} rel_l2 {rec['rel_l2']:.3e} finite {rec['finite']} "
              f"median {rec.get('median_ms', float('nan')):.1f} ms delegate {rec.get('delegate')}", flush=True)
        del g
    if a.out:
        json.dump(rows, open(a.out, 'w'), indent=1)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
