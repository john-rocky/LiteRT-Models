"""Order a g1 plan by priority for the hold clock (no leg starts after 25 min; legs left over carry to the next hold):
background removal, DINOv3, VAE, the denoiser's distinct graphs (context, input, first, one middle and the last joint
block), the octree buckets and the Gaussian decoder first; the 21 middle joint blocks that share den_joint_01's
structure last.

  python order_plan.py device/plan_g1_all.tsv device/plan_g1.tsv [--skip-done device/g1]
"""
import argparse
import os
import re

FIRST = (['bir'] + [f'dino{i}' for i in range(4)] + ['vae0', 'vae1', 'vae2', 'vae3', 'vaeattn', 'vae4']
         + ['denctx', 'denin', 'dj00', 'dj01', 'dj23'] + [f'octL{n}' for n in (64, 256, 1024, 4096, 8192)]
         + [f'gs{i}' for i in range(8)])


def rank(tag):
    if tag in FIRST:
        return (0, FIRST.index(tag))
    m = re.fullmatch(r'dj(\d+)', tag)
    return (1, int(m.group(1))) if m else (2, tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--skip-done', default=None, help='a round dir: drop legs whose <tag>.leg.json says DONE')
    a = ap.parse_args()
    head, rows = [], []
    for line in open(a.src):
        (head if line.startswith('#') or not line.strip() else rows).append(line.rstrip('\n'))
    done = set()
    if a.skip_done:
        import json
        for r in rows:
            tag = r.split()[0]
            p = os.path.join(a.skip_done, f'{tag}.leg.json')
            if os.path.exists(p) and json.load(open(p)).get('status') == 'DONE':
                done.add(tag)
    rows = sorted((r for r in rows if r.split()[0] not in done), key=lambda r: rank(r.split()[0]))
    with open(a.dst, 'w') as f:
        f.write('\n'.join(head) + '\n# order: priority for the hold clock (device/order_plan.py)\n')
        f.write('\n'.join(rows) + '\n')
    print(a.dst, len(rows), 'legs', f'({len(done)} done legs dropped)' if a.skip_done else '')


if __name__ == '__main__':
    main()
