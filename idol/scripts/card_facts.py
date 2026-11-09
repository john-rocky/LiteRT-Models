"""Render the facts tables of the model card from the raw result files (never typed by hand).

  python card_facts.py --contract contract --out card/README.facts.md [--device device/results]

Reads contract/a_<input>_mac.json (line a on the Mac), contract/mat_bir_<input>_mac.json (the BiRefNet graph vs the
author's alpha), contract/standin_bir/<input>/b.json (line b on the Mac, the shipped pipeline),
and, when present, the phone's parity dumps compared on the Mac (contract/a_<input>_<device>.json) and its
end-to-end rows (contract/b_<input>_<device>.json, contract/c_<device>.json, contract/e_<device>.json).
"""
import argparse
import glob
import json
import os
import statistics

INPUTS = ('armchair', 'axolotl', 'succulent', 'desk')


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def load_a(contract, t, where):
    """Line (a) rows of input t: for the Mac, the v2 run with the v3 re-run of the rebuilt parts (matting, VAE) over
    it (graph set v3 = v2's files for every other part, same SHA-256)."""
    d = load(f'{contract}/a_{t}_{where}.json')
    if where == 'mac':
        v3 = load(f'{contract}/a_v3_{t}_mac.json')
        if d and v3:
            d = dict(d, **{k: v for k, v in v3.items() if isinstance(v, dict)})
    return d


def line_a_table(contract, where):
    found = [t for t in INPUTS if load_a(contract, t, where)]
    if not found:
        return ''
    rows = ['| Part | Compared tensor | ' + ' | '.join(found) + ' |', '|---|---|' + '---|' * len(found)]
    data = {t: load_a(contract, t, where) for t in found}
    groups = [
        ('DINOv3 encoder', 'feature1', ['dinov3 feature1']),
        ('VAE encoder', 'latent tokens', ['vae latent tokens']),
        ('Denoiser (5 steps)', 'latent velocity, worst step', [f'denoiser step {i} latent velocity' for i in range(5)]),
        ('Octree decoder (8 levels)', 'child probabilities, worst level', [f'octree level {l} probs' for l in range(1, 9)]),
        ('Gaussian decoder', 'xyz', ['gaussians xyz']),
        ('', 'colour (f_dc)', ['gaussians features_dc']),
        ('', 'scale', ['gaussians scaling']),
        ('', 'rotation', ['gaussians rotation']),
        ('', 'opacity', ['gaussians opacity']),
    ]
    for part, what, keys in groups:
        cells = []
        for t in found:
            vals = [data[t][k]['cos'] for k in keys if k in data[t]]
            cells.append(f'{min(vals):.5f}' if vals else '—')
        rows.append(f'| {part} | {what} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(rows)


def short_where(where):
    """A phone record's 'where' as a table cell: the phone and 'the app' (the full string stays in the record)."""
    for phone in ('Galaxy S26', 'Pixel 11 Pro'):
        if phone in where:
            return f'{phone}, the app'
    return where


def phone_order(path):
    """Mac first, then the Galaxy S26, then the Pixel 11 Pro."""
    return (0 if '_mac' in path else 1 if '_s26' in path else 2, path)


def matting_table(contract):
    rows = ['| Input | uint8 alpha pixels that differ from the author\'s | largest difference | crop square (author / here) |',
            '|---|---|---|---|']
    rows[0] = rows[0].replace('| Input |', '| Input | Where |')
    rows[1] = '|---|---|---|---|---|'
    for t in INPUTS:
        sources = [('Mac, LiteRT CPU', load(f'{contract}/a_v3_{t}_mac.json') or load(f'{contract}/mat_bir_{t}_mac.json'))]
        sources += [(short_where(load(p).get('where', p)), load(p))
                    for p in sorted(glob.glob(f'{contract}/a_{t}_*.json'), key=phone_order)
                    if not p.endswith('_mac.json') and not p.endswith('_mac_v1.json')]
        for where, d in sources:
            if not d or 'matting' not in d or d['matting'].get('alpha_uint8_pixels_differing') is None:
                continue
            m = d['matting']
            same = 'same' if m['box_birefnet'] == m['box_dis'] else f'{m["box_birefnet"]} / {m["box_dis"]}'
            total = m.get('alpha_pixels', 1024 * 1024)   # inputs 1-3: 1024 x 1024 after the short-side resize
            rows.append(f'| {t} | {where} | {m["alpha_uint8_pixels_differing"]:,} of {total:,} | '
                        f'{m["alpha_uint8_max_abs_diff"]} / 255 | {same} |')
    return '\n'.join(rows)


def line_b_table(contract):
    rows = ['| Input | Where | PSNR, 4 views pooled (threshold: 25 dB) | per view: 0°, 90°, 180°, 270° |', '|---|---|---|---|']
    for t in INPUTS:
        d = load(f'{contract}/standin_v3/{t}/b.json') or load(f'{contract}/standin_bir/{t}/b.json')
        if d:
            rows.append(f'| {t} | Mac, LiteRT GPU (Metal, fp16) | {d["psnr_db"]:.2f} dB | '
                        + ', '.join(f'{v:.2f}' for v in d['psnr_per_view_db']) + ' |')
        for p in sorted(glob.glob(f'{contract}/b_{t}_*.json'), key=phone_order):
            d = load(p)
            rows.append(f'| {t} | {short_where(d.get("where", os.path.basename(p)))} | {d["psnr_db"]:.2f} dB | '
                        + ', '.join(f'{v:.2f}' for v in d['psnr_per_view_db']) + ' |')
    return '\n'.join(rows)


def speed_tables(contract):
    """Line (c) and (e) per phone, one column per run of the e2e plan (contract/c_<device>.json and e_<device>.json from
    score_phone_ce.py): run 0 is the first photo after the app compiled its graphs at start, the later runs are the
    photos after it."""
    out = []
    names = {'matting': 'background removal (BiRefNet, CPU)', 'dinov3': 'DINOv3 encoder', 'vae': 'VAE encoder',
             'denoiser': 'denoiser, 5 steps', 'octree': 'octree decoder', 'gaussian_decoder': 'Gaussian decoder',
             'ply': 'PLY writer'}
    order = ['matting', 'dinov3', 'vae', 'denoiser', 'octree', 'gaussian_decoder', 'ply']
    for c_path in sorted(glob.glob(f'{contract}/c_*.json'), key=lambda p: (0 if 'cool' in p else 1, p)):
        dev = os.path.basename(c_path)[2:-5]
        c, e = load(c_path), load(f'{contract}/e_{dev}.json')
        if not c or 'per_run' not in c:
            continue   # c_s26_cool_thermal.json and other side records
        runs = sorted(c['per_run'], key=lambda r: r['run'])
        mem = {m['run']: m for m in (e or {}).get('per_run', [])}
        heads = ['first photo after start' if r['warmup'] else f'photo {r["run"] + 1}' for r in runs]
        rows = [f'| {c["where"]} | ' + ' | '.join(heads) + ' |', '|---|' + '---|' * len(runs),
                '| **photo → .ply written** | ' + ' | '.join(f'**{r["total_s"]:.0f} s**' for r in runs) + ' |']
        for k in order:
            if all(k in (r['stages'] or {}) for r in runs):
                rows.append(f'| {names[k]} | ' + ' | '.join(f'{r["stages"][k]:.1f} s' for r in runs) + ' |')
        rows.append('| graphs compiled during the photo | ' + ' | '.join(
            f'{r["compiles"]["count"]} ({r["compiles"]["seconds"]:.0f} s)' for r in runs) + ' |')
        if mem:
            rows.append('| peak resident memory (VmHWM) | ' + ' | '.join(
                f'{mem[r["run"]]["vmhwm_kb"] / 1024 / 1024:.2f} GiB' for r in runs) + ' |')
            rows.append('| peak GPU memory (kgsl) | ' + ' | '.join(
                f'{mem[r["run"]]["gpu_page_alloc_max_bytes"] / 2**30:.2f} GiB' for r in runs) + ' |')
        out += ['\n'.join(rows)]
    return out


def render(contract, with_speed=False):
    """The facts tables as markdown: line (a) on the Mac and the phones, background removal, line (b); with_speed adds
    the (c) / (e) tables (the card puts those in its Speed section)."""
    parts = ['### Per-part agreement on the Mac (LiteRT GPU, Metal, fp16): cosine vs the author\'s model',
             line_a_table(contract, 'mac')]
    for where, phone in (('s26', 'a Galaxy S26'), ('p11p', 'a Pixel 11 Pro')):
        t = line_a_table(contract, where)
        if t:
            parts += [f'### Per-part agreement on {phone} (the app; GPU fp16 with fp32 accumulation, BiRefNet on the '
                      'CPU): cosine vs the author\'s model', t]
    parts += ['### Background removal', matting_table(contract), '### Whole pipeline: 4-view render PSNR',
              line_b_table(contract)]
    if with_speed:
        st = speed_tables(contract)
        if st:
            parts += ['### Speed and memory'] + st
    return '\n\n'.join(p for p in parts if p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contract', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    text = render(a.contract, with_speed=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, 'w').write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
