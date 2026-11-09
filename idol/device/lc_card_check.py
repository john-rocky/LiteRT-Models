"""Card vs records: every number of litert-community/Idol's card that a litert-compat record of this lane carries, read
from the live card at the Hub main that device/lc/hub_api_idol.json names, against the two records
(idol__galaxy-s26.json, idol__pixel-11-pro.json) under a snapshot root. Each number is formatted the way the card
prints it and compared as text. Numbers of the card that no record carries (the 262k/ form, kgsl GPU memory, the warm
photos' stages, per-view PSNR, the Mac rows) are not read.

  python3 -I device/lc_card_check.py <root>              root = a snapshot root holding 2.3.0/<date>/idol__<device>.json
  python3 -I device/lc_card_check.py <root> --negative   the same with three record values moved: exactly 3 FAIL expected
"""

import glob
import json
import os
import re
import subprocess
import sys

K = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE_CARD = os.path.expanduser('~/code/idol_work/idol_hf_H/main/README.md')

STAGES = [('background removal (BiRefNet, CPU)', 'stage_matting_s'), ('DINOv3 encoder', 'stage_dinov3_s'),
          ('VAE encoder', 'stage_vae_s'), ('denoiser, 5 steps', 'stage_denoiser_s'),
          ('octree decoder', 'stage_octree_s'), ('Gaussian decoder', 'stage_gaussian_decoder_s'),
          ('PLY writer', 'stage_ply_s')]


def table(lines, header_prefix):
    """-> rows (lists of cells) of the first markdown table whose header line starts with header_prefix."""
    for i, line in enumerate(lines):
        if line.startswith(header_prefix):
            rows = []
            for row in lines[i + 2:]:
                if not row.startswith('|'):
                    break
                rows.append([c.strip() for c in row.strip().strip('|').split('|')])
            return rows
    raise SystemExit(f'STOP: no table starting {header_prefix!r}')


def cell(rows, label, col=1):
    for r in rows:
        if r[0].strip('*') == label:
            return r[col].strip('*')
    raise SystemExit(f'STOP: no row {label!r}')


def gib(kb):
    return f'{kb / 1048576:.2f} GiB'


def main():
    root, negative = sys.argv[1], '--negative' in sys.argv
    hub = json.load(open(os.path.join(K, 'device/lc/hub_api_idol.json')))
    card = subprocess.run(['curl', '-sf', f'https://huggingface.co/litert-community/Idol/raw/{hub["sha"]}/README.md'],
                          capture_output=True, check=True).stdout.decode()
    assert card == open(STAGE_CARD).read(), 'the live card differs from the staged card'
    lines = card.splitlines()
    rec = {}
    for slug in ('galaxy-s26', 'pixel-11-pro'):
        f = glob.glob(os.path.join(root, '2.3.0', '*', f'idol__{slug}.json'))
        assert len(f) == 1, (slug, f)
        rec[slug] = json.load(open(f[0]))['results'][0]['metrics']
    s, p = rec['galaxy-s26'], rec['pixel-11-pro']
    if negative:   # three values that the card prints once each
        s['stage_vae_s'] += 0.3
        s['warm_photo3_wall_s'] += 2.0
        p['vmhwm_kb'] *= 1.02
    checks = []   # (what, card text, record text)

    def phone_table(prefix, m, name):
        rows = table(lines, prefix)
        checks.append((f'{name}: photo -> .ply', cell(rows, 'photo → .ply written'), f'{m["first_photo_wall_s"]:.0f} s'))
        for label, key in STAGES:
            checks.append((f'{name}: {label}', cell(rows, label), f'{m[key]:.1f} s'))
        checks.append((f'{name}: graphs compiled at app start',
                       cell(rows, 'graphs compiled at app start, before the first photo'),
                       f'{m["graphs_compiled_at_app_start"]:.0f} ({m["compile_s_at_app_start"]:.0f} s)'))
        checks.append((f'{name}: graphs compiled during the photo', cell(rows, 'graphs compiled during the photo'),
                       f'{m["graphs_compiled_in_photo"]:.0f} ({m["compile_s_in_photo"]:.0f} s)'))
        checks.append((f'{name}: VmHWM', cell(rows, 'peak resident memory (VmHWM)'), gib(m['vmhwm_kb'])))

    phone_table('| Galaxy S26 (SM-S942Q), the sample app, LiteRT 2.3.0; from a cool phone', s, 'S26 cool')
    phone_table('| Pixel 11 Pro, the sample app, LiteRT 2.3.0', p, 'Pixel')
    warm = table(lines, "| Galaxy S26 (SM-S942Q), the sample app, LiteRT 2.3.0; right after the cool photo's hold")
    for i in range(1, 5):
        checks.append((f'S26 warm photo {i}', cell(warm, 'photo → .ply written', i), f'{s[f"warm_photo{i}_wall_s"]:.0f} s'))
    # whole-pipeline PSNR
    psnr = table(lines, '| Input | Where | PSNR, 4 views pooled')
    for r in psnr:
        where = {'Galaxy S26, the app': s, 'Pixel 11 Pro, the app': p}.get(r[1])
        if where is not None:
            checks.append((f'PSNR {r[0]} {r[1]}', r[2], f'{where[f"{r[0]}_psnr_db"]:.2f} dB'))
    # per-part agreement: the table's smallest cosine = the record's line_a_min_cos
    for prefix, m, name in (('### Per-part agreement on a Galaxy S26', s, 'S26'),
                            ('### Per-part agreement on a Pixel 11 Pro', p, 'Pixel')):
        i = next(n for n, l in enumerate(lines) if l.startswith(prefix))
        rows = table(lines[i:], '| Part | Compared tensor')
        checks.append((f'{name}: smallest per-part cosine', f'{min(float(r[2]) for r in rows):.5f}',
                       f'{m["line_a_min_cos"]:.5f}'))
    # the two-forms table, default column
    forms = table(lines, '| | default (the repo')
    checks.append(('two forms: S26 cool', cell(forms, 'Galaxy S26: the first photo after start, cool phone '
                                                      '(thermal status 0 at the start)'), f'{s["first_photo_wall_s"]:.0f} s'))
    w = [s[f'warm_photo{i}_wall_s'] for i in range(1, 5)]
    checks.append(('two forms: S26 warm', cell(forms, 'Galaxy S26: photos on a phone already warm').split(' (')[0],
                   f'{min(w):.0f}–{max(w):.0f} s'))
    checks.append(('two forms: Pixel', cell(forms, 'Pixel 11 Pro: the first photo after start'),
                   f'{p["first_photo_wall_s"]:.0f} s'))
    ps = [s[f'{t}_psnr_db'] for t in ('armchair', 'axolotl', 'succulent')]
    checks.append(('two forms: PSNR', cell(forms, '4-view render PSNR against the author\'s default on the Galaxy S26 '
                                                  '(threshold 25 dB)'), f'{min(ps):.1f}–{max(ps):.1f} dB'))
    fails = 0
    for what, c, r in checks:
        ok = c == r
        fails += not ok
        print(('PASS ' if ok else 'FAIL ') + f'{what}: card {c!r} record {r!r}')
    print(f'card {hub["sha"][:8]} vs records: {len(checks) - fails} PASS, {fails} FAIL'
          + (' (negative control: 3 expected)' if negative else ''))
    sys.exit(0 if (fails == 3 if negative else fails == 0) else 1)


if __name__ == '__main__':
    main()
