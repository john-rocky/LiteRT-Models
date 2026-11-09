"""Fill the main model card for form H (round 2's shipped form) from the result files (no number typed by hand).

Form H = 2,048 anchors (65,536 Gaussians), every GPU graph at FP16, one shared graph for the denoiser's middle blocks
(den_joint_mid, attention query chunk 1,024) and one for the Gaussian decoder's middle graphs (gs_mid). The template is
card/README.md; the round-1 helpers of fill_card.py and card_facts.py are reused with form H's files:

  line (a), Mac     contract/a_v3_<input>_mac.json over a_<input>_mac.json (DINOv3, VAE, matting: the same graph bytes),
                    then a_H_den_<input>_mac.json (denoiser) and a_H_gs_<input>_mac.json (octree and Gaussian decoder
                    against the author's run at 65,536 Gaussians)
  line (a), phones  contract/a_armchair_<phone>_H.json
  line (b)          contract/standin_H/<input>/b.json (Mac), contract/b_<input>_<phone>_H.json
  speed, memory     contract/c_s26_H_cool.json, c_s26_H_warm.json, c_p11p_H.json (+ e_*.json; score_phone_ce.py)
  GPU clock cap     the S26 thermal trace (s26_thermal_log.sh) inside the cool photo's plan window (hold.log)
  Mac run           the staged repo's python/run_idol.py log (times line and run_capped's peak footprint)

  python fill_card_h.py --contract contract --thermal-csv <trace> --cool-hold device/results/s26/H \
      --warm-hold device/results/s26/H_warm \
      --mac-log <run_idol log> --prev-sha <commit> --bir-leg device/g2/bir3.leg.json --hero-caption "..." \
      --out-main <staged main>/README.md
"""
import argparse
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import card_facts as CF  # noqa: E402
import fill_card as FC  # noqa: E402

INPUTS = ('armchair', 'axolotl', 'succulent')
PHONES = (('s26', 'a Galaxy S26'), ('p11p', 'a Pixel 11 Pro'))
LOAD_A_V3 = CF.load_a   # round 1's reader (main() points CF.load_a at load_a_h for the tables)


def load_a_h(contract, t, where):
    if where == 'mac':
        d = LOAD_A_V3(contract, t, 'mac')
        if not d:
            return None
        for f in (f'{contract}/a_H_den_{t}_mac.json', f'{contract}/a_H_gs_{t}_mac.json'):
            x = CF.load(f)
            assert x, f
            d = dict(d, **{k: v for k, v in x.items() if isinstance(v, dict) and k != 'matting'})
        return d
    return CF.load(f'{contract}/a_{t}_{where}_H.json')


def matting_table_h(contract):
    rows = ['| Input | Where | uint8 alpha pixels that differ from the author\'s | largest difference | crop square (author / here) |',
            '|---|---|---|---|---|']
    for t in INPUTS:
        sources = [('Mac, LiteRT CPU', CF.load(f'{contract}/a_v3_{t}_mac.json') or CF.load(f'{contract}/mat_bir_{t}_mac.json'))]
        sources += [(CF.short_where(CF.load(p).get('where', p)), CF.load(p))
                    for p in (f'{contract}/a_{t}_{dev}_H.json' for dev, _ in PHONES) if os.path.exists(p)]
        for where, d in sources:
            if not d or 'matting' not in d or d['matting'].get('alpha_uint8_pixels_differing') is None:
                continue
            m = d['matting']
            same = 'same' if m['box_birefnet'] == m['box_dis'] else f'{m["box_birefnet"]} / {m["box_dis"]}'
            total = m.get('alpha_pixels', 1024 * 1024)
            rows.append(f'| {t} | {where} | {m["alpha_uint8_pixels_differing"]:,} of {total:,} | '
                        f'{m["alpha_uint8_max_abs_diff"]} / 255 | {same} |')
    return '\n'.join(rows)


def line_b_table_h(contract):
    rows = ['| Input | Where | PSNR, 4 views pooled (threshold: 25 dB) | per view: 0°, 90°, 180°, 270° |', '|---|---|---|---|']
    for t in INPUTS:
        d = CF.load(f'{contract}/standin_H/{t}/b.json')
        assert d, f'standin_H/{t}/b.json'
        rows.append(f'| {t} | Mac, LiteRT GPU (Metal, fp16) | {d["psnr_db"]:.2f} dB | '
                    + ', '.join(f'{v:.2f}' for v in d['psnr_per_view_db']) + ' |')
        for dev, _ in PHONES:
            p = f'{contract}/b_{t}_{dev}_H.json'
            if os.path.exists(p):
                d = CF.load(p)
                assert d['pass_'] is True, p
                rows.append(f'| {t} | {CF.short_where(d["where"])} | {d["psnr_db"]:.2f} dB | '
                            + ', '.join(f'{v:.2f}' for v in d['psnr_per_view_db']) + ' |')
    return '\n'.join(rows)


def speed_table_h(contract, name):
    names = {'matting': 'background removal (BiRefNet, CPU)', 'dinov3': 'DINOv3 encoder', 'vae': 'VAE encoder',
             'denoiser': 'denoiser, 5 steps', 'octree': 'octree decoder', 'gaussian_decoder': 'Gaussian decoder',
             'ply': 'PLY writer'}
    c, e = CF.load(f'{contract}/c_{name}.json'), CF.load(f'{contract}/e_{name}.json')
    assert c, f'c_{name}.json'
    runs = sorted(c['per_run'], key=lambda r: r['run'])
    mem = {m['run']: m for m in (e or {}).get('per_run', [])}
    heads = ['first photo after start' if r['warmup'] else f'photo {r["run"] + 1}' for r in runs]
    rows = [f'| {c["where"]} | ' + ' | '.join(heads) + ' |', '|---|' + '---|' * len(runs),
            '| **photo → .ply written** | ' + ' | '.join(f'**{r["total_s"]:.0f} s**' for r in runs) + ' |']
    for k in ('matting', 'dinov3', 'vae', 'denoiser', 'octree', 'gaussian_decoder', 'ply'):
        if all(k in (r['stages'] or {}) for r in runs):
            rows.append(f'| {names[k]} | ' + ' | '.join(f'{r["stages"][k]:.1f} s' for r in runs) + ' |')
    pre = c.get('preload') or {}
    if pre.get('graphs') is not None:
        rows.append('| graphs compiled at app start, before the first photo | '
                    + f'{pre["graphs"]} ({pre["seconds"]:.0f} s)' + ' | —' * (len(runs) - 1) + ' |')
    rows.append('| graphs compiled during the photo | ' + ' | '.join(
        f'{r["compiles"]["count"]} ({r["compiles"]["seconds"]:.0f} s)' for r in runs) + ' |')
    if mem:
        rows.append('| peak resident memory (VmHWM) | ' + ' | '.join(
            f'{mem[r["run"]]["vmhwm_kb"] / 2**20:.2f} GiB' for r in runs) + ' |')
        if all((mem[r['run']].get('gpu_page_alloc_max_bytes') or -1) > 0 for r in runs):
            rows.append('| peak GPU memory (kgsl) | ' + ' | '.join(
                f'{mem[r["run"]]["gpu_page_alloc_max_bytes"] / 2**30:.2f} GiB' for r in runs) + ' |')
    return '\n'.join(rows)


def thermal_window(csv_path, hold_dir, plan):
    """GPU clock cap and thermal status of the S26 trace inside one plan's window (hold.log start / end lines)."""
    log = open(os.path.join(hold_dir, 'hold.log')).read()
    t0 = re.search(rf'^(\d\d:\d\d:\d\d) plan {plan}: start', log, re.M).group(1)
    t1 = re.search(rf'^(\d\d:\d\d:\d\d) plan {plan}: end', log, re.M).group(1)
    wrap = t1 < t0
    rows = []
    for r in csv.DictReader(open(csv_path)):
        t = r['time']
        inside = (t0 <= t or t <= t1) if wrap else (t0 <= t <= t1)
        if inside and r['max_gpuclk_mhz'] and r['skin_status']:
            rows.append(r)
    assert len(rows) >= 10, (t0, t1, len(rows))
    caps = [float(r['max_gpuclk_mhz']) for r in rows]
    st = [int(r['skin_status']) for r in rows]
    out = {'run': f'{plan} run 0 ({os.path.basename(hold_dir)}, plan {t0}-{t1})', 'samples': len(rows),
           'cap_mhz_start': caps[0], 'cap_mhz_min': min(caps), 'status_start': st[0], 'status_max': max(st),
           'skin_c_start': float(rows[0]['skin_c']), 'skin_c_max': max(float(r['skin_c']) for r in rows)}
    return out


def cap_spread(csv_path, hold_dir, plan):
    """The GPU clock cap readings inside one plan's window -> (low, high) of the caps read at least twice, and the
    readings below that range ([(MHz, count)]); single readings above it (the idle sample before the first compile)
    are left out."""
    log = open(os.path.join(hold_dir, 'hold.log')).read()
    t0 = re.search(rf'^(\d\d:\d\d:\d\d) plan {plan}: start', log, re.M).group(1)
    t1 = re.search(rf'^(\d\d:\d\d:\d\d) plan {plan}: end', log, re.M).group(1)
    wrap = t1 < t0
    n = {}
    for r in csv.DictReader(open(csv_path)):
        t = r['time']
        if ((t0 <= t or t <= t1) if wrap else (t0 <= t <= t1)) and r['max_gpuclk_mhz']:
            c = float(r['max_gpuclk_mhz'])
            n[c] = n.get(c, 0) + 1
    main_caps = [c for c, k in n.items() if k >= 2]
    lo, hi = min(main_caps), max(main_caps)
    return lo, hi, sorted((c, k) for c, k in n.items() if c < lo)


def mac_run(log_path):
    text = open(log_path, errors='replace').read()
    times = json.loads(re.findall(r'^(\{"preprocess".*\})$', text, re.M)[-1])
    peak = int(re.findall(r'^run_capped: .* rc=0 peak_footprint_mb=(\d+)', text, re.M)[-1])
    return (f'On an M4 Max Mac (Metal GPU) one photo took {times["all"]:.0f} s and up to {peak / 1024:.0f} GB of memory '
            '(the process footprint).'), times, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contract', required=True)
    ap.add_argument('--thermal-csv', required=True)
    ap.add_argument('--cool-hold', required=True)
    ap.add_argument('--warm-hold', required=True, help='the hold of the warm photos (device/results/s26/H_warm)')
    ap.add_argument('--mac-log', required=True)
    ap.add_argument('--mac-log-262k', default=None, help='the Mac run of the staged runner with --models 262k')
    ap.add_argument('--prev-sha', required=True, help='the Hub commit that holds the 262,144-Gaussian form')
    ap.add_argument('--bir-leg', required=True)
    ap.add_argument('--g1', required=True, help='the first form\'s S26 per-graph sweep (contract/g1_s26.json)')
    ap.add_argument('--g2', required=True, help='its VAE re-sweep (device/g2)')
    ap.add_argument('--hero-caption', required=True)
    ap.add_argument('--out-main', required=True)
    a = ap.parse_args()
    ct = a.contract
    cool, warm = CF.load(f'{ct}/c_s26_H_cool.json'), CF.load(f'{ct}/c_s26_H_warm.json')
    cr = next(r for r in cool['per_run'] if r['warmup'])
    tw = thermal_window(a.thermal_csv, a.cool_hold, 'e2e_cool')
    json.dump(tw, open(f'{ct}/c_s26_H_cool_thermal.json', 'w'), indent=1)
    ww = thermal_window(a.thermal_csv, a.warm_hold, 'e2e_armchair')
    json.dump(ww, open(f'{ct}/c_s26_H_warm_thermal.json', 'w'), indent=1)
    warm_totals = [r['total_s'] for r in warm['per_run']]
    # minutes of back-to-back runs on the phone before the warm photos: the cool photo's start to the warm plan's start
    hms = lambda t: int(t[:2]) * 3600 + int(t[3:5]) * 60 + int(t[6:8])
    t_cool = re.search(r'^(\d\d:\d\d:\d\d) plan e2e_cool: start', open(os.path.join(a.cool_hold, 'hold.log')).read(), re.M).group(1)
    t_warm = re.search(r'^(\d\d:\d\d:\d\d) plan e2e_armchair: start', open(os.path.join(a.warm_hold, 'hold.log')).read(), re.M).group(1)
    busy_min = ((hms(t_warm) - hms(t_cool)) % 86400) / 60
    # the first release's form (262k/): its cool first photo with this app from the same start as form H's
    # (c_s26_262k_cool.json, hold D1) when that exists, else round 1's (c_s26_cool.json, the first release's app)
    matched = os.path.exists(f'{ct}/c_s26_262k_cool.json')
    v3 = CF.load(f'{ct}/c_s26_262k_cool.json') if matched else CF.load(f'{ct}/c_s26_cool.json')
    v3_first = next(r for r in v3['per_run'] if r['warmup'])
    r1 = CF.load(f'{ct}/c_s26_cool_thermal.json')    # round 1's cool photo: thermal status and skin at its start
    r1_first = next(r for r in CF.load(f'{ct}/c_s26_cool.json')['per_run'] if r['warmup'])
    if matched:
        t1 = thermal_window(a.thermal_csv, os.path.join(MODULE, 'device/results/s26/D1'), 'e2e_cool')
        json.dump(t1, open(f'{ct}/c_s26_262k_cool_thermal.json', 'w'), indent=1)
        assert t1['status_start'] == tw['status_start'], (t1, tw)
    # round 1: photos on a phone already warm (thermal status 2): the 3-photo series and the hold right after it
    v3_warm_totals = ([r['total_s'] for r in CF.load(f'{ct}/c_s26.json')['per_run']]
                      + [r['total_s'] for r in CF.load(f'{ct}/e2e_s26_hold3.json')['runs']])
    v3_warm_inputs = len({'armchair'} | {r['input'] for r in CF.load(f'{ct}/e2e_s26_hold3.json')['runs']})
    p11 = CF.load(f'{ct}/c_p11p_H.json')
    p11_first = next(r for r in p11['per_run'] if r['warmup'])
    b_h = [CF.load(f'{ct}/b_{t}_s26_H.json')['psnr_db'] for t in INPUTS]
    b_v3 = [CF.load(f'{ct}/b_{t}_s26.json')['psnr_db'] for t in INPUTS]
    gauss = cool['gaussians']
    assert gauss == 65536, gauss
    npre = cool['preload']['graphs']
    wlo, whi, wlow = cap_spread(a.thermal_csv, a.warm_hold, 'e2e_armchair')
    # D0: this app's one photo of the 262k/ files from thermal status 1 (round 1's manifest; the same .ply as round 1)
    d0 = json.load(open(os.path.join(MODULE, 'device/results/s26/D0/e2e_cool/e2e_cool.0.json')))
    d0_skin = float([l for l in open(os.path.join(MODULE, 'device/results/s26/D0/form.log')) if 'cool after' in l][0]
                    .strip().split(': ', 1)[1].split(',')[3])
    # the Python runner on 262k/: claimed only once its like-for-like Mac run (the reference's mode and seed) passes
    # line (b) against the author's output (contract/b_armchair_mac_runner_262k_s7.json, ~/code/idol_work/run_mac_262k_s7.sh)
    r262 = CF.load(f'{ct}/b_armchair_mac_runner_262k_s7.json')
    py262 = bool(r262 and r262.get('pass_') is True)
    speed_par = (f'On a Galaxy S26, the sample app turns one photo into a splat of {gauss:,} Gaussians in '
                 f'{cr["total_s"]:.0f} s, starting from a cool phone (the first photo after start). The phone throttles '
                 f'its GPU during that photo: the GPU clock cap fell from {tw["cap_mhz_start"]:,.0f} to '
                 f'{tw["cap_mhz_min"]:.0f} MHz (thermal status {tw["status_start"]} → {tw["status_max"]}). After '
                 f'{busy_min:.0f} minutes of back-to-back runs the phone was at thermal status {ww["status_start"]}; '
                 f'through the next {len(warm_totals)} photos the GPU clock cap stayed at {wlo:.0f}–{whi:.0f} MHz'
                 + (' (' + ', '.join(f'{"one reading" if k == 1 else f"{k} readings"} of {c:.0f} MHz' for c, k in wlow) + ')'
                    if wlow else '')
                 + f', and they took {min(warm_totals):.0f}–{max(warm_totals):.0f} s. At start the '
                 f'app compiles {npre} graph{"" if npre == 1 else "s"} before the photo ({cool["preload"]["seconds"]:.0f} s) '
                 f'and {cr["compiles"]["count"]} more during it ({cr["compiles"]["seconds"]:.0f} s): the denoiser\'s 22 '
                 'middle blocks share one graph and the Gaussian decoder\'s six middle graphs share another, so each is '
                 f'compiled once per photo. The first release\'s form in `262k/` (262,144 Gaussians, a graph per block, '
                 f'GPU FP16 with fp32 accumulation) took {v3_first["total_s"]:.0f} s '
                 + (f'the same way, also from thermal status {t1["status_start"]} (skin {t1["skin_c_start"]:.1f} °C at the '
                    f'start against {tw["skin_c_start"]:.1f} °C).' if matched else
                    f'the same way with the first release\'s app, from a warmer start (thermal status '
                    f'{r1["status_start"]}, skin {r1["skin_c_start"]:.1f} °C); this app\'s one photo of the `262k/` files '
                    f'from thermal status {d0["thermal_start"]} (skin {d0_skin:.1f} °C) took {d0["total"]:.0f} s.'))
    seat = ('Idol (speridlabs/idol with the idol-turbo LoRA) as LiteRT graphs for `CompiledModel`, converted with '
            f'litert-torch: on a Galaxy S26, the Android sample in `android/` turns one photo into a 3D Gaussian splat of '
            f'{gauss:,} Gaussians in {cr["total_s"]:.0f} s with the default form, or of 262,144 Gaussians in '
            f'{v3_first["total_s"]:.0f} s with the first release\'s form in `262k/` (the first photo after start on a '
            'cool phone, LiteRT 2.3.0' + ('' if matched else f'; the {v3_first["total_s"]:.0f} s is the first release\'s '
                                          'measurement of 2026-11-09') + ').')
    enc512 = [CF.load(f'{ct}/shape_probe/enc512_{t}_render.json')['psnr_db'] for t in INPUTS]
    forms_sentence = (f'The default form (the repo\'s root) is the fastest of the forms measured on a Galaxy S26: it makes '
                      f'a splat in {cr["total_s"]:.0f} s against {v3_first["total_s"]:.0f} s for the first release\'s '
                      f'form in `262k/`, while its renders agree less closely with the author\'s default output: '
                      f'{min(b_h):.1f}–{max(b_h):.1f} dB PSNR against {min(b_v3):.1f}–{max(b_v3):.1f} dB (higher is '
                      'closer). Of the other forms tried, int8 weights did not run on the phone GPUs, a 512 × 512 input '
                      f'made even the author\'s own PyTorch pipeline miss the 25 dB bar ({min(enc512):.1f}–'
                      f'{max(enc512):.1f} dB against its 1,024 × 1,024 output), the denoiser did not compile for the '
                      'Galaxy S26\'s NPU on the phone, and the CPU alone was slower.')
    forms_table = '\n'.join([
        '| | default (the repo\'s root) | first release (`262k/`) |',
        '|---|---|---|',
        '| Gaussians per splat | 65,536 (2,048 anchors) | 262,144 (8,192 anchors, the author\'s default) |',
        '| GPU precision | FP16 | FP16 with fp32 accumulation |',
        '| Denoiser blocks 1–22 and Gaussian decoder graphs 1–6 | one shared graph per chain; the app copies each '
        'block\'s weights in before the call | a graph per block |',
        f'| Galaxy S26: the first photo after start, cool phone (thermal status {tw["status_start"]} at the start) | '
        f'**{cr["total_s"]:.0f} s** | '
        + (f'{v3_first["total_s"]:.0f} s |' if matched else
           f'{v3_first["total_s"]:.0f} s (the first release\'s app, thermal status {r1["status_start"]} at the start; '
           f'this app\'s one photo of the `262k/` files from thermal status {d0["thermal_start"]}: {d0["total"]:.0f} s) |'),
        f'| Galaxy S26: photos on a phone already warm | {min(warm_totals):.0f}–{max(warm_totals):.0f} s '
        f'({len(warm_totals)} photos of the armchair, thermal status {ww["status_start"]}) | '
        f'{min(v3_warm_totals):.0f}–{max(v3_warm_totals):.0f} s ({len(v3_warm_totals)} photos of {v3_warm_inputs} '
        'inputs, thermal status 2) |',
        f'| Pixel 11 Pro: the first photo after start | {p11_first["total_s"]:.0f} s | not measured |',
        f'| 4-view render PSNR against the author\'s default on the Galaxy S26 (threshold 25 dB) | '
        f'{min(b_h):.1f}–{max(b_h):.1f} dB | {min(b_v3):.1f}–{max(b_v3):.1f} dB |'])
    forms_table += ('\n\nThermal status is Android\'s: 0 none, 1 light, 2 moderate, 3 severe. Both forms run with the '
                    'same app' + (' and the same Python runner: point them at the repo\'s root or at `262k/`.' if py262
                                  else ': point `install_to_device.sh` at the repo\'s root or at `262k/`.'))
    if matched:
        forms_table += (f'\n\nThe first release\'s card gives {r1_first["total_s"]:.0f} s for the `262k/` form: that photo '
                        f'started on a warmer phone (thermal status {r1["status_start"]}, skin {r1["skin_c_start"]:.1f} °C, '
                        f'against {t1["skin_c_start"]:.1f} °C here). With this app the `262k/` form makes the same splat as '
                        'with the first release\'s, byte for byte.')
    leg = json.load(open(a.bir_leg))
    assert leg['graph'] == 'bir' and leg['status'] == 'DONE', leg['status']
    bir_hwm_kb = int(re.search(r'VmHWM_max_kB (\d+)', leg['memory']).group(1))
    CF.load_a = load_a_h
    facts = ['### Per-part agreement on the Mac (LiteRT GPU, Metal, fp16): cosine vs the author\'s model',
             CF.line_a_table(ct, 'mac')]
    for where, phone in PHONES:
        t = CF.line_a_table(ct, where)
        if t:
            facts += [f'### Per-part agreement on {phone} (the app; GPU FP16, BiRefNet on the CPU): cosine vs the '
                      'author\'s model', t]
    facts += ['### Background removal', matting_table_h(ct), '### Whole pipeline: 4-view render PSNR',
              line_b_table_h(ct)]
    tables = [speed_table_h(ct, n) for n in ('s26_H_cool', 's26_H_warm', 'p11p_H') if os.path.exists(f'{ct}/c_{n}.json')]
    mac_text, _, _ = mac_run(a.mac_log)
    lo, hi, n, n_graphs = FC.call_range(a.g1, a.g2)
    vals = {
        'SEAT': seat,
        'HERO_CAPTION': a.hero_caption,
        'SPEED_PARAGRAPH': speed_par,
        'BIR_S26': f'{cr["stages"]["matting"]:.1f}',
        'FORMS_SENTENCE': forms_sentence,
        'FORMS_TABLE': forms_table,
        'PY_262K': '; --models 262k for the other form' if py262 else '',
        'PREV_SHORT': a.prev_sha[:8],
        'PREV_CARD_URL': f'https://huggingface.co/litert-community/Idol/blob/{a.prev_sha}/README.md',
        'SPEED_TABLE': '\n\n'.join(tables),
        'FACTS': '\n\n'.join(facts),
        'MAC_RUN': mac_text,
        'PYTHON_SNIPPET': open(os.path.join(MODULE, 'card/snippets/bir_python.py')).read().rstrip(),
        'KOTLIN_SNIPPET': FC.kotlin_block("// ---- the card's block, verbatim", "// ---- end of the card's block"),
        'BIR_PEAK': f'{bir_hwm_kb / 2**20:.2f} GiB of resident memory (VmHWM) in one call',
        'CALL_LENGTHS': (f'with GPU fp16 with fp32 accumulation, every GPU call of the first form took {lo:.2f}–{hi:.2f} s '
                         f'on the Galaxy S26 ({n_graphs} graphs, {n} signatures); this form calls the same blocks, or the '
                         'Gaussian decoder\'s at a quarter of the anchors, at FP16'),
    }
    text = FC.fill(open(os.path.join(MODULE, 'card/README.md')).read(), vals)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_main)), exist_ok=True)
    open(a.out_main, 'w').write(text)
    print(a.out_main, f'{len(text):,} chars')


if __name__ == '__main__':
    main()
