"""Fill the two model cards from the result files and the tested snippets (no number typed by hand).

  python fill_card.py --contract contract --g1 contract/g1_s26.json --g2 device/g2 --bir-leg device/g2/bir3.leg.json \
      --hero-caption "..." --out-main <staged main>/README.md --out-dino <staged dino>/README.md.pre-gate

Main card (card/README.md): speed paragraph and table from contract/c_s26.json and e_s26.json (score_phone_ce.py),
the facts tables from card_facts.py, the per-call range from the S26 graph sweeps (g1, and g2 for the rebuilt VAE),
the Python block from card/snippets/bir_python.py (run on the Mac as written) and the Kotlin block cut from
app/src/main/java/com/idol/CardSnippet.kt (run on the phone as written, plans/snippet.json).
DINOv3 card (card/dino_README.md): its line from contract/a_<input>_mac.json and a_armchair_s26.json, the blocks from
card/snippets/dino_python.py and CardSnippet.kt's DINOv3 block. Its gate front matter is gate_front_matter.py's step.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import card_facts  # noqa: E402

STAGES_FOR_FACTS = ['armchair', 'axolotl', 'succulent']


def kotlin_block(begin, end):
    src = open(os.path.join(MODULE, 'app/src/main/java/com/idol/CardSnippet.kt')).read()
    m = re.search(re.escape(begin) + r'[^\n]*\n(.*?)\n[ \t]*' + re.escape(end), src, re.S)
    assert m, (begin, end)
    return textwrap.dedent(m.group(1)).rstrip()


def call_range(g1_path, g2_dir):
    """Per-call median range of the shipped GPU graphs on the S26: g1 legs, with the VAE legs taken from g2."""
    vals = {}
    for r in json.load(open(g1_path)):
        if r.get('accel') == 'gpu' and r.get('median_ms') and not r['graph'].startswith('vae_'):
            vals[r['leg']] = r['median_ms']
    for p in glob.glob(os.path.join(g2_dir, 'vae*.stdout.txt')):
        m = re.search(r'median=([\d.]+)', open(p, errors='replace').read())
        if m:
            vals[os.path.basename(p).split('.')[0]] = float(m.group(1))
    return min(vals.values()) / 1000, max(vals.values()) / 1000, len(vals)


def fill(text, values):
    for k, v in values.items():
        assert text.count('{' + k + '}') == 1, k
        text = text.replace('{' + k + '}', v)
    left = re.findall(r'\{[A-Z][A-Z0-9_]+\}', text)
    assert not left, left
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contract', required=True)
    ap.add_argument('--g1', required=True)
    ap.add_argument('--g2', required=True)
    ap.add_argument('--bir-leg', required=True, help='the S26 leg record of one BiRefNet call (device/g2/bir3.leg.json)')
    ap.add_argument('--hero-caption', required=True)
    ap.add_argument('--out-main', required=True)
    ap.add_argument('--out-dino', required=True)
    a = ap.parse_args()
    c = json.load(open(os.path.join(a.contract, 'c_s26.json')))
    lo, hi, n = call_range(a.g1, a.g2)
    leg = json.load(open(a.bir_leg))
    assert leg['graph'] == 'bir' and leg['status'] == 'DONE', leg['status']
    bir_hwm_kb = int(re.search(r'VmHWM_max_kB (\d+)', leg['memory']).group(1))
    facts = card_facts.render(a.contract)
    speed = '\n\n'.join(card_facts.speed_tables(a.contract))
    warm_runs = sorted(c['per_run'], key=lambda r: r['run'])
    warm_lo, warm_hi = min(r['total_s'] for r in warm_runs), max(r['total_s'] for r in warm_runs)
    comp_n = statistics.median(r['compiles']['count'] for r in warm_runs)
    cool_path = os.path.join(a.contract, 'c_s26_cool.json')
    therm = list(csv.DictReader(open(os.path.join(a.contract, 's26_thermal.csv'))))
    cap_min = min(float(r['max_gpuclk_mhz']) for r in therm if r['max_gpuclk_mhz'])
    cap_top = max(int(f) for f in open(os.path.join(a.contract, 's26_gpu_available_frequencies.txt')).read().split()) / 1e6
    if os.path.exists(cool_path):
        cool = json.load(open(cool_path))
        cr = sorted(cool['per_run'], key=lambda r: r['run'])
        head = (f'On a Galaxy S26, the sample app makes a splat of {c["gaussians"]:,} Gaussians from a photo in '
                f'{cr[0]["total_s"]:.0f} s, starting from a cool phone (the first photo after start'
                + (f'; {cr[1]["total_s"]:.0f} s for the next one' if len(cr) > 1 else '') + '). ')
        bir_s = cr[0]['stages']['matting']
    else:
        head = f'On a Galaxy S26, the sample app makes a splat of {c["gaussians"]:,} Gaussians from a photo. '
        bir_s = c['median_stage_s']['matting']
    speed_par = (head + f'Photo after photo, the phone throttles its GPU: with its thermal status at 2 (moderate), '
                 f'photos took {warm_lo:.0f}–{warm_hi:.0f} s, and after half an hour of them the GPU clock was capped '
                 f'at {cap_min:.0f} of {cap_top:,.0f} MHz. The graphs do not all fit in the phone\'s memory at once, so '
                 'the app closes graphs and compiles them again when the run comes back to them: about '
                 f'{comp_n:.0f} compiles per photo.')
    main_vals = {
        'HERO_CAPTION': a.hero_caption,
        'SPEED_PARAGRAPH': speed_par,
        'BIR_S26': f'{bir_s:.1f}',
        'BIR_PEAK': f'{bir_hwm_kb / 2**20:.2f} GiB of resident memory (VmHWM) in one call',
        'SPEED_TABLE': speed,
        'FACTS': facts,
        'PYTHON_SNIPPET': open(os.path.join(MODULE, 'card/snippets/bir_python.py')).read().rstrip(),
        'KOTLIN_SNIPPET': kotlin_block("// ---- the card's block, verbatim", "// ---- end of the card's block"),
        'CALL_LENGTHS': f'on the Galaxy S26 each of the {n} GPU graphs takes {lo:.2f}–{hi:.2f} s per call',
    }
    main = fill(open(os.path.join(MODULE, 'card/README.md')).read(), main_vals)
    mac = []
    for t in STAGES_FOR_FACTS:
        d = card_facts.load_a(a.contract, t, 'mac')
        if d and 'dinov3 feature1' in d:
            mac.append(f'{d["dinov3 feature1"]["cos"]:.6f} ({t})')
    dino_line = 'cosine ' + ', '.join(mac)
    phones = []
    for dev, phone in (('s26', 'a Galaxy S26'), ('p11p', 'a Pixel 11 Pro')):
        d = card_facts.load(os.path.join(a.contract, f'a_armchair_{dev}.json'))
        if d and 'dinov3 feature1' in d:
            phones.append(f'{d["dinov3 feature1"]["cos"]:.6f} on {phone}')
    if phones:
        dino_line += '; on the phones (GPU, fp16 with fp32 accumulation, armchair) ' + ', '.join(phones)
    dino_vals = {
        'DINO_FACTS': dino_line,
        'DINO_PYTHON_SNIPPET': open(os.path.join(MODULE, 'card/snippets/dino_python.py')).read().rstrip(),
        'DINO_KOTLIN_SNIPPET': kotlin_block("// ---- the DINOv3 card's block, verbatim",
                                            "// ---- end of the DINOv3 card's block"),
    }
    dino_text = open(os.path.join(MODULE, 'card/dino_README.md')).read()
    dino_text = dino_text.replace('{GATE_FRONT_MATTER}', '\x00GATE\x00')
    dino = fill(dino_text, dino_vals).replace('\x00GATE\x00', '{GATE_FRONT_MATTER}')
    for path, text in ((a.out_main, main), (a.out_dino, dino)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        open(path, 'w').write(text)
        print(path, f'{len(text):,} chars')


if __name__ == '__main__':
    main()
