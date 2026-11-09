"""Card vs record for the 262k/ form: every 262k/ number of litert-community/Idol's card (the live card at the Hub main
that device/lc/hub_api_idol.json names) against the record idol-262k__galaxy-s26.json under a snapshot root, formatted
the way the card prints it and compared as text. The default form's numbers are lc_card_check.py's.

  python3 -I device/lc_card_check_262k.py <root>              root = a snapshot root holding 2.3.0/<date>/idol-262k__galaxy-s26.json
  python3 -I device/lc_card_check_262k.py <root> --negative   three record values moved: exactly 3 FAIL expected
"""

import glob
import json
import os
import re
import subprocess
import sys

K = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def one(pattern, text, what):
    m = re.findall(pattern, text)
    if len(m) != 1:
        raise SystemExit(f'STOP: {what}: {len(m)} matches of {pattern!r}')
    return m[0]


def main():
    root, negative = sys.argv[1], '--negative' in sys.argv
    hub = json.load(open(os.path.join(K, 'device/lc/hub_api_idol.json')))
    card = subprocess.run(['curl', '-sf', f'https://huggingface.co/litert-community/Idol/raw/{hub["sha"]}/README.md'],
                          capture_output=True, check=True).stdout.decode()
    f = glob.glob(os.path.join(root, '2.3.0', '*', 'idol-262k__galaxy-s26.json'))
    assert len(f) == 1, f
    m = json.load(open(f[0]))['results'][0]['metrics']
    if negative:   # three values that the card prints once each
        m['first_release_app_first_photo_wall_s'] += 2.0
        m['first_release_app_warm_photo4_wall_s'] += 2.0
        m['first_release_app_first_photo_skin_temp_c_start'] += 0.2
    wall = f'{m["first_photo_wall_s"]:.0f}'
    warm = [m[f'first_release_app_warm_photo{i}_wall_s'] for i in range(1, 6)]
    psnr = [m['armchair_psnr_db'], m['axolotl_psnr_db'], m['first_release_app_succulent_psnr_db']]
    checks = []
    seat = one(r'or of ([\d,]+) Gaussians in (\d+) s with the first release\'s form in `262k/`', card, 'seat')
    checks += [('seat: Gaussians', seat[0], f'{m["gaussians"]:,.0f}'), ('seat: wall', seat[1], wall)]
    forms = one(r'against (\d+) s for the first release\'s form in `262k/`.*?against ([\d.]+)–([\d.]+) dB', card,
                'two-forms sentence')
    checks += [('two-forms sentence: wall', forms[0], wall),
               ('two-forms sentence: PSNR low', forms[1], f'{min(psnr):.1f}'),
               ('two-forms sentence: PSNR high', forms[2], f'{max(psnr):.1f}')]
    cool = one(r'\| Galaxy S26: the first photo after start, cool phone \(thermal status 0 at the start\) \| [^|]+ \| '
               r'(\d+) s \|', card, 'table: cool')
    checks.append(('table: cool', cool, wall))
    w = one(r'\| Galaxy S26: photos on a phone already warm \| [^|]+ \| (\d+)–(\d+) s \((\d) photos of 3 inputs, '
            r'thermal status (\d)\) \|', card, 'table: warm')
    checks += [('table: warm low', w[0], f'{min(warm):.0f}'), ('table: warm high', w[1], f'{max(warm):.0f}'),
               ('table: warm photos', w[2], str(len(warm))),
               ('table: warm thermal status', w[3],
                '/'.join(sorted({f'{m[f"first_release_app_warm_photo{i}_thermal_status_start"]:.0f}' for i in range(1, 6)})))]
    p = one(r'\| 4-view render PSNR against the author\'s default on the Galaxy S26 \(threshold 25 dB\) \| [^|]+ \| '
            r'([\d.]+)–([\d.]+) dB \|', card, 'table: PSNR')
    checks += [('table: PSNR low', p[0], f'{min(psnr):.1f}'), ('table: PSNR high', p[1], f'{max(psnr):.1f}')]
    r1 = one(r'The first release\'s card gives (\d+) s for the `262k/` form: that photo started on a warmer phone '
             r'\(thermal status (\d), skin ([\d.]+) °C, against ([\d.]+) °C here\)', card, 'first release paragraph')
    checks += [('paragraph: first release wall', r1[0], f'{m["first_release_app_first_photo_wall_s"]:.0f}'),
               ('paragraph: first release thermal status', r1[1],
                f'{m["first_release_app_first_photo_thermal_status_start"]:.0f}'),
               ('paragraph: first release skin', r1[2], f'{m["first_release_app_first_photo_skin_temp_c_start"]:.1f}'),
               ('paragraph: skin here', r1[3], f'{m["skin_temp_c_start"]:.1f}')]
    sp = one(r'took (\d+) s the same way, also from thermal status (\d) \(skin ([\d.]+) °C at the start', card,
             'speed sentence')
    checks += [('speed: wall', sp[0], wall), ('speed: thermal status', sp[1], f'{m["thermal_status_start"]:.0f}'),
               ('speed: skin', sp[2], f'{m["skin_temp_c_start"]:.1f}')]
    fails = 0
    for what, c, r in checks:
        ok = c == r
        fails += not ok
        print(('PASS ' if ok else 'FAIL ') + f'{what}: card {c!r} record {r!r}')
    print(f'card {hub["sha"][:8]} vs the 262k/ record: {len(checks) - fails} PASS, {fails} FAIL'
          + (' (negative control: 3 expected)' if negative else ''))
    sys.exit(0 if (fails == 3 if negative else fails == 0) else 1)


if __name__ == '__main__':
    main()
