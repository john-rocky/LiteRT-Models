"""CONTRACT.md lines (c) speed and (e) peak memory for a phone: the app's e2e rows (AutoRun row kind "e2e", one
<id>.<run>.json per run, run 0 = warm-up) and the hold's memory samples (device/results/<device>/<plan>.mem.csv).

  python score_phone_ce.py --run device/results/s26/e2e_armchair --id e2e_armchair \
      --mem device/results/s26/e2e_armchair.mem.csv --where "Galaxy S26 (SM-S942Q), the app, LiteRT 2.3.0" \
      --c contract/c_s26.json --e contract/e_s26.json

(c): median over the measured runs (1, 2, 3) of `generate` (the preprocessed 1024² image ready -> .ply written) and of
each stage; the graphs compiled inside each run are reported apart (count and seconds), as is the warm-up run.
(e): the largest VmHWM and GPU page count (kgsl page_alloc) any run recorded, and the hold sampler's maxima.
"""
import argparse
import csv
import glob
import json
import os
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='the pulled run dir of the e2e plan')
    ap.add_argument('--id', required=True)
    ap.add_argument('--mem', default=None, help='the hold sampler csv of the plan')
    ap.add_argument('--where', required=True)
    ap.add_argument('--c', required=True)
    ap.add_argument('--e', required=True)
    a = ap.parse_args()
    runs = []
    for p in sorted(glob.glob(os.path.join(a.run, f'{a.id}.*.json')), key=lambda s: int(s.rsplit('.', 2)[-2])):
        runs.append(json.load(open(p)))
    done = [r for r in runs if r.get('status') == 'done']
    measured = [r for r in done if not r.get('warmup')]
    first_only = not measured and len(done) == 1
    if first_only:
        measured = done   # a one-run plan (e2e_cool): the first photo after start is the measurement
    if not measured:
        raise SystemExit(f'no measured run done in {a.run}: statuses {[r.get("status") for r in runs]}')
    stage_keys = sorted({k for r in measured for k in r.get('stages', {})})

    def med(vals):
        return statistics.median(vals) if vals else None

    c = {
        'where': a.where, 'run_dir': a.run, 'runs_done': len(done), 'runs_measured': len(measured), 'first_only': first_only,
        'median_generate_s': med([r['generate'] for r in measured]),
        'generate_s': [r['generate'] for r in measured],
        'total_s': [r.get('total') for r in measured],
        'median_total_s': med([r['total'] for r in measured if r.get('total') is not None]),
        'median_stage_s': {k: med([r['stages'][k] for r in measured if k in r.get('stages', {})]) for k in stage_keys},
        'compiles_in_run': [{'run': r['run'], 'count': r['compiles_in_run']['count'],
                             'seconds': r['compiles_in_run']['seconds']} for r in done],
        'thermal': [{'run': r['run'], 'start': r.get('thermal_start'), 'end': r.get('thermal_end')} for r in done],
        'warmup': next(({'generate_s': r['generate'], 'total_s': r.get('total'), 'stages': r.get('stages')}
                        for r in done if r.get('warmup')), None),
        'runs_not_done': [{'run': r.get('run'), 'status': r.get('status')} for r in runs if r.get('status') != 'done'],
        'per_run': [{'run': r['run'], 'warmup': bool(r.get('warmup')), 'total_s': r.get('total'),
                     'generate_s': r['generate'], 'stages': r.get('stages'),
                     'compiles': {'count': r['compiles_in_run']['count'], 'seconds': r['compiles_in_run']['seconds']}}
                    for r in done],
        'gaussians': measured[0].get('gaussians'), 'handoff': measured[0].get('handoff'),
        'options': measured[0].get('options'),
    }
    e = {
        'where': a.where, 'run_dir': a.run,
        'vmhwm_kb_max': max(r.get('vmhwm_kb', -1) for r in done),
        'gpu_page_alloc_max_bytes': max(r.get('gpu_page_alloc_max_bytes', -1) for r in done),
        'per_run': [{'run': r['run'], 'vmhwm_kb': r.get('vmhwm_kb'), 'vmswap_kb': r.get('vmswap_kb'),
                     'gpu_page_alloc_max_bytes': r.get('gpu_page_alloc_max_bytes'),
                     'mem_available_kb_start': r.get('mem_available_kb_start'),
                     'mem_available_kb_end': r.get('mem_available_kb_end')} for r in done],
    }
    if a.mem and os.path.exists(a.mem):
        hwm, kgsl, avail = [], [], []
        for row in csv.DictReader(open(a.mem)):
            if not (row.get('app_pid') or '').strip():
                continue   # a sample taken before the app had a pid (its columns are not the app's)
            for key, out in (('vmhwm_kb', hwm), ('kgsl_page_alloc', kgsl), ('mem_available_kb', avail)):
                try:
                    out.append(int(row[key]))
                except (KeyError, TypeError, ValueError):
                    pass
        e['sampler'] = {'vmhwm_kb_max': max(hwm, default=None), 'kgsl_page_alloc_max_bytes': max(kgsl, default=None),
                        'mem_available_kb_min': min(avail, default=None), 'samples': len(avail)}
    head = os.path.join(a.run, 'header.json')
    if os.path.exists(head):
        h = json.load(open(head))
        pre = [g for rec in h.get('load', []) for g in rec.get('record', {}).get('graphs', []) if g.get('phase') == 'preload']
        c['preload'] = {'graphs': len(pre), 'seconds': sum(g['seconds'] for g in pre)}
        c['device'] = {k: h.get(k) for k in ('device', 'android_sdk', 'litert', 'mem_total_kb', 'thermal')}
    for path, obj in ((a.c, c), (a.e, e)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        json.dump(obj, open(path, 'w'), indent=1)
    print(f'(c) {a.where}: median generate {c["median_generate_s"]:.1f} s over {len(measured)} runs '
          f'{[round(v, 1) for v in c["generate_s"]]}; stages {json.dumps({k: round(v, 1) for k, v in c["median_stage_s"].items()})}; '
          f'compiles in run {[(x["count"], round(x["seconds"], 1)) for x in c["compiles_in_run"]]}')
    print(f'(e) VmHWM max {e["vmhwm_kb_max"] / 1024 / 1024:.2f} GiB, GPU pages max {e["gpu_page_alloc_max_bytes"] / 2**30:.2f} GiB'
          + (f'; sampler {json.dumps(e["sampler"])}' if 'sampler' in e else ''))


if __name__ == '__main__':
    main()
