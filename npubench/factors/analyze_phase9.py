"""Parse phase9.log into the analysis tables for the report.

  python analyze_phase9.py <probe_dir>/phase9.log
"""
import re
import sys

# pf probe architecture (probe_batch4.py): L=2, D=1024, DFF=2816, NH=16, NKV=4, HD=64
def pf_gflops(t):
    linear = 2 * (2 * 1024 * 1024 + 2 * 2 * 1024 * 256 + 2 * 1024 * 1024 + 3 * 2 * 1024 * 2816)
    attn = 4 * 16 * 64 * t * t
    return 2 * (linear / 2 * t + attn) / 1e9  # careful: linear above is per-token for ONE block *2 blocks

def pf_gf(t):
    per_block_per_tok = (2 * 1024 * 1024) + 2 * (2 * 1024 * 256) + (2 * 1024 * 1024) + 3 * (2 * 1024 * 2816)
    per_block_attn = 4 * 16 * 64 * t * t
    return 2 * (per_block_per_tok * t + per_block_attn) / 1e9


def main(path):
    rows = {}
    for line in open(path):
        m = re.match(r'^(c|m)_(\S+?)_(npu|gpu)\|gate_status=(\S+)\|(.*)$', line.strip())
        if not m:
            continue
        phase, arm, accel, gate, rest = m.groups()
        r = {'gate': gate, 'raw': rest}
        mm = re.search(r'median=([\d.]+)ms min=([\d.]+)ms max=([\d.]+)ms load=([\d.]+)ms runs=(\d+) thermal=(\S+)', rest)
        if mm:
            r.update(median=float(mm.group(1)), mn=float(mm.group(2)), mx=float(mm.group(3)),
                     load=float(mm.group(4)), runs=int(mm.group(5)), thermal=mm.group(6))
        elif 'FAILED' in rest:
            r['failed'] = rest
        elif rest.strip() == '':
            r['empty'] = True
        rows[f'{phase}_{arm}_{accel}'] = r

    def med(arm, accel='npu', phase='m'):
        r = rows.get(f'{phase}_{arm}_{accel}', {})
        return r.get('median')

    print('== prefill T sweep (NPU, measure pass) ==')
    print(f'{"T":>6} {"ms":>9} {"GF":>7} {"GF/ms":>7} {"compile_load_ms":>15}')
    for t in [1151, 1152, 1153, 1279, 1280, 1281, 1407, 1408, 1409, 1500, 1535, 1536, 1537, 1663, 1664, 1665]:
        ms = med(f'pf{t}')
        c = rows.get(f'c_pf{t}_npu', {})
        if ms:
            gf = pf_gf(t)
            print(f'{t:>6} {ms:>9.2f} {gf:>7.1f} {gf/ms:>7.2f} {c.get("load", float("nan")):>15.0f}')
        else:
            print(f'{t:>6} {"FAIL/empty":>9} {rows.get(f"m_pf{t}_npu",{}).get("raw","-")[:60]}')

    base = med('pf1536')
    print('\n== RMSNorm flavors @T=1536 (NPU) ==')
    for arm, label in [('pf1536', 'naive (baseline)'), ('pfrmssafe1536', 'SafeRMS s=64'),
                       ('pfrmsmax1536', 'max-norm SafeRMS')]:
        ms = med(arm)
        if ms and base:
            print(f'{label:<20} {ms:>8.2f} ms  x{ms/base:.2f} vs naive')
        else:
            print(f'{label:<20} {rows.get(f"m_{arm}_npu",{}).get("raw","missing")[:70]}')

    print('\n== fusion-breaker fx arms (NPU, T=1024 d=384 probe) ==')
    fb = med('fx_relu')
    for arm in ['fx_relu', 'fx_sig', 'fx_abs', 'fx_max', 'fx_exp', 'fx_sqrt', 'fx_rsqrt', 'fx_pow', 'fx_erf', 'fx_tanh']:
        ms = med(arm)
        if ms:
            rel = f'x{ms/fb:.2f} vs fx_relu' if fb else ''
            print(f'{arm:<10} {ms:>8.2f} ms  {rel}')
        else:
            print(f'{arm:<10} {rows.get(f"m_{arm}_npu",{}).get("raw","missing")[:70]}')

    print('\n== granularity curve (NPU) ==')
    prev = None
    for arm, n in [('gr10', 10), ('gr30', 30), ('gr100', 100), ('gr300', 300), ('gr1000', 1000), ('gr3000', 3000)]:
        ms = med(arm)
        if ms:
            gf = n * 2 * 32 * 32 * 1024 / 1e9
            print(f'N={n:<5} {ms:>9.2f} ms  {gf:.2f} GF  {gf/ms:.3f} GF/ms  us/op={1000*ms/n:.1f}')
        else:
            print(f'N={n:<5} {rows.get(f"m_{arm}_npu",{}).get("raw","missing")[:70]}')
    ms = med('grmono')
    if ms:
        gf = 2 * 1792 * 1792 * 1024 / 1e9
        print(f'mono  {ms:>9.2f} ms  {gf:.2f} GF  {gf/ms:.3f} GF/ms (single conv)')

    print('\n== GEMV dtype (NPU, T=1) ==')
    for arm, mb in [('gemv_fp32', 537), ('gemv_fp16', 268), ('gemv_int8', 134), ('gemv_int4', 67)]:
        ms = med(arm)
        if ms:
            print(f'{arm:<10} {ms:>8.2f} ms  weights {mb} MB  -> {mb/ms:.1f} GB/s effective')
        else:
            print(f'{arm:<10} {rows.get(f"m_{arm}_npu",{}).get("raw","missing")[:70]}')

    print('\n== whisper pad flip ==')
    for arm in ['wh_ctrl', 'wh_pad1536']:
        for accel in ['npu', 'gpu']:
            ms = med(arm, accel)
            print(f'{arm:<11} {accel} {ms if ms else rows.get(f"m_{arm}_{accel}",{}).get("raw","missing")}')

    print('\n== memorize bisect (NPU) ==')
    for arm in ['mem_k100', 'mem_k140', 'mem_k186', 'mem_k240', 'mem_k310', 'mem_k380', 'mem_k440', 'mem_k530', 'mem_full']:
        ms = med(arm)
        print(f'{arm:<10} {ms if ms is not None else rows.get(f"m_{arm}_npu",{}).get("raw","missing")}')

    print('\n== GPU reference arms ==')
    for key, r in rows.items():
        if key.startswith('m_') and key.endswith('_gpu'):
            print(f'{key:<22} {r.get("median", r.get("raw","?"))}')

    n_empty = sum(1 for r in rows.values() if r.get('empty'))
    n_fail = sum(1 for r in rows.values() if r.get('failed'))
    print(f'\n{len(rows)} rows parsed, {n_fail} FAILED, {n_empty} empty')


if __name__ == '__main__':
    main(sys.argv[1])
