"""CONTRACT.md lines (a) and (a') for a phone: the app's parity run (AutoRun row kind "parity": each part on the
fixture's reference input, outputs dumped as <id>.<name>.f32, the raw matting alpha as <id>.alpha.u8) against the
reference (ref/<tag>/ref.npz). Same rows, names and lines as parity_mac.py.

  python score_phone_a.py --ref ref/armchair --run device/results/s26/parity_armchair --id parity_armchair
      --where "Galaxy S26, LiteRT GPU fp16 with fp32 accumulation" --out contract/a_armchair_s26.json
"""
import argparse
import json
import os

import numpy as np

LINE_FP16 = 0.99
LINE_IOU = 0.95


def cos_rel(got, ref):
    a, b = np.asarray(got, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    return {'cos': float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
            'rel_l2': float(np.linalg.norm(a - b) / np.linalg.norm(b)), 'max_abs_diff': float(np.abs(a - b).max()),
            'max_abs_ref': float(np.abs(b).max()), 'finite': bool(np.isfinite(a).all())}


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--run', required=True, help='the pulled run dir of the parity plan')
    ap.add_argument('--id', required=True, help='the parity row id (file prefix)')
    ap.add_argument('--where', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ref = dict(np.load(os.path.join(a.ref, 'ref.npz')))
    rowj = json.load(open(os.path.join(a.run, f'{a.id}.json')))
    arrays = rowj['arrays']

    def phone(name):
        spec = arrays[name]
        return np.fromfile(os.path.join(a.run, spec['file']), '<f4').reshape(spec['shape'])

    res = {'where': a.where, 'ref': a.ref, 'run': a.run, 'app_row': {k: v for k, v in rowj.items() if k != 'arrays'}}

    def row(name, r, line=True):
        r['line'] = LINE_FP16 if line else None
        r['pass'] = (r['cos'] >= LINE_FP16 and r['finite']) if line else None
        res[name] = r
        tag = '' if not line else ('PASS' if r['pass'] else 'FAIL')
        print(f'{name:34s} cos {r["cos"]:.6f}  rel_l2 {r["rel_l2"]:.3e}  max|d| {r["max_abs_diff"]:.3e}  {tag}', flush=True)

    al = rowj.get('alpha')
    if al:
        got = np.fromfile(os.path.join(a.run, al['file']), np.uint8).reshape(al['height'], al['width'])
        want = ref['alpha_birefnet']
        ab, ad = want >= 128, got >= 128
        iou = float((ab & ad).sum() / max(1, (ab | ad).sum())) if got.shape == want.shape else 0.0
        box_ref = [int(v) for v in ref['box']]
        box = [int(v) for v in al['box']]
        res['matting'] = {'iou': iou, 'line': LINE_IOU, 'pass': iou >= LINE_IOU, 'box_birefnet': box_ref, 'box_dis': box,
                          'box_corner_diff_px': [x - y for x, y in zip(box, box_ref)],
                          'alpha_nonzero_birefnet': int((want > 0).sum()), 'alpha_nonzero_dis': int((got > 0).sum()),
                          'app_matting': rowj.get('matting'), 'alpha_pixels': int(want.size),
                          'alpha_uint8_pixels_differing': int((got != want).sum()) if got.shape == want.shape else None,
                          'alpha_uint8_max_abs_diff': int(np.abs(got.astype(int) - want.astype(int)).max())
                          if got.shape == want.shape else None}
        print(f'matting IoU {iou:.4f} ({"PASS" if iou >= LINE_IOU else "FAIL"}), boxes {box_ref} vs {box}', flush=True)
    row('dinov3 feature1', cos_rel(phone('feature1'), ref['feature1']))
    row('vae moments (metered)', cos_rel(phone('vae_moments'), ref['vae_moments']), line=False)
    row('vae latent tokens', cos_rel(phone('vae_tokens'), ref['feature2'][:, 5:]))
    tg = ref['t_grid']
    for i in range(len(tg) - 1):
        row(f'denoiser step {i} latent velocity', cos_rel(phone(f'v_latent_{i}'), ref[f'v_latent_{i}']))
        row(f'denoiser step {i} camera (metered)', cos_rel(phone(f'v_camera_{i}'), ref[f'v_camera_{i}']), line=False)
    level = 1
    while f'oct_logits_{level}' in arrays:
        row(f'octree level {level} probs', cos_rel(softmax(phone(f'oct_logits_{level}').astype(np.float64)),
                                                    ref[f'oct_probs_{level}']))
        level += 1
    row('gaussian features (metered)', cos_rel(phone('gs_features'), ref['gs_features']), line=False)
    for k in ('xyz', 'features_dc', 'scaling', 'rotation', 'opacity'):
        row(f'gaussians {k}', cos_rel(phone(f'g_{k}'), ref[f'g_{k}']))
    lines = [v for v in res.values() if isinstance(v, dict) and v.get('pass') is not None]
    res['all_pass'] = all(v['pass'] for v in lines)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(f'ALL {"PASS" if res["all_pass"] else "FAIL"} ({len(lines)} lines incl. matting)', flush=True)


if __name__ == '__main__':
    main()
