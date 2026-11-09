"""CONTRACT.md lines (a) and (a') on the Mac: the shipped graphs on LiteRT GPU (Metal, ML Drift fp16) vs the reference.

(a)  each part gets the reference's own input for it and is compared with the reference's output (no chaining).
(a') the matting graph's uint8 alpha vs BiRefNet's, thresholded at >= 128: IoU; the crop squares of both.

  python parity_mac.py --models models/v1 --ref ref/armchair --image IMG --mode synthetic --out contract/a_armchair.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import idol_glue as GL  # noqa: E402
import idol_graphs as IG  # noqa: E402
from idol_runner import IdolLiteRT  # noqa: E402

LINE_FP16 = 0.99


def cos_rel(got, ref):
    a, b = np.asarray(got, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    return {'cos': float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
            'rel_l2': float(np.linalg.norm(a - b) / np.linalg.norm(b)), 'max_abs_diff': float(np.abs(a - b).max()),
            'max_abs_ref': float(np.abs(b).max()), 'finite': bool(np.isfinite(a).all())}


def row(res, name, r, line=True):
    r['line'] = LINE_FP16 if line else None
    r['pass'] = (r['cos'] >= LINE_FP16 and r['finite']) if line else None
    res[name] = r
    tag = '' if not line else ('PASS' if r['pass'] else 'FAIL')
    print(f'{name:34s} cos {r["cos"]:.6f}  rel_l2 {r["rel_l2"]:.3e}  max|d| {r["max_abs_diff"]:.3e}  {tag}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--mode', required=True)
    ap.add_argument('--matting', default='birefnet', choices=('birefnet', 'dis'))
    ap.add_argument('--dis', default=None, help='dis.tflite (with --matting dis)')
    ap.add_argument('--accel', default='gpu')
    ap.add_argument('--parts', default='mat,dino,vae,den,oct,gs')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    ref = dict(np.load(os.path.join(a.ref, 'ref.npz')))
    noise = dict(np.load(os.path.join(a.ref, 'noise.npz')))
    p = IdolLiteRT(a.models, a.accel, dis_path=a.dis, matting=a.matting)
    res = {'models': a.models, 'ref': a.ref, 'accel': a.accel, 'precision': 'metal default (fp16)'}
    parts = a.parts.split(',')
    t0 = time.time()
    if 'mat' in parts:
        raw = Image.open(a.image)
        _, alpha_dis, box_dis = GL.preprocess(raw, p.matting_alpha, GL.IMAGE_SIZE, GL.CROP_MARGIN[a.mode])
        ab, ad = ref['alpha_birefnet'] >= 128, alpha_dis >= 128
        iou = float((ab & ad).sum() / max(1, (ab | ad).sum()))
        box_ref = [int(v) for v in ref['box']]
        res['matting'] = {'iou': iou, 'line': 0.95, 'pass': iou >= 0.95, 'box_birefnet': box_ref, 'box_dis': box_dis,
                          'box_corner_diff_px': [int(x - y) for x, y in zip(box_dis, box_ref)],
                          'alpha_nonzero_birefnet': int((ref['alpha_birefnet'] > 0).sum()),
                          'alpha_nonzero_dis': int((alpha_dis > 0).sum()), 'app_matting': a.matting,
                          'alpha_uint8_pixels_differing': int((alpha_dis != ref['alpha_birefnet']).sum()),
                          'alpha_pixels': int(ref['alpha_birefnet'].size),
                          'alpha_uint8_max_abs_diff': int(np.abs(alpha_dis.astype(int) - ref['alpha_birefnet'].astype(int)).max())}
        print(f'matting IoU {iou:.4f} ({"PASS" if iou >= 0.95 else "FAIL"}), boxes {box_ref} vs {box_dis}', flush=True)
        p.release('dis', 'bir')
    if 'dino' in parts:
        row(res, 'dinov3 feature1', cos_rel(p.dinov3(ref['pixels']), ref['feature1']))
        p.release('dino_')
    if 'vae' in parts:
        mom = p.vae_moments(ref['pixels'])
        row(res, 'vae moments (metered)', cos_rel(mom, ref['vae_moments']), line=False)
        row(res, 'vae latent tokens', cos_rel(p.vae_tokens(mom, noise['vae_noise']), ref['feature2'][:, 5:]))
        p.release('vae_')
    if 'den' in parts:
        cache = {}
        tg = ref['t_grid']
        for i in range(len(tg) - 1):
            v, vc = p.velocity(ref[f'x_latent_{i}'], ref[f'x_camera_{i}'], float(tg[i]), ref['feature1'],
                               ref['feature2'], cache)
            row(res, f'denoiser step {i} latent velocity', cos_rel(v, ref[f'v_latent_{i}']))
            row(res, f'denoiser step {i} camera (metered)', cos_rel(vc, ref[f'v_camera_{i}']), line=False)
        p.release('den_')
    if 'oct' in parts:
        for l in range(1, GL.MAX_LEVEL + 1):
            logits = p.octree_logits(ref[f'oct_cells_{l}'], l, ref['latent'])
            probs = torch.softmax(torch.from_numpy(logits).float(), -1).numpy()
            row(res, f'octree level {l} probs', cos_rel(probs, ref[f'oct_probs_{l}']))
        p.release('oct')
    if 'gs' in parts:
        feats, g = p.gaussians(ref['anchors'], ref['latent'])
        row(res, 'gaussian features (metered)', cos_rel(feats, ref['gs_features']), line=False)
        for k in ('xyz', 'features_dc', 'scaling', 'rotation', 'opacity'):
            row(res, f'gaussians {k}', cos_rel(g[k].numpy(), ref[f'g_{k}']))
    lines = [v for v in res.values() if isinstance(v, dict) and v.get('pass') is not None]
    res['all_pass'] = all(v['pass'] for v in lines)
    res['wall_s'] = round(time.time() - t0, 1)
    res['stage_s'] = {k: round(v, 2) for k, v in p.times.items()}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(f'ALL {"PASS" if res["all_pass"] else "FAIL"} ({len(lines)} lines) in {res["wall_s"]} s', flush=True)


if __name__ == '__main__':
    main()
