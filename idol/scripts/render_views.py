"""CONTRACT.md line (b) renderer: a 3DGS .ply -> 4 fixed views, 512², on the Mac CPU.

Projection, covariance and tile intersection are gsplat 1.5.3's pure-PyTorch functions (nerfstudio-project/gsplat,
Apache-2.0, `gsplat/cuda/_torch_impl.py`, loaded from the installed wheel, unmodified). Compositing follows gsplat's
`accumulate`: alpha = min(0.999, opacity * exp(-sigma)), weight = alpha * prod(1 - alpha in front), over every
Gaussian whose 3.33-sigma box touches the 16x16 tile, front to back by camera depth.

  python render_views.py --ply A.ply [--ply B.ply] --out DIR [--psnr]   (with two .ply: PSNR of B vs A + side-by-side)
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import idol_glue as GL  # noqa: E402

_p = glob.glob(os.path.join(os.path.dirname(torch.__file__), '..', 'gsplat', 'cuda', '_torch_impl.py'))[0]
_spec = importlib.util.spec_from_file_location('gsplat_torch_impl', _p)
GS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GS)

C0 = 0.28209479177387814
SIZE, TILE = 512, 16
AZIMUTHS = (0.0, 90.0, 180.0, 270.0)
ELEVATION, DISTANCE, FOV_Y = 20.0, 2.2, 40.0


def cameras():
    viewmats, Ks = [], []
    # idol's PLY is y-down, as 3DGS viewers read it (the object's top is at -y): image up = -y, cameras above at -y
    up = np.array([0.0, -1.0, 0.0])
    f = (SIZE / 2) / math.tan(math.radians(FOV_Y) / 2)
    for az in AZIMUTHS:
        a, e = math.radians(az), math.radians(ELEVATION)
        c = DISTANCE * np.array([math.cos(e) * math.sin(a), -math.sin(e), math.cos(e) * math.cos(a)])
        z = -c / np.linalg.norm(c)                     # forward (OpenCV camera: x right, y down, z forward)
        x = np.cross(z, up); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        m = np.eye(4); m[:3, :3] = R; m[:3, 3] = -R @ c
        viewmats.append(m)
        Ks.append(np.array([[f, 0, SIZE / 2], [0, f, SIZE / 2], [0, 0, 1.0]]))
    return torch.tensor(np.stack(viewmats), dtype=torch.float32), torch.tensor(np.stack(Ks), dtype=torch.float32)


def load(path):
    names, rows = GL.read_ply(path)
    col = {n: i for i, n in enumerate(names)}
    t = torch.from_numpy(np.ascontiguousarray(rows))
    means = t[:, [col['x'], col['y'], col['z']]]
    quats = torch.nn.functional.normalize(t[:, [col['rot_0'], col['rot_1'], col['rot_2'], col['rot_3']]], dim=-1)
    scales = torch.exp(t[:, [col['scale_0'], col['scale_1'], col['scale_2']]])
    opac = torch.sigmoid(t[:, col['opacity']])
    colors = (0.5 + C0 * t[:, [col['f_dc_0'], col['f_dc_1'], col['f_dc_2']]]).clamp_min(0.0)
    return means, quats, scales, opac, colors


@torch.no_grad()
def render(path):
    means, quats, scales, opac, colors = load(path)
    viewmats, Ks = cameras()
    covars, _ = GS._quat_scale_to_covar_preci(quats, scales, compute_covar=True, compute_preci=False)
    radii, means2d, depths, conics, _ = GS._fully_fused_projection(means, covars, viewmats, Ks, SIZE, SIZE,
                                                                   eps2d=0.3, near_plane=0.01, far_plane=1e10)
    tw = th = SIZE // TILE
    _, isect_ids, flatten_ids = GS._isect_tiles(means2d, radii, depths, TILE, tw, th, sort=True)
    offsets = GS._isect_offset_encode(isect_ids, len(AZIMUTHS), tw, th).flatten().tolist() + [len(flatten_ids)]
    N = means.shape[0]
    flat = flatten_ids.long()
    out = torch.zeros(len(AZIMUTHS), SIZE, SIZE, 3)
    py, px = torch.meshgrid(torch.arange(TILE, dtype=torch.float32), torch.arange(TILE, dtype=torch.float32),
                            indexing='ij')
    for k in range(len(offsets) - 1):
        s, e = offsets[k], offsets[k + 1]
        if e <= s:
            continue
        cam, rem = divmod(k, tw * th)
        ty, tx = divmod(rem, tw)
        gid = flat[s:e] - cam * N
        m2 = means2d[cam, gid]
        cn = conics[cam, gid]
        dx = (px.flatten() + tx * TILE + 0.5)[None, :] - m2[:, 0:1]
        dy = (py.flatten() + ty * TILE + 0.5)[None, :] - m2[:, 1:2]
        sigma = 0.5 * (cn[:, 0:1] * dx * dx + cn[:, 2:3] * dy * dy) + cn[:, 1:2] * dx * dy
        alpha = torch.clamp_max(opac[gid][:, None] * torch.exp(-sigma), 0.999)
        trans = torch.cumprod(torch.cat([torch.ones(1, alpha.shape[1]), 1 - alpha[:-1]], dim=0), dim=0)
        w = alpha * trans
        rgb = (w.t() @ colors[gid]).reshape(TILE, TILE, 3)
        out[cam, ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE] = rgb
    return out.clamp(0, 1)


def psnr(a, b):
    mse = float(((a.double() - b.double()) ** 2).mean())
    return 10 * math.log10(1.0 / max(mse, 1e-12)), mse


def to_u8(img):
    return (img.numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', action='append', required=True, help='first = reference (author), second = ours')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    views = [render(p) for p in a.ply]
    for i, v in enumerate(views):
        for j, az in enumerate(AZIMUTHS):
            Image.fromarray(to_u8(v[j])).save(os.path.join(a.out, f'view{i}_az{int(az):03d}.png'))
    res = {'ply': a.ply, 'cameras': {'azimuth_deg': AZIMUTHS, 'elevation_deg': ELEVATION, 'distance': DISTANCE,
                                     'fov_y_deg': FOV_Y, 'size': SIZE}, 'renderer': 'gsplat 1.5.3 _torch_impl'}
    if len(views) == 2:
        p_all, mse = psnr(views[0], views[1])
        res.update(psnr_db=p_all, mse=mse, psnr_per_view_db=[psnr(views[0][j], views[1][j])[0]
                                                            for j in range(len(AZIMUTHS))])
        grid = np.concatenate([np.concatenate([to_u8(v[j]) for j in range(len(AZIMUTHS))], axis=1) for v in views])
        Image.fromarray(grid).save(os.path.join(a.out, f'b_{a.tag or "pair"}.png'))
    json.dump(res, open(os.path.join(a.out, 'render.json'), 'w'), indent=1)
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
