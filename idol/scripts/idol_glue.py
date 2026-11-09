"""Host-side steps of the Idol pipeline that are not tensor programs, as the phone runs them (the Kotlin code in the
sample mirrors these functions line by line). Each follows idol-sdk @4702a7f; the comments name the source.

- preprocess: idol/preprocess.py preprocess_image, with the alpha predictor passed in (BiRefNet or the matting graph)
- shared randomness: CONTRACT.md "Shared randomness"
- systematic_rows / octree_descend: idol/models/decoder.py sample + systematic_sample, with the per-row pool draws
- to_gaussians / write_ply: GaussianFixedlenDecoder.to_representation, idol/gaussian.py Gaussian + save_ply
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

CROP_MARGIN = {'synthetic': 1.2, 'real': 1.7}
IMAGE_SIZE = 1024
MAX_LEVEL = 8
PER_ANCHOR = 32
PHASE_THRESHOLD = 0.5


# ---------------------------------------------------------------- preprocessing

def has_real_alpha(image):
    return image.mode == 'RGBA' and np.array(image.getchannel(3), dtype=np.int32).min() < 255


def preprocess(image, alpha_fn, image_size=IMAGE_SIZE, crop_margin=1.2):
    """-> (composite RGB image the encoders see, uint8 alpha at the resized size before erosion, crop box).
    alpha_fn(rgb PIL image) -> uint8 [H, W] alpha."""
    width, height = image.size
    scale = image_size / min(width, height)
    image = image.resize((max(1, int(round(width * scale))), max(1, int(round(height * scale)))), Image.LANCZOS)
    if not has_real_alpha(image):
        rgb = image.convert('RGB')
        image = rgb.copy()
        image.putalpha(Image.fromarray(alpha_fn(rgb), mode='L'))
    alpha_raw = np.array(image.getchannel(3))
    image.putalpha(image.getchannel(3).filter(ImageFilter.MinFilter(3)))
    alpha = np.array(image.getchannel(3))
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        raise ValueError('no object found in the image (the alpha channel is empty)')
    left, top, right, bottom = xs.min(), ys.min(), xs.max(), ys.max()
    center_x, center_y = (left + right) / 2, (top + bottom) / 2
    half = max(right - left, bottom - top) / 2 * crop_margin
    box = [int(center_x - half), int(center_y - half), int(center_x + half), int(center_y + half)]
    image = image.crop(box)
    image = image.resize((image_size, image_size), Image.LANCZOS)
    composite = Image.new('RGB', (image_size, image_size), (0, 0, 0))
    composite.paste(image, mask=image.split()[3])
    return composite, alpha_raw, box


def birefnet_alpha_fn(birefnet):
    return lambda rgb: np.array(birefnet.remove_background(rgb).getchannel(3))


BIREFNET_SIZE = 1024
BIREFNET_MEAN = (0.485, 0.456, 0.406)
BIREFNET_STD = (0.229, 0.224, 0.225)


def birefnet_alpha(rgb, run_bir):
    """The author's BiRefNet.remove_background (idol/models/birefnet.py) around the graph: RGB / 255, bilinear
    (align_corners) to 1024 x 1024, ImageNet mean / std; the graph returns the sigmoid map [1, 1, 1024, 1024];
    bilinear (align_corners) back to the image size; x 255 truncated to uint8."""
    w, h = rgb.size
    t = torch.from_numpy(np.array(rgb.convert('RGB'), dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(BIREFNET_SIZE, BIREFNET_SIZE), mode='bilinear', align_corners=True)
    t = (t - torch.tensor(BIREFNET_MEAN).view(1, 3, 1, 1)) / torch.tensor(BIREFNET_STD).view(1, 3, 1, 1)
    a = torch.from_numpy(np.asarray(run_bir(t.numpy()), np.float32).reshape(1, 1, BIREFNET_SIZE, BIREFNET_SIZE))
    a = F.interpolate(a, size=(h, w), mode='bilinear', align_corners=True)[0, 0]
    return (a.clamp(0, 1) * 255).to(torch.uint8).numpy()


# uint8 DIS alpha values below this floor are set to 0 before the author's preprocessing (0 = off). DIS leaves faint
# responses BiRefNet does not (1-5 of 255 at the image corners, along thin wisps); the author's rule crops to the
# nonzero alpha, so they stretch the crop. The value is calibrated against BiRefNet's crop on held-out images
# (scripts/calibrate_matting_floor.py, contract/matting_floor.json).
DIS_ALPHA_FLOOR = 8   # calibrated 2026-11-09 on 31 held-out gallery images: crop box error mean 0.14 %, max 0.59 % of the side


def dis_alpha(rgb, run_dis, floor=None):
    """IS-Net DIS general-use as its authors run it (xuebinqin/DIS inference): bilinear resize to 1024², x/255 - 0.5,
    the sigmoid map back to the image size (bilinear), min-max normalised, x255 truncated to uint8.
    run_dis(x [1, 3, 1024, 1024] float32 numpy) -> [1, 1, 1024, 1024] sigmoid alpha."""
    w, h = rgb.size
    t = torch.from_numpy(np.array(rgb, dtype=np.float32)).permute(2, 0, 1)[None]
    t = F.interpolate(t, size=(1024, 1024), mode='bilinear', align_corners=False)
    x = (t / 255.0 - 0.5).numpy().astype(np.float32)
    a = torch.from_numpy(np.asarray(run_dis(x), dtype=np.float32).reshape(1, 1, 1024, 1024))
    a = F.interpolate(a, size=(h, w), mode='bilinear', align_corners=False)[0, 0]
    mi, ma = a.min(), a.max()
    a = (a - mi) / (ma - mi).clamp_min(1e-12)
    a = (a.clamp(0, 1) * 255).to(torch.uint8).numpy()
    floor = DIS_ALPHA_FLOOR if floor is None else floor
    if floor > 0:
        a = np.where(a >= floor, a, 0).astype(np.uint8)
    return a


def pixels_of(composite):
    """The encoders' input: [1, 3, 1024, 1024] float32 in [0, 1]."""
    return (np.asarray(composite, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0).astype(np.float32)


# ---------------------------------------------------------------- shared randomness

def make_noise(seed, q_tokens=8192, in_ch=16, cam_ch=5):
    g = torch.Generator().manual_seed(seed)
    vae = torch.randn(1, 32, 128, 128, generator=g)
    latent = torch.randn(1, q_tokens, in_ch, generator=g)
    camera = torch.randn(1, 1, cam_ch, generator=g)
    g1 = torch.Generator().manual_seed(seed + 1)
    pool = torch.rand(MAX_LEVEL, q_tokens, generator=g1)
    leaf = torch.rand(1, q_tokens, 3, generator=g1)
    return {k: v.numpy() for k, v in
            dict(vae_noise=vae, latent_noise=latent, camera_noise=camera, octree_pool=pool, leaf_pool=leaf).items()}


def time_grid(steps=5, rescale_t=3.0):
    t = np.linspace(1, 0, steps + 1)
    return rescale_t * t / (1 + (rescale_t - 1) * t)


# ---------------------------------------------------------------- octree sampling

def systematic_rows(probs, counts, u):
    """idol systematic_sample for one batch, row i's offset u0 = u[i] / n_i. probs [Np, P] float32 torch, counts [Np]
    int64 torch, u [Np] float32 torch -> [Np, P] int64."""
    P = probs.size(-1)
    probs = probs.to(torch.float32).clamp_min(0)
    row_sums = probs.sum(1, keepdim=True)
    zero_mask = row_sums.eq(0)
    probs = probs / row_sums.clamp_min(1)
    if zero_mask.any():
        probs = probs.clone()
        probs[zero_mask.expand_as(probs)] = 1.0 / P
    out = torch.zeros(probs.shape[0], P, dtype=torch.long)
    cdf = probs.cumsum(dim=1).clamp(max=1.0 - 1e-12)
    for n in torch.unique(counts).tolist():
        if n == 0:
            continue
        rows = (counts == n).nonzero(as_tuple=False).squeeze(1)
        u0 = u.index_select(0, rows)[:, None] / float(n)
        grid = torch.arange(n, dtype=torch.float32)[None, :] / float(n)
        us = (u0 + grid).clamp(max=1.0 - 1e-12)
        idx = torch.searchsorted(cdf.index_select(0, rows), us).clamp_max(P - 1)
        buf = torch.zeros(rows.numel(), P, dtype=torch.float32)
        buf.scatter_add_(1, idx, torch.ones_like(idx, dtype=buf.dtype))
        out.index_copy_(0, rows, buf.to(torch.long))
    return out


CHILD_OFFSET = torch.tensor([[i, j, k] for k in [0, 1] for j in [0, 1] for i in [0, 1]], dtype=torch.long)


def octree_descend(probs_fn, pool, leaf, num_points=8192, level=MAX_LEVEL, rec=None):
    """idol OctreeProbabilityFixedlenDecoder.sample for one batch. probs_fn(cells [1, L, 3] float32 torch, l) ->
    child probabilities [1, L, 8] (the softmax of the decoder's logits). pool [8, >= rows] and leaf [1, num_points, 3]
    float32 torch. -> anchors [1, num_points, 3] in [0, 1]^3."""
    prev = torch.zeros(1, 3, dtype=torch.long)
    counts = torch.full((1,), num_points, dtype=torch.long)
    for l in range(1, level + 1):
        res_p = 1 << (l - 1)
        cells = (prev.to(torch.float32) + 0.5) / res_p
        probs = probs_fn(cells[None], l)[0]
        sampled = systematic_rows(probs, counts, pool[l - 1, :prev.shape[0]]).flatten()
        child = (prev[:, None, :] * 2 + CHILD_OFFSET[None]).flatten(0, 1)
        mask = sampled > 0
        if rec is not None:
            rec[f'oct_cells_{l}'] = cells.numpy()
            rec[f'oct_probs_{l}'] = probs.float().numpy()
            rec[f'oct_counts_{l}'] = counts.numpy()
        prev, counts = child[mask], sampled[mask]
    coords = torch.repeat_interleave(prev, counts, dim=0)
    return ((coords.to(torch.float32) + leaf[0]) / (1 << level))[None]


# ---------------------------------------------------------------- Gaussians and PLY

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


LAYOUT = [('_xyz', 3), ('_features_dc', 3), ('_scaling', 3), ('_rotation', 4), ('_opacity', 1), ('_offset_scale', 1)]


def to_gaussians(points, features, rep, perturbation, base_offset_scale):
    """GaussianFixedlenDecoder.to_representation + Gaussian.__init__ for one batch, float32.
    points [1, L, 3], features [1, L, 480] torch -> dict of activated tensors (z-up): xyz [N, 3], features_dc [N, 1, 3],
    scaling [N, 3], rotation [N, 4] (w, x, y, z), opacity [N, 1]."""
    features = features.to(torch.float32)
    L = features.shape[1]
    sl, start = {}, 0
    for name, size in LAYOUT:
        sl[name] = features[0, :, start:start + PER_ANCHOR * size].reshape(L, PER_ANCHOR, size)
        start += PER_ANCHOR * size
    lr = rep['lr']
    offset_scale = F.softplus(sl['_offset_scale'] + base_offset_scale)
    offset = sl['_xyz'] * lr['_xyz'] + perturbation
    offset = torch.tanh(offset) * 0.5 * rep['perturbe_size'] * offset_scale
    raw_xyz = (offset + points[0, :, None, :]).flatten(0, 1)
    scaling_offset = inverse_softplus(torch.tensor(rep['scaling_bias']))
    opacity_offset = inverse_sigmoid(torch.tensor(rep['opacity_bias']))
    scaling = F.softplus(sl['_scaling'].flatten(0, 1) * lr['_scaling'] + scaling_offset)
    return {
        'xyz': raw_xyz * 1.0 - 0.5,
        'features_dc': (sl['_features_dc'].flatten(0, 1) * lr['_features_dc'])[:, None, :],
        'scaling': torch.sqrt(torch.square(scaling) + rep['3d_filter_kernel_size'] ** 2),
        'rotation': F.normalize(sl['_rotation'].flatten(0, 1) * lr['_rotation'] + torch.tensor([1.0, 0.0, 0.0, 0.0])),
        'opacity': torch.sigmoid(sl['_opacity'].flatten(0, 1) * lr['_opacity'] + opacity_offset),
    }


def quaternion_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=1)


PLY_NAMES = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
             'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']


def ply_rows(g):
    """idol Gaussian.save_ply's attribute rows [N, 17] float32 (z-up turned to y-up)."""
    xyz = g['xyz'].detach().cpu().numpy()
    f_dc = g['features_dc'].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    eps = 1e-6
    opacities = inverse_sigmoid(g['opacity'].clamp(eps, 1 - eps)).detach().cpu().numpy()
    scale = torch.log(g['scaling'].clamp_min(eps)).detach().cpu().numpy()
    rotation = g['rotation'].detach().cpu().numpy()
    xyz = np.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], axis=1)
    rotation = quaternion_multiply(np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]), rotation)
    return np.concatenate((xyz, np.zeros_like(xyz), f_dc, opacities, scale, rotation), axis=1).astype(np.float32)


def write_ply(rows, path):
    """Binary little-endian PLY with the 17 float properties of save_ply (the same bytes plyfile writes)."""
    header = ['ply', 'format binary_little_endian 1.0', f'element vertex {rows.shape[0]}']
    header += [f'property float {n}' for n in PLY_NAMES] + ['end_header']
    with open(path, 'wb') as f:
        f.write(('\n'.join(header) + '\n').encode('ascii'))
        f.write(np.ascontiguousarray(rows, dtype='<f4').tobytes())


def read_ply(path):
    with open(path, 'rb') as f:
        data = f.read()
    end = data.index(b'end_header\n') + len(b'end_header\n')
    head = data[:end].decode('ascii').splitlines()
    names = [l.split()[-1] for l in head if l.startswith('property')]
    n = int([l for l in head if l.startswith('element vertex')][0].split()[-1])
    return names, np.frombuffer(data[end:], dtype='<f4').reshape(n, len(names))
