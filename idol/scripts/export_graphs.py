"""Export the shipped Idol graph set (idol_graphs.py) with litert-torch, after checking each part's GPU form in
PyTorch float32 against the reference run's own tensors (ref/<input>/ref.npz from ref_pipeline.py).

  python export_graphs.py --parts den,oct,gs,dino,vae --out models/v1 --ref ref/armchair [--no-export]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser(os.environ.get('IDOL_SDK', 'idol-sdk')))  # github.com/speridlabs/idol-sdk @4702a7f
import f0_export as F0  # noqa: E402
import idol_graphs as IG  # noqa: E402
import idol_glue as GL  # noqa: E402


def cos_rel(got, ref):
    a, b = np.asarray(got, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    return {'cos': float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
            'rel_l2': float(np.linalg.norm(a - b) / np.linalg.norm(b)), 'max_abs_diff': float(np.abs(a - b).max()),
            'max_abs_ref': float(np.abs(b).max())}


def show(name, r):
    print(f'[check] {name}: cos {r["cos"]:.9f} rel_l2 {r["rel_l2"]:.3e} max|d| {r["max_abs_diff"]:.3e} '
          f'max|ref| {r["max_abs_ref"]:.3e}', flush=True)
    return r


def t32(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def export(module, args, path):
    t0 = time.time()
    edge = __import__('litert_torch').convert(module.eval(), args)
    edge.export(path)
    info = F0.scan_tflite(path)
    info['export_s'] = round(time.time() - t0, 1)
    print(f'[export] {os.path.basename(path)}: {info["bytes"]:,} B, {info["ops"]} ops, rank<= {info["max_rank"]}, '
          f'BMM ranks {info["bmm_ranks"]}, {info["export_s"]} s', flush=True)
    return info


def export_multi(named, path):
    import litert_torch
    t0 = time.time()
    conv = None
    for name, module, args in named:
        conv = litert_torch.signature(name, module.eval(), args) if conv is None else conv.signature(name, module, args)
    conv.convert().export(path)
    info = F0.scan_tflite(path)
    info['export_s'] = round(time.time() - t0, 1)
    print(f'[export] {os.path.basename(path)}: {info["bytes"]:,} B, {info["ops"]} ops ({len(named)} signatures), '
          f'{info["export_s"]} s', flush=True)
    return info


# ---------------------------------------------------------------- denoiser

def part_den(out, ref, do_export, chunk=512):
    den, _ = F0.load_denoiser()
    den = den.float()
    lora = load_file(F0.snap('speridlabs/idol-turbo', 'idol_turbo_lora.safetensors'))
    ctx, dinp = IG.DenCtx(den, lora, chunk).eval(), IG.DenIn(den, lora, chunk).eval()
    joints = [IG.DenJoint(den, k, lora, chunk).eval() for k in range(len(den.blocks))]
    pos_emb = den.pos_embedder(den.pos_pe).float()
    pos_emb.numpy().astype(np.float32).tofile(os.path.join(out, 'den_pos_emb.f32'))
    res = {}
    f1, f2 = t32(ref['feature1']), t32(ref['feature2'])
    tg = ref['t_grid']
    for i in (0, len(tg) - 2):
        t = float(tg[i])
        gate = IG.lora_gate('slow' if t >= GL.PHASE_THRESHOLD else 'fast')
        h_cond = ctx(f1, f2, gate)
        h_x, h_cam, t_mod, t_emb = dinp(t32(ref[f'x_latent_{i}']), t32(ref[f'x_camera_{i}']),
                                        IG.sinusoid(1000 * t), pos_emb, gate)
        h = joints[0](h_x, h_cond, h_cam, t_mod, gate)
        for k in range(1, len(joints) - 1):
            h = joints[k](h, t_mod, gate)
        v, vc = joints[-1](h, t_mod, t_emb, gate)
        res[f'step{i}_latent'] = show(f'denoiser step {i} (t={t:.4f}) latent velocity', cos_rel(v, ref[f'v_latent_{i}']))
        res[f'step{i}_camera'] = show(f'denoiser step {i} camera velocity', cos_rel(vc, ref[f'v_camera_{i}']))
    if do_export:
        g = IG.lora_gate('slow')
        L = IG.DEN_TOKENS + IG.COND_TOKENS + 1
        res['export'] = {
            'den_ctx': export(ctx, (f1, f2, g), os.path.join(out, 'den_ctx.tflite')),
            'den_in': export(dinp, (torch.randn(1, 8192, 16), torch.randn(1, 1, 5), IG.sinusoid(700.0), pos_emb, g),
                             os.path.join(out, 'den_in.tflite')),
        }
        t_mod = torch.randn(1, 6144)
        for k, jb in enumerate(joints):
            if k == 0:
                args = (torch.randn(1, 8192, 1024), torch.randn(1, 4101, 1024), torch.randn(1, 1, 1024), t_mod, g)
            elif k == len(joints) - 1:
                args = (torch.randn(1, L, 1024), t_mod, torch.randn(1, 1024), g)
            else:
                args = (torch.randn(1, L, 1024), t_mod, g)
            res['export'][f'den_joint_{k:02d}'] = export(jb, args, os.path.join(out, f'den_joint_{k:02d}.tflite'))
    return res


# ---------------------------------------------------------------- octree

def load_decoder(name):
    from idol.models.decoder import GaussianFixedlenDecoder, OctreeProbabilityFixedlenDecoder
    cfg = json.load(open(F0.snap('speridlabs/idol', 'config.json')))
    cls = OctreeProbabilityFixedlenDecoder if name == 'decoder' else GaussianFixedlenDecoder
    m = cls(**cfg[name]['args'])
    m.load_state_dict(load_file(F0.snap('speridlabs/idol', f'{name}.safetensors')), strict=True)
    m = m.float().eval()
    m.dtype = torch.float32
    return m


def bucket(n):
    for b in IG.OCT_BUCKETS:
        if n <= b:
            return b
    raise ValueError(n)


def oct_inputs(cells, level):
    """cells [1, n, 3] -> padded graph inputs at the bucket size."""
    n = cells.shape[1]
    b = bucket(n)
    c = torch.zeros(1, b, 3)
    c[:, :n] = cells
    return (c, IG.point_pe(c[0])[None], IG.sinusoid(float(1 << level), max_period=1024.0, two_pi=True)), b


def part_oct(out, ref, do_export, chunk=512):
    dec = load_decoder('decoder')
    g = IG.Octree(dec, chunk).eval()
    latent = t32(ref['latent'])
    res = {}
    for l in range(1, GL.MAX_LEVEL + 1):
        cells = t32(ref[f'oct_cells_{l}'])
        (c, pe, lf), b = oct_inputs(cells, l)
        logits = g(c, pe, lf, latent)[:, :cells.shape[1]]
        res[f'level{l}'] = show(f'octree level {l} ({cells.shape[1]} rows, bucket {b}) probs',
                                cos_rel(torch.softmax(logits, -1), ref[f'oct_probs_{l}']))
    if do_export:
        named = [(f'L{b}', g, (torch.rand(1, b, 3), torch.randn(1, b, 1024), IG.sinusoid(64.0, max_period=1024.0,
                  two_pi=True), latent)) for b in IG.OCT_BUCKETS]
        res['export'] = {'oct': export_multi(named, os.path.join(out, 'oct.tflite'))}
    return res


# ---------------------------------------------------------------- Gaussian decoder

GS_SPLIT = [(2 * i, 2 * i + 2) for i in range(8)]


def part_gs(out, ref, do_export, chunk=512):
    dgs = load_decoder('decoder_gs')
    parts = [IG.Gauss(dgs, lo, hi, chunk).eval() for lo, hi in GS_SPLIT]
    anchors, latent = t32(ref['anchors']), t32(ref['latent'])
    pe = IG.point_pe(anchors[0])[None]
    h = parts[0](anchors, pe, latent)
    for p in parts[1:]:
        h = p(h, latent)
    res = {'features': show('gaussian decoder features', cos_rel(h, ref['gs_features']))}
    gg = GL.to_gaussians(anchors, h, dgs.rep_config, dgs.points_offset_perturbation.float(),
                         dgs.base_offset_scale.float())
    for k in ('xyz', 'features_dc', 'scaling', 'rotation', 'opacity'):
        res[k] = show(f'gaussians {k}', cos_rel(gg[k], ref[f'g_{k}']))
    np.save(os.path.join(out, 'gs_perturbation.npy'), dgs.points_offset_perturbation.float().numpy())
    json.dump({'base_offset_scale': float(dgs.base_offset_scale), 'rep': dgs.rep_config},
              open(os.path.join(out, 'gs_rep.json'), 'w'), indent=1)
    if do_export:
        res['export'] = {}
        for i, p in enumerate(parts):
            args = (anchors, pe, latent) if i == 0 else (torch.randn(1, 8192, 1024), latent)
            res['export'][f'gs_{i}'] = export(p, args, os.path.join(out, f'gs_{i}.tflite'))
    return res


# ---------------------------------------------------------------- DINOv3

DINO_SPLIT = [(8 * i, 8 * i + 8) for i in range(4)]


def load_dino():
    from transformers import DINOv3ViTModel
    cfg = F0.dino_config()
    m = DINOv3ViTModel(cfg)
    sd = load_file(F0.snap('Comfy-Org/Idol', 'clip_vision/dino_v3_vit_h_bf16.safetensors'))
    sd = {('model.' + k if k.startswith('layer.') else k): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m.float().eval(), cfg


def part_dino(out, ref, do_export, chunk=1024):
    model, cfg = load_dino()
    _, _, cos, sin = F0.dino_rope(cfg, 64)
    cos.numpy().astype(np.float32).tofile(os.path.join(out, 'dino_rope_cos.f32'))
    sin.numpy().astype(np.float32).tofile(os.path.join(out, 'dino_rope_sin.f32'))
    parts = [IG.Dino(model, lo, hi, chunk).eval() for lo, hi in DINO_SPLIT]
    pixels = t32(ref['pixels'])
    h = pixels
    for p in parts:
        h = p(h, cos, sin)
    res = {'feature1': show('dinov3 feature1', cos_rel(h, ref['feature1']))}
    if do_export:
        res['export'] = {}
        for i, p in enumerate(parts):
            args = (pixels if i == 0 else torch.randn(1, 4101, 1280), cos, sin)
            res['export'][f'dino_{i}'] = export(p, args, os.path.join(out, f'dino_{i}.tflite'))
    return res


# ---------------------------------------------------------------- BiRefNet (the author's background remover)

class BirefnetGraph(torch.nn.Module):
    """BiRefNet.forward: normalised image [1, 3, 1024, 1024] -> sigmoid map [1, 1, 1024, 1024]."""

    def __init__(self, bir):
        super().__init__()
        self.m = bir

    def forward(self, x):
        return self.m.forward(x)


def part_bir(out, ref, do_export):
    """The author's BiRefNet with its 20 deformable convolutions in gather form (deform_conv.py) and the memory-lean
    rewrites of bir_lean.py, one graph for the CPU (the gather is GATHER_ND, which the LiteRT GPU does not run).
    Checked on the reference's own input image:
    the gather form vs the author's forward, then the uint8 alpha after the glue's host steps vs the reference's."""
    from PIL import Image
    import idol.models.birefnet as BR
    import bir_lean
    root = os.path.dirname(F0.snap('speridlabs/idol', 'config.json'))
    bir = BR.BiRefNet()
    bir.load_state_dict(load_file(os.path.join(root, 'birefnet.safetensors')), strict=True)
    bir = bir.eval().float()
    meta = json.load(open(os.path.join(os.path.dirname(os.path.abspath(ref['__path__'])), 'meta.json')))
    raw = Image.open(meta['image'])
    w, h = raw.size
    scale = GL.IMAGE_SIZE / min(w, h)
    rgb = raw.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.LANCZOS).convert('RGB')
    captured = {}

    def run_author(x):
        captured['x'] = torch.from_numpy(x)
        captured['map'] = bir.forward(captured['x'])
        return captured['map'].numpy()

    a_author = GL.birefnet_alpha(rgb, run_author)
    # the memory-lean form (bir_lean.py): row-tiled gathers, no ASPP concatenation, the last 1x1 conv folded
    # through the resize; the first form (one-tile gathers, the author's concatenations) peaked at 9.3 GB per call
    restore = bir_lean.apply(BR)
    try:
        graph = BirefnetGraph(bir).eval()
        res = {'map': show('birefnet map, gather form vs author', cos_rel(graph(captured['x']), captured['map']))}
        a_glue = GL.birefnet_alpha(rgb, lambda x: graph(torch.from_numpy(x)).numpy())
        d = np.abs(a_glue.astype(int) - ref['alpha_birefnet'].astype(int))
        res['alpha_uint8_vs_reference'] = {'pixels_differing': int((d > 0).sum()), 'max_abs_diff': int(d.max()),
                                           'author_forward_equal_reference': bool((a_author == ref['alpha_birefnet']).all())}
        print(f'[check] birefnet uint8 alpha vs reference: {res["alpha_uint8_vs_reference"]}', flush=True)
        if do_export:
            res['export'] = {'bir': export(graph, (captured['x'],), os.path.join(out, 'bir.tflite'))}
    finally:
        restore()
    return res


# ---------------------------------------------------------------- VAE encoder

def latent_tokens(moments, noise, vae):
    """ImageEncoder._flux2_tokens after the encoder, with the shared posterior noise -> [1, 4096, 128]."""
    m, logvar = torch.chunk(moments, 2, dim=1)
    lat = m + torch.exp(0.5 * torch.clamp(logvar, -30.0, 20.0)) * noise
    b, c, h, w = lat.shape
    lat = lat.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    bm = vae.bn.running_mean.view(1, -1, 1, 1)
    bs = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    return ((lat - bm) / bs).flatten(2).transpose(1, 2)


def part_vae(out, ref, noise, do_export, chunk=1024):
    vae = F0.load_vae()
    json.dump({'bn_mean': vae.bn.running_mean.float().tolist(), 'bn_var': vae.bn.running_var.float().tolist(),
               'bn_eps': vae.config.batch_norm_eps}, open(os.path.join(out, 'vae_bn.json'), 'w'))
    parts = [IG.Vae(F0.load_vae(), s, chunk).eval() for s in IG.Vae.PARTS]
    x = t32(ref['pixels'])
    h, ins = x, []
    for p in parts:
        ins.append(h)
        h = p(h)
    res = {'moments': show('vae moments', cos_rel(h, ref['vae_moments']))}
    tok = latent_tokens(h, t32(noise['vae_noise']), vae)
    res['tokens'] = show('vae latent tokens', cos_rel(tok, ref['feature2'][:, 5:]))
    if do_export:
        res['export'] = {}
        for s, p, a in zip(IG.Vae.PARTS, parts, ins):
            res['export'][f'vae_{s}'] = export(p, (a,), os.path.join(out, f'vae_{s}.tflite'))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', default='den,oct,gs,dino,vae')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--no-export', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_grad_enabled(False)
    ref = dict(np.load(os.path.join(a.ref, 'ref.npz')))
    ref['__path__'] = os.path.join(a.ref, 'ref.npz')
    noise = dict(np.load(os.path.join(a.ref, 'noise.npz')))
    jpath = os.path.join(a.out, 'export.json')
    results = json.load(open(jpath)) if os.path.exists(jpath) else {}
    for p in a.parts.split(','):
        t0 = time.time()
        if p == 'den':
            r = part_den(a.out, ref, not a.no_export)
        elif p == 'oct':
            r = part_oct(a.out, ref, not a.no_export)
        elif p == 'gs':
            r = part_gs(a.out, ref, not a.no_export)
        elif p == 'dino':
            r = part_dino(a.out, ref, not a.no_export)
        elif p == 'vae':
            r = part_vae(a.out, ref, noise, not a.no_export)
        elif p == 'bir':
            r = part_bir(a.out, ref, not a.no_export)
        else:
            raise SystemExit(p)
        r['wall_s'] = round(time.time() - t0, 1)
        results[p] = r
        json.dump(results, open(jpath, 'w'), indent=1)
        print(f'== {p} done in {r["wall_s"]} s', flush=True)


if __name__ == '__main__':
    main()
