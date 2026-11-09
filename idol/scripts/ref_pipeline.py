"""The author's Idol pipeline (idol-sdk modules) as the reference, with every random draw explicit and recorded.

Differences from idol-sdk's IdolPipeline.__call__, all on purpose and none in the math:
- the gated encoder repos are not read: DINOv3 ViT-H+/16 comes from Comfy-Org/Idol (bf16 repack, transformers
  layout) and the FLUX.2 VAE from black-forest-labs/FLUX.2-klein-4B `vae/` (Apache-2.0); `require_encoder_access`
  is never called;
- precision: every module in float32 (the SDK's GPU defaults are bf16 encoders and fp16 transformer blocks), unless
  --native (bf16 encoders, fp16 blocks, as the SDK);
- the random draws are the contract's shared randomness (idol_glue.make_noise): VAE posterior noise, latent and
  camera noise, and per-row octree offsets + leaf positions handed to idol-sdk's own sampler through a torch.rand hook;
- turbo (--turbo): idol-turbo's phase LoRA (rank 32, alpha 128, slow for t >= 0.5, fast below), guidance-free Euler,
  re-implemented from turbo.py (qkv / out / mlp[0] / mlp[2] of every refiner and joint block).

  python ref_pipeline.py --image IMG --mode real --seed 42 --out DIR --turbo [--steps 5]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser(os.environ.get('IDOL_SDK', 'idol-sdk')))  # github.com/speridlabs/idol-sdk @4702a7f

from idol.camera import decode_camera  # noqa: E402
from idol.models import BiRefNet, GaussianFixedlenDecoder, LatentSeqMMFlowModel, OctreeProbabilityFixedlenDecoder  # noqa: E402
from idol.models.decoder import systematic_sample  # noqa: E402
from idol.models.layers import RePo3DRotaryEmbedding  # noqa: E402
from idol.preprocess import CROP_MARGIN, preprocess_image  # noqa: E402

import f0_export as F0  # noqa: E402  (snap, dino_config, load_vae)
import idol_glue as GL  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PHASE_THRESHOLD = 0.5
LORA_RANK, LORA_ALPHA = 32, 128.0


class LoRALinear(nn.Module):
    def __init__(self, base, slow_a, slow_b, fast_a, fast_b, scale):
        super().__init__()
        self.base = base
        self.slow_a, self.slow_b, self.fast_a, self.fast_b = slow_a, slow_b, fast_a, fast_b
        self.scale = scale
        self.active = 'slow'

    def forward(self, x):
        out = self.base(x)
        if self.active == 'off':
            return out
        a, b = (self.slow_a, self.slow_b) if self.active == 'slow' else (self.fast_a, self.fast_b)
        return out + ((x.to(a.dtype) @ a.t()) @ b.t()).to(out.dtype) * self.scale


def inject_turbo(denoiser, lora_path):
    sd = load_file(lora_path)
    n = 0
    for group in ('noise_refiner', 'context_refiner', 'blocks'):
        for i, block in enumerate(getattr(denoiser, group)):
            for attr, path in (('attn.qkv', 'attn.qkv'), ('attn.out', 'attn.out'), ('mlp.mlp.0', 'mlp.mlp.0'),
                               ('mlp.mlp.2', 'mlp.mlp.2')):
                parent_path, leaf = attr.rsplit('.', 1)
                parent = block
                for p in parent_path.split('.'):
                    parent = getattr(parent, p)
                base = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
                key = f'{group}.{i}.{path}'
                ws = {k: sd[f'{key}.{k}'].float() for k in ('slow_A', 'slow_B', 'fast_A', 'fast_B')}
                lin = LoRALinear(base, ws['slow_A'], ws['slow_B'], ws['fast_A'], ws['fast_B'], LORA_ALPHA / LORA_RANK)
                if leaf.isdigit():
                    parent[int(leaf)] = lin
                else:
                    setattr(parent, leaf, lin)
                n += 1
    return n, sorted(sd.keys())[:2]


def set_phase(denoiser, phase):
    for m in denoiser.modules():
        if isinstance(m, LoRALinear):
            m.active = phase


def to_dtype_all(model, dtype):
    model.to(dtype)
    if hasattr(model, 'dtype'):
        model.dtype = dtype
    return model


def load_models(device, native):
    cfg = json.load(open(F0.snap('speridlabs/idol', 'config.json')))
    root = os.path.dirname(F0.snap('speridlabs/idol', 'config.json'))

    def load(model, name):
        model.load_state_dict(load_file(os.path.join(root, f'{name}.safetensors')), strict=True)
        return model.eval().requires_grad_(False)

    den = load(LatentSeqMMFlowModel(**cfg['denoiser']['args']), 'denoiser')
    dec = load(OctreeProbabilityFixedlenDecoder(**cfg['decoder']['args']), 'decoder')
    dgs = load(GaussianFixedlenDecoder(**cfg['decoder_gs']['args']), 'decoder_gs')
    bir = load(BiRefNet(), 'birefnet')
    if not native:
        for m in (den, dec, dgs):
            to_dtype_all(m, torch.float32)
    if str(device) == 'mps':  # RePo's real-valued (cos, sin) path: same rotation, MPS has no complex64 matmul
        for m in den.modules():
            if isinstance(m, RePo3DRotaryEmbedding):
                m.real_valued = True
    from transformers import DINOv3ViTModel
    dcfg = F0.dino_config()
    dino = DINOv3ViTModel(dcfg)
    sd = load_file(F0.snap('Comfy-Org/Idol', 'clip_vision/dino_v3_vit_h_bf16.safetensors'))
    # the repack names the blocks layer.N.*; DINOv3ViTModel keeps them under model.layer.N.*
    sd = {('model.' + k if k.startswith('layer.') else k): v for k, v in sd.items()}
    missing, unexpected = dino.load_state_dict(sd, strict=False)
    assert not unexpected and all('rope' in k for k in missing), (missing, unexpected)
    dino = dino.eval().to(torch.bfloat16 if native else torch.float32)
    vae = F0.load_vae()
    if native:
        vae = vae.to(torch.bfloat16)
    return cfg, den.to(device), dec, dgs.to(device), bir, dino.to(device), vae.to(device)


@torch.no_grad()
def encode(dino, vae, pixels, vae_noise, native):
    """ImageEncoder.__call__ with the VAE posterior noise given (latent_dist.sample() = mean + std * noise)."""
    mean = torch.tensor(IMAGENET_MEAN, device=pixels.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=pixels.device).view(1, 3, 1, 1)
    x = (pixels.float() - mean) / std
    dt = next(dino.parameters()).dtype
    tokens = dino(pixel_values=x.to(dt)).last_hidden_state.float()
    feature1 = F.layer_norm(tokens, tokens.shape[-1:])
    vdt = next(vae.parameters()).dtype
    moments = vae.quant_conv(vae.encoder(pixels.to(vdt) * 2 - 1))
    m, logvar = torch.chunk(moments, 2, dim=1)
    logvar = torch.clamp(logvar, -30.0, 20.0)
    latents = m + torch.exp(0.5 * logvar) * vae_noise.to(m.device, m.dtype)
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    bm = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bs = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(latents.device, latents.dtype)
    tokens2 = ((latents - bm) / bs).float().flatten(2).transpose(1, 2).contiguous()
    feature2 = torch.cat([torch.zeros(b, 5, tokens2.shape[2], device=tokens2.device), tokens2], dim=1)
    return feature1, feature2, moments.float()


@torch.no_grad()
def sample(den, latent_noise, cam_noise, cond, steps, rescale_t, cfg_strength, turbo, rec):
    from tensordict import TensorDict
    x = TensorDict({'latent': latent_noise, 'camera': cam_noise}, batch_size=1)
    neg = cond.apply(torch.zeros_like)
    ts = GL.time_grid(steps, rescale_t)
    for i in range(steps):
        t, tp = ts[i], ts[i + 1]
        tt = torch.tensor([1000 * t], device=latent_noise.device, dtype=torch.float32)
        rec[f'x_latent_{i}'] = x['latent'].float().cpu().numpy()
        rec[f'x_camera_{i}'] = x['camera'].float().cpu().numpy()
        if turbo:
            set_phase(den, 'slow' if t >= PHASE_THRESHOLD else 'fast')
            v = den(x, tt, cond)
        else:
            v = den(x, tt, cond)
            if cfg_strength > 0:
                vn = den(x, tt, neg)
                v = TensorDict({k: (1 + cfg_strength) * v[k] - cfg_strength * vn[k] for k in v.keys()}, batch_size=1)
        rec[f'v_latent_{i}'] = v['latent'].float().cpu().numpy()
        rec[f'v_camera_{i}'] = v['camera'].float().cpu().numpy()
        x = TensorDict({k: x[k] - v[k] * (t - tp) for k in x.keys()}, batch_size=1)
        print(f'step {i} t={t:.4f} done', flush=True)
    rec['t_grid'] = np.asarray(ts, dtype=np.float64)
    return x


@torch.no_grad()
def author_octree(dec, latent, pool, leaf, num_points, level, rec):
    """idol-sdk's OctreeProbabilityFixedlenDecoder.sample itself; torch.rand (systematic_sample's per-group offsets)
    and torch.rand_like (leaf positions) return the contract's per-row pool values instead of fresh draws."""
    import idol.models.decoder as D
    state = {'level': 0, 'queue': []}
    orig_ss, orig_rand, orig_rand_like = D.systematic_sample, torch.rand, torch.rand_like

    def ss_hook(probs, counts):
        state['level'] += 1
        flat = counts.reshape(-1)
        unique_n, _ = flat.unique(sorted=False, return_inverse=True)   # the same call idol makes -> the same order
        u = pool[state['level'] - 1, :flat.numel()]
        state['queue'] = [u.index_select(0, (flat == n).nonzero(as_tuple=False).squeeze(1))[:, None]
                          for n in unique_n.tolist() if n != 0]
        return orig_ss(probs, counts)

    def rand_hook(*args, **kwargs):
        q = state['queue'].pop(0)
        shape = tuple(args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)
        assert tuple(q.shape) == shape, (q.shape, shape)
        return q

    def rand_like_hook(t, dtype=None, **kwargs):
        assert tuple(t.shape) == tuple(leaf.shape), (t.shape, leaf.shape)
        return leaf.clone()

    calls = []
    orig_forward = dec.forward

    def fwd(x, lvl, cond):
        out = orig_forward(x, lvl, cond)
        calls.append((x.detach().clone(), torch.softmax(out, dim=-1).detach().clone()))
        return out

    dec.forward = fwd
    D.systematic_sample, torch.rand, torch.rand_like = ss_hook, rand_hook, rand_like_hook
    try:
        anchors = D.OctreeProbabilityFixedlenDecoder.sample(dec, latent, num_points, level)
    finally:
        D.systematic_sample, torch.rand, torch.rand_like = orig_ss, orig_rand, orig_rand_like
        dec.forward = orig_forward
    assert not state['queue']
    for l, (cells, probs) in enumerate(calls, start=1):
        rec[f'oct_cells_{l}'] = cells.numpy()
        rec[f'oct_probs_{l}'] = probs.float().numpy()
    return anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--mode', default='real', choices=['real', 'synthetic'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--turbo', action='store_true')
    ap.add_argument('--cfg', type=float, default=3.0)
    ap.add_argument('--rescale-t', type=float, default=3.0)
    ap.add_argument('--num-gaussians', type=int, default=262144)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--native', action='store_true', help='the SDK precisions (bf16 encoders, fp16 blocks)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--encode-size', type=int, default=0,
                    help='speed-ladder probe: the encoders see the preprocessed picture resized (Lanczos) to this size; '
                         'matting and crop stay at the model size; 0 = the model size')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_grad_enabled(False)
    dev = torch.device(a.device)
    t0 = time.time()
    cfg, den, dec, dgs, bir, dino, vae = load_models(dev, a.native)
    bir = bir.float()
    if a.turbo:
        n, _ = inject_turbo(den, F0.snap('speridlabs/idol-turbo', 'idol_turbo_lora.safetensors'))
        print(f'turbo LoRA on {n} linears', flush=True)
        den.to(dev)
    print(f'loaded in {time.time() - t0:.1f} s', flush=True)

    # preprocessing: the author's function, and the glue's copy of it (which also keeps the alpha and the box)
    raw = Image.open(a.image)
    author = preprocess_image(raw, background_remover=bir, image_size=cfg['image_size'],
                              crop_margin=CROP_MARGIN[a.mode])
    composite, alpha, box = GL.preprocess(raw, GL.birefnet_alpha_fn(bir), cfg['image_size'], CROP_MARGIN[a.mode])
    assert np.array_equal(np.asarray(author), np.asarray(composite)), 'glue preprocessing differs from idol-sdk'
    composite.save(os.path.join(a.out, 'input.png'))
    Image.fromarray(alpha, mode='L').save(os.path.join(a.out, 'alpha_birefnet.png'))
    enc_img = composite
    if a.encode_size and a.encode_size != cfg['image_size']:
        enc_img = composite.resize((a.encode_size, a.encode_size), Image.LANCZOS)
        enc_img.save(os.path.join(a.out, f'input_{a.encode_size}.png'))
    pixels = torch.from_numpy(GL.pixels_of(enc_img)).to(dev)

    noise = GL.make_noise(a.seed, den.q_token_length, den.in_channels, den.cam_channels)
    if enc_img is not composite:  # the VAE posterior noise of the smaller latent: its top-left corner
        h = a.encode_size // 8
        noise['vae_noise'] = np.ascontiguousarray(noise['vae_noise'][:, :, :h, :h])
    np.savez(os.path.join(a.out, 'noise.npz'), **noise)
    rec = {'pixels': pixels.cpu().numpy(), 'alpha_birefnet': alpha, 'box': np.asarray(box)}

    t1 = time.time()
    f1, f2, moments = encode(dino, vae, pixels, torch.from_numpy(noise['vae_noise']), a.native)
    from tensordict import TensorDict
    cond = TensorDict({'feature1': f1, 'feature2': f2}, batch_size=1)
    rec.update(feature1=f1.cpu().numpy(), feature2=f2.cpu().numpy(), vae_moments=moments.cpu().numpy())
    t2 = time.time()
    print(f'encoders {t2 - t1:.1f} s', flush=True)
    x = sample(den, torch.from_numpy(noise['latent_noise']).to(dev), torch.from_numpy(noise['camera_noise']).to(dev),
               cond, a.steps, a.rescale_t, a.cfg, a.turbo, rec)
    latent = x['latent'].float().cpu()
    rec['latent'] = latent.numpy()
    rec['camera'] = x['camera'].float().cpu().numpy()
    t3 = time.time()
    num_points = max(1, a.num_gaussians // dgs.rep_config['num_gaussians'])
    pool, leaf = torch.from_numpy(noise['octree_pool']), torch.from_numpy(noise['leaf_pool'][:, :num_points])
    anchors = author_octree(dec, latent, pool, leaf, num_points, cfg['max_voxel_level'], rec)
    # the glue's copy of the sampler must give the same anchors from the same probabilities
    probs_by_level = {l: torch.from_numpy(rec[f'oct_probs_{l}']) for l in range(1, cfg['max_voxel_level'] + 1)}
    glue_anchors = GL.octree_descend(lambda cells, l: probs_by_level[l], pool, leaf, num_points,
                                     cfg['max_voxel_level'])
    assert torch.equal(glue_anchors, anchors), 'glue octree sampler differs from idol-sdk'
    rec['anchors'] = anchors.numpy()
    t4 = time.time()
    feats = dgs(anchors.to(dev), latent.to(dev)).float().cpu()
    rec['gs_features'] = feats.numpy()
    gauss = dgs.float().cpu().to_representation(anchors, feats)[0]
    gauss.save_ply(os.path.join(a.out, 'gaussian_author.ply'))
    g = GL.to_gaussians(anchors, feats, dgs.rep_config, dgs.points_offset_perturbation.float().cpu(),
                        dgs.base_offset_scale.float().cpu())
    for k in ('xyz', 'features_dc', 'scaling', 'rotation', 'opacity'):
        ref_v = getattr(gauss, k).float()
        assert torch.allclose(g[k], ref_v, rtol=0, atol=1e-6), (k, (g[k] - ref_v).abs().max())
        rec[f'g_{k}'] = ref_v.numpy()
    rows = GL.ply_rows({k: getattr(gauss, k) for k in ('xyz', 'features_dc', 'scaling', 'rotation', 'opacity')})
    GL.write_ply(rows, os.path.join(a.out, 'gaussian.ply'))
    same_bytes = open(os.path.join(a.out, 'gaussian.ply'), 'rb').read() == \
        open(os.path.join(a.out, 'gaussian_author.ply'), 'rb').read()
    t5 = time.time()
    np.savez(os.path.join(a.out, 'ref.npz'), **rec)
    cam = decode_camera(x['camera'][0, 0].float().cpu())
    meta = {'image': a.image, 'mode': a.mode, 'seed': a.seed, 'steps': a.steps, 'turbo': a.turbo,
            'cfg': 0.0 if a.turbo else a.cfg, 'rescale_t': a.rescale_t, 'num_gaussians': a.num_gaussians,
            'encode_size': a.encode_size or cfg['image_size'],
            'device': a.device, 'native': a.native, 'box': box, 'camera': cam, 'glue_ply_bytes_equal': same_bytes,
            'octree_rows': [int(rec[f'oct_cells_{l}'].shape[1]) for l in range(1, cfg['max_voxel_level'] + 1)],
            'seconds': {'load': round(t1 - t0, 1), 'encode': round(t2 - t1, 1), 'sample': round(t3 - t2, 1),
                        'octree': round(t4 - t3, 1), 'gaussians': round(t5 - t4, 1)}}
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1, default=str)
    print(json.dumps(meta, default=str), flush=True)


if __name__ == '__main__':
    main()
