"""The phone pipeline in Python: the shipped LiteRT graphs + the host glue, step for step what the Kotlin app does.
Used for the Mac stand-in of CONTRACT.md lines (a), (a') and (b) (graphs on Metal = ML Drift fp16 on the Mac).

  stages: matting (the author's BiRefNet on the CPU; IS-Net DIS as an option) -> preprocessing -> DINOv3 (4 graphs) + VAE (4 graphs) -> 5 Euler steps (den_ctx once per
  LoRA phase, den_in, 24 joint graphs) -> octree (8 levels, row buckets) -> Gaussian decoder (8 graphs) -> PLY
"""
import gc
import json
import os
import time

import numpy as np
import torch

import idol_glue as GL
import idol_graphs as IG
from lrt import Graph


class IdolLiteRT:
    def __init__(self, models, accel='gpu', dis_path=None, enforce_f32=False, stream=False, dino_dir=None,
                 matting='birefnet'):
        self.dir = models
        self.matting = matting   # 'birefnet' = the author's background remover (bir.tflite, CPU, the default);
        # 'dis' = IS-Net DIS + the alpha floor (dis.tflite, GPU; CONTRACT.md: its edges fail line (b) on input 3)
        self.dino_dir = dino_dir or models   # the DINOv3 graphs and RoPE tables ship in their own (gated) repo
        self.accel, self.f32 = accel, enforce_f32
        self.stream = stream   # load each graph for one call and release it (fp32 on the Mac: 26 resident graphs > 32 GB)
        self.g = {}
        f = lambda n: os.path.join(models, n)  # noqa: E731
        if os.path.exists(f('den_pos_emb.f32')):
            self.pos_emb = np.fromfile(f('den_pos_emb.f32'), np.float32).reshape(1, 8192, 1024)
        rope = lambda n: os.path.join(self.dino_dir, n)  # noqa: E731
        if os.path.exists(rope('dino_rope_cos.f32')):
            self.rope_cos = np.fromfile(rope('dino_rope_cos.f32'), np.float32).reshape(1, 1, 4101, 64)
            self.rope_sin = np.fromfile(rope('dino_rope_sin.f32'), np.float32).reshape(1, 1, 4101, 64)
        if os.path.exists(f('vae_bn.json')):
            bn = json.load(open(f('vae_bn.json')))
            self.bn_mean = np.asarray(bn['bn_mean'], np.float32).reshape(1, 128, 1, 1)
            self.bn_std = np.sqrt(np.asarray(bn['bn_var'], np.float32) + bn['bn_eps']).reshape(1, 128, 1, 1)
        if os.path.exists(f('gs_rep.json')):
            rep = json.load(open(f('gs_rep.json')))
            self.rep, self.base_offset_scale = rep['rep'], rep['base_offset_scale']
            if os.path.exists(f('gs_perturbation.npy')):
                pert = np.load(f('gs_perturbation.npy'))
            else:   # the released folder carries the raw little-endian float32 file only
                pert = np.fromfile(f('gs_perturbation.f32'), '<f4').reshape(-1, 3)
            self.perturbation = torch.from_numpy(pert)
        self.dis_path = dis_path
        self.times = {}
        # graphs that run in fp32 precision (their fp16 form fails a CONTRACT.md line): the manifest's, as the app does
        self.fp32_graphs = set()
        if os.path.exists(f('manifest.json')):
            man = json.load(open(f('manifest.json')))
            self.fp32_graphs = {n for n, g in man['graphs'].items() if g['precision'] == 'fp32'}

    def graph(self, name):
        if name not in self.g:
            base = self.dino_dir if name.startswith('dino_') else self.dir
            path = self.dis_path if name == 'dis' else os.path.join(base, f'{name}.tflite')
            # BiRefNet's deformable convolutions sample at data-dependent positions (GATHER_ND): no GPU path, so CPU
            accel = 'cpu' if name == 'bir' else self.accel
            self.g[name] = Graph(path, accel, self.f32 or name in self.fp32_graphs)
        return self.g[name]

    def release(self, *prefixes):
        """Close the graphs whose names start with any of the prefixes: one stage's graphs at a time, as the app's
        preload = false mode does (all 46 graphs loaded at once pass 32 GB of footprint on the Mac)."""
        for n in [n for n in self.g if n.startswith(prefixes)]:
            del self.g[n]
        gc.collect()

    def run1(self, name, inputs, key=None):
        out = self.graph(name).run(inputs, key)
        if self.stream and name != 'oct':
            self.release(name)
        return [out[k] for k in sorted(out, key=lambda s: int(s.split('_')[-1]))]

    def _t(self, stage, t0):
        self.times[stage] = self.times.get(stage, 0.0) + time.time() - t0

    # ---------------------------------------------------------------- stages
    def matting_alpha(self, rgb):
        if self.matting == 'birefnet':
            return GL.birefnet_alpha(rgb, lambda x: self.run1('bir', [x])[0])
        return GL.dis_alpha(rgb, lambda x: self.run1('dis', [x])[0])

    def dinov3(self, pixels):
        t0 = time.time()
        h = pixels
        for i in range(4):
            h = self.run1(f'dino_{i}', [h, self.rope_cos, self.rope_sin])[0]
        self._t('dinov3', t0)
        return h

    def vae_moments(self, pixels):
        t0 = time.time()
        h = pixels
        for s in IG.Vae.PARTS:
            h = self.run1(f'vae_{s}', [h])[0]
        self._t('vae', t0)
        return h

    def vae_tokens(self, moments, noise):
        m, logvar = moments[:, :32], moments[:, 32:]
        lat = m + np.exp(0.5 * np.clip(logvar, -30.0, 20.0)) * noise
        b, c, h, w = lat.shape
        lat = lat.reshape(b, c, h // 2, 2, w // 2, 2).transpose(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
        tok = ((lat - self.bn_mean) / self.bn_std).reshape(b, c * 4, -1).transpose(0, 2, 1)
        return np.ascontiguousarray(tok, np.float32)

    def cond(self, pixels, vae_noise):
        f1 = self.dinov3(pixels)
        tok = self.vae_tokens(self.vae_moments(pixels), vae_noise)
        f2 = np.concatenate([np.zeros((1, 5, 128), np.float32), tok], axis=1)
        return f1, f2

    def velocity(self, x_lat, x_cam, t, f1, f2, ctx_cache):
        phase = 'slow' if t >= GL.PHASE_THRESHOLD else 'fast'
        gate = IG.lora_gate(phase).numpy()
        t0 = time.time()
        if phase not in ctx_cache:
            ctx_cache[phase] = self.run1('den_ctx', [f1, f2, gate])[0]
        h_cond = ctx_cache[phase]
        h_x, h_cam, t_mod, t_emb = self.run1('den_in', [x_lat, x_cam, IG.sinusoid(1000 * t).numpy(), self.pos_emb,
                                                         gate])
        h = self.run1('den_joint_00', [h_x, h_cond, h_cam, t_mod, gate])[0]
        for k in range(1, 23):
            h = self.run1(f'den_joint_{k:02d}', [h, t_mod, gate])[0]
        v, vc = self.run1('den_joint_23', [h, t_mod, t_emb, gate])
        self._t('denoiser', t0)
        return v, vc

    def sample(self, noise, f1, f2, steps=5, rescale_t=3.0, rec=None):
        x_lat, x_cam = noise['latent_noise'].copy(), noise['camera_noise'].copy()
        ts = GL.time_grid(steps, rescale_t)
        cache = {}
        for i in range(steps):
            v, vc = self.velocity(x_lat, x_cam, float(ts[i]), f1, f2, cache)
            if rec is not None:
                rec[f'v_latent_{i}'], rec[f'v_camera_{i}'] = v, vc
            dt = np.float32(ts[i] - ts[i + 1])
            x_lat = (x_lat - v * dt).astype(np.float32)
            x_cam = (x_cam - vc * dt).astype(np.float32)
        return x_lat, x_cam

    def octree_logits(self, cells, level, latent):
        n = cells.shape[1]
        b = next(bb for bb in IG.OCT_BUCKETS if n <= bb)
        c = np.zeros((1, b, 3), np.float32)
        c[:, :n] = cells
        pe = IG.point_pe(torch.from_numpy(c[0])).numpy()[None]
        lf = IG.sinusoid(float(1 << level), max_period=1024.0, two_pi=True).numpy()
        return self.run1('oct', [c, pe, lf, latent], key=f'L{b}')[0][:, :n]

    def anchors(self, latent, noise, rec=None):
        t0 = time.time()
        pool, leaf = torch.from_numpy(noise['octree_pool']), torch.from_numpy(noise['leaf_pool'])

        def probs_fn(cells, level):
            logits = torch.from_numpy(self.octree_logits(cells.numpy(), level, latent))
            return torch.softmax(logits.float(), dim=-1)
        out = GL.octree_descend(probs_fn, pool, leaf, rec=rec).numpy()
        self._t('octree', t0)
        return out

    def gaussians(self, anchors, latent):
        t0 = time.time()
        pe = IG.point_pe(torch.from_numpy(anchors[0])).numpy()[None]
        h = self.run1('gs_0', [anchors, pe, latent])[0]
        for i in range(1, 8):
            h = self.run1(f'gs_{i}', [h, latent])[0]
        self._t('gaussian_decoder', t0)
        g = GL.to_gaussians(torch.from_numpy(anchors), torch.from_numpy(h), self.rep, self.perturbation,
                            torch.tensor(self.base_offset_scale))
        return h, g

    def run(self, image, mode, noise, out_dir, rec=None):
        """raw image -> PLY, the app's whole pipeline."""
        from PIL import Image
        t_all = time.time()
        raw = Image.open(image)
        t0 = time.time()
        composite, alpha, box = GL.preprocess(raw, self.matting_alpha, GL.IMAGE_SIZE, GL.CROP_MARGIN[mode])
        self._t('preprocess', t0)
        self.release('dis', 'bir')
        os.makedirs(out_dir, exist_ok=True)
        composite.save(os.path.join(out_dir, 'input.png'))
        Image.fromarray(alpha, mode='L').save(os.path.join(out_dir, f'alpha_{self.matting}.png'))
        pixels = GL.pixels_of(composite)
        t_gen = time.time()
        f1, f2 = self.cond(pixels, noise['vae_noise'])
        self.release('dino_', 'vae_')
        lat, cam = self.sample(noise, f1, f2, rec=rec)
        self.release('den_')
        anchors = self.anchors(lat, noise)
        self.release('oct')
        _, g = self.gaussians(anchors, lat)
        self.release('gs_')
        t0 = time.time()
        GL.write_ply(GL.ply_rows(g), os.path.join(out_dir, 'gaussian.ply'))
        self._t('ply', t0)
        self.times['generate_total'] = time.time() - t_gen
        self.times['all'] = time.time() - t_all
        return {'box': box, 'times': {k: round(v, 2) for k, v in self.times.items()}}
