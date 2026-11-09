"""F0: compare each S26 leg's dumped output with the same GPU-form block in float32 on the Mac (PyTorch).

  python f0_dump_check.py --lane <K> --round f0 [--legs den16,dino16,...]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser('~/code/idol_work/src/idol-sdk'))
import f0_export as F  # noqa: E402
import gpu_blocks as G  # noqa: E402


def module_for(graph):
    from safetensors.torch import load_file
    if graph.startswith('f0_den_block'):
        m, _ = F.load_denoiser()
        return G.DenoiserBlockGPU(m.blocks[0].float(), m.repo_layers[0].float(), chunk=512)
    if graph.startswith('f0_dino_block'):
        from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTLayer
        layer = DINOv3ViTLayer(F.dino_config())
        sd = load_file(F.snap('Comfy-Org/Idol', 'clip_vision/dino_v3_vit_h_bf16.safetensors'))
        layer.load_state_dict({k[len('layer.0.'):]: v.float() for k, v in sd.items() if k.startswith('layer.0.')})
        return G.DinoBlockGPU(layer.float(), chunk=1024)
    if graph.startswith('f0_gs_block'):
        from idol.models.decoder import GaussianFixedlenDecoder
        cfg = json.load(open(F.snap('speridlabs/idol', 'config.json')))
        m = GaussianFixedlenDecoder(**cfg['decoder_gs']['args'])
        m.load_state_dict(load_file(F.snap('speridlabs/idol', 'decoder_gs.safetensors')))
        return G.GaussianBlockGPU(m.blocks[0].float(), chunk=512)
    if graph.startswith('f0_oct_block'):
        from idol.models.decoder import OctreeProbabilityFixedlenDecoder
        cfg = json.load(open(F.snap('speridlabs/idol', 'config.json')))
        m = OctreeProbabilityFixedlenDecoder(**cfg['decoder']['args'])
        m.load_state_dict(load_file(F.snap('speridlabs/idol', 'decoder.safetensors')))
        return G.OctreeBlockGPU(m.blocks[0].float(), chunk=512)
    if graph.startswith('f0_vae_enc'):
        return F.VaeEncoderGPU(F.load_vae(), chunk=1024)
    raise SystemExit(graph)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lane', required=True)
    ap.add_argument('--round', default='f0')
    ap.add_argument('--legs', default='')
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    rdir = os.path.join(a.lane, 'device', a.round)
    out = {}
    cache = {}
    for leg_json in sorted(f for f in os.listdir(rdir) if f.endswith('.leg.json')):
        leg = json.load(open(os.path.join(rdir, leg_json)))
        tag = leg['tag']
        if a.legs and tag not in a.legs.split(','):
            continue
        dump = os.path.join(rdir, f'{tag}.dump', '000.f32')
        if not os.path.exists(dump):
            dump = os.path.join(rdir, f'{tag}.dump', 'model0', '000.f32')
        if not os.path.exists(dump):
            out[tag] = {'status': leg.get('status'), 'dump': os.path.exists(dump)}
            continue
        man = json.load(open(os.path.join(a.lane, leg['manifest'])))
        fx_dir = os.path.join(a.lane, man['dir'])
        ins = [torch.from_numpy(np.fromfile(os.path.join(fx_dir, f"000_{x['name']}.f32"), dtype=np.float32)
                                .reshape(x['shape'])) for x in man['order']['inputs_subgraph_order']]
        graph = leg['graph']
        if graph not in cache:
            mod = module_for(graph).eval()
            cache[graph] = mod(*ins).float().numpy().ravel().astype(np.float64)
        ref = cache[graph]
        got = np.fromfile(dump, dtype=np.float32).astype(np.float64)
        cos = float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got)))
        d = np.abs(ref - got)
        out[tag] = {'precision': leg['precision'], 'cos': cos, 'max_abs_diff': float(d.max()),
                    'mean_abs_diff': float(d.mean()), 'max_abs_ref': float(np.abs(ref).max()),
                    'rel_l2': float(np.linalg.norm(ref - got) / np.linalg.norm(ref)), 'finite': bool(np.isfinite(got).all())}
        print(tag, json.dumps(out[tag]), flush=True)
    json.dump(out, open(os.path.join(rdir, 'dump_check.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
