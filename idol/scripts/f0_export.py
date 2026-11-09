"""F0: one block of each heavy Idol part as a LiteRT graph, for the on-device ms / peak numbers.

For each part: build the GPU form of one block from the real weights (gpu_blocks.py), check it against the
idol-sdk / transformers / diffusers module at a reduced length on the CPU in float32, then export it at the
pipeline's full length with litert-torch and scan the flatbuffer (op histogram, tensor ranks, BATCH_MATMUL ranks).

  python f0_export.py --parts den,dino,vae,gs,oct --out <dir>
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SDK = os.path.expanduser(os.environ.get('IDOL_SDK', 'idol-sdk'))  # github.com/speridlabs/idol-sdk @4702a7f
sys.path.insert(0, SDK)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # litert_gpu_toolkit (zoo root)

import gpu_blocks as G  # noqa: E402

HUB = os.path.expanduser('~/.cache/huggingface/hub')


def snap(repo, rel):
    hits = glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*/{rel}")
    assert hits, f'missing {repo}/{rel}'
    return hits[0]


def cos_max(a, b):
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    return float((a @ b) / (a.norm() * b.norm())), float((a - b).abs().max()), float(b.abs().max())


def report(name, ref, got):
    c, m, s = cos_max(got, ref)
    print(f'[check] {name}: cos {c:.9f}  max|diff| {m:.3e}  max|ref| {s:.3e}', flush=True)
    return {'cos': c, 'max_abs_diff': m, 'max_abs_ref': s}


def scan_tflite(path):
    from ai_edge_litert import schema_py_generated as schema
    buf = open(path, 'rb').read()
    model = schema.ModelT.InitFromPackedBuf(buf, 0)
    names = {v: k for k, v in schema.BuiltinOperator.__dict__.items() if not k.startswith('_')}
    hist, bmm, max_rank = {}, [], 0
    for g in model.subgraphs:
        for t in g.tensors:
            if t.shape is not None:
                max_rank = max(max_rank, len(t.shape))
        for op in g.operators:
            code = model.operatorCodes[op.opcodeIndex]
            name = names.get(max(code.builtinCode, code.deprecatedBuiltinCode), str(code.builtinCode))
            if name == 'CUSTOM' or name == 'STABLEHLO_COMPOSITE':
                name = f'{name}:{code.customCode}'
            hist[name] = hist.get(name, 0) + 1
            if name == 'BATCH_MATMUL':
                bmm.append([list(g.tensors[i].shape) for i in op.inputs])
    big5 = sum(1 for g in model.subgraphs for t in g.tensors if t.shape is not None and len(t.shape) > 4)
    return {'ops': sum(hist.values()), 'hist': dict(sorted(hist.items())), 'max_rank': max_rank,
            'tensors_rank_gt4': big5, 'bmm_ranks': sorted({len(s[0]) for s in bmm}), 'bmm_first': [[list(map(int, x)) for x in b] for b in bmm[:2]],
            'bytes': os.path.getsize(path)}


def export(module, args, path):
    import litert_torch
    t0 = time.time()
    module.eval()
    edge = litert_torch.convert(module, args)
    edge.export(path)
    info = scan_tflite(path)
    info['export_s'] = round(time.time() - t0, 1)
    print(f'[export] {os.path.basename(path)}: {info["bytes"]:,} B, {info["ops"]} ops, max rank {info["max_rank"]}, '
          f'rank>4 tensors {info["tensors_rank_gt4"]}, BMM ranks {info["bmm_ranks"]}, {info["export_s"]} s', flush=True)
    print('         ' + ', '.join(f'{k} {v}' for k, v in info['hist'].items()), flush=True)
    return info


# -- parts ---------------------------------------------------------------------------------------------------

def load_denoiser():
    from idol.models.dit import LatentSeqMMFlowModel
    cfg = json.load(open(snap('speridlabs/idol', 'config.json')))
    m = LatentSeqMMFlowModel(**cfg['denoiser']['args'])
    m.load_state_dict(load_file(snap('speridlabs/idol', 'denoiser.safetensors')), strict=True)
    return m.eval(), cfg


def part_den(out, chunk, full_len):
    m, _ = load_denoiser()
    block, repo = m.blocks[0].float(), m.repo_layers[0].float()
    gpu = G.DenoiserBlockGPU(block, repo, chunk=chunk).eval()
    torch.manual_seed(0)
    l_small = 600
    x = torch.randn(1, l_small, 1024) * 3
    t = torch.tensor([1000 * 0.7])
    mod = m.adaLN_modulation(m.t_embedder(t)).float()
    with torch.no_grad():
        repo.real_valued = True
        ref = block(x, mod=mod, rotary_emb=repo(x))
        got = G.DenoiserBlockGPU(block, repo, chunk=128)(x, mod)
    res = {'check': report('den joint block (L=600, chunk 128)', ref, got)}
    xf = torch.randn(1, full_len, 1024)
    res['export'] = export(gpu, (xf, mod), os.path.join(out, f'f0_den_block_L{full_len}_c{chunk}.tflite'))
    return res


def part_gs(out, chunk, anchors):
    from idol.models.decoder import GaussianFixedlenDecoder
    cfg = json.load(open(snap('speridlabs/idol', 'config.json')))
    m = GaussianFixedlenDecoder(**cfg['decoder_gs']['args'])
    m.load_state_dict(load_file(snap('speridlabs/idol', 'decoder_gs.safetensors')), strict=True)
    block = m.blocks[0].float().eval()
    torch.manual_seed(0)
    x, ctx = torch.randn(1, 300, 1024) * 2, torch.randn(1, 500, 16)
    with torch.no_grad():
        ref = block(x, ctx)
        got = G.GaussianBlockGPU(block, chunk=64)(x, ctx)
    res = {'check': report('gs block (L=300, ctx 500, chunk 64)', ref, got)}
    gpu = G.GaussianBlockGPU(block, chunk=chunk).eval()
    res['export'] = export(gpu, (torch.randn(1, anchors, 1024), torch.randn(1, 8192, 16)),
                           os.path.join(out, f'f0_gs_block_A{anchors}_c{chunk}.tflite'))
    return res


def part_oct(out, chunk, cells):
    from idol.models.decoder import OctreeProbabilityFixedlenDecoder
    cfg = json.load(open(snap('speridlabs/idol', 'config.json')))
    m = OctreeProbabilityFixedlenDecoder(**cfg['decoder']['args'])
    m.load_state_dict(load_file(snap('speridlabs/idol', 'decoder.safetensors')), strict=True)
    m = m.float().eval()
    block = m.blocks[0]
    torch.manual_seed(0)
    x, ctx = torch.randn(1, 300, 1024) * 2, torch.randn(1, 500, 16)
    with torch.no_grad():
        mod = m.adaLN_modulation(m.l_embedder(torch.tensor([64.0]))).float()
        ref = block(x, mod, ctx)
        got = G.OctreeBlockGPU(block, chunk=64)(x, mod, ctx)
    res = {'check': report('octree block (L=300, ctx 500, chunk 64)', ref, got)}
    gpu = G.OctreeBlockGPU(block, chunk=chunk).eval()
    res['export'] = export(gpu, (torch.randn(1, cells, 1024), mod, torch.randn(1, 8192, 16)),
                           os.path.join(out, f'f0_oct_block_L{cells}_c{chunk}.tflite'))
    return res


def dino_config():
    from transformers import DINOv3ViTConfig
    raw = json.load(open(snap('ddalcu/Idol-MLX-Serve-8bit', 'dinov3/config.json')))
    keep = {k: v for k, v in raw.items() if k not in ('architectures', 'torch_dtype', 'transformers_version')}
    return DINOv3ViTConfig(**keep)


def dino_rope(cfg, grid):
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTRopePositionEmbedding
    rope = DINOv3ViTRopePositionEmbedding(cfg).eval()   # train mode would add DINOv3's random coordinate augmentation
    cos, sin = rope(torch.zeros(1, 3, grid * cfg.patch_size, grid * cfg.patch_size))
    prefix = 1 + cfg.num_register_tokens
    d = cos.shape[-1]
    cos_all = torch.cat([torch.ones(prefix, d), cos], 0)[None, None]
    sin_all = torch.cat([torch.zeros(prefix, d), sin], 0)[None, None]
    return cos, sin, cos_all, sin_all


def part_dino(out, chunk, grid):
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTLayer
    cfg = dino_config()
    cfg._attn_implementation = 'sdpa'
    layer = DINOv3ViTLayer(cfg)
    sd = load_file(snap('Comfy-Org/Idol', 'clip_vision/dino_v3_vit_h_bf16.safetensors'))
    layer.load_state_dict({k[len('layer.0.'):]: v.float() for k, v in sd.items() if k.startswith('layer.0.')},
                          strict=True)
    layer = layer.float().eval()
    torch.manual_seed(0)
    g_small = 8
    cos, sin, cos_all, sin_all = dino_rope(cfg, g_small)
    x = torch.randn(1, 5 + g_small * g_small, 1280) * 2
    with torch.no_grad():
        ref = layer(x, position_embeddings=(cos, sin))
        got = G.DinoBlockGPU(layer, chunk=16)(x, cos_all, sin_all)
    res = {'check': report(f'dinov3 layer (grid {g_small}, chunk 16)', ref, got)}
    _, _, cos_all, sin_all = dino_rope(cfg, grid)
    gpu = G.DinoBlockGPU(layer, chunk=chunk).eval()
    n = 5 + grid * grid
    res['export'] = export(gpu, (torch.randn(1, n, 1280), cos_all, sin_all),
                           os.path.join(out, f'f0_dino_block_L{n}_c{chunk}.tflite'))
    return res


class VaeAttnGPU(nn.Module):
    """diffusers Attention (VAE mid block, AttnProcessor2_0 semantics) with chunked rank-4 attention."""

    def __init__(self, attn, chunk):
        super().__init__()
        from litert_gpu_toolkit.patches import ManualGroupNorm
        self.gn = ManualGroupNorm(attn.group_norm)
        self.q, self.k, self.v, self.o = attn.to_q, attn.to_k, attn.to_v, attn.to_out[0]
        self.heads = attn.heads
        self.scale = attn.scale
        self.residual = attn.residual_connection
        self.rescale = attn.rescale_output_factor
        self.chunk = chunk

    def forward(self, x, *args, **kwargs):
        b, c, hh, ww = x.shape
        h = self.gn(x).reshape(b, c, hh * ww).transpose(1, 2)
        d = c // self.heads
        q = (self.q(h) * self.scale).reshape(b, hh * ww, self.heads, d).permute(0, 2, 1, 3)
        k = self.k(h).reshape(b, hh * ww, self.heads, d).permute(0, 2, 1, 3)
        v = self.v(h).reshape(b, hh * ww, self.heads, d).permute(0, 2, 1, 3)
        o = G.attention(q, k, v, self.chunk).reshape(b, hh * ww, c)
        o = self.o(o).transpose(1, 2).reshape(b, c, hh, ww)
        if self.residual:
            o = o + x
        return o / self.rescale


class VaeEncoderGPU(nn.Module):
    """FLUX.2 VAE encoder + quant_conv -> the 64 posterior moments [1, 64, H/8, W/8] (mean = first 32)."""

    def __init__(self, vae, chunk):
        super().__init__()
        from litert_gpu_toolkit.patches import patch_groupnorm
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv
        mid = self.encoder.mid_block
        mid.attentions = nn.ModuleList([VaeAttnGPU(a, chunk) for a in mid.attentions])
        patch_groupnorm(self.encoder)

    def forward(self, x):
        return self.quant_conv(self.encoder(x))


def load_vae():
    from diffusers.models import AutoencoderKLFlux2
    root = os.path.dirname(os.path.dirname(snap('black-forest-labs/FLUX.2-klein-4B', 'vae/config.json')))
    vae = AutoencoderKLFlux2.from_pretrained(root, subfolder='vae', torch_dtype=torch.float32)
    return vae.float().eval()


def part_vae(out, chunk, size):
    vae = load_vae()
    torch.manual_seed(0)
    x = torch.rand(1, 3, 128, 128) * 2 - 1
    with torch.no_grad():
        ref = vae.quant_conv(vae.encoder(x))
        gpu = VaeEncoderGPU(load_vae(), chunk=64).eval()
        got = gpu(x)
    res = {'check': report('flux2 vae encoder (128^2, chunk 64)', ref, got)}
    gpu = VaeEncoderGPU(load_vae(), chunk=chunk).eval()
    res['export'] = export(gpu, (torch.rand(1, 3, size, size) * 2 - 1,),
                           os.path.join(out, f'f0_vae_enc_{size}_c{chunk}.tflite'))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', default='den,dino,vae,gs,oct')
    ap.add_argument('--out', required=True)
    ap.add_argument('--den-chunk', type=int, default=512)
    ap.add_argument('--den-len', type=int, default=8192 + 4101 + 1)
    ap.add_argument('--dino-chunk', type=int, default=1024)
    ap.add_argument('--dino-grid', type=int, default=64)
    ap.add_argument('--vae-chunk', type=int, default=1024)
    ap.add_argument('--vae-size', type=int, default=1024)
    ap.add_argument('--gs-chunk', type=int, default=512)
    ap.add_argument('--gs-anchors', type=int, default=8192)
    ap.add_argument('--oct-chunk', type=int, default=512)
    ap.add_argument('--oct-cells', type=int, default=8192)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))   # litert_gpu_toolkit
    torch.set_grad_enabled(False)
    jpath = os.path.join(a.out, 'f0_export.json')
    results = json.load(open(jpath)) if os.path.exists(jpath) else {}
    for p in a.parts.split(','):
        t0 = time.time()
        if p == 'den':
            results[p] = part_den(a.out, a.den_chunk, a.den_len)
        elif p == 'dino':
            results[p] = part_dino(a.out, a.dino_chunk, a.dino_grid)
        elif p == 'vae':
            results[p] = part_vae(a.out, a.vae_chunk, a.vae_size)
        elif p == 'gs':
            results[p] = part_gs(a.out, a.gs_chunk, a.gs_anchors)
        elif p == 'oct':
            results[p] = part_oct(a.out, a.oct_chunk, a.oct_cells)
        else:
            raise SystemExit(f'unknown part {p}')
        results[p]['wall_s'] = round(time.time() - t0, 1)
        with open(jpath, 'w') as f:
            json.dump(results, f, indent=1)
    print('done', json.dumps({k: v.get('check') for k, v in results.items()}))


if __name__ == '__main__':
    main()
