"""The shipped Idol graph set: GPU-shaped modules built from the real weights, one module per LiteRT graph.

Every GPU call stays well under 2 s on the S26 (CONTRACT.md "Forms"), so the big parts are split:
  denoiser   den_ctx (condition refiner, once per LoRA phase), den_in (time embedding, noise refiner, camera MLP),
             den_joint_00 .. den_joint_23 (one joint block each; _00 concatenates the streams, _23 adds the heads)
  octree     oct (one graph, one signature per row bucket 64 / 256 / 1024 / 4096 / 8192: the weights are shared)
  gaussians  gs_0 .. gs_7 (two blocks each; gs_0 embeds the anchors, gs_7 projects to the 480 features)
  dinov3     dino_0 .. dino_3 (eight layers each; dino_0 normalises and embeds the pixels, dino_3 ends in both norms)
  vae        vae_0 .. vae_2 (one down stage each), vae_3 (down 3 + mid resnet 0), vae_attn (mid attention, fp32),
             vae_4 (mid resnet 1, head, quant_conv)

Host inputs instead of in-graph math, for fp16 range: sinusoidal features of the diffusion time and the octree level
(args up to 1000 rad), the point position embeddings (args up to 2^16 * 2pi), the denoiser's fixed latent-token
position embedding, and DINOv3's RoPE tables. The turbo LoRA is kept as side matmuls: one graph set, and a gate input
[1, 64] = (4 * slow, 4 * fast) per 32 ranks selects the phase.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import gpu_blocks as G

LORA_SCALE = 128.0 / 32.0
DEN_TOKENS, COND_TOKENS = 8192, 4101
OCT_BUCKETS = (64, 256, 1024, 4096, 8192)


def lora_pair(sd, key):
    """-> A [64, in] (slow rows then fast rows), B [out, 64] (slow columns then fast columns)."""
    a = torch.cat([sd[f'{key}.slow_A'], sd[f'{key}.fast_A']], 0).float()
    b = torch.cat([sd[f'{key}.slow_B'], sd[f'{key}.fast_B']], 1).float()
    return a, b


def lora_gate(phase):
    g = torch.zeros(1, 64)
    g[:, :32] = LORA_SCALE if phase == 'slow' else 0.0
    g[:, 32:] = LORA_SCALE if phase == 'fast' else 0.0
    return g


class DenBlock(G.DenoiserBlockGPU):
    """DenoiserBlockGPU + the turbo LoRA on qkv, out, fc1, fc2: forward(x, mod or None, gate [1, 64])."""

    def __init__(self, block, repo, lora_sd, key, chunk):
        super().__init__(block, repo, chunk=chunk)
        c, h, d = self.c, self.h, self.d
        perm = G.deinterleave_perm(d)
        a, b = lora_pair(lora_sd, f'{key}.attn.qkv')
        self.qkv_a = G._linear(a, None)

        def rows(i, permute):
            bi = b[i * c:(i + 1) * c].reshape(h, d, 64)
            if permute:
                bi = bi[:, perm]
            return G._linear(bi.reshape(c, 64), None)

        self.q_b, self.k_b, self.v_b = rows(0, True), rows(1, True), rows(2, False)
        self.out_a, self.out_b = [G._linear(t, None) for t in lora_pair(lora_sd, f'{key}.attn.out')]
        self.fc1_a, self.fc1_b = [G._linear(t, None) for t in lora_pair(lora_sd, f'{key}.mlp.mlp.0')]
        self.fc2_a, self.fc2_b = [G._linear(t, None) for t in lora_pair(lora_sd, f'{key}.mlp.mlp.2')]

    def attn_l(self, hh, cos, sin, gate):
        b, l, _ = hh.shape
        half = self.d // 2

        def rope(t):
            t0, t1 = t[..., :half], t[..., half:]
            return torch.cat([t0 * cos - t1 * sin, t0 * sin + t1 * cos], dim=-1)

        ax = self.qkv_a(hh) * gate
        q = G.safe_rms_unit(rope((self.q(hh) + self.q_b(ax)).reshape(b, l, self.h, self.d))) * self.gq
        k = G.safe_rms_unit(rope((self.k(hh) + self.k_b(ax)).reshape(b, l, self.h, self.d))) * self.gk
        v = (self.v(hh) + self.v_b(ax)).reshape(b, l, self.h, self.d)
        o = G.attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), self.chunk)
        o = o.reshape(b, l, self.c)
        return self.out(o) + self.out_b(self.out_a(o) * gate)

    def mlp_l(self, hh, gate):
        u = self.fc1(hh) + self.fc1_b(self.fc1_a(hh) * gate)
        u = G._gelu_tanh(u)
        return self.fc2(u) + self.fc2_b(self.fc2_a(u) * gate)

    def forward(self, x, mod, gate):
        cos, sin = self.repo(x)
        if not self.modulation:
            x = x + self.attn_l(G.safe_layer_norm(x, self.eps1, self.n1w, self.n1b), cos, sin, gate)
            return x + self.mlp_l(G.safe_layer_norm(x, self.eps2, self.n2w, self.n2b), gate)
        m = mod + self.shift_table
        c = self.c
        shift_msa, scale_msa, gate_msa = m[:, 0:c], m[:, c:2 * c], m[:, 2 * c:3 * c]
        shift_mlp, scale_mlp, gate_mlp = m[:, 3 * c:4 * c], m[:, 4 * c:5 * c], m[:, 5 * c:6 * c]
        hh = G.safe_layer_norm(x, self.eps1) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + self.attn_l(hh, cos, sin, gate) * gate_msa.unsqueeze(1)
        hh = G.safe_layer_norm(x, self.eps2) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        return x + self.mlp_l(hh, gate) * gate_mlp.unsqueeze(1)


def _mlp_seq(seq):
    """nn.Sequential(Linear, SiLU or GELU(tanh), Linear) -> a float copy."""
    return nn.Sequential(*[G._linear(m.weight, m.bias) if isinstance(m, nn.Linear) else m for m in seq])


class DenCtx(nn.Module):
    """(feature1 [1, 4101, 1280], feature2 [1, 4101, 128], gate) -> h_cond [1, 4101, 1024]."""

    def __init__(self, den, lora_sd, chunk):
        super().__init__()
        self.e1 = G._linear(den.cond_embedder.weight, den.cond_embedder.bias)
        self.e2 = G._linear(den.cond_embedder2.weight, den.cond_embedder2.bias)
        self.blocks = nn.ModuleList([DenBlock(b.float(), r.float(), lora_sd, f'context_refiner.{i}', chunk)
                                     for i, (b, r) in enumerate(zip(den.context_refiner, den.context_repo_layers))])

    def forward(self, feature1, feature2, gate):
        h = self.e1(feature1) + self.e2(feature2)
        for blk in self.blocks:
            h = blk(h, None, gate)
        return h


class DenIn(nn.Module):
    """(latent [1, 8192, 16], camera [1, 1, 5], t_freq [1, 256], pos_emb [1, 8192, 1024], gate)
    -> h_x [1, 8192, 1024], h_cam [1, 1, 1024], t_mod [1, 6144], t_emb [1, 1024]."""

    def __init__(self, den, lora_sd, chunk):
        super().__init__()
        self.t_mlp = _mlp_seq(den.t_embedder.mlp)
        self.ada = _mlp_seq(den.adaLN_modulation)
        self.inp = G._linear(den.input_layer.weight, den.input_layer.bias)
        self.blocks = nn.ModuleList([DenBlock(b.float(), r.float(), lora_sd, f'noise_refiner.{i}', chunk)
                                     for i, (b, r) in enumerate(zip(den.noise_refiner, den.noise_repo_layers))])
        self.cam = _mlp_seq(den.cam_refiner.mlp)

    def forward(self, latent, camera, t_freq, pos_emb, gate):
        t_emb = self.t_mlp(t_freq)
        t_mod = self.ada(t_emb)
        h = self.inp(latent) + pos_emb
        for blk in self.blocks:
            h = blk(h, t_mod, gate)
        return h, self.cam(camera), t_mod, t_emb


class DenJoint(nn.Module):
    """Joint block k. first: (h_x, h_cond, h_cam, t_mod, gate) -> h; middle: (h, t_mod, gate) -> h;
    last: (h, t_mod, t_emb, gate) -> v_latent [1, 8192, 16], v_camera [1, 1, 5]."""

    def __init__(self, den, k, lora_sd, chunk):
        super().__init__()
        self.first, self.last = k == 0, k == len(den.blocks) - 1
        self.block = DenBlock(den.blocks[k].float(), den.repo_layers[k].float(), lora_sd, f'blocks.{k}', chunk)
        if self.last:
            self.shift_table = nn.Parameter(den.shift_table.detach().float().clone(), requires_grad=False)
            self.out = G._linear(den.out_layer.weight, den.out_layer.bias)
            self.cam_out = G._linear(den.cam_out_layer.weight, den.cam_out_layer.bias)

    def forward(self, *args):
        if self.first:
            h_x, h_cond, h_cam, t_mod, gate = args
            h = torch.cat([h_x, h_cond, h_cam], dim=1)
        elif self.last:
            h, t_mod, t_emb, gate = args
        else:
            h, t_mod, gate = args
        h = self.block(h, t_mod, gate)
        if not self.last:
            return h
        st = self.shift_table + t_emb.unsqueeze(1)
        shift, scale = st[:, 0:1], st[:, 1:2]
        hx = G.safe_layer_norm(h[:, :DEN_TOKENS], 1e-5) * (1 + scale) + shift
        hc = G.safe_layer_norm(h[:, -1:], 1e-5) * (1 + scale) + shift
        return self.out(hx), self.cam_out(hc)


class Octree(nn.Module):
    """(cells [1, L, 3], cells_pe [1, L, 1024], l_freq [1, 256], latent [1, 8192, 16]) -> logits [1, L, 8]."""

    def __init__(self, dec, chunk):
        super().__init__()
        self.in_proj = G._linear(dec.in_proj.weight, dec.in_proj.bias)
        self.inp = G._linear(dec.input_layer.weight, dec.input_layer.bias)
        self.l_mlp = _mlp_seq(dec.l_embedder.mlp)
        self.ada = _mlp_seq(dec.adaLN_modulation)
        self.blocks = nn.ModuleList([G.OctreeBlockGPU(b.float(), chunk) for b in dec.blocks])
        self.out = G._linear(dec.out_proj.weight, dec.out_proj.bias)

    def forward(self, cells, cells_pe, l_freq, latent):
        h = self.inp(self.in_proj(cells) + cells_pe)
        mod = self.ada(self.l_mlp(l_freq))
        for blk in self.blocks:
            h = blk(h, mod, latent)
        return self.out(G.safe_layer_norm(h, 1e-5))


class Gauss(nn.Module):
    """Blocks [lo, hi) of the Gaussian decoder. first: (anchors [1, A, 3], anchors_pe [1, A, 1024], latent) -> h;
    others: (h, latent) -> h; the last ends in the output projection -> features [1, A, 480]."""

    def __init__(self, dgs, lo, hi, chunk):
        super().__init__()
        self.first, self.last = lo == 0, hi == len(dgs.blocks)
        if self.first:
            self.in_proj = G._linear(dgs.in_proj.weight, dgs.in_proj.bias)
            self.inp = G._linear(dgs.input_layer.weight, dgs.input_layer.bias)
        self.blocks = nn.ModuleList([G.GaussianBlockGPU(dgs.blocks[i].float(), chunk) for i in range(lo, hi)])
        if self.last:
            self.out = G._linear(dgs.out_proj.weight, dgs.out_proj.bias)

    def forward(self, *args):
        if self.first:
            anchors, anchors_pe, latent = args
            h = self.inp(self.in_proj(anchors) + anchors_pe)
        else:
            h, latent = args
        for blk in self.blocks:
            h = blk(h, latent)
        return self.out(G.safe_layer_norm(h, 1e-5)) if self.last else h


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Dino(nn.Module):
    """DINOv3 layers [lo, hi). first: (pixels [1, 3, 1024, 1024] in [0, 1], cos, sin) -> h [1, 4101, 1280];
    others: (h, cos, sin) -> h; the last ends in the model norm and idol's F.layer_norm -> feature1."""

    def __init__(self, model, lo, hi, chunk):
        super().__init__()
        self.first, self.last = lo == 0, hi == len(model.model.layer)
        emb = model.embeddings
        if self.first:
            self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer('inv_std', 1.0 / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
            pe = emb.patch_embeddings
            self.patch = nn.Conv2d(3, pe.out_channels, pe.kernel_size, pe.stride, bias=True)
            self.patch.weight.data = pe.weight.detach().float().clone()
            self.patch.bias.data = pe.bias.detach().float().clone()
            self.cls = nn.Parameter(emb.cls_token.detach().float().clone(), requires_grad=False)
            self.reg = nn.Parameter(emb.register_tokens.detach().float().clone(), requires_grad=False)
        self.layers = nn.ModuleList([G.DinoBlockGPU(model.model.layer[i].float(), 20, chunk) for i in range(lo, hi)])
        if self.last:
            self.nw = nn.Parameter(model.norm.weight.detach().float().clone(), requires_grad=False)
            self.nb = nn.Parameter(model.norm.bias.detach().float().clone(), requires_grad=False)
            self.neps = model.norm.eps

    def forward(self, x, cos, sin):
        if self.first:
            p = self.patch((x - self.mean) * self.inv_std)            # [1, 1280, 64, 64]
            p = p.flatten(2).transpose(1, 2)
            x = torch.cat([self.cls, self.reg, p], dim=1)
        for layer in self.layers:
            x = layer(x, cos, sin)
        if self.last:
            x = G.safe_layer_norm(x, self.neps, self.nw, self.nb)
            x = G.safe_layer_norm(x, 1e-5)
        return x


# -- FLUX.2 VAE encoder ----------------------------------------------------------------------------------------

class PooledGroupNorm(nn.Module):
    """GroupNorm without moving channels into the batch: per-channel spatial means and abs-maxes by a hierarchy of
    pools (no reduction over more than 1024 values at once), group statistics on [1, C] vectors, and the centred map
    divided by its per-group abs-max before squaring so the fp16 squares cannot overflow. Exact in real arithmetic.

    The abs-max goes through a max pool first (exact: a max of maxes): one REDUCE_MAX over the 1024 x 1024 map of
    the VAE's first stage made each call of that graph 8 s on the Galaxy S26 GPU (vae_1 at 512 x 512: 1.6 s)."""

    def __init__(self, gn, hw):
        super().__init__()
        self.g, self.c, self.eps = gn.num_groups, gn.num_channels, gn.eps
        self.w = nn.Parameter(gn.weight.detach().float().clone().view(1, -1, 1, 1), requires_grad=False)
        self.b = nn.Parameter(gn.bias.detach().float().clone().view(1, -1, 1, 1), requires_grad=False)
        self.k = max(1, hw // 32)
        # group -> channel expansion as a one-hot FULLY_CONNECTED (an expand or a broadcast MUL by ones becomes
        # BROADCAST_TO, which has no GPU op); exact: each output is 1 * one group value
        self.expand = nn.Linear(self.g, self.c, bias=False)
        self.expand.weight.data = (torch.arange(self.c)[:, None] // (self.c // self.g) ==
                                   torch.arange(self.g)[None, :]).float()

    def _chan_mean(self, x):
        if self.k > 1:
            x = F.avg_pool2d(x, self.k)
        return x.mean(dim=(2, 3), keepdim=True)                         # [1, C, 1, 1]

    def _group(self, v):                                                # [1, C, 1, 1] -> group mean, per channel
        g = v.reshape(1, self.g, self.c // self.g).mean(-1)
        return self.expand(g).reshape(1, self.c, 1, 1)

    def forward(self, x):
        d = x - self._group(self._chan_mean(x))
        m = d.abs()
        if self.k > 1:
            m = F.max_pool2d(m, self.k)
        m = m.amax(dim=(2, 3), keepdim=True)
        s = m.reshape(1, self.g, self.c // self.g).amax(-1)
        s = self.expand(s).reshape(1, self.c, 1, 1).clamp(min=1e-2)
        ds = d / s
        var = self._group(self._chan_mean(ds * ds))
        return ds * torch.rsqrt(var + self.eps / (s * s)) * self.w + self.b


def _patch_gn(module, hw):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, PooledGroupNorm(child, hw))
        else:
            _patch_gn(child, hw)


class VaeAttn(nn.Module):
    """diffusers Attention of the VAE mid block (one head, residual, rescale 1) with chunked rank-4 attention."""

    def __init__(self, attn, chunk, hw):
        super().__init__()
        self.gn = PooledGroupNorm(attn.group_norm, hw)
        self.q, self.k, self.v = [G._linear(m.weight, m.bias) for m in (attn.to_q, attn.to_k, attn.to_v)]
        self.o = G._linear(attn.to_out[0].weight, attn.to_out[0].bias)
        self.heads, self.scale, self.chunk = attn.heads, attn.scale, chunk
        self.residual, self.rescale = attn.residual_connection, attn.rescale_output_factor

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


class Vae(nn.Module):
    """FLUX.2 VAE encoder, part p:
    0: x in [0, 1] [1, 3, 1024, 1024] -> conv_in, down 0 -> [1, 128, 512, 512];  1: down 1 -> [1, 256, 256, 256];
    2: down 2 -> [1, 512, 128, 128];  3: down 3, mid resnet 0 -> [1, 512, 128, 128];
    'attn': the mid block's attention (one head over 16,384 tokens, its own graph to keep each GPU call short);
    4: mid resnet 1, norm, SiLU, conv_out, quant_conv -> moments [1, 64, 128, 128]."""

    SIZES = {0: 1024, 1: 512, 2: 256, 3: 128, 'attn': 128, 4: 128}
    PARTS = (0, 1, 2, 3, 'attn', 4)

    def __init__(self, vae, part, chunk):
        super().__init__()
        enc = vae.encoder
        self.part = part
        hw = self.SIZES[part]
        if part in (0, 1, 2, 3):
            self.down = enc.down_blocks[part]
            _patch_gn(self.down, hw)
        if part == 0:
            self.conv_in = enc.conv_in
        mid = enc.mid_block
        if part == 3:
            self.res = mid.resnets[0]
            _patch_gn(self.res, hw)
        if part == 'attn':
            self.attn = VaeAttn(mid.attentions[0], chunk, hw)
        if part == 4:
            self.res = mid.resnets[1]
            _patch_gn(self.res, hw)
            self.norm_out = PooledGroupNorm(enc.conv_norm_out, hw)
            self.conv_out = enc.conv_out
            self.quant_conv = vae.quant_conv

    def forward(self, x):
        p = self.part
        if p == 0:
            x = self.conv_in(x * 2 - 1)
        if p in (0, 1, 2, 3):
            x = self.down(x)
        if p == 3:
            x = self.res(x, None)
        if p == 'attn':
            x = self.attn(x)
        if p == 4:
            x = self.res(x, None)
            x = self.quant_conv(self.conv_out(F.silu(self.norm_out(x))))
        return x


# -- host-side features (float32 on the CPU; the Kotlin glue computes the same) ----------------------------------

def sinusoid(value, dim=256, max_period=10000.0, two_pi=False):
    """SinusoidalEmbedder's frequency features of one scalar -> [1, dim] float32."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
    args = torch.tensor([float(value)], dtype=torch.float32)[:, None] * freqs[None]
    if two_pi:
        args = args * 2 * torch.pi
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def point_pe(points, channels=1024, max_res=10):
    """PcdAbsolutePositionEmbedderv2 of points [N, 3] in [0, 1] -> [N, 1024] float32."""
    freq_dim = channels // 3 // 2
    freqs = torch.pow(2.0, torch.linspace(0.0, float(max_res), steps=freq_dim, dtype=torch.float32))
    ang = points.float().unsqueeze(-1) * freqs * torch.pi
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).reshape(points.shape[0], -1)
    return torch.cat([emb, torch.zeros(points.shape[0], channels - emb.shape[1])], dim=-1)


# ---------------------------------------------------------------- shared middle joint block (state lever, round 2)

# The weight inputs of the shared joint graph after (h, t_mod, gate), in input order args_3 .. args_32. Linear weights
# are passed transposed ([1, in, out]), so each projection is one BATCH_MATMUL with a runtime operand.
SHARED_WEIGHT_NAMES = ['repo_norm_w', 'repo_norm_b', 'repo_gate_t', 'repo_content_t', 'repo_ang_t',
                       'q_t', 'q_b', 'k_t', 'k_b', 'v_t', 'v_b', 'gq', 'gk', 'out_t', 'out_b', 'fc1_t', 'fc1_b',
                       'fc2_t', 'fc2_b', 'shift', 'lqkv_a_t', 'lq_b_t', 'lk_b_t', 'lv_b_t', 'lout_a_t', 'lout_b_t',
                       'lfc1_a_t', 'lfc1_b_t', 'lfc2_a_t', 'lfc2_b_t']


def shared_block_weights(jb):
    """A middle DenJoint's weights in SHARED_WEIGHT_NAMES order (float32 tensors, the shapes of the graph inputs)."""
    def tw(lin):
        return lin.weight.detach().float().t().contiguous()[None]

    def row(v):
        return v.detach().float()[None, None]

    b = jb.block
    r = b.repo
    return [row(r.norm_w), row(r.norm_b), tw(r.gate), tw(r.content), tw(r.ang),
            tw(b.q), row(b.q.bias), tw(b.k), row(b.k.bias), tw(b.v), row(b.v.bias), b.gq.detach().float(),
            b.gk.detach().float(), tw(b.out), row(b.out.bias), tw(b.fc1), row(b.fc1.bias), tw(b.fc2), row(b.fc2.bias),
            b.shift_table.detach().float(), tw(b.qkv_a), tw(b.q_b), tw(b.k_b), tw(b.v_b), tw(b.out_a), tw(b.out_b),
            tw(b.fc1_a), tw(b.fc1_b), tw(b.fc2_a), tw(b.fc2_b)]


class DenJointShared(nn.Module):
    """One graph for every middle joint block: (h [1, L, 1024], t_mod [1, 6144], gate [1, 64], *weights) -> h, the
    block's weights as runtime inputs (shared_block_weights order). Same arithmetic as DenBlock.forward; compiled once
    per photo on the phone instead of once per block."""

    def __init__(self, ref_joint):
        super().__init__()
        b = ref_joint.block
        self.c, self.h, self.d, self.chunk = b.c, b.h, b.d, b.chunk
        self.eps1, self.eps2, self.repo_eps, self.repo_half = b.eps1, b.eps2, b.repo.eps, b.repo.half

    def forward(self, x, mod, gate, *w):
        W = dict(zip(SHARED_WEIGHT_NAMES, w))
        c, H, d = self.c, self.h, self.d
        bsz, L, _ = x.shape
        mm = torch.matmul
        g = gate[:, None, :]
        hr = G.safe_layer_norm(x, self.repo_eps, W['repo_norm_w'], W['repo_norm_b'])
        a = mm(F.silu(mm(hr, W['repo_gate_t'])) * mm(hr, W['repo_content_t']), W['repo_ang_t'])
        a = a.reshape(bsz, L, H, self.repo_half)
        cos, sin = torch.cos(a), torch.sin(a)
        m = mod + W['shift']
        shift_msa, scale_msa, gate_msa = m[:, 0:c], m[:, c:2 * c], m[:, 2 * c:3 * c]
        shift_mlp, scale_mlp, gate_mlp = m[:, 3 * c:4 * c], m[:, 4 * c:5 * c], m[:, 5 * c:6 * c]
        hh = G.safe_layer_norm(x, self.eps1) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        half = d // 2

        def rope(t):
            t0, t1 = t[..., :half], t[..., half:]
            return torch.cat([t0 * cos - t1 * sin, t0 * sin + t1 * cos], dim=-1)

        ax = mm(hh, W['lqkv_a_t']) * g
        q = G.safe_rms_unit(rope((mm(hh, W['q_t']) + W['q_b'] + mm(ax, W['lq_b_t'])).reshape(bsz, L, H, d))) * W['gq']
        k = G.safe_rms_unit(rope((mm(hh, W['k_t']) + W['k_b'] + mm(ax, W['lk_b_t'])).reshape(bsz, L, H, d))) * W['gk']
        v = (mm(hh, W['v_t']) + W['v_b'] + mm(ax, W['lv_b_t'])).reshape(bsz, L, H, d)
        o = G.attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), self.chunk)
        o = o.reshape(bsz, L, c)
        x = x + (mm(o, W['out_t']) + W['out_b'] + mm(mm(o, W['lout_a_t']) * g, W['lout_b_t'])) * gate_msa.unsqueeze(1)
        hh = G.safe_layer_norm(x, self.eps2) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        u = G._gelu_tanh(mm(hh, W['fc1_t']) + W['fc1_b'] + mm(mm(hh, W['lfc1_a_t']) * g, W['lfc1_b_t']))
        return x + (mm(u, W['fc2_t']) + W['fc2_b'] + mm(mm(u, W['lfc2_a_t']) * g, W['lfc2_b_t'])) * gate_mlp.unsqueeze(1)


# ---------------------------------------------------------------- shared middle Gaussian decoder graph (state lever)

# One GaussianBlockGPU's weights as graph inputs, in input order: self-attention, cross-attention, norm2's affine, MLP.
# Linear weights are passed transposed ([1, in, out]), as for the shared joint block.
GAUSS_BLOCK_WEIGHT_NAMES = ['sa_q_t', 'sa_q_b', 'sa_k_t', 'sa_k_b', 'sa_v_t', 'sa_v_b', 'sa_gq', 'sa_gk', 'sa_out_t',
                            'sa_out_b', 'ca_q_t', 'ca_q_b', 'ca_k_t', 'ca_k_b', 'ca_v_t', 'ca_v_b', 'ca_gq', 'ca_gk',
                            'ca_out_t', 'ca_out_b', 'n2_w', 'n2_b', 'fc1_t', 'fc1_b', 'fc2_t', 'fc2_b']


def gauss_block_weights(gb):
    """One GaussianBlockGPU's weights in GAUSS_BLOCK_WEIGHT_NAMES order (float32 tensors, the graph input shapes)."""
    def tw(lin):
        return lin.weight.detach().float().t().contiguous()[None]

    def row(v):
        return v.detach().float().reshape(1, 1, -1)

    def mha(m):
        return [tw(m.q), row(m.q.bias), tw(m.k), row(m.k.bias), tw(m.v), row(m.v.bias), m.gq.detach().float(),
                m.gk.detach().float(), tw(m.out), row(m.out.bias)]

    assert gb.n2w is not None and gb.n2b is not None
    return (mha(gb.self_attn) + mha(gb.cross_attn) +
            [row(gb.n2w), row(gb.n2b), tw(gb.fc1), row(gb.fc1.bias), tw(gb.fc2), row(gb.fc2.bias)])


def gauss_shared_weights(g):
    """A middle Gauss graph's weights: its blocks' gauss_block_weights one after another."""
    return [t for b in g.blocks for t in gauss_block_weights(b)]


class GaussShared(nn.Module):
    """One graph for the Gaussian decoder's middle graphs (gs_1 .. gs_6, two blocks each): (h [1, A, 1024],
    latent [1, 8192, 16], *weights) -> h, the blocks' weights as runtime inputs (gauss_shared_weights order). Same
    arithmetic as Gauss.forward for a middle graph; compiled once per photo on the phone instead of six times."""

    def __init__(self, ref):
        super().__init__()
        assert not ref.first and not ref.last
        b0 = ref.blocks[0]
        self.n = len(ref.blocks)
        self.eps = (b0.e1, b0.e2, b0.e3)
        self.sa = (b0.self_attn.h, b0.self_attn.c, b0.self_attn.d, b0.self_attn.chunk)
        self.ca = (b0.cross_attn.h, b0.cross_attn.c, b0.cross_attn.d, b0.cross_attn.chunk)

    @staticmethod
    def _mha(x, src, W, p, heads, c, d, chunk):
        mm = torch.matmul
        b, lq, _ = x.shape
        lk = src.shape[1]
        q = G.safe_rms_unit((mm(x, W[p + 'q_t']) + W[p + 'q_b']).reshape(b, lq, heads, d)) * W[p + 'gq']
        k = G.safe_rms_unit((mm(src, W[p + 'k_t']) + W[p + 'k_b']).reshape(b, lk, heads, d)) * W[p + 'gk']
        v = (mm(src, W[p + 'v_t']) + W[p + 'v_b']).reshape(b, lk, heads, d)
        o = G.attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), chunk)
        return mm(o.reshape(b, lq, c), W[p + 'out_t']) + W[p + 'out_b']

    def forward(self, h, latent, *w):
        n = len(GAUSS_BLOCK_WEIGHT_NAMES)
        assert len(w) == n * self.n, len(w)
        e1, e2, e3 = self.eps
        for i in range(self.n):
            W = dict(zip(GAUSS_BLOCK_WEIGHT_NAMES, w[i * n:(i + 1) * n]))
            x = G.safe_layer_norm(h, e1)
            h = h + self._mha(x, x, W, 'sa_', *self.sa)
            h = h + self._mha(G.safe_layer_norm(h, e2, W['n2_w'], W['n2_b']), latent, W, 'ca_', *self.ca)
            u = G._gelu_tanh(torch.matmul(G.safe_layer_norm(h, e3), W['fc1_t']) + W['fc1_b'])
            h = h + (torch.matmul(u, W['fc2_t']) + W['fc2_b'])
        return h
