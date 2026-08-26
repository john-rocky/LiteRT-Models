#!/usr/bin/env python3
"""Pocket TTS (Kyutai, 100M) -> LiteRT CompiledModel graphs + parity.

Pocket TTS is a flow-matching LM over continuous 32-dim Mimi latents:
per 12.5 Hz frame a 6-layer/1024-wide causal transformer conditions a
6-block AdaLN MLP flow head that turns one Gaussian noise draw into the
next latent (LSD, 1 step); a 20M tiny Mimi (2-layer transformer + SEANet,
ratios 6*5*4, x16 upsample) decodes latents to 24 kHz audio.

Graphs (all fixed-shape, stateless; KV caches and streaming state live on
the host, following the dia2/vibevoice packed-KV pattern):

  pt_flowlm_step   emb[1,1,1024] + cos/sin[1,1,1,64] + mask[1,16,1,PMAX+1]
                   + pk/pv[1,96,PMAX,64] -> cond[1,1024], eos[1,1],
                   nk/nv[1,96,1,64].  RoPE pairs de-interleaved by baking the
                   [even|odd] row permutation into in_proj (bit-exact:
                   q,k share the permutation so q.k is unchanged).
  pt_flow_head     cond[1,1024] + noise[1,32] -> latent[1,32].  The two LSD
                   time embeddings (s=0, t=1) are constants and are baked
                   into cond_embed's bias; output includes the +noise.
  pt_mimi_dec_tx   lat[1,65,32] -> feat[1,512,1024]. denorm + 1x1 proj +
                   x16 ZeroStuffConvT1d + 2-layer transformer (context 250,
                   banded const bias). Slot 0 is the previous frame (the x16
                   ConvT kernel reaches one frame back); blocks slide by 32
                   frames and keep the right 512 positions, exact because the
                   2-layer stacked window reaches 2*249 = 498 < 512 back.
  pt_mimi_deconly  feat[1,512,4096] -> audio[1,1,491520]. SEANet decoder,
                   one-shot 256-frame window; causal => exact per frame
                   (dia2 t256 precedent).

Host-side assets: token-embedding table (fp16), input_linear, bos vector,
preset voice states repacked (k de-interleaved) as fp16 blobs.

Numerics: the only non-exact rewrite is erf-GELU -> fitted tanh-polynomial
(|gelu err| <= 7.1e-5, measured below); everything else is bit-exact.

Run:  PYTHONPATH=<pocket-tts clone> python build_pockettts.py [stage]
      stage in {flowlm, head, dectx, deconly, assets, pipeline, all}
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("PT_OUT", os.path.join(HERE, "out"))
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

# ---------------------------------------------------------------- constants
D_MODEL = 1024
N_LAYERS = 6
N_HEADS = 16
HD = 64
FFN = 4096
LDIM = 32
PMAX = 512            # flow-LM KV capacity: voice (~142) + text (~55) + gen (~235)
FLOW_DIM = 512
FLOW_DEPTH = 6

MIMI_D = 512
MIMI_HEADS = 8
MIMI_HD = 64
MIMI_LAYERS = 2
MIMI_CTX = 250
UPS = 16              # 12.5 Hz -> 200 Hz
SPF = 1920            # samples per 12.5 Hz frame

# dec_tx block: the 2-layer sliding-window (250) transformer has a stacked
# receptive field of 2*249 = 498 positions, so kept queries must sit >= 498
# positions into the block: 64-frame blocks (1024 positions), keep the right
# 32 frames (512 positions >= 498).
F_BLK = 64
S_BLK = F_BLK * UPS
F_HOP = 32
DEC_FRAMES = 256      # deconly window
S_DEC = DEC_FRAMES * UPS

BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP",
          "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM"}

MASK_NEG = -1e4       # fp16-safe additive mask


# ------------------------------------------------------------------ helpers
def corr(a, b):
    a = np.asarray(a).ravel().astype(np.float64)
    b = np.asarray(b).ravel().astype(np.float64)
    n = min(a.size, b.size)
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def maxd(a, b):
    a = np.asarray(a).ravel().astype(np.float64)
    b = np.asarray(b).ravel().astype(np.float64)
    return float(np.abs(a - b).max())


def convert(mod, example_inputs, out):
    import litert_torch
    litert_torch.convert(mod.eval(), example_inputs).export(out)
    print(f"exported {out} ({os.path.getsize(out)/1e6:.1f} MB)")
    return out


def to_fp16(fp32_path, fp16_path):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT),
        algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16_path):
        os.remove(fp16_path)
    qt = quantizer.Quantizer(float_model=fp32_path)
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(fp16_path)
    print(f"exported {fp16_path} ({os.path.getsize(fp16_path)/1e6:.1f} MB)")
    return fp16_path


def opcheck(path, label):
    import collections
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    ops = collections.Counter(d.get("op_name", "?") for d in it._get_ops_details())
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    over = sum(1 for d in it.get_tensor_details() if len(d.get("shape", [])) > 4)
    print(f"[{label}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} "
          f"size {os.path.getsize(path)/1e6:.1f}MB "
          f"VERDICT: {'GPU-CLEAN' if not bad and not over else 'BLOCKERS'}")
    return not bad and not over


class CM:
    """ai_edge_litert CompiledModel wrapper (CPU on host) with named-order I/O."""

    def __init__(self, path):
        from ai_edge_litert.compiled_model import CompiledModel
        self.m = CompiledModel.from_file(path)
        self.inb = self.m.create_input_buffers(0)
        self.outb = self.m.create_output_buffers(0)
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=path)
        self.in_shapes = [tuple(d["shape"]) for d in it.get_input_details()]
        self.out_shapes = [tuple(d["shape"]) for d in it.get_output_details()]

    def __call__(self, *arrays):
        for buf, a in zip(self.inb, arrays):
            buf.write(np.ascontiguousarray(a, dtype=np.float32).ravel())
        self.m.run_by_index(0, self.inb, self.outb)
        outs = []
        for buf, shp in zip(self.outb, self.out_shapes):
            outs.append(np.array(buf.read(int(np.prod(shp)), np.float32)).reshape(shp))
        return outs


def rope_cos_sin_deint(pos):
    """cos/sin (each [64]) for one absolute position, de-interleaved [c|c],[s|s]."""
    inv = 10000.0 ** (-np.arange(32, dtype=np.float64) / 32.0)
    ang = pos * inv
    c = np.cos(ang).astype(np.float32)
    s = np.sin(ang).astype(np.float32)
    return np.concatenate([c, c]), np.concatenate([s, s])


class ErfGELU(nn.Module):
    """erf-GELU via a fitted odd tanh-polynomial (MUL/ADD/TANH/RELU only):
    erf(z) ~ tanh(z(c1 + z^2(c3 + z^2(c5 + z^2 c7)))), max |gelu err| 7.1e-5
    over |x| <= 8 -- ~15x closer to erf than the classic tanh-GELU. z is
    clamped to +-5.5 (erf there is 1 - 4e-8) so the fp16 Horner never sees
    large powers."""
    INV_SQRT2 = 0.7071067811865476
    C1 = 1.1280827604e+00
    C3 = 1.0434222081e-01
    C5 = -1.9996018773e-03
    C7 = 4.5717509263e-05
    CLAMP = 5.5

    def forward(self, x):
        z = x * self.INV_SQRT2
        z = z - torch.relu(z - self.CLAMP) + torch.relu(-z - self.CLAMP)
        z2 = z * z
        p = z * (self.C1 + z2 * (self.C3 + z2 * (self.C5 + z2 * self.C7)))
        return 0.5 * x * (1.0 + torch.tanh(p))


class CleanELU(nn.Module):
    """ELU(x) = relu(x) - relu(1 - exp(min(x,0))); SELECT-free, exact."""

    def forward(self, x):
        xm = -torch.relu(-x)
        return torch.relu(x) - torch.relu(1.0 - torch.exp(xm))


class ZeroStuffConvT1d(nn.Module):
    """Exact ConvTranspose1d via 2D nearest upsample x const zero-stuff mask +
    flipped (grouped) conv1d, then a crop reproducing StreamingConvTranspose1d's
    fresh-state one-shot output (full conv trimmed by kernel-stride)."""

    def __init__(self, ct, L):
        super().__init__()
        self.s = ct.stride[0]
        self.k = ct.kernel_size[0]
        self.g = ct.groups
        self.L = L
        cin, cout = ct.in_channels, ct.out_channels
        w = ct.weight.detach()
        w = (w.view(self.g, cin // self.g, cout // self.g, self.k)
             .permute(0, 2, 1, 3).reshape(cout, cin // self.g, self.k))
        self.register_buffer("w", w.flip(2).contiguous())
        self.register_buffer("b", ct.bias.detach().clone() if ct.bias is not None
                             else torch.zeros(cout))
        mk = np.zeros((L * self.s,), np.float32)
        mk[::self.s] = 1.0
        self.register_buffer("mask", torch.from_numpy(mk)[None, None])

    def forward(self, x):
        xn = F.interpolate(x.unsqueeze(2), size=(1, self.L * self.s),
                           mode="nearest").squeeze(2) * self.mask
        y = F.conv1d(xn, self.w, bias=self.b, padding=self.k - 1, groups=self.g)
        # full ConvT output = (L-1)*s + k; streaming one-shot trims the k-s tail
        return y[:, :, :self.L * self.s]


def load_eager():
    from pocket_tts import TTSModel
    model = TTSModel.load_model()
    model.eval()
    return model


def deint_perm():
    """Per-head permutation putting even (real) dims first: [0,2,..,62,1,3,..,63]."""
    return torch.cat([torch.arange(0, HD, 2), torch.arange(1, HD, 2)])


# ================================================================ flow-LM step
class FlowLMStep(nn.Module):
    """One AR step of the pocket-tts flow-LM backbone with packed-KV I/O."""

    def __init__(self, flow_lm):
        super().__init__()
        perm = deint_perm()
        self.layers = nn.ModuleList()
        for lyr in flow_lm.transformer.layers:
            m = nn.Module()
            w = lyr.self_attn.in_proj.weight.detach().clone()   # [3072, 1024]
            w3 = w.view(3, N_HEADS, HD, D_MODEL)
            w3[0] = w3[0][:, perm]                              # q rows
            w3[1] = w3[1][:, perm]                              # k rows
            m.in_w = nn.Parameter(w3.reshape(3 * D_MODEL, D_MODEL))
            m.out_w = nn.Parameter(lyr.self_attn.out_proj.weight.detach().clone())
            m.n1_w = nn.Parameter(lyr.norm1.weight.detach().clone())
            m.n1_b = nn.Parameter(lyr.norm1.bias.detach().clone())
            m.n2_w = nn.Parameter(lyr.norm2.weight.detach().clone())
            m.n2_b = nn.Parameter(lyr.norm2.bias.detach().clone())
            m.l1_w = nn.Parameter(lyr.linear1.weight.detach().clone())
            m.l2_w = nn.Parameter(lyr.linear2.weight.detach().clone())
            self.layers.append(m)
        self.on_w = nn.Parameter(flow_lm.out_norm.weight.detach().clone())
        self.on_b = nn.Parameter(flow_lm.out_norm.bias.detach().clone())
        self.eos_w = nn.Parameter(flow_lm.out_eos.weight.detach().clone())
        self.eos_b = nn.Parameter(flow_lm.out_eos.bias.detach().clone())
        self.gelu = ErfGELU()
        self.eps = 1e-5

    def ln(self, x, w, b):
        mu = x.mean(-1, keepdim=True)
        d = x - mu
        var = (d * d).mean(-1, keepdim=True)
        return d * torch.rsqrt(var + self.eps) * w + b

    def forward(self, x, cos, sin, mask, pk, pv):
        # x[1,1,1024] cos/sin[1,1,1,64] mask[1,16,1,PMAX+1] pk/pv[1,96,PMAX,64]
        nk, nv = [], []
        scale = 1.0 / math.sqrt(HD)
        for i, m in enumerate(self.layers):
            h = self.ln(x, m.n1_w, m.n1_b)
            proj = F.linear(h, m.in_w)                              # [1,1,3072]
            # slice thirds, not view(...,3,H,D): the GPU's maximum tensor rank is 4
            q = proj[..., :D_MODEL].view(1, 1, N_HEADS, HD)
            k = proj[..., D_MODEL:2 * D_MODEL].view(1, 1, N_HEADS, HD)
            v = proj[..., 2 * D_MODEL:].view(1, 1, N_HEADS, HD)
            qh = q.transpose(1, 2)                                  # [1,16,1,64]
            kh = k.transpose(1, 2)
            vh = v.transpose(1, 2)

            def rot(t):
                a, b = torch.chunk(t, 2, dim=-1)
                return t * cos + torch.cat([-b, a], dim=-1) * sin

            qh = rot(qh)
            kh = rot(kh)
            base = i * N_HEADS
            k_all = torch.cat([pk[:, base:base + N_HEADS], kh], dim=2)  # [1,16,P+1,64]
            v_all = torch.cat([pv[:, base:base + N_HEADS], vh], dim=2)
            scores = torch.matmul(qh, k_all.transpose(-1, -2)) * scale + mask
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v_all)                         # [1,16,1,64]
            ctx = ctx.transpose(1, 2).reshape(1, 1, D_MODEL)
            x = x + F.linear(ctx, m.out_w)
            h2 = self.ln(x, m.n2_w, m.n2_b)
            x = x + F.linear(self.gelu(F.linear(h2, m.l1_w)), m.l2_w)
            nk.append(kh)
            nv.append(vh)
        cond = self.ln(x, self.on_w, self.on_b)[:, 0]               # [1,1024]
        eos = F.linear(cond, self.eos_w, self.eos_b)                # [1,1]
        return cond, eos, torch.cat(nk, dim=1), torch.cat(nv, dim=1)


# ================================================================= flow head
class FlowHead(nn.Module):
    """SimpleMLPAdaLN with LSD (s=0, t=1) time embeddings baked into cond bias.
    Output includes the LSD 1-step integration: latent = noise + v."""

    def __init__(self, flow_lm):
        super().__init__()
        net = flow_lm.flow_net
        with torch.no_grad():
            z = torch.zeros(1, 1)
            o = torch.ones(1, 1)
            tconst = (net.time_embed[0](z) + net.time_embed[1](o)) / 2.0   # [1,512]
        ce = nn.Linear(D_MODEL, FLOW_DIM)
        ce.weight = nn.Parameter(net.cond_embed.weight.detach().clone())
        ce.bias = nn.Parameter(net.cond_embed.bias.detach().clone() + tconst[0])
        self.cond_embed = ce
        self.input_proj = net.input_proj
        self.res_blocks = net.res_blocks
        self.final_layer = net.final_layer
        self.eps = 1e-6

    def forward(self, cond, noise):
        # cond [1,1024], noise [1,32] -> latent [1,32]
        y = self.cond_embed(cond)
        x = self.input_proj(noise)
        for blk in self.res_blocks:
            mod = blk.adaLN_modulation(y)
            shift, scale, gate = torch.chunk(mod, 3, dim=-1)
            h = blk.in_ln(x) * (1 + scale) + shift
            x = x + gate * blk.mlp(h)
        mod = self.final_layer.adaLN_modulation(y)
        shift, scale = torch.chunk(mod, 2, dim=-1)
        h = self.final_layer.norm_final(x) * (1 + scale) + shift
        return noise + self.final_layer.linear(h)


class FusedStep(nn.Module):
    """flow-LM step + flow head in ONE graph with ONE output tensor.

    On Mali the per-step cost is dispatch/sync-bound, not FLOP-bound: two
    CompiledModel invocations per frame (step + head) plus four separate
    output readbacks cost more than the math. This variant runs the whole
    frame in one invocation and concatenates every result into a single
    [1, 1+32+6144+6144] vector: eos | latent | new-k | new-v. During text
    prompting the host feeds zero noise and ignores the latent slot.
    """

    def __init__(self, flow_lm):
        super().__init__()
        self.step = FlowLMStep(flow_lm)
        self.head = FlowHead(flow_lm)

    def forward(self, x, cos, sin, mask, pk, pv, noise):
        cond, eos, nk, nv = self.step(x, cos, sin, mask, pk, pv)
        lat = self.head(cond, noise)
        return torch.cat(
            [eos, lat, nk.reshape(1, -1), nv.reshape(1, -1)], dim=-1)


def stage_fused(model):
    print("\n=== fused step graph (step + head, single output) ===")
    flm = model.flow_lm
    fused = FusedStep(flm).eval()
    step = fused.step
    head = fused.head

    ks, vs, off0 = load_voice_state("alba")
    pk, pv = pack_voice(ks, vs, off0)
    emb_w = flm.conditioner.embed.weight.detach()
    in_w = flm.input_linear.weight.detach()
    bos_in = (flm.bos_emb.detach() @ in_w.T)

    # teacher-forced 12-step free-run vs the separate modules
    torch.manual_seed(3)
    noises = [torch.randn(1, LDIM) * math.sqrt(0.3) for _ in range(12)]
    off_a = off0
    pk_a, pv_a = pk.clone(), pv.clone()
    lat_sep = []
    x_in = bos_in.view(1, 1, -1)
    with torch.no_grad():
        for i in range(12):
            c, s = rope_cos_sin_deint(off_a)
            cond, eos, nk, nv = step(x_in, torch.from_numpy(c).view(1, 1, 1, HD),
                                     torch.from_numpy(s).view(1, 1, 1, HD),
                                     torch.from_numpy(make_mask(off_a)), pk_a, pv_a)
            pk_a[0, :, off_a] = nk[0, :, 0]
            pv_a[0, :, off_a] = nv[0, :, 0]
            off_a += 1
            lat = head(cond, noises[i])
            lat_sep.append(lat)
            x_in = (lat[0] @ in_w.T).view(1, 1, -1)
    off_b = off0
    pk_b, pv_b = pk.clone(), pv.clone()
    lat_fused = []
    x_in = bos_in.view(1, 1, -1)
    with torch.no_grad():
        for i in range(12):
            c, s = rope_cos_sin_deint(off_b)
            out = fused(x_in, torch.from_numpy(c).view(1, 1, 1, HD),
                        torch.from_numpy(s).view(1, 1, 1, HD),
                        torch.from_numpy(make_mask(off_b)), pk_b, pv_b, noises[i])
            lat = out[:, 1:1 + LDIM]
            nk = out[:, 1 + LDIM:1 + LDIM + G_KV].reshape(1, N_LAYERS * N_HEADS, 1, HD)
            nv = out[:, 1 + LDIM + G_KV:].reshape(1, N_LAYERS * N_HEADS, 1, HD)
            pk_b[0, :, off_b] = nk[0, :, 0]
            pv_b[0, :, off_b] = nv[0, :, 0]
            off_b += 1
            lat_fused.append(lat)
            x_in = (lat[0] @ in_w.T).view(1, 1, -1)
    d = max(maxd(a.numpy(), b.numpy()) for a, b in zip(lat_sep, lat_fused))
    print(f"fused vs separate modules over 12 free-run steps: max|d| {d:.2e}")

    example = (torch.zeros(1, 1, D_MODEL), torch.zeros(1, 1, 1, HD),
               torch.zeros(1, 1, 1, HD), torch.zeros(1, N_HEADS, 1, PMAX + 1),
               torch.zeros(1, N_LAYERS * N_HEADS, PMAX, HD),
               torch.zeros(1, N_LAYERS * N_HEADS, PMAX, HD),
               torch.zeros(1, LDIM))
    p = convert(fused, example, os.path.join(OUT, "pt_flowlm_fused.tflite"))
    opcheck(p, "flowlm_fused")
    to_fp16(p, os.path.join(OUT, "pt_flowlm_fused_fp16.tflite"))
    opcheck(os.path.join(OUT, "pt_flowlm_fused_fp16.tflite"), "flowlm_fused_fp16")

    cm = CM(p)
    c, s = rope_cos_sin_deint(off0)
    with torch.no_grad():
        ref = fused(bos_in.view(1, 1, -1), torch.from_numpy(c).view(1, 1, 1, HD),
                    torch.from_numpy(s).view(1, 1, 1, HD),
                    torch.from_numpy(make_mask(off0)), pk, pv, noises[0])
    outs = cm(bos_in.view(1, 1, -1).numpy(), c.reshape(1, 1, 1, HD),
              s.reshape(1, 1, 1, HD), make_mask(off0), pk.numpy(), pv.numpy(),
              noises[0].numpy())
    print(f"tflite fused one-step corr {corr(outs[0], ref.numpy()):.6f} "
          f"max|d| {maxd(outs[0], ref.numpy()):.2e}")


G_KV = N_LAYERS * N_HEADS * HD


# ============================================================ mimi dec graphs
def banded_bias(seq, window, neg=MASK_NEG):
    i = torch.arange(seq)[:, None]
    j = torch.arange(seq)[None, :]
    mask = (j <= i) & (j > i - window)
    return torch.where(mask, torch.zeros(()), torch.full((), neg)).float()[None, None]


def rope_table(seq):
    inv = 10000.0 ** (-torch.arange(32, dtype=torch.float64) / 32.0)
    ang = torch.arange(seq, dtype=torch.float64)[:, None] * inv[None, :]
    c = torch.cat([ang.cos(), ang.cos()], dim=-1).float()   # de-interleaved [seq,64]
    s = torch.cat([ang.sin(), ang.sin()], dim=-1).float()
    return c[None, None], s[None, None]                     # [1,1,seq,64]


class MimiTx(nn.Module):
    """Pocket-Mimi decoder transformer, stateless fixed length, GPU-clean.
    LayerScale baked into out/fc2; RoPE de-interleaved via in_proj bake;
    banded causal const bias; SafeLayerNorm-style scaled reduction."""

    def __init__(self, ptx, seq):
        super().__init__()
        perm = deint_perm()
        cos, sin = rope_table(seq)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("bias", banded_bias(seq, MIMI_CTX))
        self.layers = nn.ModuleList()
        for lyr in ptx.transformer.layers:
            m = nn.Module()
            w = lyr.self_attn.in_proj.weight.detach().clone()
            w3 = w.view(3, MIMI_HEADS, MIMI_HD, MIMI_D)
            w3[0] = w3[0][:, perm[:MIMI_HD]]
            w3[1] = w3[1][:, perm[:MIMI_HD]]
            m.in_w = nn.Parameter(w3.reshape(3 * MIMI_D, MIMI_D))
            g1 = lyr.layer_scale_1.scale.detach()
            g2 = lyr.layer_scale_2.scale.detach()
            m.out_w = nn.Parameter(lyr.self_attn.out_proj.weight.detach() * g1[:, None])
            m.l1_w = nn.Parameter(lyr.linear1.weight.detach().clone())
            m.l2_w = nn.Parameter(lyr.linear2.weight.detach() * g2[:, None])
            m.n1_w = nn.Parameter(lyr.norm1.weight.detach().clone())
            m.n1_b = nn.Parameter(lyr.norm1.bias.detach().clone())
            m.n2_w = nn.Parameter(lyr.norm2.weight.detach().clone())
            m.n2_b = nn.Parameter(lyr.norm2.bias.detach().clone())
            self.layers.append(m)
        self.seq = seq
        self.gelu = ErfGELU()
        self.eps = 1e-5
        self.lnscale = 0.0625     # exact: LN is scale-invariant (eps scaled too)

    def ln(self, x, w, b):
        xs = x * self.lnscale
        mu = xs.mean(-1, keepdim=True)
        d = xs - mu
        var = (d * d).mean(-1, keepdim=True)
        return d * torch.rsqrt(var + self.eps * self.lnscale * self.lnscale) * w + b

    def forward(self, x):
        # x [1,seq,512]
        scale = 1.0 / math.sqrt(MIMI_HD)
        for m in self.layers:
            h = self.ln(x, m.n1_w, m.n1_b)
            proj = F.linear(h, m.in_w)            # [1,seq,1536]
            q = proj[..., :MIMI_D].view(1, self.seq, MIMI_HEADS, MIMI_HD).transpose(1, 2)
            k = proj[..., MIMI_D:2 * MIMI_D].view(1, self.seq, MIMI_HEADS, MIMI_HD).transpose(1, 2)
            v = proj[..., 2 * MIMI_D:].view(1, self.seq, MIMI_HEADS, MIMI_HD).transpose(1, 2)

            def rot(t):
                a, b = torch.chunk(t, 2, dim=-1)
                return t * self.cos + torch.cat([-b, a], dim=-1) * self.sin

            q = rot(q)
            k = rot(k)
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale + self.bias
            attn = torch.softmax(scores, dim=-1)
            ctx = attn.matmul(v).transpose(1, 2).reshape(1, self.seq, MIMI_D)
            x = x + F.linear(ctx, m.out_w)
            h2 = self.ln(x, m.n2_w, m.n2_b)
            x = x + F.linear(self.gelu(F.linear(h2, m.l1_w)), m.l2_w)
        return x


class MimiDecTx(nn.Module):
    """lat[1,1+F_BLK,32] -> denorm -> 1x1 proj -> x16 zero-stuff upsample ->
    decoder transformer -> feat[1,512,S_BLK].

    Slot 0 is the PREVIOUS latent frame (left context): the x16 ConvTranspose
    (kernel 32) reaches one frame back, so without it the first 16 positions
    of the block would be wrong -- and those positions sit inside the 250-wide
    attention window of kept queries. For the very first block the host passes
    the neutral latent -emb_mean/emb_std, whose denormalized features are
    exactly zero (= no frame). Slots 1..F_BLK are the block's frames; the
    upsampled sequence is cropped to their 512 positions."""

    def __init__(self, model):
        super().__init__()
        mimi = model.mimi
        self.register_buffer("std", model.flow_lm.emb_std.detach().view(1, 1, LDIM))
        self.register_buffer("mean", model.flow_lm.emb_mean.detach().view(1, 1, LDIM))
        self.qw = nn.Parameter(mimi.quantizer.output_proj.weight.detach().clone())  # [512,32,1]
        self.up = ZeroStuffConvT1d(mimi.upsample.convtr.convtr, F_BLK + 1)
        self.tx = MimiTx(mimi.decoder_transformer, S_BLK)

    def forward(self, lat):
        x = lat * self.std + self.mean
        x = x.transpose(1, 2)                     # [1,32,F+1]
        x = F.conv1d(x, self.qw)                  # [1,512,F+1]
        x = self.up(x)                            # [1,512,(F+1)*16]
        x = x[:, :, UPS:]                         # crop the context frame -> [1,512,S]
        x = self.tx(x.transpose(1, 2))            # [1,S,512]
        return x.transpose(1, 2)                  # [1,512,S]


class MimiDecOnly(nn.Module):
    """feat[1,512,S_DEC] -> SEANet decoder (streaming semantics baked) -> audio."""

    def __init__(self, model):
        super().__init__()
        from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
        from pocket_tts.modules.seanet import SEANetResnetBlock
        dec = model.mimi.decoder
        steps = []
        for layer in dec.model:
            if isinstance(layer, StreamingConv1d):
                conv = layer.conv
                pad = layer._effective_kernel_size - layer._stride
                steps.append(("conv", conv, pad))
            elif isinstance(layer, StreamingConvTranspose1d):
                steps.append(("convtr", layer.convtr, None))
            elif isinstance(layer, SEANetResnetBlock):
                sub = []
                for el in layer.block:
                    if isinstance(el, nn.ELU):
                        sub.append(("elu", None, None))
                    else:
                        pad = el._effective_kernel_size - el._stride
                        sub.append(("conv", el.conv, pad))
                steps.append(("res", sub, None))
            elif isinstance(layer, nn.ELU):
                steps.append(("elu", None, None))
            else:
                raise RuntimeError(f"unexpected layer {type(layer)}")
        self.mods = nn.ModuleList()
        self.plan = []
        L = S_DEC

        def add_conv(conv, pad, length):
            self.mods.append(conv)
            self.plan.append(("conv", len(self.mods) - 1, pad))
            return (length + pad - ((conv.kernel_size[0] - 1) * conv.dilation[0] + 1)) \
                // conv.stride[0] + 1

        i = 0
        flat = []
        for kind, obj, pad in steps:
            if kind == "res":
                flat.append(("res_open", None, None))
                for k2, o2, p2 in obj:
                    flat.append((k2, o2, p2))
                flat.append(("res_close", None, None))
            else:
                flat.append((kind, obj, pad))
        for kind, obj, pad in flat:
            if kind == "conv":
                L = add_conv(obj, pad, L)
            elif kind == "convtr":
                self.mods.append(ZeroStuffConvT1d(obj, L))
                self.plan.append(("zs", len(self.mods) - 1, None))
                L = L * obj.stride[0]
            elif kind == "elu":
                self.plan.append(("elu", None, None))
            elif kind == "res_open":
                self.plan.append(("res_open", None, None))
            elif kind == "res_close":
                self.plan.append(("res_close", None, None))
        self.elu = CleanELU()
        self.out_len = L

    def forward(self, x):
        stack = []
        for kind, idx, pad in self.plan:
            if kind == "conv":
                x = self.mods[idx](F.pad(x, (pad, 0)))
            elif kind == "zs":
                x = self.mods[idx](x)
            elif kind == "elu":
                x = self.elu(x)
            elif kind == "res_open":
                stack.append(x)
            elif kind == "res_close":
                x = x + stack.pop()
        return x


# ============================================================== eager tracing
def record_reference(model, voice, text, seed=1234):
    """Run the real generate_audio while recording noises, latents, conds, tokens."""
    rec = {"noise": [], "latents": [], "conds": [], "eos": [], "tokens": None}
    orig_normal = torch.nn.init.normal_
    orig_trunc = torch.nn.init.trunc_normal_

    def rec_normal(t, mean=0.0, std=1.0):
        r = orig_normal(t, mean=mean, std=std)
        rec["noise"].append(t.detach().clone())
        return r

    def rec_trunc(t, mean=0.0, std=1.0, a=-2.0, b=2.0):
        r = orig_trunc(t, mean=mean, std=std, a=a, b=b)
        rec["noise"].append(t.detach().clone())
        return r

    flm = model.flow_lm
    orig_fwd = flm.forward

    def wrap_fwd(sequence, text_embeddings, model_state, sampler_decode_steps,
                 temp, noise_clamp, eos_threshold):
        out, eos = orig_fwd(sequence, text_embeddings, model_state,
                            sampler_decode_steps, temp, noise_clamp, eos_threshold)
        rec["latents"].append(out.detach().clone())
        rec["eos"].append(bool(eos.item()))
        return out, eos

    cond = flm.conditioner
    orig_c = cond.forward

    def wrap_c(inputs):
        if rec["tokens"] is None and inputs[0].numel() > 0:
            rec["tokens"] = inputs[0].detach().clone()
        return orig_c(inputs)

    torch.manual_seed(seed)
    torch.nn.init.normal_ = rec_normal
    torch.nn.init.trunc_normal_ = rec_trunc
    flm.forward = wrap_fwd
    cond.forward = wrap_c
    try:
        state = model.get_state_for_audio_prompt(voice)
        audio = model.generate_audio(state, text)
    finally:
        torch.nn.init.normal_ = orig_normal
        torch.nn.init.trunc_normal_ = orig_trunc
        flm.forward = orig_fwd
        cond.forward = orig_c
    return rec, audio


def load_voice_state(name):
    from huggingface_hub import hf_hub_download
    import safetensors
    p = hf_hub_download("kyutai/pocket-tts-without-voice-cloning",
                        f"languages/english/embeddings/{name}.safetensors",
                        revision="e81d79e8194ad4c7ce879c87a4258ef20cbf2487")
    ks, vs, off = [], [], None
    with safetensors.safe_open(p, framework="pt") as f:
        for li in range(N_LAYERS):
            cache = f.get_tensor(f"transformer.layers.{li}.self_attn/cache")
            ks.append(cache[0, 0])      # [T,16,64]
            vs.append(cache[1, 0])
            off = int(f.get_tensor(f"transformer.layers.{li}.self_attn/offset")[0])
    return ks, vs, off


def pack_voice(ks, vs, off):
    """[T,16,64] x6 -> packed pk/pv [1,96,PMAX,64], k de-interleaved."""
    perm = deint_perm()
    pk = torch.zeros(1, N_LAYERS * N_HEADS, PMAX, HD)
    pv = torch.zeros(1, N_LAYERS * N_HEADS, PMAX, HD)
    for li in range(N_LAYERS):
        k = ks[li][:off][:, :, perm].permute(1, 0, 2)    # [16,T,64] de-interleaved
        v = vs[li][:off].permute(1, 0, 2)
        pk[0, li * N_HEADS:(li + 1) * N_HEADS, :off] = k
        pv[0, li * N_HEADS:(li + 1) * N_HEADS, :off] = v
    return pk, pv


def make_mask(off):
    m = np.full((1, N_HEADS, 1, PMAX + 1), MASK_NEG, np.float32)
    m[:, :, :, :off] = 0.0
    m[:, :, :, PMAX] = 0.0
    return m


# =================================================================== stages
def stage_flowlm(model):
    print("\n=== flow-LM step graph ===")
    flm = model.flow_lm
    step = FlowLMStep(flm).eval()

    # teacher-forced replay of a recorded reference trajectory
    rec, ref_audio = record_reference(model, "alba",
                                      "Hello world. I am Pocket TTS running on a phone.")
    tokens = rec["tokens"][0].tolist()
    print(f"tokens={len(tokens)} latents={len(rec['latents'])} noises={len(rec['noise'])}")

    ks, vs, off0 = load_voice_state("alba")
    pk, pv = pack_voice(ks, vs, off0)
    emb_w = flm.conditioner.embed.weight.detach()
    in_w = flm.input_linear.weight.detach()          # [1024,32]
    bos_in = (flm.bos_emb.detach() @ in_w.T)         # [1024]

    off = off0
    conds = []
    with torch.no_grad():
        # text prompt replay (per-step S=1)
        for t in tokens:
            c, s = rope_cos_sin_deint(off)
            cond, eos, nk, nv = step(emb_w[t].view(1, 1, -1),
                                     torch.from_numpy(c).view(1, 1, 1, HD),
                                     torch.from_numpy(s).view(1, 1, 1, HD),
                                     torch.from_numpy(make_mask(off)),
                                     pk, pv)
            for li in range(N_LAYERS):
                pk[0, li * N_HEADS:(li + 1) * N_HEADS, off] = nk[0, li * N_HEADS:(li + 1) * N_HEADS, 0]
                pv[0, li * N_HEADS:(li + 1) * N_HEADS, off] = nv[0, li * N_HEADS:(li + 1) * N_HEADS, 0]
            off += 1
        # generation replay, teacher-forced on the recorded latents
        x_in = bos_in.view(1, 1, -1)
        gen_noises = rec["noise"][1:]  # noise[0] is the text-step draw whose latent is discarded
        for gi, ref_lat in enumerate(rec["latents"][1:]):
            c, s = rope_cos_sin_deint(off)
            cond, eos, nk, nv = step(x_in,
                                     torch.from_numpy(c).view(1, 1, 1, HD),
                                     torch.from_numpy(s).view(1, 1, 1, HD),
                                     torch.from_numpy(make_mask(off)),
                                     pk, pv)
            conds.append(cond)
            for li in range(N_LAYERS):
                pk[0, li * N_HEADS:(li + 1) * N_HEADS, off] = nk[0, li * N_HEADS:(li + 1) * N_HEADS, 0]
                pv[0, li * N_HEADS:(li + 1) * N_HEADS, off] = nv[0, li * N_HEADS:(li + 1) * N_HEADS, 0]
            off += 1
            x_in = (ref_lat[0] @ in_w.T).view(1, 1, -1)  # teacher forcing

    # eager cond reference for the same trajectory: recompute via flow_lm modules
    head = FlowHead(flm).eval()
    lat_err = []
    with torch.no_grad():
        for gi, ref_lat in enumerate(rec["latents"][1:]):
            noise = gen_noises[gi].view(1, LDIM)
            lat = head(conds[gi], noise)
            lat_err.append(maxd(lat.numpy(), ref_lat[0].numpy()))
    print(f"teacher-forced latent max|d| over {len(lat_err)} steps: {max(lat_err):.3e}"
          f"  (the fitted tanh-poly erf-GELU is the only approximation)")

    example = (torch.zeros(1, 1, D_MODEL), torch.zeros(1, 1, 1, HD),
               torch.zeros(1, 1, 1, HD), torch.zeros(1, N_HEADS, 1, PMAX + 1),
               torch.zeros(1, N_LAYERS * N_HEADS, PMAX, HD),
               torch.zeros(1, N_LAYERS * N_HEADS, PMAX, HD))
    p = convert(step, example, os.path.join(OUT, "pt_flowlm_step.tflite"))
    opcheck(p, "flowlm_step")
    to_fp16(p, os.path.join(OUT, "pt_flowlm_step_fp16.tflite"))
    opcheck(os.path.join(OUT, "pt_flowlm_step_fp16.tflite"), "flowlm_step_fp16")

    # tflite parity on one step
    cm = CM(p)
    c, s = rope_cos_sin_deint(off0)
    pk0, pv0 = pack_voice(ks, vs, off0)
    with torch.no_grad():
        ref = step(emb_w[tokens[0]].view(1, 1, -1),
                   torch.from_numpy(c).view(1, 1, 1, HD),
                   torch.from_numpy(s).view(1, 1, 1, HD),
                   torch.from_numpy(make_mask(off0)), pk0, pv0)
    outs = cm(emb_w[tokens[0]].view(1, 1, -1).numpy(), c.reshape(1, 1, 1, HD),
              s.reshape(1, 1, 1, HD), make_mask(off0), pk0.numpy(), pv0.numpy())
    by_shape = {}
    for o in outs:
        by_shape.setdefault(tuple(o.shape), []).append(o)
    cond_t = by_shape[(1, D_MODEL)][0]
    print(f"tflite one-step cond corr {corr(cond_t, ref[0].numpy()):.6f} "
          f"max|d| {maxd(cond_t, ref[0].numpy()):.2e}")
    kv = by_shape[(1, N_LAYERS * N_HEADS, 1, HD)]
    print(f"kv order check: out0-vs-nk {maxd(kv[0], ref[2].numpy()):.2e} "
          f"out0-vs-nv {maxd(kv[0], ref[3].numpy()):.2e} "
          f"out1-vs-nk {maxd(kv[1], ref[2].numpy()):.2e} "
          f"out1-vs-nv {maxd(kv[1], ref[3].numpy()):.2e}")
    return step, head, rec, ref_audio


def stage_head(model, head=None):
    print("\n=== flow head graph ===")
    flm = model.flow_lm
    if head is None:
        head = FlowHead(flm).eval()
    cond = torch.randn(1, D_MODEL)
    noise = torch.randn(1, LDIM) * math.sqrt(0.3)
    with torch.no_grad():
        ref = head(cond, noise)
        # eager reference via the model's own lsd path
        from pocket_tts.models.flow_lm import lsd_decode
        from functools import partial
        eager = lsd_decode(partial(flm.flow_net, cond), noise, 1)
    print(f"baked head vs eager lsd: max|d| {maxd(ref.numpy(), eager.numpy()):.2e}")
    p = convert(head, (cond, noise), os.path.join(OUT, "pt_flow_head.tflite"))
    opcheck(p, "flow_head")
    to_fp16(p, os.path.join(OUT, "pt_flow_head_fp16.tflite"))
    cm = CM(p)
    outs = cm(cond.numpy(), noise.numpy())
    print(f"tflite head corr {corr(outs[0], ref.numpy()):.6f} max|d| {maxd(outs[0], ref.numpy()):.2e}")
    return head


def eager_decode(model, latents):
    """Reference: eager one-shot decode_from_latent from a fresh streaming state."""
    from pocket_tts.modules.stateful_module import init_states
    with torch.no_grad():
        st = init_states(model.mimi, batch_size=1,
                         sequence_length=latents.shape[1] * UPS)
        x = latents * model.flow_lm.emb_std + model.flow_lm.emb_mean
        audio = model.mimi.decode_from_latent(x, st)
    return audio


def stage_dectx(model):
    print("\n=== mimi dec_tx block graph ===")
    g = MimiDecTx(model).eval()
    T = 96
    torch.manual_seed(7)
    lat = torch.randn(1, T, LDIM) * 0.8

    # reference: eager full-sequence transformer path
    from pocket_tts.modules.stateful_module import init_states
    mimi = model.mimi
    with torch.no_grad():
        x = lat * model.flow_lm.emb_std + model.flow_lm.emb_mean
        z = mimi.quantizer(x.transpose(1, 2))
        st = init_states(mimi, 1, T * UPS)
        z = mimi.upsample(z, st)
        (ref_feat,) = mimi.decoder_transformer(z, None)   # stateless = banded causal

    # block-composed
    neutral = neutral_latent(model)
    with torch.no_grad():
        feat = compose_dectx(g, lat, neutral)
    print(f"block dec_tx vs eager: corr {corr(feat.numpy(), ref_feat.numpy()):.6f} "
          f"max|d| {maxd(feat.numpy(), ref_feat.numpy()):.2e}")

    p = convert(g, (torch.zeros(1, 1 + F_BLK, LDIM),), os.path.join(OUT, "pt_mimi_dec_tx.tflite"))
    opcheck(p, "dec_tx")
    to_fp16(p, os.path.join(OUT, "pt_mimi_dec_tx_fp16.tflite"))
    cm = CM(p)
    blk = neutral.view(1, 1, LDIM).repeat(1, 1 + F_BLK, 1).clone()
    blk[0, 1:1 + min(F_BLK, T)] = lat[0, :min(F_BLK, T)]
    with torch.no_grad():
        ref_blk = g(blk)
    outs = cm(blk.numpy())
    print(f"tflite dec_tx corr {corr(outs[0], ref_blk.numpy()):.6f}")
    return g


def neutral_latent(model):
    """The latent whose denormalized features are exactly zero (= 'no frame')."""
    return (-model.flow_lm.emb_mean / model.flow_lm.emb_std).detach()


def compose_dectx(g, lat, neutral):
    """Slide the (1+F_BLK)-frame graph; first block keeps all 512 positions,
    later blocks (payload overlapping 16 frames back) keep the right 256."""
    T = lat.shape[1]
    feat = torch.zeros(1, MIMI_D, T * UPS)
    blk = neutral.view(1, 1, LDIM).repeat(1, 1 + F_BLK, 1).clone()
    n0 = min(F_BLK, T)
    blk[0, 1:1 + n0] = lat[0, :n0]
    out = g(blk)
    feat[:, :, :n0 * UPS] = out[:, :, :n0 * UPS]
    kept = F_BLK
    while kept < T:
        start = kept - F_HOP                     # payload start
        blk = neutral.view(1, 1, LDIM).repeat(1, 1 + F_BLK, 1).clone()
        blk[0, 0] = lat[0, start - 1]
        n = min(F_BLK, T - start)
        blk[0, 1:1 + n] = lat[0, start:start + n]
        out = g(blk)
        keep_n = (n - F_HOP) * UPS
        feat[:, :, kept * UPS:kept * UPS + keep_n] = \
            out[:, :, F_HOP * UPS:F_HOP * UPS + keep_n]
        kept += n - F_HOP
    return feat


def stage_deconly(model):
    print("\n=== mimi SEANet deconly graph ===")
    g = MimiDecOnly(model).eval()
    print(f"deconly window: feat[1,512,{S_DEC}] -> audio[1,1,{g.out_len}]")
    T = 96
    torch.manual_seed(7)
    lat = torch.randn(1, T, LDIM) * 0.8
    ref_audio = eager_decode(model, lat)

    with torch.no_grad():
        feat = compose_dectx(MimiDecTx(model).eval(), lat, neutral_latent(model))
        featp = torch.zeros(1, MIMI_D, S_DEC)
        featp[:, :, :feat.shape[-1]] = feat
        audio = g(featp)[:, :, :T * SPF]
    print(f"deconly+dectx vs eager decode: corr {corr(audio.numpy(), ref_audio.numpy()):.6f} "
          f"max|d| {maxd(audio.numpy(), ref_audio.numpy()):.2e}")

    p = convert(g, (torch.zeros(1, MIMI_D, S_DEC),), os.path.join(OUT, "pt_mimi_deconly.tflite"))
    opcheck(p, "deconly")
    to_fp16(p, os.path.join(OUT, "pt_mimi_deconly_fp16.tflite"))
    return g


def stage_assets(model):
    print("\n=== host assets ===")
    flm = model.flow_lm
    emb = flm.conditioner.embed.weight.detach().numpy().astype(np.float16)
    emb.tofile(os.path.join(OUT, "pt_embed_f16.bin"))
    inw = flm.input_linear.weight.detach().numpy().astype(np.float32)   # [1024,32]
    inw.tofile(os.path.join(OUT, "pt_input_linear_f32.bin"))
    bos = (flm.bos_emb.detach() @ flm.input_linear.weight.detach().T).numpy().astype(np.float32)
    bos.tofile(os.path.join(OUT, "pt_bos_input_f32.bin"))
    neutral_latent(model).numpy().astype(np.float32).tofile(
        os.path.join(OUT, "pt_neutral_latent_f32.bin"))
    print(f"embed {emb.shape} fp16, input_linear {inw.shape}, bos [1024], neutral [32]")

    # Licensing gate: only CC-BY-4.0 (alba-mackenna, VCTK) and CC0
    # (voice-donations, voice-zero) voices ship. Expresso (cosette) and EARS
    # (jean) are CC-BY-NC in kyutai/tts-voices, so they are NOT bundled.
    voices = ["alba", "marius", "javert", "charles", "mary", "eve"]
    perm = deint_perm().numpy()
    for name in voices:
        ks, vs, off = load_voice_state(name)
        k = np.stack([ks[li][:off][:, :, perm].permute(1, 0, 2).numpy().astype(np.float16)
                      for li in range(N_LAYERS)])    # [6,16,T,64]
        v = np.stack([vs[li][:off].permute(1, 0, 2).numpy().astype(np.float16)
                      for li in range(N_LAYERS)])
        path = os.path.join(OUT, f"pt_voice_{name}.bin")
        with open(path, "wb") as f:
            np.array([off], dtype=np.int32).tofile(f)
            k.reshape(N_LAYERS * N_HEADS, off, HD).tofile(f)
            v.reshape(N_LAYERS * N_HEADS, off, HD).tofile(f)
        print(f"{name}: T={off} -> {os.path.getsize(path)/1e6:.1f} MB")

    # tokenizer: pieces + scores + types for the Kotlin unigram encoder
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb
    from huggingface_hub import hf_hub_download
    mp = hf_hub_download("kyutai/pocket-tts-without-voice-cloning",
                         "languages/english/tokenizer.model",
                         revision="d29db7978e464fb90cb3359ee0c69a273b9142cc")
    m = pb.ModelProto()
    m.ParseFromString(open(mp, "rb").read())
    with open(os.path.join(OUT, "pt_tokenizer.tsv"), "w") as f:
        for i, p in enumerate(m.pieces):
            piece = p.piece.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")
            f.write(f"{i}\t{p.type}\t{p.score}\t{piece}\n")
    print(f"tokenizer.tsv: {len(m.pieces)} pieces")


def stage_pipeline(model):
    """Full free-running generation through the tflite graphs vs eager audio."""
    print("\n=== full pipeline on tflite (CPU CompiledModel) ===")
    text = "Hello world. I am Pocket TTS running on a phone."
    rec, ref_audio = record_reference(model, "alba", text, seed=1234)
    tokens = rec["tokens"][0].tolist()
    n_gen = len(rec["latents"]) - 1

    step = CM(os.path.join(OUT, "pt_flowlm_step.tflite"))
    headg = CM(os.path.join(OUT, "pt_flow_head.tflite"))
    dectx = CM(os.path.join(OUT, "pt_mimi_dec_tx.tflite"))
    deconly = CM(os.path.join(OUT, "pt_mimi_deconly.tflite"))

    flm = model.flow_lm
    emb_w = flm.conditioner.embed.weight.detach().numpy()
    in_w = flm.input_linear.weight.detach().numpy()
    bos_in = (flm.bos_emb.detach().numpy() @ in_w.T)

    ks, vs, off0 = load_voice_state("alba")
    pk, pv = pack_voice(ks, vs, off0)
    pk = pk.numpy()
    pv = pv.numpy()
    off = off0

    def run_step(x):
        nonlocal off
        c, s = rope_cos_sin_deint(off)
        outs = step(x.reshape(1, 1, D_MODEL), c.reshape(1, 1, 1, HD),
                    s.reshape(1, 1, 1, HD), make_mask(off), pk, pv)
        by = {}
        for o in outs:
            by.setdefault(tuple(o.shape), []).append(o)
        cond = by[(1, D_MODEL)][0]
        eos = by[(1, 1)][0]
        nkv = by[(1, N_LAYERS * N_HEADS, 1, HD)]
        # outputs come back in graph order: nk first, nv second (verified on export)
        nk, nv = nkv[0], nkv[1]
        pk[0, :, off] = nk[0, :, 0]
        pv[0, :, off] = nv[0, :, 0]
        off += 1
        return cond, eos

    for t in tokens:
        run_step(emb_w[t])

    latents = []
    x = bos_in
    gen_noises = rec["noise"][1:]
    match = 0
    for gi in range(n_gen):
        cond, eos = run_step(x)
        noise = gen_noises[gi].numpy().reshape(1, LDIM)
        lat = headg(cond, noise)[0]
        latents.append(lat[0])
        ref_lat = rec["latents"][1 + gi][0, 0].numpy()
        if gi < 5 or gi == n_gen - 1:
            print(f"  step {gi}: latent max|d| {maxd(lat, ref_lat):.3e} eos {float(eos[0,0]):+.2f}")
        x = lat[0] @ in_w.T

    lat_arr = torch.from_numpy(np.stack(latents)[None])   # [1,T,32]
    T = lat_arr.shape[1]
    feat = compose_dectx_np(dectx, lat_arr.numpy(), neutral_latent(model).numpy())
    featp = np.zeros((1, MIMI_D, S_DEC), np.float32)
    featp[:, :, :feat.shape[-1]] = feat
    audio = deconly(featp)[0][:, :, :T * SPF]
    ref = ref_audio.numpy()
    n = min(audio.size, ref.size)
    print(f"pipeline audio: {audio.size/24000:.2f}s vs eager {ref.size/24000:.2f}s, "
          f"corr {corr(audio.ravel()[:n], ref.ravel()[:n]):.6f}")
    import scipy.io.wavfile
    scipy.io.wavfile.write(os.path.join(OUT, "pipeline_tflite.wav"), 24000,
                           audio.ravel().astype(np.float32))
    scipy.io.wavfile.write(os.path.join(OUT, "pipeline_eager.wav"), 24000,
                           ref.ravel().astype(np.float32))
    print("wrote pipeline_tflite.wav / pipeline_eager.wav")


def compose_dectx_np(cm, lat, neutral):
    T = lat.shape[1]
    feat = np.zeros((1, MIMI_D, T * UPS), np.float32)
    blk = np.tile(neutral.reshape(1, 1, LDIM), (1, 1 + F_BLK, 1)).astype(np.float32)
    n0 = min(F_BLK, T)
    blk[0, 1:1 + n0] = lat[0, :n0]
    out = cm(blk)[0]
    feat[:, :, :n0 * UPS] = out[:, :, :n0 * UPS]
    kept = F_BLK
    while kept < T:
        start = kept - F_HOP
        blk = np.tile(neutral.reshape(1, 1, LDIM), (1, 1 + F_BLK, 1)).astype(np.float32)
        blk[0, 0] = lat[0, start - 1]
        n = min(F_BLK, T - start)
        blk[0, 1:1 + n] = lat[0, start:start + n]
        out = cm(blk)[0]
        keep_n = (n - F_HOP) * UPS
        feat[:, :, kept * UPS:kept * UPS + keep_n] = \
            out[:, :, F_HOP * UPS:F_HOP * UPS + keep_n]
        kept += n - F_HOP
    return feat


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    model = load_eager()
    if stage in ("flowlm", "all"):
        stage_flowlm(model)
    if stage in ("head", "all"):
        stage_head(model)
    if stage in ("fused", "all"):
        stage_fused(model)
    if stage in ("dectx", "all"):
        stage_dectx(model)
    if stage in ("deconly", "all"):
        stage_deconly(model)
    if stage in ("assets", "all"):
        stage_assets(model)
    if stage in ("pipeline", "all"):
        stage_pipeline(model)


if __name__ == "__main__":
    main()
