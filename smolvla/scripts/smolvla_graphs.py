"""Re-authored SmolVLA (lerobot/smolvla_base) modules for fixed-shape LiteRT export.

Three graphs, all float32, batch 1, every tensor rank <= 4:

  VisionGraph      image [1,3,512,512], pos_embed [1,1024,768] -> img_emb [1,64,960]
                   SigLIP-B/16 (12 layers, exact tanh-GELU, which litert-torch emits
                   as the native GELU op) + post LayerNorm + pixel-shuffle x4
                   connector + Linear(12288 -> 960). The zoo's smolvlm vision recipe
                   (smolvlm/scripts/convert_smolvlm.py) without its SigmoidGELU and
                   with rank-4 (einsum) attention products. The output is NOT yet
                   multiplied by sqrt(960) (PrefixGraph does that).
  PrefixGraph      img_emb [1,64N,960], lang_emb [1,48,960], state [1,32],
                   attn_bias [1,1,S,S], rope_cos/rope_sin [1,1,S,32]
                   -> k_all [1,80,S,64], v_all [1,80,S,64]   (S = 64N + 48 + 1)
                   The 16 SmolVLM2 text layers lerobot keeps; outputs the post-RoPE
                   K and the V of every layer (layer l = channels 5l..5l+4).
  ExpertStepGraph  x_t [1,50,32], time_emb [1,1,720], k_all, v_all,
                   bias_self [1,1,50,S+50], bias_cross [1,1,50,S],
                   rope_cos_s/rope_sin_s/rope_cos_c/rope_sin_c [1,1,50,32]
                   -> v_t [1,50,32]   (one Euler velocity evaluation)

Every rewrite relative to lerobot 0.6.1 (`smolvlm_with_expert.py`,
`modeling_smolvla.py`) is exact up to float summation order:
  * apply_rope's empty_like + slice assignment -> cat([x1*c - x2*s, x2*c + x1*s]);
    cos/sin come from host tables built with lerobot's own formula (base 10 000).
  * GQA 5 -> 15 heads without expand/5D: q [1,15,S,64] -> [1,5,3S,64] (q head
    3g+r = row r*S+s of group g); the additive bias is tiled to [1,5,3S,K] with
    CONCATENATIONs. The two attention products are einsums so that the converter
    keeps rank-4 BATCH_MATMULs (torch.matmul is lowered to rank-3 [H,N,d] ones).
  * where(mask, w, finfo.min) -> w + bias with bias in {0, -30000}: identical
    softmax for every row that has at least one visible key (exp underflows to 0
    in float32); fully masked rows only occur for pad tokens, which are never read.
  * cat([action_emb, time_emb]) @ W^T + b -> action_emb @ Wa^T + (time_emb @ Wt^T + b)
    with a per-channel broadcast ADD (no expand / BROADCAST_TO); the suffix MLP runs
    as 1x1 convs so every operand stays 4D (see ExpertStepGraph.suffix_embedding).
  * Layer 15 of the prefix: only its K/V are consumed downstream, so its attention
    output and MLP are not computed (dead code in lerobot's prefill as well).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 960            # SmolVLM2 text hidden size
EXPERT_HIDDEN = 720     # action expert hidden size (0.75 x 960)
N_HEADS = 15
N_KV = 5
HEAD_DIM = 64
N_LAYERS = 16
CHUNK = 50              # action chunk (suffix length)
MAX_DIM = 32            # max_state_dim == max_action_dim
LANG_LEN = 48           # tokenizer_max_length
IMG_TOKENS = 64         # tokens per camera after the x4 pixel shuffle
VIS_PATCHES = 1024      # 32 x 32 patches of 16 px at 512 x 512
VIS_HIDDEN = 768
VIS_HEADS = 12
MASK_NEG = -30000.0     # additive mask value (fp16-representable)


def prefix_len(num_cameras):
    return IMG_TOKENS * num_cameras + LANG_LEN + 1


# ------------------------------------------------------------------ helpers
def _param(t):
    return nn.Parameter(t.detach().to(torch.float32).clone(), requires_grad=False)


def make_linear(weight, bias):
    """nn.Linear holding float32 copies of weight [out, in] and bias [out] (or None)."""
    lin = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    lin.weight = _param(weight)
    if bias is not None:
        lin.bias = _param(bias)
    return lin


def copy_linear(src):
    return make_linear(src.weight, src.bias)


def conv1x1(weight, bias):
    """1x1 nn.Conv2d computing the Linear (weight [out, in], bias [out] or None)."""
    conv = nn.Conv2d(weight.shape[1], weight.shape[0], kernel_size=1, bias=bias is not None)
    conv.weight = _param(weight[:, :, None, None])
    if bias is not None:
        conv.bias = _param(bias)
    return conv


class LayerNorm(nn.Module):
    """nn.LayerNorm over the last dim (same F.layer_norm call, same lowering).

    fp16_safe=True (the default graphs, see set_fp16_safe_norms) switches to the exact
    down-scaled form (litert_gpu_toolkit SafeLayerNorm "adaptive_v2" with eps kept at
    its true magnitude): with S = max(1, max|x| / 8) per row and d = x/S - mean(x/S),
    y = d * rsqrt(mean(d^2) + eps / S^2). The variance is never scaled back up (a
    rebuilt var * S^2 can itself overflow fp16), and no intermediate squares a value
    above 8 * 8. Equal to LayerNorm in real arithmetic.
    """

    def __init__(self, src):
        super().__init__()
        self.normalized_shape = tuple(src.normalized_shape)
        self.weight = _param(src.weight)
        self.bias = _param(src.bias)
        self.eps = float(src.eps)
        self.fp16_safe = False

    def forward(self, x):
        if not self.fp16_safe:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        s = (x.abs().amax(-1, keepdim=True) * 0.125).clamp(min=1.0)
        xs = x / s
        d = xs - xs.mean(-1, keepdim=True)
        var = (d * d).mean(-1, keepdim=True)
        y = d * torch.rsqrt(var + self.eps / (s * s)) * self.weight + self.bias
        return y.contiguous()


def copy_layernorm(src):
    return LayerNorm(src)


class RMSNorm(nn.Module):
    """LlamaRMSNorm in float32: weight * (x * rsqrt(mean(x*x) + eps)) (x*x == pow(x, 2)).

    fp16_safe=True (the default graphs, see set_fp16_safe_norms) switches to the exact
    max-normalized form (litert_gpu_toolkit safe_rms): with m = max(1, max|x|) per
    row, weight * ((x/m) * rsqrt(mean((x/m)^2) + eps/m^2)); every square is <= 1.
    The floor of 1 (instead of safe_rms's 1e-4) keeps m^2 out of the fp16 underflow
    range; rows with max|x| < 1 then use the plain formula, which is already safe.
    """

    def __init__(self, src):
        super().__init__()
        self.weight = _param(src.weight)
        self.eps = float(src.variance_epsilon)
        self.fp16_safe = False

    def forward(self, x):
        if not self.fp16_safe:
            var = (x * x).mean(-1, keepdim=True)
            return self.weight * (x * torch.rsqrt(var + self.eps))
        m = x.abs().amax(-1, keepdim=True).clamp_min(1.0)
        xs = x / m
        var = (xs * xs).mean(-1, keepdim=True)
        return self.weight * (xs * torch.rsqrt(var + self.eps / (m * m)))


def set_fp16_safe_norms(module, enabled=True):
    """Switch every RMSNorm / LayerNorm in module to its fp16-safe exact form.

    build_smolvla.py does this for the default vision/prefix graphs (--norms fp16safe);
    --norms exact keeps the plain formulas as a reference. Returns the number switched.
    """
    count = 0
    for m in module.modules():
        if isinstance(m, (RMSNorm, LayerNorm)):
            m.fp16_safe = enabled
            count += 1
    return count


def rope(x, cos, sin):
    """lerobot apply_rope, split-half form. x [1,H,S,64]; cos/sin [1,1,S,32]."""
    half = HEAD_DIM // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def gqa_attention(q, k, v, bias_g):
    """Eager GQA attention with 15 query heads over 5 KV heads, all tensors 4D.

    q [1,15,S,64], k/v [1,5,K,64], bias_g [1,5,3S,K] (the [1,1,S,K] additive bias
    tiled 3x along rows and 5x along the KV groups, see tile_bias). Returns
    [1,S,960] in lerobot's (head, dim) order.

    The two products are einsums on purpose: litert-torch lowers torch.matmul on
    [1,H,N,d] to a rank-3 BATCH_MATMUL on [H,N,d] (and the bias ADD then mixes a
    rank-3 and a rank-4 tensor), while einsum lowers through dot_general and keeps
    rank-4 BATCH_MATMULs: [1,5,3S,64] x [1,5,K,64] (adj_y) and [1,5,3S,K] x [1,5,K,64].
    """
    s = q.shape[2]
    qg = q.reshape(1, N_KV, 3 * s, HEAD_DIM)
    w = torch.einsum("bgqd,bgkd->bgqk", qg, k) * (HEAD_DIM ** -0.5)
    p = torch.softmax(w + bias_g, dim=-1)
    o = torch.einsum("bgqk,bgkd->bgqd", p, v)                # [1,5,3S,64]
    o = o.reshape(1, N_HEADS, s, HEAD_DIM).permute(0, 2, 1, 3)
    return o.reshape(1, s, N_HEADS * HEAD_DIM)


def tile_bias(bias):
    """[1,1,S,K] -> [1,5,3S,K] with CONCATENATIONs only: rows tiled 3x for the three
    query heads of a group, then 5x along the groups, so the score ADD is same-shape."""
    rows = torch.cat([bias, bias, bias], dim=2)
    return torch.cat([rows] * N_KV, dim=1)


def heads(x, n):
    """[1,S,n*64] -> [1,n,S,64]."""
    return x.view(1, x.shape[1], n, HEAD_DIM).permute(0, 2, 1, 3)


class LlamaMLP(nn.Module):
    def __init__(self, src):
        super().__init__()
        self.gate_proj = copy_linear(src.gate_proj)
        self.up_proj = copy_linear(src.up_proj)
        self.down_proj = copy_linear(src.down_proj)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ------------------------------------------------------------------- vision
def vis_heads(x, n):
    """[1,N,768] -> [1,12,N,64]."""
    return x.view(1, n, VIS_HEADS, VIS_HIDDEN // VIS_HEADS).permute(0, 2, 1, 3)


class SiglipLayer(nn.Module):
    """SigLIP encoder layer (SmolVLMEncoderLayer), eager attention.

    The two attention products are einsums, as in gqa_attention: torch.matmul on
    [1,12,N,64] would be lowered to rank-3 [12,N,64] BATCH_MATMULs, and a collapsed
    batch dim in an attention chain is a known silent-miscompute pattern on ML Drift
    (Mali); einsum keeps rank-4 BATCH_MATMULs.
    """

    def __init__(self, src):
        super().__init__()
        a = src.self_attn
        self.ln1 = copy_layernorm(src.layer_norm1)
        self.q_proj = copy_linear(a.q_proj)
        self.k_proj = copy_linear(a.k_proj)
        self.v_proj = copy_linear(a.v_proj)
        self.out_proj = copy_linear(a.out_proj)
        self.ln2 = copy_layernorm(src.layer_norm2)
        self.fc1 = copy_linear(src.mlp.fc1)
        self.fc2 = copy_linear(src.mlp.fc2)
        self.scale = float(a.scale)

    def forward(self, x):
        n = x.shape[1]
        h = self.ln1(x)
        q = vis_heads(self.q_proj(h), n)
        k = vis_heads(self.k_proj(h), n)
        v = vis_heads(self.v_proj(h), n)
        w = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        p = torch.softmax(w, dim=-1)
        o = torch.einsum("bhqk,bhkd->bhqd", p, v).permute(0, 2, 1, 3).reshape(1, n, VIS_HIDDEN)
        x = x + self.out_proj(o)
        h = self.ln2(x)
        return x + self.fc2(F.gelu(self.fc1(h), approximate="tanh"))


def pixel_shuffle(x, scale=4):
    """transformers SmolVLMConnector.pixel_shuffle; every intermediate is <= 4D."""
    bsz, seq, dim = x.shape
    side = int(seq ** 0.5)
    x = x.view(bsz, side, side, dim)
    x = x.view(bsz, side, side // scale, dim * scale)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(bsz, side // scale, side // scale, dim * scale * scale)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(bsz, seq // (scale * scale), dim * scale * scale)


class VisionGraph(nn.Module):
    """SigLIP-B/16 @512 + connector. pos_embed is a runtime input on purpose."""

    def __init__(self, vision_model, connector):
        super().__init__()
        pe = vision_model.embeddings.patch_embedding
        self.patch_embedding = nn.Conv2d(
            pe.in_channels, pe.out_channels, kernel_size=pe.kernel_size,
            stride=pe.stride, padding=0, bias=True)
        self.patch_embedding.weight = _param(pe.weight)
        self.patch_embedding.bias = _param(pe.bias)
        self.layers = nn.ModuleList(SiglipLayer(l) for l in vision_model.encoder.layers)
        self.post_layernorm = copy_layernorm(vision_model.post_layernorm)
        self.scale_factor = 4
        self.proj = copy_linear(connector.modality_projection.proj)
        self.taps = None        # set to a list to record the residual stream

    def forward(self, image, pos_embed):
        x = self.patch_embedding(image)                 # [1,768,32,32]
        x = x.flatten(2).transpose(1, 2)                # [1,1024,768]
        x = x + pos_embed
        for layer in self.layers:
            x = layer(x)
            if self.taps is not None:
                self.taps.append(x.detach())
        x = self.post_layernorm(x)
        x = pixel_shuffle(x, self.scale_factor)         # [1,64,12288]
        return self.proj(x)                             # [1,64,960]


# ------------------------------------------------------------------- prefix
class PrefixLayer(nn.Module):
    def __init__(self, src, kv_only=False):
        super().__init__()
        a = src.self_attn
        self.input_layernorm = RMSNorm(src.input_layernorm)
        self.k_proj = copy_linear(a.k_proj)
        self.v_proj = copy_linear(a.v_proj)
        self.kv_only = kv_only
        if not kv_only:
            self.q_proj = copy_linear(a.q_proj)
            self.o_proj = copy_linear(a.o_proj)
            self.post_attention_layernorm = RMSNorm(src.post_attention_layernorm)
            self.mlp = LlamaMLP(src.mlp)

    def forward(self, x, bias_g, cos, sin):
        h = self.input_layernorm(x)
        k = rope(heads(self.k_proj(h), N_KV), cos, sin)       # post-RoPE K
        v = heads(self.v_proj(h), N_KV)
        if self.kv_only:
            return x, k, v
        q = rope(heads(self.q_proj(h), N_HEADS), cos, sin)
        x = x + self.o_proj(gqa_attention(q, k, v, bias_g))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, k, v


class PrefixGraph(nn.Module):
    """VLM prefill over [image tokens | language tokens | state token]."""

    def __init__(self, flow_model, num_cameras=1):
        super().__init__()
        vlm = flow_model.vlm_with_expert.get_vlm_model()
        src_layers = vlm.text_model.layers
        assert len(src_layers) == N_LAYERS
        self.layers = nn.ModuleList(
            PrefixLayer(l, kv_only=(i == N_LAYERS - 1)) for i, l in enumerate(src_layers))
        self.state_proj = copy_linear(flow_model.state_proj)
        self.num_cameras = num_cameras
        # embed_prefix: img_emb * torch.tensor(960 ** 0.5) and lang_emb * math.sqrt(960);
        # both round to the same float32 scalar.
        self.emb_scale = float(torch.tensor(HIDDEN ** 0.5, dtype=torch.float32))
        assert self.emb_scale == float(torch.tensor(math.sqrt(HIDDEN), dtype=torch.float32))
        self.taps = None

    def forward(self, img_emb, lang_emb, state, attn_bias, rope_cos, rope_sin):
        img = img_emb * self.emb_scale
        lang = lang_emb * self.emb_scale
        st = self.state_proj(state).unsqueeze(1)               # [1,1,960]
        x = torch.cat([img, lang, st], dim=1)                   # [1,S,960]
        bias_g = tile_bias(attn_bias)
        ks, vs = [], []
        for layer in self.layers:
            if self.taps is not None:
                self.taps.append(x.detach())
            x, k, v = layer(x, bias_g, rope_cos, rope_sin)
            ks.append(k)
            vs.append(v)
        return torch.cat(ks, dim=1), torch.cat(vs, dim=1)       # [1,80,S,64] x 2


# ------------------------------------------------------------------- expert
class ExpertLayer(nn.Module):
    def __init__(self, src, self_attn):
        super().__init__()
        a = src.self_attn
        self.self_attn = self_attn
        self.input_layernorm = RMSNorm(src.input_layernorm)
        self.q_proj = copy_linear(a.q_proj)
        self.k_proj = copy_linear(a.k_proj)          # 720->320 (self) / 320->320 (cross)
        self.v_proj = copy_linear(a.v_proj)
        self.o_proj = copy_linear(a.o_proj)
        self.post_attention_layernorm = RMSNorm(src.post_attention_layernorm)
        self.mlp = LlamaMLP(src.mlp)

    def forward(self, x, kp, vp, bias_self_g, bias_cross_g, cos_s, sin_s, cos_c, sin_c):
        h = self.input_layernorm(x)
        q = heads(self.q_proj(h), N_HEADS)
        if self.self_attn:
            k = rope(heads(self.k_proj(h), N_KV), cos_s, sin_s)
            v = heads(self.v_proj(h), N_KV)
            q = rope(q, cos_s, sin_s)
            kk = torch.cat([kp, k], dim=2)                       # [1,5,S+50,64]
            vv = torch.cat([vp, v], dim=2)
            a = gqa_attention(q, kk, vv, bias_self_g)
        else:
            s = kp.shape[2]
            kf = kp.permute(0, 2, 1, 3).reshape(1, s, N_KV * HEAD_DIM)   # (head, dim) order
            vf = vp.permute(0, 2, 1, 3).reshape(1, s, N_KV * HEAD_DIM)
            kk = heads(self.k_proj(kf), N_KV)                    # no RoPE on these keys
            vv = heads(self.v_proj(vf), N_KV)
            q = rope(q, cos_c, sin_c)
            a = gqa_attention(q, kk, vv, bias_cross_g)
        x = x + self.o_proj(a)
        return x + self.mlp(self.post_attention_layernorm(x))


class ExpertStepGraph(nn.Module):
    """One denoise_step: suffix embedding + 16 expert layers + action_out_proj."""

    def __init__(self, flow_model, self_attn_every_n_layers=2):
        super().__init__()
        vwe = flow_model.vlm_with_expert
        src_layers = vwe.lm_expert.layers
        assert len(src_layers) == N_LAYERS
        self.layers = nn.ModuleList(
            ExpertLayer(l, self_attn=(i % self_attn_every_n_layers == 0))
            for i, l in enumerate(src_layers))
        self.norm = RMSNorm(vwe.lm_expert.norm)
        w_in = flow_model.action_time_mlp_in.weight               # [720, 1440]
        # action_time_mlp_in(cat([action_emb, time_emb])) split into its two halves;
        # the per-token Linears run as 1x1 convs (see suffix_embedding).
        self.action_in_proj = conv1x1(flow_model.action_in_proj.weight,
                                      flow_model.action_in_proj.bias)
        self.time_mlp_in_action = conv1x1(w_in[:, :EXPERT_HIDDEN], None)
        self.time_mlp_in_time = conv1x1(w_in[:, EXPERT_HIDDEN:],
                                        flow_model.action_time_mlp_in.bias)
        self.time_mlp_out = conv1x1(flow_model.action_time_mlp_out.weight,
                                    flow_model.action_time_mlp_out.bias)
        self.action_out_proj = copy_linear(flow_model.action_out_proj)
        self.taps = None

    def suffix_embedding(self, x_t, time_emb):
        """lerobot embed_suffix: action_in_proj, cat with the time embedding,
        action_time_mlp_in, SiLU, action_time_mlp_out -> [1,50,720].

        The Linears run as 1x1 convs over NCHW views ([1,C,1,50] for the 50 tokens,
        [1,720,1,1] for the time half, which then adds as a per-channel bias). As
        plain Linears the converter flattens their outputs to rank-2 [50,720] and
        mixes them with rank-3 tensors in broadcast ADD/MUL ops (the FC [N,C] layout
        conflict class); as convs every operand stays 4D.
        """
        xc = x_t.permute(0, 2, 1).reshape(1, MAX_DIM, 1, CHUNK)             # [1,32,1,50]
        t = self.time_mlp_in_time(time_emb.reshape(1, EXPERT_HIDDEN, 1, 1))  # [1,720,1,1]
        h = self.time_mlp_in_action(self.action_in_proj(xc)) + t             # [1,720,1,50]
        x = self.time_mlp_out(F.silu(h))
        return x.reshape(1, EXPERT_HIDDEN, CHUNK).permute(0, 2, 1)          # [1,50,720]

    def forward(self, x_t, time_emb, k_all, v_all, bias_self, bias_cross,
                rope_cos_s, rope_sin_s, rope_cos_c, rope_sin_c):
        x = self.suffix_embedding(x_t, time_emb)                  # [1,50,720]
        b_self = tile_bias(bias_self)
        b_cross = tile_bias(bias_cross)
        for i, layer in enumerate(self.layers):
            if self.taps is not None:
                self.taps.append(x.detach())
            kp = k_all[:, N_KV * i:N_KV * (i + 1)]
            vp = v_all[:, N_KV * i:N_KV * (i + 1)]
            x = layer(x, kp, vp, b_self, b_cross, rope_cos_s, rope_sin_s, rope_cos_c, rope_sin_c)
        if self.taps is not None:
            self.taps.append(x.detach())
        return self.action_out_proj(self.norm(x))                 # [1,50,32]


# ------------------------------------------------------------ lerobot load
SMOLVLA_REPO = "lerobot/smolvla_base"


def load_policy(repo=SMOLVLA_REPO, float32=True):
    """lerobot SmolVLAPolicy on CPU; float32=True upcasts every module (the reference)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = PreTrainedConfig.from_pretrained(repo)
    cfg.device = "cpu"
    policy = SmolVLAPolicy.from_pretrained(repo, config=cfg)
    policy.to("cpu")
    if float32:
        policy.model.float()
    policy.eval()
    return policy


def load_processors(repo=SMOLVLA_REPO, policy_config=None):
    from lerobot.policies.factory import make_pre_post_processors
    over = {"device_processor": {"device": "cpu"}}
    return make_pre_post_processors(policy_config, pretrained_path=repo,
                                    preprocessor_overrides=over,
                                    postprocessor_overrides=over)


def norm_stats_from_processors(pre, post):
    """The MEAN_STD statistics lerobot's processor steps will actually apply.

    lerobot looks stats up by exact feature key ("observation.state", "action");
    a key that is absent means the step returns the tensor unchanged. Stats stored
    under other keys (smolvla_base: "so100.buffer.action", ...) are reported as
    unmatched and are NOT applied by lerobot 0.6.1.
    """
    from lerobot.processor.normalize_processor import NormalizerProcessorStep
    from lerobot.processor.normalize_processor import UnnormalizerProcessorStep

    out = {"eps": None, "state": None, "action": None, "unmatched_stat_keys": {},
           "norm_map": None}
    for step in list(pre.steps) + list(post.steps):
        stats = getattr(step, "_tensor_stats", None)
        if stats is None:
            continue
        out["eps"] = float(step.eps)
        out["norm_map"] = {str(getattr(k, "value", k)): str(getattr(v, "value", v))
                           for k, v in step.norm_map.items()}
        for key, entry in stats.items():
            vals = {k: v.detach().cpu().float().flatten().tolist() for k, v in entry.items()}
            if key == "observation.state" and isinstance(step, NormalizerProcessorStep):
                out["state"] = vals
            elif key == "action" and isinstance(step, UnnormalizerProcessorStep):
                out["action"] = vals
            elif key not in ("observation.state", "action"):
                out["unmatched_stat_keys"][key] = vals
    return out
