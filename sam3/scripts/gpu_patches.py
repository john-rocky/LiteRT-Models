"""GPU-clean (<=4-D, no data-dependent / gather / select ops) re-authoring of the SAM 3.1
text encoder and detector head for LiteRT CompiledModel GPU. Everything here is
numerically exact w.r.t. the stock modules (verified by build_sam3.py parity checks):

  text   : token embeddings looked up on the HOST (fp16 table, klein/RWKV pattern);
           graph input = (1,32,1024) embeddings; nn.MultiheadAttention -> 4-D manual
           attention with the causal mask as an additive constant.
  head   : sam3 MultiheadAttention -> 4-D manual attention with float additive masks
           (key padding mask arrives as float 1.0 = pad; -1e4 additive is exact for
           fp32 softmax and fp16-safe); log-RPB matrix built with 4-D broadcasting;
           inverse_sigmoid clamp(0,1) -> relu(x)-relu(x-1) (no RELU_0_TO_1);
           gen_sineembed dim_t as a constant; GroupNorm -> 4-D manual; mask einsum ->
           matmul; the tensor-index gathers of Sam3Image._get_img_feats /
           _encode_prompt bypassed by a re-implemented text-only forward_grounding.
"""
import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e4  # additive mask value: exp(NEG - max) == 0 in fp32 and fp16, no NaN rows
# Key-mask formulation. "pre": scores + NEG*mask -> SOFTMAX (what SAM2 shipped on Mali).
# "safe": explicit max-over-valid-keys softmax -- the Metal delegate mis-executes SOFTMAX
# fed by an elementwise op with a broadcast CONSTANT (text causal mask; diag_text_gpu.py),
# and this form is also fp16-safe (max taken over valid keys only, exponent clamped <= 0).
MASK_STYLE = "safe"


def masked_softmax(scores, keep):
    """softmax over the last dim restricted to keep==1 (keep: float 1=valid, broadcastable)."""
    if MASK_STYLE == "pre":
        return torch.softmax(scores + (1.0 - keep) * NEG, dim=-1)
    # max over the VALID keys only (the masked form feeds nothing but the reduce), then
    # exp((s - m) * keep): masked keys get exp(0) -> no fp16 overflow, killed by * keep.
    # NB: clamp/min/relu on (s - m) or an additive-mask tensor with two consumers are
    # silently mis-executed by the Metal delegate (diag_text_gpu.py C/D/E/F variants).
    m = (scores * keep + (1.0 - keep) * NEG).max(dim=-1, keepdim=True).values
    e = torch.exp((scores - m) * keep) * keep
    return e / e.sum(dim=-1, keepdim=True)


# ----------------------------------------------------------------------------- attention
def _mask_to_additive(mask, dtype):
    """bool (True = masked) or float (1.0 = masked) -> float additive (0 / NEG)."""
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        return mask.to(dtype) * NEG
    if mask.is_floating_point():
        # float additive masks (-inf / 0) from the stock code pass through unchanged
        # unless they look like 0/1 indicator masks
        return mask
    raise TypeError(mask.dtype)


def biased_softmax(scores, bias):
    """softmax(scores + bias) written so that no ADD output has two consumers and no
    elementwise op feeds SOFTMAX directly (both mis-execute on the Metal delegate):
    m = max(scores + bias) [single consumer]; e = exp((scores - m) + bias) <= 1."""
    if MASK_STYLE == "pre":
        return torch.softmax(scores + bias, dim=-1)
    m = (scores + bias).max(dim=-1, keepdim=True).values
    e = torch.exp((scores - m) + bias)
    return e / e.sum(dim=-1, keepdim=True)


def mha_core_bf(m, query, key, value, key_padding_mask=None, attn_mask=None, attn_bias=None):
    """Batch-first 4-D attention using the weights of a sam3 / torch MultiheadAttention
    module `m`. query (B,Lq,E), key/value (B,Lk,E); key_padding_mask (B,Lk) bool/float
    (1 = pad); attn_mask bool or float additive bias ((Lq,Lk), (B*H,Lq,Lk) or (B,H,Lq,Lk))."""
    B, Lq, E = query.shape
    Lk = key.shape[1]
    H = m.num_heads
    hd = E // H
    if m._qkv_same_embed_dim:
        w, b = m.in_proj_weight, m.in_proj_bias
        q = F.linear(query, w[:E], None if b is None else b[:E])
        k = F.linear(key, w[E:2 * E], None if b is None else b[E:2 * E])
        v = F.linear(value, w[2 * E:], None if b is None else b[2 * E:])
    else:
        b = m.in_proj_bias
        q = F.linear(query, m.q_proj_weight, None if b is None else b[:E])
        k = F.linear(key, m.k_proj_weight, None if b is None else b[E:2 * E])
        v = F.linear(value, m.v_proj_weight, None if b is None else b[2 * E:])
    q = q.reshape(B, Lq, H, hd).transpose(1, 2)          # (B,H,Lq,hd)
    k = k.reshape(B, Lk, H, hd).transpose(1, 2)
    v = v.reshape(B, Lk, H, hd).transpose(1, 2)
    scores = torch.matmul(q * (1.0 / math.sqrt(hd)), k.transpose(-2, -1))  # (B,H,Lq,Lk)
    bias = None
    if attn_mask is not None:          # bool (True = masked) or float additive bias
        am = _mask_to_additive(attn_mask, scores.dtype)
        if am.dim() == 2:
            am = am.reshape(1, 1, Lq, Lk)
        elif am.dim() == 3:  # (B*H, Lq, Lk) or (1, Lq, Lk)
            am = am.reshape(-1, H, Lq, Lk) if am.shape[0] == B * H else am.reshape(1, 1, Lq, Lk)
        bias = am
    if attn_bias is not None:
        bias = attn_bias if bias is None else bias + attn_bias
    if Lk == 1:
        # single key (geometry cls self-attention): softmax is identically 1 -> output = v.
        # (the converter otherwise emits DIV(e, e) which the GPU delegates refuse)
        o = v.expand(B, H, Lq, hd)
    else:
        if key_padding_mask is not None:   # bool True = pad, or float 1.0 = pad (indicator)
            assert bias is None
            keep = 1.0 - key_padding_mask.to(scores.dtype).reshape(B, 1, 1, Lk)
            attn = masked_softmax(scores, keep)
        elif bias is not None:
            attn = biased_softmax(scores, bias)
        else:
            attn = torch.softmax(scores, dim=-1)
        o = torch.matmul(attn, v)
    o = o.transpose(1, 2).reshape(B, Lq, E)
    return m.out_proj(o)


def sam3_mha_forward_4d(self, query, key, value, key_padding_mask=None, need_weights=False,
                        attn_mask=None, average_attn_weights=True, attn_bias=None):
    """Drop-in for sam3.model.model_misc.MultiheadAttention.forward and torch
    nn.MultiheadAttention.forward (Vanilla, eval)."""
    assert not self.training
    seq_first = not self.batch_first
    if seq_first:  # (L, B, E) -> (B, L, E)
        query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
    o = mha_core_bf(self, query, key, value, key_padding_mask, attn_mask, attn_bias)
    if seq_first:
        o = o.transpose(0, 1)
    return o, None


def text_attention_4d(self, q_x, k_x=None, v_x=None, attn_mask=None):
    """Drop-in for text_encoder_ve.ResidualAttentionBlock.attention (torch nn.MHA,
    batch_first=True, self-attention, causal float mask). Runs on a rank-4 (1,1,L,E)
    activation: rank-3 [1,L,C] tensors that fan out to several consumers are silently
    mis-computed by the GPU delegates (Metal here; the Mali "[1,N,C] fanout" class)."""
    a = self.attn
    x4 = q_x if q_x.dim() == 4 else q_x.unsqueeze(1)      # (1,1,L,E)
    B, _, L, E = x4.shape
    H = a.num_heads
    hd = E // H
    w, b = a.in_proj_weight, a.in_proj_bias
    q = F.linear(x4, w[:E], b[:E]).reshape(B, L, H, hd).transpose(1, 2)
    k = F.linear(x4, w[E:2 * E], b[E:2 * E]).reshape(B, L, H, hd).transpose(1, 2)
    v = F.linear(x4, w[2 * E:], b[2 * E:]).reshape(B, L, H, hd).transpose(1, 2)
    scores = torch.matmul(q * (1.0 / math.sqrt(hd)), k.transpose(-2, -1))
    attn = masked_softmax(scores, self.causal_keep)       # (1,1,L,L) constant buffer, 1 = valid
    o = torch.matmul(attn, v).transpose(1, 2).reshape(B, 1, L, E)
    o = a.out_proj(o)
    return o if q_x.dim() == 4 else o.squeeze(1)


# ----------------------------------------------------------------------------- small ops
def inverse_sigmoid_clean(x, eps=1e-3):
    x = F.relu(x) - F.relu(x - 1.0)                       # == clamp(0, 1)
    x1 = torch.clamp(x, min=eps)
    x2 = torch.clamp(1 - x, min=eps)
    return torch.log(x1 / x2)


class GroupNorm4d(nn.Module):
    """nn.GroupNorm(G, C) on NCHW with only 4-D tensors (fp16-safe scaled variance)."""

    def __init__(self, gn, scale=1.0 / 8):
        super().__init__()
        self.G, self.C, self.eps, self.scale = gn.num_groups, gn.num_channels, gn.eps, scale
        self.register_buffer("w", gn.weight.detach().clone().view(1, -1, 1, 1))
        self.register_buffer("b", gn.bias.detach().clone().view(1, -1, 1, 1))

    def forward(self, x):
        N, C, H, W = x.shape
        xs = (x * self.scale).reshape(N, self.G, (C // self.G) * H, W)
        # hierarchical (per-row then per-group) means: a single reduce over ~2.6M elements
        # overflows the delegate's fp16 accumulator (zoo C29 class)
        mu = xs.mean(3, keepdim=True).mean(2, keepdim=True)
        d = xs - mu
        var = (d * d).mean(3, keepdim=True).mean(2, keepdim=True)
        y = (d * torch.rsqrt(var + self.eps * self.scale * self.scale)).reshape(N, C, H, W)
        return y * self.w + self.b


def mask_predictor_forward_4d(self, obj_queries, pixel_embed):
    """MaskPredictor.forward for the (bqc, bchw) case with a plain matmul. obj_queries may be
    rank-4 (1,B,Q,C): keeping the decoder output rank-4 avoids the Mali ML Drift bug where a
    rank-3 [1,N,C] token tensor fanning out to several heads is silently corrupted."""
    assert pixel_embed.dim() == 4
    B, C, H, W = pixel_embed.shape
    q = self.mask_embed(obj_queries)                      # (B, Q, C) or (1, B, Q, C)
    if q.dim() == 4:
        q = q.reshape(1, B, -1, C)
        return torch.matmul(q, pixel_embed.reshape(1, B, C, H * W)).reshape(B, -1, H, W)
    return torch.matmul(q, pixel_embed.reshape(B, C, H * W)).reshape(B, -1, H, W)


def rpb_matrix_4d(self, reference_boxes, feat_size, batch_first=False):
    """TransformerDecoder._get_rpb_matrix (boxRPB='log') with 4-D broadcasting only.
    Fixed 72x72 feature grid (1008/14). reference_boxes (nq,bs,4) or, batch_first, (bs,nq,4)."""
    H = W = 72
    from sam3.model.box_ops import box_cxcywh_to_xyxy
    boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes)
    if not batch_first:
        boxes_xyxy = boxes_xyxy.transpose(0, 1)                          # bs, nq, 4
    bs, nq, _ = boxes_xyxy.shape
    bx = boxes_xyxy.reshape(bs * nq, 1, 4)
    ys = torch.cat([bx[:, :, 1:2], bx[:, :, 3:4]], -1)               # (bs*nq, 1, 2)
    xs = torch.cat([bx[:, :, 0:1], bx[:, :, 2:3]], -1)
    dy = (self.rpb_coords_h - ys).view(bs, nq, H, 2)
    dx = (self.rpb_coords_w - xs).view(bs, nq, W, 2)
    if self.boxRPB in ["log", "both"]:
        dxl = dx * 8
        dxl = torch.sign(dxl) * torch.log2(torch.abs(dxl) + 1.0) / 3.0
        dyl = dy * 8
        dyl = torch.sign(dyl) * torch.log2(torch.abs(dyl) + 1.0) / 3.0
        if self.boxRPB == "log":
            dx, dy = dxl, dyl
        else:
            dx, dy = torch.cat([dx, dxl], -1), torch.cat([dy, dyl], -1)
    ex = self.boxRPB_embed_x(dx)                                       # bs, nq, W, nh
    ey = self.boxRPB_embed_y(dy)                                       # bs, nq, H, nh
    nh = ex.shape[-1]
    ey_ = ey.permute(0, 3, 1, 2).reshape(bs * nh, nq, H, 1)
    ex_ = ex.permute(0, 3, 1, 2).reshape(bs * nh, nq, 1, W)
    return (ey_ + ex_).reshape(bs, nh, nq, H * W)


def _sine_dim_t_half(nf=128):
    import numpy as np
    d = np.arange(nf // 2, dtype=np.float64)
    return torch.tensor((10000.0 ** (2 * d / nf)).astype(np.float32))


_SINE_DIM_T_HALF = _sine_dim_t_half()   # built at import time (never inside inference_mode)


def gen_sineembed_clean(pos_tensor, num_feats=256):
    """model_misc.gen_sineembed_for_position without strided slices (they lower to
    GATHER_ND) and without POW/FLOOR_DIV: dim_t[2i] == dim_t[2i+1], so the interleaved
    [sin(v/dim_t[0]), cos(v/dim_t[1]), sin(v/dim_t[2]), ...] is stack((sin, cos), -1) over
    the 64 distinct frequencies. dim_t is a numpy constant."""
    assert num_feats % 2 == 0
    nf = num_feats // 2
    assert _SINE_DIM_T_HALF.numel() == nf // 2
    dim_t = _SINE_DIM_T_HALF.to(pos_tensor.device)
    scale = 2 * math.pi

    def emb(v):
        p = v[:, :, None] / dim_t                                   # (nq, bs, 64)
        return torch.stack((p.sin(), p.cos()), dim=3).flatten(2)   # (nq, bs, 128)

    x = emb(pos_tensor[:, :, 0] * scale)
    y = emb(pos_tensor[:, :, 1] * scale)
    if pos_tensor.size(-1) == 2:
        return torch.cat((y, x), dim=2)
    w = emb(pos_tensor[:, :, 2] * scale)
    h = emb(pos_tensor[:, :, 3] * scale)
    return torch.cat((y, x, w, h), dim=2)


def dot_prod_mean_pool_text(self, prompt, prompt_mask):
    is_valid = (1.0 - prompt_mask.to(prompt.dtype)).permute(1, 0)[..., None]   # (seq, bs, 1)
    num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
    return (prompt * is_valid).sum(dim=0) / num_valid


def geometry_encode_bf(ge, g, memory_bf, pos_bf):
    """SequenceGeometryEncoder `encode` layers (encoder.TransformerEncoderLayer, pre-norm,
    pos only on cross-attn keys) batch-first: g (1,1,C) cls token, memory/pos (1,HW,C).
    Keeps the (HW,1,C) seq-first image tensor single-consumer (the Metal delegate
    mis-executes a rank-3 dim0>1 tensor that fans out; diag_head_gpu.py geoenc)."""
    for lay in ge.encode:
        t2 = lay.norm1(g)
        g = g + mha_core_bf(lay.self_attn, t2, t2, t2)              # 1 token: softmax == 1
        t2 = lay.norm2(g)
        g = g + mha_core_bf(lay.cross_attn_image, t2, memory_bf + pos_bf, memory_bf)
        t2 = lay.norm3(g)
        g = g + lay.linear2(lay.activation(lay.linear1(t2)))
    return ge.encode_norm(g)


def decoder_forward_bf(dec, tgt, memory, pos, ref, text, text_pad):
    """TransformerDecoder.forward + TransformerDecoderLayer.forward (eval, box refine,
    log-RPB, presence token, text cross-attn) re-implemented BATCH-FIRST: every activation
    is (1, n, C). The stock seq-first (n, 1, C) layout has n>1 in dim 0, which the Metal
    delegate mis-executes for broadcast elementwise ops (diag_head_gpu.py b_add_out).
    Returns hs_last (1,nq,C), ref_in_last (1,nq,4), presence_logit_last (1,1)."""
    output = tgt                                                       # (1,nq,C)
    presence = dec.presence_token.weight[None]                         # (1,1,C)
    nq = tgt.shape[1]
    for layer in dec.layers:
        sine = gen_sineembed_clean(ref, dec.d_model)                   # (1,nq,2C)
        qpos = dec.ref_point_head(sine)                                # (1,nq,C)
        mm = rpb_matrix_4d(dec, ref, (72, 72), batch_first=True)       # (1,nh,nq,HW)
        cam = torch.cat([torch.zeros_like(mm[:, :, :1, :]), mm], 2)    # presence row
        # self-attention over [presence | queries]
        t = torch.cat([presence, output], 1)                           # (1,1+nq,C)
        qp = torch.cat([torch.zeros_like(presence), qpos], 1)
        q = t + qp
        t = t + mha_core_bf(layer.self_attn, q, q, t)
        t = layer.norm2(t)
        # text cross-attention
        t = layer.catext_norm(t + mha_core_bf(layer.ca_text, t + qp, text, text,
                                                key_padding_mask=text_pad))
        # image cross-attention with the log-RPB bias
        t = layer.norm1(t + mha_core_bf(layer.cross_attn, t + qp, memory + pos, memory,
                                          attn_bias=cam))
        t = layer.forward_ffn(t)
        presence, output = t[:, :1], t[:, 1:]
        # box refinement (use_normed_output_consistently=True)
        normed = dec.norm(output)
        ref_in = ref
        ref = torch.sigmoid(dec.bbox_embed(normed) + inverse_sigmoid_clean(ref))
    presence_logit = dec.presence_token_head(dec.presence_token_out_norm(presence)).reshape(1, 1)
    # rank-4 (1,1,nq,C): the three consumers (score / box / mask heads) must not fan out from a
    # rank-3 [1,N,C] tensor (Mali ML Drift clobbers the later branch -> constant logits on device)
    return normed.unsqueeze(0), ref_in, presence_logit


class SafeLayerNormLast(nn.Module):
    """LayerNorm over the last dim with scale-before-square (fp16 accumulator safety);
    exact (LN is scale invariant, eps scaled accordingly)."""

    def __init__(self, ln, scale):
        super().__init__()
        self.scale = scale
        self.eps = ln.eps * scale * scale
        self.register_buffer("w", ln.weight.detach().clone())
        self.register_buffer("b", ln.bias.detach().clone())

    def forward(self, x):
        xs = x * self.scale
        mu = xs.mean(-1, keepdim=True)
        d = xs - mu
        var = (d * d).mean(-1, keepdim=True)
        return d * torch.rsqrt(var + self.eps) * self.w + self.b



# ----------------------------------------------------------------------------- apply
def apply_head_patches(det):
    from sam3.model import model_misc, decoder as decoder_mod, maskformer_segmentation as ms
    from sam3.model.model_misc import MultiheadAttention, DotProductScoring
    n = 0
    for m in det.modules():
        # sam3's MultiheadAttention (encoder / geometry / seg head) AND torch's
        # nn.MultiheadAttention (decoder self_attn + ca_text) share the attribute layout
        if isinstance(m, (MultiheadAttention, nn.MultiheadAttention)):
            m.forward = types.MethodType(sam3_mha_forward_4d, m)
            n += 1
    model_misc.inverse_sigmoid = inverse_sigmoid_clean
    decoder_mod.inverse_sigmoid = inverse_sigmoid_clean
    import sam3.model.sam3_image as si
    si.inverse_sigmoid = inverse_sigmoid_clean
    decoder_mod.gen_sineembed_for_position = gen_sineembed_clean
    decoder_mod.TransformerDecoder._get_rpb_matrix = rpb_matrix_4d
    dec = det.transformer.decoder
    dec.register_buffer("rpb_coords_h", (torch.arange(72, dtype=torch.float32) / 72).view(1, 72, 1),
                        persistent=False)
    dec.register_buffer("rpb_coords_w", (torch.arange(72, dtype=torch.float32) / 72).view(1, 72, 1),
                        persistent=False)
    DotProductScoring.mean_pool_text = dot_prod_mean_pool_text
    ms.MaskPredictor.forward = mask_predictor_forward_4d
    pd = det.segmentation_head.pixel_decoder
    pd.norms = nn.ModuleList([GroupNorm4d(g) for g in pd.norms])
    # every LayerNorm of the head -> scale-before-square form: activations reach |x|~25 and
    # sum_256 (x-mean)^2 overflows the delegate's fp16 accumulator (deep-ViT rule)
    n_ln = 0
    for root in (det.transformer, det.geometry_encoder, det.segmentation_head, det.dot_prod_scoring):
        for name, m in list(root.named_modules()):
            for cname, c in list(m.named_children()):
                if type(c) is nn.LayerNorm:
                    setattr(m, cname, SafeLayerNormLast(c, 1.0 / 16))
                    n_ln += 1
    print(f"[patch] head LayerNorm -> SafeLayerNorm: {n_ln}")
    return n


def apply_text_patches(det, safe_ln_scale=1.0 / 64):
    te = det.backbone.language_backbone.encoder
    if safe_ln_scale:
        # CLIP-L text residual stream reaches |x| ~ 1.2e3 (massive activations): the plain
        # LayerNorm variance overflows the delegate's fp16 accumulators.
        for blk in te.transformer.resblocks:
            blk.ln_1 = SafeLayerNormLast(blk.ln_1, safe_ln_scale)
            blk.ln_2 = SafeLayerNormLast(blk.ln_2, safe_ln_scale)
        te.ln_final = SafeLayerNormLast(te.ln_final, safe_ln_scale)
    L = te.context_length
    causal = te.attn_mask[:L, :L].detach().clone()       # 0 / -inf upper triangle
    keep = (causal == 0).float()
    for blk in te.transformer.resblocks:
        blk.register_buffer("causal_keep", keep.view(1, 1, L, L), persistent=False)
        blk.attention = types.MethodType(text_attention_4d, blk)
    return len(te.transformer.resblocks)


# ----------------------------------------------------------------------------- wrappers
class TextFlat4d(nn.Module):
    """(1,32,1024) token embeddings (host lookup) -> (1, 32*256) text memory."""

    def __init__(self, det):
        super().__init__()
        self.te = det.backbone.language_backbone
        self.L = self.te.encoder.context_length

    def forward(self, emb):
        enc = self.te.encoder
        x = (emb + enc.positional_embedding[:self.L]).unsqueeze(1)   # (1,1,L,E): stay rank-4
        x = enc.transformer(x, attn_mask=None)            # causal bias is baked per block
        x = enc.ln_final(x)
        return self.te.resizer(x).flatten(1)


class HeadFlat4d(nn.Module):
    """[fpn288 | fpn144 | fpn72 | text_mem(32*256) | text_pad(32)] ->
       [logits(200) | boxes(800, cxcywh norm) | presence(1) | masks(200*288*288)]
    Text-only prompting (0 boxes / 0 points); re-implements Sam3Image.forward_grounding
    without the tensor-index gathers."""

    def __init__(self, det, sizes=((288, 288), (144, 144), (72, 72))):
        super().__init__()
        self.det = det
        self.sizes = list(sizes)
        self.n = [256 * h * w for h, w in self.sizes]
        self.L = det.backbone.language_backbone.encoder.context_length
        pe = det.backbone.vision_backbone.position_encoding
        self.pos = nn.ParameterList([
            nn.Parameter(pe(torch.zeros(1, 256, h, w)).detach(), requires_grad=False)
            for h, w in self.sizes])
        d = det.transformer.decoder
        self.register_buffer("query_embed", d.query_embed.weight.detach().clone(), persistent=False)
        self.register_buffer("ref_boxes0", d.reference_points.weight.detach().clone().sigmoid(),
                             persistent=False)                                       # (200,4)
        # geometry encoder, text-only prompt: cls token -> final_proj -> norm is a CONSTANT
        # (the cross-attention `encode` layers that follow are image dependent and stay in graph)
        ge = det.geometry_encoder
        with torch.no_grad():
            cls = ge.cls_embed.weight.view(1, 1, ge.d_model)
            pre = ge.norm(ge.final_proj(cls)) if ge.final_proj is not None else cls
        self.register_buffer("geo_pre", pre.detach().clone(), persistent=False)       # (1,1,256)

    def forward(self, flat):
        det = self.det
        L = self.L
        off = 0
        fpn = []
        for (h, w), n in zip(self.sizes, self.n):
            fpn.append(flat[:, off:off + n].reshape(1, 256, h, w))
            off += n
        text_mem = flat[:, off:off + L * 256].reshape(1, L, 256)                    # (1,L,256)
        off += L * 256
        text_pad = flat[:, off:off + L]                                              # (1,L) float
        dev = flat.device
        pos = list(self.pos)
        # num_feature_levels == 1: encoder / geometry / decoder see only the 72x72 level.
        # Seq-first (HW,1,C) tensors are made for the stock encoder ONLY (single consumer);
        # everything else uses batch-first (1,HW,C) views of the same 4-D maps.
        vis_feat_sizes = [tuple(pos[-1].shape[-2:])]
        img_feats = [fpn[-1].flatten(2).permute(2, 0, 1)]                            # (HW,1,C)
        img_pos = [pos[-1].flatten(2).permute(2, 0, 1)]
        feat_bf = fpn[-1].flatten(2).transpose(1, 2)                                 # (1,HW,C)
        pos_bf0 = pos[-1].flatten(2).transpose(1, 2)
        # Constants that would otherwise start all-constant op chains (the GPU delegates
        # refuse ops whose every input is constant, and litert-torch does not fold them):
        # make them runtime tensors by adding an input-derived zero.
        zero = torch.clamp(flat[:, :1], min=0.0, max=0.0)   # (1,1) runtime zero (x*0 gets folded)
        # --- prompt = text + geometry(cls only) ---
        ge = det.geometry_encoder
        g = self.geo_pre + zero                                                     # (1,1,256)
        if ge.encode is not None:
            g = geometry_encode_bf(ge, g, feat_bf, pos_bf0)
        geo_feats = g                                                               # (1,1,256)
        prompt = torch.cat([text_mem, geo_feats], 1)                                # (1,L+1,256)
        prompt_mask = torch.cat([text_pad, torch.zeros(1, 1, device=dev)], 1)       # (1,L+1)
        prompt_sf = prompt.transpose(0, 1)                                          # (L+1,1,256)
        # --- encoder (batch_first inside) ---
        memory = det.transformer.encoder(
            src=img_feats.copy(), src_key_padding_mask=None, src_pos=img_pos.copy(),
            prompt=prompt_sf, prompt_pos=torch.zeros_like(prompt_sf),
            prompt_key_padding_mask=prompt_mask, feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=None)
        enc_hs = memory["memory"].transpose(0, 1)                                   # (1,HW,256)
        pos_bf = memory["pos_embed"].transpose(0, 1)                                # (1,HW,256)
        # --- decoder (batch-first re-implementation) ---
        dec = det.transformer.decoder
        tgt = self.query_embed.unsqueeze(0) + zero                                  # (1,200,256)
        ref0 = self.ref_boxes0.unsqueeze(0) + zero                                  # (1,200,4)
        hs4, ref_in, presence = decoder_forward_bf(dec, tgt, enc_hs, pos_bf, ref0,
                                                   prompt, prompt_mask)             # (1,1,200,256)
        # --- scores / boxes (Sam3Image._update_scores_and_boxes, eval, last layer) ---
        outputs_class = det.dot_prod_scoring(hs4, prompt_sf, prompt_mask)           # (1,1,200,1)
        anchor = dec.bbox_embed(hs4)                                                # (1,1,200,4)
        boxes = (inverse_sigmoid_clean(ref_in.unsqueeze(0)) + anchor).sigmoid()     # (1,1,200,4)
        prob_pres = presence.sigmoid()                                              # (1,1)
        logits = inverse_sigmoid_clean(outputs_class.sigmoid() * prob_pres.reshape(1, 1, 1, 1))
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        # --- masks (UniversalSegmentationHead, bs=1), batch-first ---
        sh = det.segmentation_head
        e = enc_hs
        if sh.cross_attend_prompt is not None:
            t2 = sh.cross_attn_norm(e)
            t2 = mha_core_bf(sh.cross_attend_prompt, t2, prompt, prompt, key_padding_mask=prompt_mask)
            e = t2 + e
        e = e.transpose(1, 2).reshape(1, 256, *self.sizes[-1])                      # (1,256,72,72)
        feats = [fpn[0], fpn[1], e]
        pixel_embed = sh.pixel_decoder(feats)
        inst = sh.instance_seg_head(pixel_embed)
        masks = sh.mask_predictor(hs4, inst)                                        # (1,200,288,288)
        return torch.cat([logits.reshape(1, -1), boxes.reshape(1, -1),
                          presence.reshape(1, -1), masks.reshape(1, -1)], 1)
