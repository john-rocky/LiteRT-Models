# RF-DETR-Seg-Nano (roboflow/rf-detr 1.9.3, Apache-2.0) 2-graph split for LiteRT CompiledModel GPU.
#   GraphA (GPU) = DINOv2-S/12 @312 backbone (num_windows=1 -> ALL-GLOBAL attn) + projector(P4)
#                  + enc_output norm + enc heads
#                  -> enc_class[1,676,91], enc_coord[1,676,4], memory[1,676,256]
#   host         = scores = max(enc_class,-1) -> topk-100 -> gather enc_coord -> ts[1,100,4]
#   GraphB (GPU) = two-stage reparam combine + decoder(4L, deformable tent-matmul cross-attn)
#                  + bbox/class heads + seg mask branch (spatial map rebuilt from memory:
#                  memory.T -> [1,256,26,26] -> bilinear x3 -> 4 DepthwiseConvBlocks -> 1x1 proj
#                  -> rank-4 matmul with query feats)
#                  -> boxes[1,100,4], logits[1,100,91], masks[1,100,78,78]
# Ported from the RF-DETR Nano detection recipe (rfdetr-work/build_rfdetr_{bb,full,split}.py,
# device-verified 2026-06-29) onto rfdetr 1.9.3; SafeLayerNorm upgraded to v2 (never rebuild the
# variance -- Parakeet lesson) and the einsum mask head kept rank-4 (Mali rank-3 traps).
import sys, os, math, collections
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tfm_compat  # noqa: F401  transformers 4.57 <-> 5.x shims (no-op on >=5.1)

torch._shape_as_tensor = lambda t: torch.tensor(list(t.shape), dtype=torch.long)  # untraceable private op -> const (fixed res)
torch._assert = lambda *a, **k: None

R = int(os.environ.get("RF_RES", "312"))
NQ, NCLS, HID = 100, 91, 256
GH = GW = R // 12                     # 26x26 single deformable level
MH = MW = R // 4                      # 78x78 mask grid (mask_downsample_ratio=4)
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP", "RFFT2D",
          "FFT", "STFT", "COMPLEX", "CUMSUM", "MIRROR_PAD"}

# ---- backbone SDPA -> manual rank-4 matmuls (Mali rank-3 SDPA miscompute + no SDPA lowering) ----
from rfdetr.models.backbone import dinov2_with_windowed_attn as D
def _sdpa_manual(self, hidden_states, output_attentions=False):
    q = self.transpose_for_scores(self.query(hidden_states))
    k = self.transpose_for_scores(self.key(hidden_states))
    v = self.transpose_for_scores(self.value(hidden_states))
    scale = 1.0 / (q.shape[-1] ** 0.5)
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    a = torch.softmax(s, dim=-1)
    c = torch.matmul(a, v).permute(0, 2, 1, 3).contiguous()
    c = c.view(c.size()[:-2] + (self.all_head_size,))
    return c, None
D.Dinov2WithRegistersSdpaSelfAttention.forward = _sdpa_manual

# ---- SafeLayerNorm v2 for nn.LayerNorm (channels-last) --------------------------------------------
# Adaptive per-row down-scale, and NEVER reconstruct the large variance: the scale cancels in
# y = d/sqrt(var), so every intermediate stays O(1)..O(amax) -- fp16-safe at any magnitude
# (Parakeet witness: the RF-DETR det-era var*(S*S) form overflows when amax ~ 7000).
def _safe_ln_forward(self, x):
    amax = x.abs().amax(-1, keepdim=True)
    S = (amax * (1.0 / 8.0)).clamp(min=1.0)
    xs = x / S
    mu = xs.mean(-1, keepdim=True)
    d = xs - mu
    var = (d * d).mean(-1, keepdim=True)
    y = d * torch.rsqrt(var + self.eps)
    if self.elementwise_affine:
        y = y * self.weight + self.bias
    return y
nn.LayerNorm.forward = _safe_ln_forward

# ---- projector channels-first LayerNorm -> same v2 math, via a 3D detour --------------------------
# litert-torch's NHWC layout pass has no rewriter for amax on layout-tracked 4D tensors, so drop to
# [B, HW, C] (3D leaves the conv-layout domain) before the adaptive reduction; math is unchanged.
from rfdetr.models.backbone import projector as _PROJ
def _safe_ln_proj_forward(s, x):
    B, C, H, W = x.shape
    t = x.reshape(B, C, H * W).transpose(1, 2)         # [B, HW, C]
    amax = t.abs().amax(-1, keepdim=True)
    S = (amax * (1.0 / 8.0)).clamp(min=1.0)
    xs = t / S
    mu = xs.mean(-1, keepdim=True)
    d = xs - mu
    var = (d * d).mean(-1, keepdim=True)
    y = d * torch.rsqrt(var + s.eps) * s.weight + s.bias
    return y.transpose(1, 2).reshape(B, C, H, W)
_PROJ.LayerNorm.forward = _safe_ln_proj_forward

# ---- grid_sample -> GATHER/CAST-free bilinear (tent weights + matmul, exact: MAE ~1e-8) ------------
def _gs(input, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
    N, C, H, W = input.shape
    Hg, Wg = grid.shape[1], grid.shape[2]
    ac = bool(align_corners)
    if ac:
        ix = (grid[..., 0] + 1) * (W - 1) / 2; iy = (grid[..., 1] + 1) * (H - 1) / 2
    else:
        ix = (grid[..., 0] + 1) * W / 2 - 0.5; iy = (grid[..., 1] + 1) * H / 2 - 0.5
    ix = ix.reshape(N, Hg * Wg, 1); iy = iy.reshape(N, Hg * Wg, 1)
    xs = torch.arange(W, dtype=input.dtype).reshape(1, 1, W)
    ys = torch.arange(H, dtype=input.dtype).reshape(1, 1, H)
    wx = torch.relu(1 - (ix - xs).abs()); wy = torch.relu(1 - (iy - ys).abs())
    Wm = (wy.unsqueeze(-1) * wx.unsqueeze(-2)).reshape(N, 1, Hg * Wg, H * W)
    # rank-4 BMM: ML Drift silently mis-executes rank-3 batched matmuls (#8619 family)
    out = torch.matmul(input.reshape(N, 1, C, H * W), Wm.transpose(-1, -2))
    return out.reshape(N, C, Hg, Wg)
F.grid_sample = _gs
from rfdetr.utilities import tensors as _T
_T._bilinear_grid_sample = lambda input, grid, padding_mode="zeros", align_corners=False: _gs(
    input, grid, padding_mode=padding_mode, align_corners=align_corners)
from rfdetr.models.ops.functions import ms_deform_attn_func as _MSF
_MSF._bilinear_grid_sample = _T._bilinear_grid_sample

# ---- MSDeformAttn.forward re-authored <=4D (n_levels=1) with the tent-matmul sampler --------------
import rfdetr.models.ops.modules.ms_deform_attn as _MSMOD
def _msda_forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                  input_level_start_index, input_padding_mask=None, input_spatial_shapes_hw=None, **kw):
    bs = query.shape[0]; len_q = query.shape[1]
    nh = self.n_heads; npnt = self.n_points; dm = self.d_model; hd = dm // nh
    H, W = (input_spatial_shapes_hw[0] if input_spatial_shapes_hw else (GH, GW))
    value = self.value_proj(input_flatten)                                    # [bs, HW, dm]
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], 0.0)
    so = self.sampling_offsets(query).view(bs, len_q, nh, npnt * 2).permute(0, 2, 1, 3).reshape(bs * nh, len_q, npnt, 2)
    aw = self.attention_weights(query).view(bs, len_q, nh, npnt)
    aw = torch.softmax(aw, -1).permute(0, 2, 1, 3).reshape(bs * nh, 1, len_q, npnt)
    ref = reference_points[:, :, 0, :]                                        # squeeze n_levels=1
    rxy = ref[..., :2].unsqueeze(1).repeat(1, nh, 1, 1).reshape(bs * nh, len_q, 1, 2)
    if ref.shape[-1] == 4:
        rwh = ref[..., 2:].unsqueeze(1).repeat(1, nh, 1, 1).reshape(bs * nh, len_q, 1, 2)
        loc = rxy + so / npnt * rwh * 0.5
    else:
        norm = torch.tensor([W, H], dtype=value.dtype).reshape(1, 1, 1, 2)
        loc = rxy + so / norm
    val = value.transpose(1, 2).reshape(bs * nh, hd, H, W)
    sampled = _gs(val, 2 * loc - 1, padding_mode="zeros", align_corners=False)  # [bs*nh, hd, len_q, npnt]
    out = (sampled * aw).sum(-1).reshape(bs, dm, len_q).transpose(1, 2)
    return self.output_proj(out)
_MSMOD.MSDeformAttn.forward = _msda_forward

# ---- sine pos-embed: bake dim_t (no POW/FLOOR_DIV) + reshape interleave (no strided GATHER_ND) -----
import rfdetr.models.transformer as _TR
_DIMT = {}
def _gen_sine(pos_tensor, dim=128):
    scale = 2 * math.pi
    if dim not in _DIMT:
        dt = torch.arange(dim, dtype=torch.float32)
        _DIMT[dim] = (10000.0 ** (2 * (dt // 2) / dim)).detach()
    dim_t = _DIMT[dim]
    def il(emb):
        p = emb[:, :, None] * scale / dim_t
        pr = p.reshape(p.shape[0], p.shape[1], dim // 2, 2)
        return torch.stack((pr[..., 0].sin(), pr[..., 1].cos()), -1).flatten(2)
    pos_x = il(pos_tensor[:, :, 0]); pos_y = il(pos_tensor[:, :, 1])
    if pos_tensor.size(-1) == 2:
        return torch.cat((pos_y, pos_x), dim=2)
    return torch.cat((pos_y, pos_x, il(pos_tensor[:, :, 2]), il(pos_tensor[:, :, 3])), dim=2)
_TR.gen_sineembed_for_position = _gen_sine

# ---- seg head DepthwiseConvBlock: plain F.conv2d (no cuDNN autograd Function) and the LN via the
# same 3D detour (its permute(0,2,3,1) LayerNorm is a layout-tracked 4D tensor -> amax rewriter gap).
from rfdetr.models.heads import segmentation as _SEG
def _dwblock_forward(self, x):
    inp = x
    x = F.conv2d(x, self.dwconv.weight, self.dwconv.bias, self.dwconv.stride,
                 self.dwconv.padding, self.dwconv.dilation, self.dwconv.groups)
    B, C, H, W = x.shape
    x = x.reshape(B, C, H * W).transpose(1, 2)          # [B, HW, C] 3D
    x = self.norm(x)
    x = self.pwconv1(x)
    x = self.act(x)
    if self.gamma is not None:
        x = self.gamma * x
    x = x.transpose(1, 2).reshape(B, C, H, W)
    return x + inp
_SEG.DepthwiseConvBlock.forward = _dwblock_forward


class TG(nn.Module):  # tanh-GELU (the DINOv2/TIPSv2 ship recipe; ERF has no GPU lowering)
    def forward(s, x):
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


class ManualMHA(nn.Module):
    """Rank-4 replacement for nn.MultiheadAttention (batch_first, self-attention use).
    torch MHA lowers to rank-3 BMMs, which ML Drift silently mis-executes (#8619 family);
    same weights, manual head-split matmuls, exact."""
    def __init__(s, mha):
        super().__init__()
        s.h = mha.num_heads
        s.e = mha.embed_dim
        s.in_w = nn.Parameter(mha.in_proj_weight.data.clone())
        s.in_b = nn.Parameter(mha.in_proj_bias.data.clone())
        s.out = mha.out_proj

    def forward(s, q, k, v, attn_mask=None, key_padding_mask=None, need_weights=False):
        E, H = s.e, s.h
        hd = E // H
        b, n, _ = q.shape
        qp = F.linear(q, s.in_w[:E], s.in_b[:E]).reshape(b, n, H, hd).permute(0, 2, 1, 3)
        kp = F.linear(k, s.in_w[E:2 * E], s.in_b[E:2 * E]).reshape(b, -1, H, hd).permute(0, 2, 1, 3)
        vp = F.linear(v, s.in_w[2 * E:], s.in_b[2 * E:]).reshape(b, -1, H, hd).permute(0, 2, 1, 3)
        a = torch.softmax(torch.matmul(qp, kp.transpose(-1, -2)) * (hd ** -0.5), -1)
        c = torch.matmul(a, vp).permute(0, 2, 1, 3).reshape(b, n, E)
        return s.out(c), None


def build_net():
    from rfdetr import RFDETRSegNano
    m = RFDETRSegNano()
    net = m.model.model.eval()
    net.export()
    bb = None
    for mod in net.modules():
        if hasattr(mod, "encoder") and hasattr(getattr(mod, "encoder"), "layer") and hasattr(mod, "embeddings"):
            bb = mod
            break
    emb = bb.embeddings
    C = emb.cls_token.shape[-1]
    nreg = getattr(emb.config, "num_register_tokens", 0)
    N = GH * GW + 1
    _pos = emb.interpolate_pos_encoding(torch.zeros(1, N, C), R, R).detach()
    emb.interpolate_pos_encoding = lambda e, h, w, _p=_pos: _p
    for mod in net.modules():
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.GELU) or type(ch).__name__ in ("GELUActivation", "QuickGELUActivation"):
                setattr(mod, cn, TG())
    # LayerScale bake (exact): Mali ML Drift mis-executes `h + lambda*f(h)` (the broadcast-const MUL
    # feeding a residual ADD; device probe n4 corr 0.62 vs n5 `h + f(h)` corr 0.9999). Fold lambda
    # into the preceding Linear (MoGe-2 DINOv2 recipe) and drop the op.
    nbaked = 0
    for mod in net.modules():
        if hasattr(mod, "layer_scale1") and hasattr(mod, "attention") and hasattr(mod, "mlp"):
            lam1 = mod.layer_scale1.lambda1.data
            mod.attention.output.dense.weight.data.mul_(lam1[:, None])
            mod.attention.output.dense.bias.data.mul_(lam1)
            mod.layer_scale1 = nn.Identity()
            lam2 = mod.layer_scale2.lambda1.data
            mod.mlp.fc2.weight.data.mul_(lam2[:, None])
            mod.mlp.fc2.bias.data.mul_(lam2)
            mod.layer_scale2 = nn.Identity()
            nbaked += 1
    print(f"  LayerScale baked into dense/fc2 weights in {nbaked} backbone layers")
    nmha = 0
    for mod in net.modules():
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.MultiheadAttention):
                setattr(mod, cn, ManualMHA(ch))
                nmha += 1
    print(f"  nn.MultiheadAttention -> rank-4 ManualMHA in {nmha} decoder layers")
    nwin = emb.config.num_windows
    print(f"  RFDETR-Seg-Nano: {sum(p.numel() for p in net.parameters())/1e6:.1f}M params; "
          f"num_windows={nwin} registers={nreg} dec_registers={net.transformer.num_registers} "
          f"seg_blocks={len(net.segmentation_head.blocks)}")
    assert nwin == 1, "SegNano expected num_windows=1"
    assert net.transformer.num_registers == 0, "decoder register tokens not handled by this split"
    assert nreg == 0, "backbone register tokens not handled by this split"
    clspos = (emb.cls_token + _pos[:, :1]).detach().clone()
    pospatch = _pos[:, 1:].detach().clone()
    return net, bb, clspos, pospatch


def build_proposals(h, w):
    """gen_encoder_output_proposals for bbox_reparam (unsigmoid=False), single level, no masking.
    For a 26x26 grid every proposal lies in (0.019, 0.981) so the (0.01, 0.99) validity mask is
    all-True and the masked_fill is a no-op; the grid is image-independent -> baked constant."""
    gy, gx = torch.meshgrid(torch.linspace(0, h - 1, h, dtype=torch.float32),
                            torch.linspace(0, w - 1, w, dtype=torch.float32), indexing="ij")
    grid = torch.cat([gx.unsqueeze(-1), gy.unsqueeze(-1)], -1)
    scale = torch.tensor([w, h], dtype=torch.float32).reshape(1, 1, 1, 2)
    cxcy = (grid.unsqueeze(0) + 0.5) / scale
    wh = torch.ones_like(cxcy) * 0.05
    return torch.cat((cxcy, wh), -1).reshape(1, -1, 4)                       # [1, h*w, 4]


class GraphA(nn.Module):
    """(image, clspos[1,1,384], pospatch[1,676,384]) -> enc_class, enc_delta, memory*2.
    The position embedding (and the cls token, pre-added into clspos) is HOST-FED: ML Drift
    mis-executes compute chains that consume large BAKED constants (device probes f_const vs
    f_input), so no big constant may live inside the graph."""
    def __init__(s, net, inner):
        super().__init__()
        s.tr = net.transformer
        s.bb0 = net.backbone[0]
        s.inner = inner                            # WindowedDinov2WithRegistersBackbone

    def forward(s, x, clspos, pospatch):
        emb = s.inner.embeddings
        pe = emb.patch_embeddings(x)               # [1,676,384]
        h = torch.cat((clspos, pe + pospatch), dim=1)
        hs_all = [h]
        for l in s.inner.encoder.layer:
            o = l(h)
            h = o[0] if isinstance(o, tuple) else o
            hs_all.append(h)
        feats = []
        for stage, hstate in zip(s.inner.stage_names, hs_all):
            if stage in s.inner.out_features:
                if s.inner.config.apply_layernorm:
                    hstate = s.inner.layernorm(hstate)
                hstate = hstate[:, 1:]             # strip cls (num_register_tokens=0)
                hstate = hstate.reshape(1, GH, GW, -1).permute(0, 3, 1, 2).contiguous()
                feats.append(hstate)
        src = s.bb0.projector(feats)[0]            # MultiScaleProjector(P4) -> [1,256,26,26]
        memory = src.flatten(2).transpose(1, 2)    # [1, 676, 256]
        om = s.tr.enc_output_norm[0](s.tr.enc_output[0](memory))
        enc_class = s.tr.enc_out_class_embed[0](om)
        delta = s.tr.enc_out_bbox_embed[0](om)     # token-pointwise -> applying pre-topk is exact
        # raw delta out; the proposal-grid combine moves to the HOST (same baked-const rule).
        # memory is consumed (enc_output) AND a graph output -> Mali zeroes the output copy
        # ([1,N,C] output-and-consumed bug). x2 forces a separate buffer (exact in fp16); host halves.
        return enc_class, delta, memory * 2.0


class GraphB(nn.Module):
    """(memory, refpoint, query_feat) -> boxes, logits, masks.
    query_feat is a HOST-fed input (not a baked const) and the two-stage reparam combine runs on
    the host: ML Drift mis-executes compute chains on large baked constants (f_const vs f_input)."""
    def __init__(s, net):
        super().__init__()
        s.net = net
        s.tr = net.transformer
        s.seg = net.segmentation_head
        s.register_buffer("ss", torch.tensor([[GH, GW]], dtype=torch.long), persistent=False)
        s.register_buffer("lsi", torch.tensor([0], dtype=torch.long), persistent=False)

    def forward(s, memory, refpoint, query_feat):
        tgt = query_feat
        dec = s.tr.decoder(tgt, memory, memory_key_padding_mask=None, pos=None,
                           refpoints_unsigmoid=refpoint, level_start_index=s.lsi,
                           spatial_shapes=s.ss, spatial_shapes_hw=[(GH, GW)], valid_ratios=None)
        hs, ref = dec[:2]                                                   # export: hs[1,100,256]
        delta = s.net.bbox_embed(hs)
        bcxcy = delta[..., :2] * ref[..., 2:] + ref[..., :2]
        bwh = delta[..., 2:].exp() * ref[..., 2:]
        boxes = torch.cat([bcxcy, bwh], -1)
        logits = s.net.class_embed(hs)
        # seg mask branch: rebuild the projector map from memory (memory IS srcs[0] flattened)
        src = memory.transpose(1, 2).reshape(1, HID, GH, GW)
        sf = F.interpolate(src, size=(MH, MW), mode="bilinear", align_corners=False)
        for blk in s.seg.blocks:
            sf = blk(sf)
        sfp = s.seg.spatial_features_proj(sf)                               # [1,256,78,78]
        q = s.seg.query_features_proj(s.seg.query_features_block(hs))       # [1,100,256]
        m4 = torch.matmul(q.unsqueeze(1), sfp.reshape(1, 1, HID, MH * MW))  # rank-4 einsum
        masks = (m4 + s.seg.bias).reshape(1, NQ, MH, MW)
        return boxes, logits, masks


def host_select(enc_class, enc_delta, proposals, rp):
    """Host glue (later done in Kotlin): proposal-grid combine + topk-100 + gather + two-stage
    reparam combine with the learned refpoints. All per-token elementwise -> exact off-GPU."""
    cxcy = enc_delta[..., :2] * proposals[..., 2:] + proposals[..., :2]
    wh = enc_delta[..., 2:].exp() * proposals[..., 2:]
    enc_coord = torch.cat([cxcy, wh], -1)
    scores = enc_class.amax(-1)
    idx = scores.topk(NQ, dim=1).indices
    ts = torch.gather(enc_coord, 1, idx.unsqueeze(-1).expand(-1, -1, 4))
    rcxcy = rp[..., :2] * ts[..., 2:] + ts[..., :2]
    rwh = rp[..., 2:].exp() * ts[..., 2:]
    refpoint = torch.cat([rcxcy, rwh], -1)
    return refpoint, ts, idx


def opcheck(p, l):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=p); it.allocate_tensors()
    ops = collections.Counter(d.get("op_name", "?") for d in it._get_ops_details())
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    over = sum(1 for d in it.get_tensor_details() if len(d.get("shape", [])) > 4)
    print(f"[{l}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"[{l}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(p)/1e6:.1f}MB",
          "GPU-CLEAN" if not bad and not over else "BLOCKERS")
    return it, (not bad and not over)


def run_tflite(path, inputs):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path); it.allocate_tensors()
    for d in it.get_input_details():
        shp = tuple(d["shape"]); matched = None
        for arr in inputs:
            if tuple(arr.shape) == shp:
                matched = arr
                break
        assert matched is not None, f"no input for slot {d['name']} shape {shp}"
        it.set_tensor(d["index"], matched.astype(d["dtype"]))
    it.invoke()
    return [it.get_tensor(od["index"]) for od in it.get_output_details()]


def to_fp16(fp32, fp16):
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
    if os.path.exists(fp16):
        os.remove(fp16)
    q = quantizer.Quantizer(float_model=fp32)
    q.load_quantization_recipe(rm.get_quantization_recipe())
    q.quantize().export_model(fp16)
    return fp16


def stats(name, a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    c = np.corrcoef(a, b)[0, 1]; md = np.abs(a - b).max()
    print(f"  {name}: corr {c:.6f}  max|diff| {md:.4e}")
    return c, md


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    net, inner, clspos, pospatch = build_net()
    x = torch.randn(1, 3, R, R) * 0.5
    ga, gb = GraphA(net, inner).eval(), GraphB(net).eval()
    proposals = build_proposals(GH, GW)
    rp = net.refpoint_embed.weight[:NQ].unsqueeze(0).detach().clone()
    qf0 = net.query_feat.weight[:NQ].unsqueeze(0).detach().clone()

    with torch.no_grad():
        ref_coord, ref_cls, ref_masks = net.forward_export(x)              # torch full reference
        ec, ed, mem2 = ga(x, clspos, pospatch)
        mem = mem2 * 0.5                                                    # invert the x2 output trick
        refpoint, ts, idx = host_select(ec, ed, proposals, rp)
        sb, sl, sm = gb(mem, refpoint, qf0)
    print("ref     :", tuple(ref_coord.shape), tuple(ref_cls.shape), tuple(ref_masks.shape))
    print("graphA  :", tuple(ec.shape), tuple(ed.shape), tuple(mem.shape))
    print("graphB  :", tuple(sb.shape), tuple(sl.shape), tuple(sm.shape))
    stats("split-vs-torch boxes ", sb, ref_coord)
    stats("split-vs-torch logits", sl, ref_cls)
    stats("split-vs-torch masks ", sm, ref_masks)
    if cmd == "forward":
        sys.stdout.flush(); sys.exit()

    import litert_torch
    pa = f"{HERE}/rfsA.tflite"
    litert_torch.convert(ga, (x, clspos, pospatch)).export(pa)
    ita, okA = opcheck(pa, "rfsA")
    oa = run_tflite(pa, [x.numpy().astype(np.float32), clspos.numpy().astype(np.float32),
                         pospatch.numpy().astype(np.float32)])
    def by_shape(outs, shp):
        return next(o for o in outs if tuple(o.shape) == shp)
    ta_ec = by_shape(oa, tuple(ec.shape)); ta_ed = by_shape(oa, tuple(ed.shape))
    ta_mem = by_shape(oa, tuple(mem.shape)) * 0.5
    stats("A enc_class", ta_ec, ec.numpy())
    stats("A enc_delta", ta_ed, ed.numpy())
    stats("A memory   ", ta_mem, mem.numpy())

    pb = f"{HERE}/rfsB.tflite"
    litert_torch.convert(gb, (mem, refpoint, qf0)).export(pb)
    itb, okB = opcheck(pb, "rfsB")
    b_ins = [mem.numpy().astype(np.float32), refpoint.numpy().astype(np.float32),
             qf0.numpy().astype(np.float32)]
    ob = run_tflite(pb, b_ins)
    tb_boxes = by_shape(ob, tuple(sb.shape)); tb_logits = by_shape(ob, tuple(sl.shape)); tb_masks = by_shape(ob, tuple(sm.shape))
    stats("B boxes ", tb_boxes, sb.numpy())
    stats("B logits", tb_logits, sl.numpy())
    stats("B masks ", tb_masks, sm.numpy())

    ref_t, _, _ = host_select(torch.from_numpy(ta_ec), torch.from_numpy(ta_ed), proposals, rp)
    ob2 = run_tflite(pb, [ta_mem.astype(np.float32), ref_t.numpy().astype(np.float32),
                          qf0.numpy().astype(np.float32)])
    stats("E2E boxes ", by_shape(ob2, tuple(sb.shape)), ref_coord.numpy())
    stats("E2E logits", by_shape(ob2, tuple(sl.shape)), ref_cls.numpy())
    stats("E2E masks ", by_shape(ob2, tuple(sm.shape)), ref_masks.numpy())

    if cmd in ("fp16", "all"):
        to_fp16(pa, f"{HERE}/rfsA_fp16.tflite"); opcheck(f"{HERE}/rfsA_fp16.tflite", "rfsA_fp16")
        to_fp16(pb, f"{HERE}/rfsB_fp16.tflite"); opcheck(f"{HERE}/rfsB_fp16.tflite", "rfsB_fp16")
        x.numpy().astype(np.float32).tofile(f"{HERE}/rfsA_in.bin")
        clspos.numpy().astype(np.float32).tofile(f"{HERE}/rfsA_in_clspos.bin")
        pospatch.numpy().astype(np.float32).tofile(f"{HERE}/rfsA_in_pospatch.bin")
        np.save(f"{HERE}/host_clspos.npy", clspos.numpy())
        np.save(f"{HERE}/host_pospatch.npy", pospatch.numpy())
        mem.numpy().astype(np.float32).tofile(f"{HERE}/rfsB_in_memory.bin")
        refpoint.numpy().astype(np.float32).tofile(f"{HERE}/rfsB_in_ref.bin")
        qf0.numpy().astype(np.float32).tofile(f"{HERE}/rfsB_in_qf.bin")
        np.save(f"{HERE}/rfsA_ec.npy", ec.numpy()); np.save(f"{HERE}/rfsA_ed.npy", ed.numpy())
        np.save(f"{HERE}/rfsB_boxes.npy", sb.numpy()); np.save(f"{HERE}/rfsB_logits.npy", sl.numpy())
        np.save(f"{HERE}/rfsB_masks.npy", sm.numpy())
        np.save(f"{HERE}/host_proposals.npy", proposals.numpy())
        np.save(f"{HERE}/host_refpoint_embed.npy", rp.numpy())
        np.save(f"{HERE}/host_query_feat.npy", qf0.numpy())
        print("saved fp16 + device-probe + host-constant artifacts")
    sys.stdout.flush()
