#!/usr/bin/env python3
"""SAM 2.1 Hiera-Tiny VIDEO path (memory attention + memory encoder) -> LiteRT GPU.

Phase 1 of the video-tracking port (2026-08-19). Builds on ``convert_sam2.py``
(image path: Hiera encoder + prompt mask decoder) and exports the four
stateless, fixed-shape per-frame sub-graphs of ``facebook/sam2.1-hiera-tiny``'s
tracking loop. The rolling memory bank and the per-frame orchestration live on
the host (Kotlin / Swift) -- see the STEP BOUNDARY block at the bottom of this
docstring and ``verify_sam2_video.py`` for the numpy reference of that host loop.

Graphs (single flat float32 input / output each, like the image-path graphs):

  sam2v_encode      image (1,3,1024,1024) -> [pix_raw(IE) | hi0(F0) | hi1(F1)]
                    pix_raw = Hiera top level WITHOUT the no-memory embedding
                    (the image-path encoder bakes it in; the video path needs the
                    raw features for memory attention and the memory encoder).
  sam2v_memcond{N}  [pix_raw(IE) | mem_spatial(N*4096*64) | slot_tpe(N*64) |
                     ptr_tok(64*64) | ptr_pos(64*64) | key_mask(N*4096+64)]
                    -> pix_feat(IE).  Memory attention over a FIXED bank of N
                    spatial memory slots + 16 object pointers (x4 tokens of 64).
                    Unused slots are masked (additive -1e9), which is numerically
                    identical to HF's variable-length bank. Exported for N=7
                    (SAM2 default num_maskmem) and N=2 (the team's "2-slot").
  sam2v_decode      [pix_feat(IE) | hi0(F0) | hi1(F1) | sparse(512) | nomem(1)]
                    -> [masks(4*256*256) | iou(4) | obj_ptr(4*256) | obj_score(1)]
                    Video mask decoder + object_pointer_proj. nomem=1 adds the
                    no-memory embedding (initial conditioning frame == image path),
                    nomem=0 for tracked frames (pix_feat from memcond). All 4
                    mask tokens are emitted; the host picks argmax IoU over 1..3
                    (multimask, what SAM2.1 uses for <=1 point incl. tracking).
  sam2v_memorize    [pix_raw(IE) | mask_for_mem(1024*1024) | occ(1)]
                    -> mem_spatial(4096*64)   (token-major: 4096 tokens x 64)
                    Memory encoder. mask_for_mem = sigmoid(hi-res logits)*20-10
                    for tracked frames, (hi-res logits>0)*20-10 on prompted
                    frames (host builds it). occ = 1 - is_obj_appearing adds the
                    occlusion spatial embedding.

Constants for the host (float32 .bin, see save_constants):
  sam2v_prompt.bin      [gaussian(2x128) | point_embed[1] | point_embed[0] | not_a_point]
  sam2v_track_sparse.bin sparse prompt for "no point" tracking frames (2x256)
  sam2v_mtpe.bin        memory temporal pos enc (7 x 64); slot at temporal offset r
                        (1..6 frames back) uses row r-1, the conditioning frame row 6
  sam2v_no_obj_ptr.bin  no-object pointer (256), used when obj_score <= 0
  sam2v_tpos_proj.bin   object-pointer temporal pos projection W(64x256) | b(64)

STEP BOUNDARY (per frame t, one object; the host keeps the state):
  1. encode(frame_t) -> pix_raw_t, hi0_t, hi1_t                (image only)
  2. t==prompt frame: pix_feat = decode(pix_raw, nomem=1, sparse=click)
     else:            pix_feat = memcond(pix_raw_t, bank)  where bank =
        {mem_spatial of the prompt frame + up to N-1 most recent tracked frames,
         each with slot_tpe = mtpe[offset-1] (prompt frame: mtpe[6]);
         obj_ptr tokens of the prompt frame + up to 15 most recent tracked frames,
         each with ptr_pos = tpos_proj(sine_1d((t - t_i)/15, 256)); key_mask}
        then decode(pix_feat, nomem=0, sparse=track_sparse)
  3. host: best = argmax iou[1:4]; mask_t = masks[best]; obj_ptr_t = ptr[best]
     if obj_score<=0: mask_t = -1024 everywhere, obj_ptr_t = no_obj_ptr
  4. memorize(pix_raw_t, mask_for_mem(mask_t), occ) -> mem_spatial_t; push
     (t, mem_spatial_t) and (t, obj_ptr_t) into the bank (rolling window).
  5. carried across frames: bank only (N x 1 MB spatial + 16 x 1 KB pointers);
     pix_raw/hi0/hi1 are per-frame and dropped after step 4.

Env: ~/venvs/ltconv040dev (litert-torch 0.9.3, transformers 5.14, ai-edge-litert
2.1.6, ai-edge-quantizer). Run:
  SAM2_OUT=sam2/scripts/output/video python sam2/scripts/convert_sam2_video.py
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_sam2 as base  # noqa: E402  (installs the Hiera 4-D patches on import)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import Sam2VideoModel  # noqa: E402
import transformers.models.sam2_video.modeling_sam2_video as MV  # noqa: E402

CKPT = os.environ.get("SAM2_CKPT", "facebook/sam2.1-hiera-tiny")
OUT = os.path.abspath(os.environ.get(
    "SAM2_OUT",
    os.path.dirname(os.path.abspath(__file__)) + "/output/video"))
os.makedirs(OUT, exist_ok=True)
NMM_LIST = [int(x) for x in os.environ.get("SAM2_NMM", "7,2").split(",")]

IE, F0, F1 = base.IE, base.F0, base.F1          # 1048576, 2097152, 1048576
HW = 4096                                       # 64x64 top-level tokens
MEMCH = 64                                      # memory channel dim
HD = 256                                        # memory-attention hidden (1 head)
NPTR_FRAMES, PTR_SPLIT = 16, 4                  # 16 pointers x 4 tokens of 64
NPTR = NPTR_FRAMES * PTR_SPLIT                  # 64 pointer tokens
MASK_NEG = -1e9
# Head-dim permutation that turns SAM2's pairwise-interleaved RoPE into the
# half-split (rotate_half) form: [even dims | odd dims]. Applied to the q/k
# projection rows, so q'.k' == q.k exactly (see reference_interleaved_rope_bake).
PERM = torch.tensor([2 * i for i in range(HD // 2)] + [2 * i + 1 for i in range(HD // 2)])


def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def deinterleave(c):
    """cos/sin (L, 256) with c[2i]==c[2i+1]  ->  (1,1,L,256) as [c_i | c_i]."""
    half = c[..., 0::2]
    return torch.cat([half, half], -1)[None, None]


class MemoryAttention4D(nn.Module):
    """Batch-first, <=4-D re-authoring of Sam2VideoMemoryAttention over a fixed bank.

    Numerically identical to HF (verified in verify_sam2_video.py) given the
    PERM-permuted q/k projections. Query = current pix_raw (4096 tokens),
    memory = N spatial slots x 4096 tokens + NPTR pointer tokens, all 64-ch.
    """

    def __init__(self, ma, nmm, vision_pos, mem_spatial_pos):
        super().__init__()
        self.layers = ma.layers
        self.layer_norm = ma.layer_norm
        self.nmm = nmm
        cos, sin = ma.rotary_emb()
        self.register_buffer("cos", deinterleave(cos.detach().clone()))     # (1,1,4096,256)
        self.register_buffer("sin", deinterleave(sin.detach().clone()))
        # 0.1 * sine position embedding of the top-level vision features (constant)
        self.register_buffer("vpos", (0.1 * vision_pos).reshape(1, HD, HW).transpose(
            1, 2).reshape(1, 1, HW, HD).contiguous())
        # constant spatial memory position encoding, token-major (1,1,4096,64)
        self.register_buffer("mpos", mem_spatial_pos.reshape(1, 1, HW, MEMCH).contiguous())

    def rope(self, x):
        return x * self.cos + rotate_half(x) * self.sin

    def forward(self, pix_raw, mem_spatial, slot_tpe, ptr_tok, ptr_pos, key_mask):
        # pix_raw (1,256,64,64) -> queries (1,1,4096,256)
        x = pix_raw.reshape(1, HD, HW).transpose(1, 2).reshape(1, 1, HW, HD) + self.vpos
        n = self.nmm
        # memory tokens (1,1,L,64) and their positions
        spatial = mem_spatial.reshape(1, n, HW, MEMCH)                       # (1,N,4096,64)
        spatial_pos = self.mpos + slot_tpe.reshape(1, n, 1, MEMCH)          # (1,N,4096,64)
        memory = torch.cat([spatial.reshape(1, 1, n * HW, MEMCH),
                            ptr_tok.reshape(1, 1, NPTR, MEMCH)], 2)
        mem_pos = torch.cat([spatial_pos.reshape(1, 1, n * HW, MEMCH),
                             ptr_pos.reshape(1, 1, NPTR, MEMCH)], 2)
        keys_in = memory + mem_pos
        km = key_mask.reshape(1, 1, 1, n * HW + NPTR)
        for layer in self.layers:
            # --- self attention (RoPE on q and k) ---
            h = layer.layer_norm1(x)
            sa = layer.self_attn
            q = self.rope(sa.q_proj(h))
            k = self.rope(sa.k_proj(h))
            v = sa.v_proj(h)
            a = torch.softmax((q @ k.transpose(-1, -2)) * sa.scaling, dim=-1)
            x = x + sa.o_proj(a @ v)
            # --- cross attention to memory (RoPE on q and spatial k only) ---
            h = layer.layer_norm2(x)
            ca = layer.cross_attn_image
            q = self.rope(ca.q_proj(h))
            k = ca.k_proj(keys_in)                                           # (1,1,L,256)
            k_sp = k[..., :n * HW, :].reshape(1, n, HW, HD)
            k_sp = (k_sp * self.cos + rotate_half(k_sp) * self.sin).reshape(1, 1, n * HW, HD)
            k = torch.cat([k_sp, k[..., n * HW:, :]], 2)
            v = ca.v_proj(memory)
            a = torch.softmax((q @ k.transpose(-1, -2)) * ca.scaling + km, dim=-1)
            x = x + ca.o_proj(a @ v)
            # --- MLP ---
            h = layer.layer_norm3(x)
            x = x + layer.linear2(layer.activation(layer.linear1(h)))
        x = self.layer_norm(x)                                               # (1,1,4096,256)
        return x.reshape(1, HW, HD).transpose(1, 2).reshape(1, IE)


def _vdec4d(self, image_embeddings, image_positional_embeddings,
            sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output,
            high_resolution_features, attention_similarity=None,
            target_embedding=None, **kwargs):
    """Sam2VideoMaskDecoder.forward, <=4-D, returning ALL 4 mask tokens.

    Same body as convert_sam2._dec4d plus the IoU head, the object-score head and
    the un-sliced mask tokens (host selects the multimask best-IoU among 1..3).
    """
    bs, nc, h, w = image_embeddings.shape
    pb = sparse_prompt_embeddings.shape[1]
    output_tokens = torch.cat(
        [self.obj_score_token.weight, self.iou_token.weight,
         self.mask_tokens.weight], 0).repeat(bs, pb, 1, 1)
    point_embeddings = torch.cat(
        (output_tokens, sparse_prompt_embeddings), 2).to(self.iou_token.weight.dtype)
    image_emb = (image_embeddings + dense_prompt_embeddings).repeat_interleave(pb, 0)
    image_pos = image_positional_embeddings.repeat_interleave(pb, 0)
    point_embeddings, image_emb = self.transformer(
        point_embeddings=point_embeddings, image_embeddings=image_emb,
        image_positional_embeddings=image_pos,
        attention_similarity=attention_similarity,
        target_embedding=target_embedding, **kwargs)
    iou_token_out = point_embeddings[:, :, 1, :]
    mask_tokens_out = point_embeddings[:, :, 2:(2 + self.num_mask_tokens), :]
    image_emb = image_emb.transpose(2, 3).view(bs * pb, nc, h, w)
    feat_s0, feat_s1 = high_resolution_features
    feat_s0 = feat_s0.repeat_interleave(pb, 0)
    feat_s1 = feat_s1.repeat_interleave(pb, 0)
    upscaled = self.activation(
        self.upscale_layer_norm(self.upscale_conv1(image_emb) + feat_s1))
    upscaled = self.activation(self.upscale_conv2(upscaled) + feat_s0)
    hyper = torch.stack(
        [self.output_hypernetworks_mlps[i](mask_tokens_out[:, :, i, :])
         for i in range(self.num_mask_tokens)], 2)
    _, nc2, h2, w2 = upscaled.shape
    batch = bs * pb
    masks = (hyper.view(batch, self.num_mask_tokens, nc2)
             @ upscaled.view(batch, nc2, h2 * w2)).view(
                 batch, self.num_mask_tokens, h2, w2)
    iou = self.iou_prediction_head(iou_token_out)
    obj = self.pred_obj_score_head(point_embeddings[:, :, 0, :])
    return masks, iou, mask_tokens_out, obj


MV.Sam2VideoMaskDecoder.forward = _vdec4d


class ConstPos(nn.Module):
    def __init__(self, const):
        super().__init__()
        self.register_buffer("c", const)

    def forward(self, *a, **k):
        return self.c


def permute_rope_projections(model):
    """Bake PERM into every memory-attention q/k projection (call exactly once)."""
    assert not getattr(model, "_rope_permuted", False)
    for layer in model.memory_attention.layers:
        for attn in (layer.self_attn, layer.cross_attn_image):
            for proj in (attn.q_proj, attn.k_proj):
                proj.weight.data = proj.weight.data[PERM].contiguous()
                proj.bias.data = proj.bias.data[PERM].contiguous()
    model._rope_permuted = True


def build(model):
    """Return (Encode, MemCond-factory, Decode, Memorize) wrapper modules."""
    permute_rope_projections(model)
    md = model.mask_decoder
    md.upscale_conv1 = base.ZeroStuffConvT(md.upscale_conv1, 64)
    md.upscale_conv2 = base.ZeroStuffConvT(md.upscale_conv2, 128)

    with torch.no_grad():
        ve = model.vision_encoder(torch.randn(1, 3, 1024, 1024), return_dict=True)
        vision_pos = ve.fpn_position_encoding[-1].detach().clone()          # (1,256,64,64)
        mem_pos = model.memory_encoder.position_encoding(
            (1, MEMCH, 64, 64), torch.device("cpu"), torch.float32).detach().clone()
        mem_pos_tok = mem_pos.reshape(1, MEMCH, HW).transpose(1, 2).contiguous()  # (1,4096,64)
        img_pos = model.get_image_wide_positional_embeddings().detach().clone()
    model.memory_encoder.position_encoding = ConstPos(mem_pos)

    class Encode(nn.Module):
        def __init__(self):
            super().__init__()
            self.ve = model.vision_encoder
            self.cs0, self.cs1 = md.conv_s0, md.conv_s1

        def forward(self, x):
            fpn = self.ve(x, return_dict=True).fpn_hidden_states
            return torch.cat([fpn[2].reshape(-1), self.cs0(fpn[0]).reshape(-1),
                              self.cs1(fpn[1]).reshape(-1)])[None]

    class MemCond(nn.Module):
        def __init__(self, nmm):
            super().__init__()
            self.nmm = nmm
            self.attn = MemoryAttention4D(model.memory_attention, nmm, vision_pos, mem_pos_tok)

        def forward(self, flat):
            n = self.nmm
            f = flat[0]
            o = 0
            pix = f[o:o + IE].reshape(1, HD, 64, 64); o += IE
            mem = f[o:o + n * HW * MEMCH]; o += n * HW * MEMCH
            tpe = f[o:o + n * MEMCH]; o += n * MEMCH
            ptr = f[o:o + NPTR * MEMCH]; o += NPTR * MEMCH
            ppos = f[o:o + NPTR * MEMCH]; o += NPTR * MEMCH
            km = f[o:o + n * HW + NPTR]
            return self.attn(pix, mem, tpe, ptr, ppos, km)

    class Decode(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = md
            self.optr = model.object_pointer_proj
            self.register_buffer("ipe", img_pos)
            self.register_buffer("dense", model.prompt_encoder.no_mask_embed.weight.reshape(
                1, -1, 1, 1).expand(1, 256, 64, 64).contiguous())
            self.register_buffer("no_mem", model.no_memory_embedding.detach().reshape(1, HD, 1, 1))

        def forward(self, flat):
            f = flat[0]
            pix = f[:IE].reshape(1, HD, 64, 64)
            h0 = f[IE:IE + F0].reshape(1, 32, 256, 256)
            h1 = f[IE + F0:IE + F0 + F1].reshape(1, 64, 128, 128)
            sparse = f[IE + F0 + F1:IE + F0 + F1 + 512].reshape(1, 1, 2, 256)
            nomem = f[IE + F0 + F1 + 512:].reshape(1, 1, 1, 1)
            pix = pix + nomem * self.no_mem
            masks, iou, tok, obj = self.d(pix, self.ipe, sparse, self.dense,
                                          multimask_output=True,
                                          high_resolution_features=[h0, h1])
            ptr = self.optr(tok)                                             # (1,1,4,256)
            return torch.cat([masks[0].reshape(-1), iou[0].reshape(-1),
                              ptr[0].reshape(-1), obj[0].reshape(-1)])[None]

    class Memorize(nn.Module):
        def __init__(self):
            super().__init__()
            self.e = model.memory_encoder
            self.register_buffer("occ", model.occlusion_spatial_embedding_parameter.detach(
            ).reshape(1, MEMCH))

        def forward(self, flat):
            f = flat[0]
            pix = f[:IE].reshape(1, HD, 64, 64)
            mfm = f[IE:2 * IE].reshape(1, 1, 1024, 1024)
            occ = f[2 * IE:].reshape(1, 1)
            mem, _ = self.e(pix, mfm)                                        # (1,64,64,64)
            mem = mem.reshape(1, MEMCH, HW).transpose(1, 2)                  # (1,4096,64)
            mem = mem + occ * self.occ
            return mem.reshape(1, HW * MEMCH)

    return Encode, MemCond, Decode, Memorize


def memcond_in_size(nmm):
    return IE + nmm * HW * MEMCH + nmm * MEMCH + 2 * NPTR * MEMCH + nmm * HW + NPTR


DEC_IN = IE + F0 + F1 + 512 + 1
DEC_OUT = 4 * 65536 + 4 + 4 * 256 + 1
MEM_IN = 2 * IE + 1
MEM_OUT = HW * MEMCH


def save_constants(model):
    pe = model.prompt_encoder
    g = pe.shared_embedding.positional_embedding.detach().numpy().flatten()
    prompt = np.concatenate([
        g, pe.point_embed.weight[1].detach().numpy(),
        pe.point_embed.weight[0].detach().numpy(),
        pe.not_a_point_embed.weight[0].detach().numpy()]).astype(np.float32)
    prompt.tofile(f"{OUT}/sam2v_prompt.bin")
    with torch.no_grad():
        ts, _ = pe(input_points=torch.zeros(1, 1, 1, 2),
                   input_labels=-torch.ones(1, 1, 1, dtype=torch.int32),
                   input_boxes=None, input_masks=None)
    ts.numpy().astype(np.float32).tofile(f"{OUT}/sam2v_track_sparse.bin")
    model.memory_temporal_positional_encoding.detach().numpy().astype(
        np.float32).tofile(f"{OUT}/sam2v_mtpe.bin")
    model.no_object_pointer.detach().numpy().astype(np.float32).tofile(
        f"{OUT}/sam2v_no_obj_ptr.bin")
    lin = model.temporal_positional_encoding_projection_layer
    np.concatenate([lin.weight.detach().numpy().flatten(),
                    lin.bias.detach().numpy()]).astype(np.float32).tofile(
                        f"{OUT}/sam2v_tpos_proj.bin")
    print("constants: prompt(1024) track_sparse(512) mtpe(448) no_obj_ptr(256) tpos_proj(16448)")


def convert(mod, example, name):
    fp32 = f"{OUT}/{name}_fp32.tflite"
    with torch.no_grad():
        mod(*example)
    base.litert_torch.convert(mod, tuple(t.detach().clone() for t in example)).export(fp32)
    base.opcheck(fp32, name)
    print(f"{name} FP16 %.1fMB" % base.fp16(fp32, f"{OUT}/{name}.tflite"))
    os.remove(fp32)


def main():
    model = Sam2VideoModel.from_pretrained(CKPT).eval()
    import litert_torch  # deferred (see convert_sam2.py)
    base.litert_torch = litert_torch
    print("baked pos_embed:", base.bake_pos_embed(model))
    Encode, MemCond, Decode, Memorize = build(model)
    save_constants(model)
    which = os.environ.get("SAM2_GRAPHS", "encode,memcond,decode,memorize").split(",")
    if "encode" in which:
        convert(Encode().eval(), (torch.randn(1, 3, 1024, 1024),), "sam2v_encode")
    if "memcond" in which:
        for n in NMM_LIST:
            convert(MemCond(n).eval(), (torch.randn(1, memcond_in_size(n)),), f"sam2v_memcond{n}")
    if "decode" in which:
        convert(Decode().eval(), (torch.randn(1, DEC_IN),), "sam2v_decode")
    if "memorize" in which:
        convert(Memorize().eval(), (torch.randn(1, MEM_IN),), "sam2v_memorize")
    print("FLAT IO: enc_out=%d | memcond_in=%s out=%d | decode_in=%d out=%d | memorize_in=%d out=%d" % (
        IE + F0 + F1, {n: memcond_in_size(n) for n in NMM_LIST}, IE, DEC_IN, DEC_OUT, MEM_IN, MEM_OUT))


if __name__ == "__main__":
    main()
