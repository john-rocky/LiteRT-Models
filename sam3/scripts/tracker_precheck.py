#!/usr/bin/env python3
"""SAM 3.1 Object-Multiplex tracker: stage-2 feasibility (fixed-shape exports of the three
per-frame sub-graphs, one bucket = 16 objects, N memory slots, no host state machine).

  memenc   [pix_raw(256*72*72) | mux_mask(16*1008*1008 fg prob*2-1 already) | cond(16*1008*1008)]
           -> maskmem (256*72*72)          (SimpleMaskEncoder, multiplexed 16+16 channels)
  maskdec  [pix_feat_with_mem(256*72*72) | hi0(32*288*288) | hi1(64*144*144) | valid(16)]
           -> [masks(16*3*288*288) | iou(16*3) | obj_score(16) | sam_tokens(16*3*256)]
           (MultiplexMaskDecoder, propagation path, multimask)
  memattn  [pix_raw(72*72*256 tokens) | mem_image(N*5184*256) | mem_pos(N*5184*256 incl tpos)
            | maskmem(N*5184*256) | obj_ptr(P*16*256) | ptr_pos(P*16*256)]
           -> pix_feat_with_mem (5184*256)   (TransformerEncoderDecoupledCrossAttention,
           4 layers d256 8 heads RoPE 72x72; fixed N slots, P pointer frames)

This script builds the tracker from the merged checkpoint (tracker.* keys), runs each
piece in torch at the real shapes, reports sizes/timing, then tries a RAW litert-torch
export + opcheck (+ Mac GPU run) for each. Patches are NOT applied (precheck only).
Usage: tracker_precheck.py [memenc|maskdec|memattn|all] [--no-convert] [--gpu-mac]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import precheck_sam3 as P  # noqa: E402  (shims + helpers)


def build_tracker(ckpt_path, n_slots=7, n_ptr_frames=16):
    from sam3 import model_builder as mb
    from sam3.model.position_encoding import PositionEmbeddingSine
    mb._create_position_encoding = lambda precompute_resolution=None: PositionEmbeddingSine(
        num_pos_feats=256, normalize=True, scale=None, temperature=10000, precompute_resolution=None)
    # the multiplex maskmem / tracker builders call PositionEmbeddingSine(precompute_resolution=1008)
    # directly (allocates on "cuda"): neutralize the precompute for every construction
    _orig_init = PositionEmbeddingSine.__init__

    def _init_no_precompute(self, *args, **kw):
        kw["precompute_resolution"] = None
        if len(args) >= 5:
            args = args[:4]
        _orig_init(self, *args, **kw)
    PositionEmbeddingSine.__init__ = _init_no_precompute
    import sam3.model.vitdet as vitdet
    vitdet.addmm_act = lambda act, linear, x: act()(linear(x))
    torch.Tensor.pin_memory = lambda self, *a, **k: self
    from sam3.model.decoder import TransformerDecoder
    _orig = TransformerDecoder._get_coords
    TransformerDecoder._get_coords = staticmethod(
        lambda H, W, device: _orig(H, W, "cpu" if device == "cuda" else device))
    t0 = time.time()
    trk = mb.build_sam3_multiplex_video_model(checkpoint_path=None, load_from_HF=False,
                                              multiplex_count=16, use_fa3=False,
                                              use_rope_real=True, strict_state_dict_loading=False,
                                              device="cpu", compile=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    sd = {k[len("tracker.model."):]: v for k, v in ckpt.items() if k.startswith("tracker.model.")}
    missing, unexpected = trk.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("backbone.") and "freqs_cis" not in k]
    print(f"[ckpt] tracker.model.* keys={len(sd)} missing(non-backbone)={len(missing)} "
          f"unexpected={len(unexpected)} ({time.time()-t0:.0f}s)")
    if missing:
        print("  missing:", missing[:12])
    if unexpected:
        print("  unexpected:", unexpected[:12])
    del trk.backbone
    trk.backbone = None
    trk.eval()
    n_cplx = 0
    for m in trk.modules():
        fc = getattr(m, "freqs_cis", None)
        if isinstance(fc, torch.Tensor) and fc.is_complex():
            # real/imag pair is what the use_rope_real path consumes; keep a real dummy with
            # the same leading length so SimpleRoPEAttention's shape check still passes
            if "freqs_cis" in m._buffers:
                m._buffers["freqs_cis"] = torch.zeros(fc.shape[0], 1)
            else:
                m.freqs_cis = torch.zeros(fc.shape[0], 1)
            if hasattr(m, "freqs_cis_real") and not isinstance(m.freqs_cis_real, torch.Tensor):
                pass
            n_cplx += 1
    print(f"[patch] complex freqs_cis dropped: {n_cplx}")
    # mask downsampler: 1008 -> 1152 is an UPSAMPLE, antialias is a numerical no-op there
    # (max|diff| 4.8e-7) but has no litert-torch lowering (_upsample_bilinear2d_aa)
    import sam3.model.memory as mem
    import torch.nn.functional as F

    def _ds_forward(self, x):
        if self.interpol_size is not None and self.interpol_size != list(x.shape[-2:]):
            x = F.interpolate(x.float(), size=self.interpol_size, align_corners=False,
                              mode="bilinear", antialias=False)
        return self.encoder(x)
    mem.SimpleMaskDownSampler.forward = _ds_forward
    return trk


class MemEnc(nn.Module):
    """[pix_raw(1,256,72,72) flat | mux_mask(1,16,1008,1008) | cond(1,16,1008,1008)] -> maskmem flat"""
    def __init__(self, trk):
        super().__init__()
        self.enc = trk.maskmem_backbone
        self.n_pix = 256 * 72 * 72
        self.n_m = 16 * 1008 * 1008

    def forward(self, flat):
        pix = flat[:, :self.n_pix].reshape(1, 256, 72, 72)
        m = flat[:, self.n_pix:self.n_pix + self.n_m].reshape(1, 16, 1008, 1008)
        c = flat[:, self.n_pix + self.n_m:].reshape(1, 16, 1008, 1008)
        out = self.enc(pix, torch.cat([m, c], 1), skip_mask_sigmoid=True)
        return out["vision_features"].flatten(1)


class MaskDec(nn.Module):
    """[pix(1,256,72,72) | hi0(1,32,288,288) | hi1(1,64,144,144) | extra_emb(16*256)] ->
       [masks(1,16,3,288,288)->flat | iou(16*3) | obj_score(16) | tokens(16*3*256)]
    extra_emb = per-object output-suppression embedding (valid/invalid), computed on host."""
    def __init__(self, trk):
        super().__init__()
        self.trk = trk
        self.dec = trk.sam_mask_decoder
        self.register_buffer("pe", trk.get_propagation_dense_pe().detach().clone(), persistent=False)
        self.n_pix = 256 * 72 * 72
        self.n_h0 = 32 * 288 * 288
        self.n_h1 = 64 * 144 * 144

    def forward(self, flat):
        pix = flat[:, :self.n_pix].reshape(1, 256, 72, 72)
        o = self.n_pix
        h0 = flat[:, o:o + self.n_h0].reshape(1, 32, 288, 288); o += self.n_h0
        h1 = flat[:, o:o + self.n_h1].reshape(1, 64, 144, 144); o += self.n_h1
        emb = flat[:, o:o + 16 * 256].reshape(1, 16, 256)
        out = self.dec(image_embeddings=pix, image_pe=self.pe, high_res_features=[h0, h1],
                       multimask_output=True, extra_per_object_embeddings=emb)
        if not hasattr(self, "_printed"):
            self._printed = True
            print("[maskdec shapes]", {k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
        return torch.cat([out["masks"].flatten(1), out["iou_pred"].flatten(1),
                          out["object_score_logits"].flatten(1), out["sam_tokens_out"].flatten(1)], 1)


class InitDec(nn.Module):
    """Interactive (SAM-style) mask decoder used to initialise new objects from a detector
    mask (mask-as-output path): per object
       [pix(1,256,72,72) | hi0(1,32,288,288) | hi1(1,64,144,144) | sparse(1,S,256) | dense(1,256,72,72)]
       -> [low_res_masks(1,1,288,288) | iou(1) | token0(256) | obj_score(1)]
    (multimask_output=False, repeat_image=True). sparse/dense come from the SAM prompt encoder
    (1 padding point + the downscaled mask) which stays on the host/torch for now."""
    def __init__(self, trk, n_sparse):
        super().__init__()
        self.dec = trk.interactive_sam_mask_decoder
        # the mask-as-output init only consumes token 0 (obj_ptr) + object score; the stability-
        # based single/multi mask selection (ARG_MAX/SELECT/GATHER_ND) is irrelevant here
        self.dec.dynamic_multimask_via_stability = False
        self.register_buffer("pe", trk.interactive_sam_prompt_encoder.get_dense_pe().detach().clone(), persistent=False)
        self.n_pix = 256 * 72 * 72
        self.n_h0 = 32 * 288 * 288
        self.n_h1 = 64 * 144 * 144
        self.n_sp = n_sparse * 256
        self.n_sparse = n_sparse

    def forward(self, flat):
        o = 0
        pix = flat[:, o:o + self.n_pix].reshape(1, 256, 72, 72); o += self.n_pix
        h0 = flat[:, o:o + self.n_h0].reshape(1, 32, 288, 288); o += self.n_h0
        h1 = flat[:, o:o + self.n_h1].reshape(1, 64, 144, 144); o += self.n_h1
        sp = flat[:, o:o + self.n_sp].reshape(1, self.n_sparse, 256); o += self.n_sp
        de = flat[:, o:o + self.n_pix].reshape(1, 256, 72, 72)
        masks, ious, toks, score = self.dec(image_embeddings=pix, image_pe=self.pe, sparse_prompt_embeddings=sp,
                                            dense_prompt_embeddings=de, multimask_output=False, repeat_image=True,
                                            high_res_features=[h0, h1])
        if not hasattr(self, "_printed"):
            self._printed = True
            print("[initdec shapes]", tuple(masks.shape), tuple(ious.shape), tuple(toks.shape), tuple(score.shape))
        return torch.cat([masks.flatten(1), ious.flatten(1), toks[:, 0].flatten(1), score.flatten(1)], 1)


class MemAttn(nn.Module):
    """fixed N memory slots + P pointer frames (x16 objects); all slots assumed valid."""
    def __init__(self, trk, N=7, Pf=16):
        super().__init__()
        self.enc = trk.transformer.encoder
        self.N, self.Pf = N, Pf
        self.L = 5184
        self.C = 256
        sizes = [self.L * self.C, N * self.L * self.C, N * self.L * self.C, N * self.L * self.C,
                 Pf * 16 * self.C, Pf * 16 * self.C, N * self.L + Pf * 16]   # + key_keep mask
        self.offs = np.cumsum([0] + sizes).tolist()
        pe = trk.maskmem_backbone.position_encoding
        self.register_buffer("img_pos", pe(torch.zeros(1, 256, 72, 72)).flatten(2).permute(2, 0, 1)
                             .detach().clone(), persistent=False)                 # (L,1,C)

    def forward(self, flat):
        o = self.offs
        L, C, N, Pf = self.L, self.C, self.N, self.Pf
        pix = flat[:, o[0]:o[1]].reshape(1, L, C).transpose(0, 1)                # (L,1,C)
        mem_img = flat[:, o[1]:o[2]].reshape(1, N * L, C).transpose(0, 1)        # (N*L,1,C)
        mem_img_pos = flat[:, o[2]:o[3]].reshape(1, N * L, C).transpose(0, 1)
        maskmem = flat[:, o[3]:o[4]].reshape(1, N * L, C).transpose(0, 1)
        ptr = flat[:, o[4]:o[5]].reshape(1, Pf * 16, C).transpose(0, 1)          # (P*16,1,C)
        ptr_pos = flat[:, o[5]:o[6]].reshape(1, Pf * 16, C).transpose(0, 1)
        prompt = torch.cat([maskmem, ptr], 0)
        prompt_pos = torch.cat([mem_img_pos, ptr_pos], 0)
        keep = flat[:, o[6]:o[7]].reshape(1, 1, 1, -1)                          # 1 = valid key
        for m in self.enc.modules():
            if getattr(m, "rope_k_repeat", False):                              # cross attentions
                m.key_keep = keep
        out = self.enc(image=pix, src=pix, memory_image=mem_img, memory=prompt,
                       image_pos=self.img_pos, src_pos=self.img_pos,
                       memory_image_pos=mem_img_pos, memory_pos=prompt_pos,
                       num_obj_ptr_tokens=Pf * 16)
        return out["memory"].reshape(1, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="all")
    ap.add_argument("--no-convert", action="store_true")
    ap.add_argument("--gpu-mac", action="store_true")
    ap.add_argument("--slots", type=int, default=7)
    ap.add_argument("--ptr-frames", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(P.ROOT, "models", "tracker_precheck"))
    ap.add_argument("--patch", action="store_true", help="apply tracker_patches (4-D RoPE attention)")
    ap.add_argument("--chunks", type=int, default=9)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    trk = build_tracker(os.path.join(P.ROOT, "models", "sam3.1_multiplex.pt"))
    for name, mod in [("maskmem_backbone", trk.maskmem_backbone), ("sam_mask_decoder", trk.sam_mask_decoder),
                      ("memattn", trk.transformer), ("interactive_sam_mask_decoder", trk.interactive_sam_mask_decoder)]:
        print(f"[params] {name}: {sum(p.numel() for p in mod.parameters())/1e6:.2f}M")
    torch.manual_seed(0)
    do = {a.what} if a.what != "all" else {"memenc", "maskdec", "memattn", "initdec"}
    jobs = []
    if "memenc" in do:
        m = MemEnc(trk)
        x = torch.cat([torch.randn(1, 256 * 72 * 72), (torch.rand(1, 16 * 1008 * 1008) > 0.7).float() * 2 - 1,
                       torch.zeros(1, 16 * 1008 * 1008)], 1)
        jobs.append(("trk_memenc", m, x))
    if "maskdec" in do:
        m = MaskDec(trk)
        x = torch.cat([torch.randn(1, 256 * 72 * 72), torch.randn(1, 32 * 288 * 288), torch.randn(1, 64 * 144 * 144),
                       trk.output_valid_embed.detach().reshape(1, -1).repeat(1, 16)], 1)
        jobs.append(("trk_maskdec", m, x))
    if "initdec" in do:
        with torch.no_grad():
            pe_mod = trk.interactive_sam_prompt_encoder
            dummy_mask = (torch.rand(1, 1, 288, 288) > 0.5).float() * 20 - 10
            sp, de = pe_mod(points=(torch.zeros(1, 1, 2), -torch.ones(1, 1, dtype=torch.int32)), boxes=None, masks=dummy_mask)
            print("[initdec] prompt-encoder sparse", tuple(sp.shape), "dense", tuple(de.shape))
        m = InitDec(trk, sp.shape[1])
        x = torch.cat([torch.randn(1, 256 * 72 * 72), torch.randn(1, 32 * 288 * 288), torch.randn(1, 64 * 144 * 144),
                       sp.reshape(1, -1), de.reshape(1, -1)], 1)
        jobs.append(("trk_initdec", m, x))
    if "memattn" in do:
        m = MemAttn(trk, a.slots, a.ptr_frames)
        x = torch.randn(1, m.offs[-1]) * 0.5
        x[:, m.offs[6]:] = 1.0                                   # all keys valid for the precheck
        jobs.append((f"trk_memattn_n{a.slots}", m, x))
    for name, m, x in jobs:
        print(f"[{name}] input {tuple(x.shape)} ({x.numel()*4/1e6:.1f} MB)")
        try:
            ref = P.timeit(m, x, n=2, tag=name)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"[{name}] torch FAILED: {type(e).__name__}: {str(e)[:400]}")
            continue
        if a.patch and name.startswith("trk_memattn"):
            from tracker_patches import patch_memattn
            print("[patch] memattn RoPE attentions:", patch_memattn(trk, q_chunks=a.chunks, n_slots=a.slots))
            with torch.inference_mode():
                y = m(x)
            d = (y - ref).abs().max().item()
            corr = float(np.corrcoef(y.reshape(-1).numpy(), ref.reshape(-1).numpy())[0, 1])
            print(f"[parity memattn 4d vs stock] corr={corr:.7f} max|diff|={d:.3g}")
            ref = P.timeit(m, x, n=2, tag=name + "-4d")
        if a.no_convert:
            continue
        p = P.convert(m, x, name, a.out)
        if p:
            P.tflite_parity(p, x, ref, name)
            if a.gpu_mac:
                P.gpu_mac(os.path.join(a.out, f"{name}.tflite"), x, ref, name)
                P.gpu_mac(os.path.join(a.out, f"{name}.tflite"), x, ref, name, f32=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
