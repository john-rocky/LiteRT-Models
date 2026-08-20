#!/usr/bin/env python3
"""Stage-wise Metal-GPU bisection of the SAM3 head graph (all gpu_patches applied)."""
import os, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precheck_sam3 as P, gpu_patches as G
from build_sam3 import load_image
from ai_edge_litert.interpreter import Interpreter
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.options import Options

det = P.build_detector(os.path.join(P.ROOT, "models", "sam3.1_multiplex.pt"))
sizes = [(288, 288), (144, 144), (72, 72)]
x = load_image(os.path.join(P.ROOT, "vendor_sam3/assets/images/truck.jpg"))
tok = det.backbone.language_backbone.tokenizer(["wheel"], context_length=32)
with torch.inference_mode():
    vis_ref = P.VisionFlat(det)(x)
    txt_full = P.TextFlat(det)(tok)
G.apply_text_patches(det); G.apply_head_patches(det)
head = G.HeadFlat4d(det, sizes)
flat = torch.cat([vis_ref, txt_full], 1)
out = os.path.join(P.ROOT, "models", "precheck")
which = sys.argv[1:] or ["all"]


def check(mod, xin, name):
    import litert_torch
    with torch.inference_mode():
        ref = mod(xin).reshape(-1).numpy()
    p = os.path.join(out, f"dhead_{name}.tflite")
    litert_torch.convert(mod.eval(), (xin,)).export(p)
    if os.environ.get("SAVE_FIXTURES"):
        xin.numpy().astype(np.float32).tofile(os.path.join(out, f"fix_{name}_in.bin"))
        np.save(os.path.join(out, f"fix_{name}_ref.npy"), ref)
        if os.environ.get("NO_GPU"):
            print(f"[{name}] exported + fixtures saved", flush=True)
            return
    it = Interpreter(model_path=p); it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], xin.numpy()); it.invoke()
    ycpu = it.get_tensor(it.get_output_details()[0]["index"]).reshape(-1)
    res = []
    for f32 in (False, True):
        try:
            if f32:
                o = Options.create(); o.hardware_accelerators = HardwareAccelerator.GPU; o.gpu_options.enforce_f32 = True
                m = CompiledModel.from_file(p, options=o)
            else:
                m = CompiledModel.from_file(p, HardwareAccelerator.GPU)
            ib = m.create_input_buffers(0); ob = m.create_output_buffers(0); ib[0].write(xin.numpy().ravel())
            m.run_by_index(0, ib, ob); y = np.array(ob[0].read(ref.size, np.float32)); m.close()
            res.append(f"{'f32' if f32 else 'fp16'} corr={np.corrcoef(y, ref)[0,1]:.6f} max|d|={np.abs(y-ref).max():.3g}")
        except Exception as e:
            res.append(f"{'f32' if f32 else 'fp16'} FAILED {type(e).__name__}")
    print(f"[{name}] cpu corr={np.corrcoef(ycpu, ref)[0,1]:.6f} |ref|max={np.abs(ref).max():.3g}  gpu " + " / ".join(res), flush=True)


# ---- torch intermediates (mirror HeadFlat4d.forward) ----
L = 32
with torch.inference_mode():
    off = 0; fpn = []
    for (h, w), n in zip(sizes, head.n):
        fpn.append(flat[:, off:off + n].reshape(1, 256, h, w)); off += n
    text_mem = flat[:, off:off + L * 256].reshape(1, L, 256).transpose(0, 1); off += L * 256
    text_pad = flat[:, off:off + L]
    pos = list(head.pos)
    img_feats = [fpn[-1].flatten(2).permute(2, 0, 1)]
    img_pos = [pos[-1].flatten(2).permute(2, 0, 1)]
    ge = det.geometry_encoder
    g = head.geo_pre.clone()
    gmask = torch.zeros(1, 1)
    for lay in ge.encode:
        g = lay(tgt=g, memory=img_feats[-1], tgt_key_padding_mask=gmask, pos=img_pos[-1])
    g = ge.encode_norm(g)
    prompt = torch.cat([text_mem, g], 0)
    prompt_mask = torch.cat([text_pad, torch.zeros(1, 1)], 1)
    memory = det.transformer.encoder(src=img_feats.copy(), src_key_padding_mask=None, src_pos=img_pos.copy(),
                                     prompt=prompt, prompt_pos=torch.zeros_like(prompt),
                                     prompt_key_padding_mask=prompt_mask, feat_sizes=[(72, 72)],
                                     encoder_extra_kwargs=None)
    enc_hs = memory["memory"]


class GeoStage(nn.Module):
    """[fpn72 flat | dummy] -> geometry cls after encode layers (1,1,256)"""
    def forward(self, f):
        zero = torch.clamp(f[:, :1], min=0.0, max=0.0)
        feat = f[:, :256 * 72 * 72].reshape(1, 256, 72, 72).flatten(2).permute(2, 0, 1)
        g = head.geo_pre + zero
        gm = torch.zeros(1, 1) + zero
        for lay in ge.encode:
            g = lay(tgt=g, memory=feat, tgt_key_padding_mask=gm, pos=img_pos[-1])
        return ge.encode_norm(g).reshape(1, -1)


class EncStage(nn.Module):
    """[fpn72 | prompt(33*256) | prompt_mask(33)] -> enc_hs (5184*256)"""
    def __init__(self, nlayers=6):
        super().__init__(); self.nl = nlayers
    def forward(self, f):
        feat = f[:, :256 * 5184].reshape(1, 256, 72, 72).flatten(2).permute(2, 0, 1)
        pr = f[:, 256 * 5184:256 * 5184 + 33 * 256].reshape(1, 33, 256).transpose(0, 1)
        pm = f[:, 256 * 5184 + 33 * 256:]
        enc = det.transformer.encoder
        layers = enc.layers
        enc.layers = layers[:self.nl]
        try:
            m = enc(src=[feat], src_key_padding_mask=None, src_pos=[img_pos[-1]], prompt=pr,
                    prompt_pos=torch.zeros_like(pr), prompt_key_padding_mask=pm, feat_sizes=[(72, 72)],
                    encoder_extra_kwargs=None)
        finally:
            enc.layers = layers
        return m["memory"].reshape(1, -1)


class DecStage(nn.Module):
    """[enc_hs | prompt | prompt_mask] -> [hs_last(200*256) | ref_in(200*4) | presence(1)] (batch-first)"""
    def __init__(self, nlayers=6):
        super().__init__(); self.nl = nlayers
    def forward(self, f):
        e = f[:, :256 * 5184].reshape(1, 5184, 256)
        pr = f[:, 256 * 5184:256 * 5184 + 33 * 256].reshape(1, 33, 256)
        pm = f[:, 256 * 5184 + 33 * 256:]
        zero = torch.clamp(f[:, :1], min=0.0, max=0.0)
        dec = det.transformer.decoder
        layers = dec.layers
        dec.layers = layers[:self.nl]
        try:
            hs, rb, pres = G.decoder_forward_bf(dec, head.query_embed.unsqueeze(0) + zero, e,
                                                img_pos[-1].transpose(0, 1), head.ref_boxes0.unsqueeze(0) + zero, pr, pm)
        finally:
            dec.layers = layers
        return torch.cat([hs.reshape(1, -1), rb.reshape(1, -1), pres.reshape(1, -1)], 1)


class SegStage(nn.Module):
    """[fpn288 | fpn144 | enc_hs | prompt | pm | hs_last] -> masks (200*288*288)"""
    def forward(self, f):
        n0, n1 = 256 * 288 * 288, 256 * 144 * 144
        f0 = f[:, :n0].reshape(1, 256, 288, 288); f1 = f[:, n0:n0 + n1].reshape(1, 256, 144, 144)
        off = n0 + n1
        e = f[:, off:off + 256 * 5184].reshape(1, 5184, 256); off += 256 * 5184
        pr = f[:, off:off + 33 * 256].reshape(1, 33, 256); off += 33 * 256
        pm = f[:, off:off + 33]; off += 33
        hs = f[:, off:off + 200 * 256].reshape(1, 200, 256)
        sh = det.segmentation_head
        t2 = sh.cross_attn_norm(e)
        t2 = G.mha_core_bf(sh.cross_attend_prompt, t2, pr, pr, key_padding_mask=pm)
        e = (t2 + e).transpose(1, 2).reshape(1, 256, 72, 72)
        pixel = sh.pixel_decoder([f0, f1, e])
        inst = sh.instance_seg_head(pixel)
        return sh.mask_predictor(hs, inst).reshape(1, -1)


fpn72 = fpn[-1].reshape(1, -1)
if "geo" in which or "all" in which:
    check(GeoStage(), fpn72, "geo")
enc_in = torch.cat([fpn72, prompt.transpose(0, 1).reshape(1, -1), prompt_mask], 1)
if "enc" in which or "all" in which:
    check(EncStage(1), enc_in, "enc1")
    check(EncStage(6), enc_in, "enc6")
dec_in = torch.cat([enc_hs.transpose(0, 1).reshape(1, -1), prompt.transpose(0, 1).reshape(1, -1), prompt_mask], 1)
if "dec" in which or "all" in which:
    check(DecStage(1), dec_in, "dec1")
    check(DecStage(6), dec_in, "dec6")
if "seg" in which or "all" in which:
    with torch.inference_mode():
        y = DecStage(6)(dec_in)
    hs_last = y[:, :200 * 256]
    seg_in = torch.cat([fpn[0].reshape(1, -1), fpn[1].reshape(1, -1), enc_hs.transpose(0, 1).reshape(1, -1),
                        prompt.transpose(0, 1).reshape(1, -1), prompt_mask, hs_last], 1)
    check(SegStage(), seg_in, "seg")

# ---------------- decoder layer-0 internals ----------------
class DecPartsOLD(nn.Module):
    def __init__(self, mode): super().__init__(); self.mode = mode
    def forward(self, f):
        e = f[:, :256 * 5184].reshape(1, 5184, 256).transpose(0, 1)          # memory (HW,1,256)
        pr = f[:, 256 * 5184:256 * 5184 + 33 * 256].reshape(1, 33, 256).transpose(0, 1)
        pm = f[:, 256 * 5184 + 33 * 256:]
        zero = torch.clamp(f[:, :1], min=0.0, max=0.0)
        dec = det.transformer.decoder; layer = dec.layers[0]
        tgt = head.query_embed.unsqueeze(1) + zero                             # (200,1,256)
        ref = head.ref_boxes0.unsqueeze(1) + zero                              # (200,1,4)
        valid = torch.ones(1, 1, 2)
        rpi = ref[:, :, None] * torch.cat([valid, valid], -1)[None, :]         # (200,1,1,4)
        sine = G.gen_sineembed_clean(rpi[:, :, 0, :], dec.d_model)             # (200,1,512)
        if self.mode == "sine": return sine.reshape(1, -1)
        qpos = dec.ref_point_head(sine)                                        # (200,1,256)
        if self.mode == "qpos": return qpos.reshape(1, -1)
        mm = dec._get_rpb_matrix(ref, (72, 72))                                # (1,8,200,5184)
        if self.mode == "rpb": return mm.reshape(1, -1)
        mm = mm.flatten(0, 1)
        pres = dec.presence_token.weight[None].expand(1, 1, -1)
        # --- layer.forward replica ---
        tgt_o2o = torch.cat([pres, tgt], 0); qp_o2o = torch.cat([torch.zeros_like(pres), qpos], 0)
        q = k = tgt_o2o + qp_o2o
        t2 = layer.self_attn(q, k, tgt_o2o, attn_mask=None)[0]
        tgt_o2o = tgt_o2o + t2
        tgt2 = layer.norm2(tgt_o2o)
        if self.mode == "sa": return tgt2.reshape(1, -1)
        t2 = layer.ca_text(tgt2 + qp_o2o, pr, pr, key_padding_mask=pm)[0]
        tgt3 = layer.catext_norm(tgt2 + t2)
        if self.mode == "catext": return tgt3.reshape(1, -1)
        cam = torch.cat([torch.zeros_like(mm[:, :1, :]), mm], 1)
        t2 = layer.cross_attn(query=tgt3 + qp_o2o, key=e + img_pos[-1], value=e, attn_mask=cam, key_padding_mask=None)[0]
        tgt4 = layer.norm1(tgt3 + t2)
        if self.mode == "ca": return tgt4.reshape(1, -1)
        if self.mode == "ca_nobias":
            t2 = layer.cross_attn(query=tgt3 + qp_o2o, key=e + img_pos[-1], value=e, attn_mask=None, key_padding_mask=None)[0]
            return layer.norm1(tgt3 + t2).reshape(1, -1)
        tgt5 = layer.forward_ffn(tgt4)
        return tgt5.reshape(1, -1)
if "partsOLD" in which:
    for mode in ["sine", "qpos", "rpb", "sa", "catext", "ca_nobias", "ca", "ffn"]:
        check(DecParts(mode), dec_in, "dec0_" + mode)

class SineVar(nn.Module):
    def __init__(self, mode): super().__init__(); self.mode = mode
    def forward(self, f):
        zero = torch.clamp(f[:, :1], min=0.0, max=0.0)
        ref = head.ref_boxes0.unsqueeze(1) + zero                              # (200,1,4)
        if self.mode == "sine_seqfirst":
            return G.gen_sineembed_clean(ref, 256).reshape(1, -1)              # (200,1,512)
        if self.mode == "sine_batchfirst":
            return G.gen_sineembed_clean(ref.transpose(0, 1), 256).reshape(1, -1)   # (1,200,512)
        if self.mode == "sin_only_seq":
            return torch.sin(ref * 6.28).reshape(1, -1)
        if self.mode == "sin_only_batch":
            return torch.sin(ref.transpose(0, 1) * 6.28).reshape(1, -1)
        if self.mode == "mlp_seq":
            return det.transformer.decoder.ref_point_head(torch.cat([ref, ref] * 64, -1)).reshape(1, -1)
        if self.mode == "mlp_batch":
            return det.transformer.decoder.ref_point_head(torch.cat([ref, ref] * 64, -1).transpose(0, 1)).reshape(1, -1)
if "sine" in which:
    for mode in ["sine_seqfirst", "sine_batchfirst", "sin_only_seq", "sin_only_batch", "mlp_seq", "mlp_batch"]:
        check(SineVar(mode), dec_in[:, :10], mode)

class ZeroVar(nn.Module):
    def __init__(self, mode): super().__init__(); self.mode = mode
    def forward(self, f):
        if self.mode == "clamp": z = torch.clamp(f[:, :1], min=0.0, max=0.0)
        elif self.mode == "mul_tiny": z = f[:, :1] * 1e-30
        elif self.mode == "relu_pair": z = torch.relu(f[:, :1]) - torch.relu(f[:, :1] + 0.0) 
        elif self.mode == "minmax": z = torch.minimum(torch.relu(f[:, :1]), torch.zeros(1, 1))
        ref = head.ref_boxes0.unsqueeze(1) + z
        return torch.cat([z.reshape(1, -1), torch.sin(ref * 6.28).reshape(1, -1)], 1)
if "zero" in which:
    for mode in ["clamp", "mul_tiny", "minmax"]:
        check(ZeroVar(mode), dec_in[:, :10], "zero_" + mode)

class BVar(nn.Module):
    def __init__(self, mode): super().__init__(); self.mode = mode
    def forward(self, f):
        z = torch.clamp(f[:, :1], min=0.0, max=0.0)   # (1,1)
        r0 = head.ref_boxes0.unsqueeze(1)             # (200,1,4) const
        if self.mode == "add_out": return (r0 + z).reshape(1, -1)
        if self.mode == "add_out_z3": return (r0 + z.reshape(1, 1, 1)).reshape(1, -1)
        if self.mode == "add_out_zfull": return (r0 + z.reshape(1, 1, 1).expand(200, 1, 4)).reshape(1, -1)
        if self.mode == "sin_input": return torch.sin(f[:, :800].reshape(200, 1, 4) * 6.28).reshape(1, -1)
        if self.mode == "sin_input_b1": return torch.sin(f[:, :800].reshape(1, 200, 4) * 6.28).reshape(1, -1)
        if self.mode == "sin_input_4d": return torch.sin(f[:, :800].reshape(1, 1, 200, 4) * 6.28).reshape(1, -1)
        if self.mode == "add_out_4d": return (head.ref_boxes0.reshape(1, 1, 200, 4) + z.reshape(1, 1, 1, 1)).reshape(1, -1)
        if self.mode == "sin_add_4d": return torch.sin((head.ref_boxes0.reshape(1, 1, 200, 4) + z.reshape(1, 1, 1, 1)) * 6.28).reshape(1, -1)
if "bvar" in which:
    for mode in ["add_out", "add_out_z3", "add_out_zfull", "sin_input", "sin_input_b1", "sin_input_4d", "add_out_4d", "sin_add_4d"]:
        check(BVar(mode), dec_in[:, :1000], "b_" + mode)

class PostStage(nn.Module):
    """[hs_last(200*256) | ref_in(200*4) | presence(1) | prompt(33*256) | pm(33)] -> [logits | boxes]"""
    def __init__(self, mode="all"): super().__init__(); self.mode = mode
    def forward(self, f):
        off = 0
        hs = f[:, :200*256].reshape(1, 1, 200, 256); off += 200*256
        ref = f[:, off:off+800].reshape(1, 1, 200, 4); off += 800
        pres = f[:, off:off+1].reshape(1, 1); off += 1
        pr = f[:, off:off+33*256].reshape(1, 33, 256).transpose(0, 1); off += 33*256
        pm = f[:, off:off+33]
        dec = det.transformer.decoder
        if self.mode == "dot": return det.dot_prod_scoring(hs, pr, pm).reshape(1, -1)
        if self.mode == "anchor": return dec.bbox_embed(hs).reshape(1, -1)
        if self.mode == "invsig": return G.inverse_sigmoid_clean(ref).reshape(1, -1)
        if self.mode == "boxes": return (G.inverse_sigmoid_clean(ref) + dec.bbox_embed(hs)).sigmoid().reshape(1, -1)
        oc = det.dot_prod_scoring(hs, pr, pm)
        lg = G.inverse_sigmoid_clean(oc.sigmoid() * pres.sigmoid().reshape(1, 1, 1, 1))
        lg = torch.clamp(lg, min=-10.0, max=10.0)
        bx = (G.inverse_sigmoid_clean(ref) + dec.bbox_embed(hs)).sigmoid()
        return torch.cat([lg.reshape(1, -1), bx.reshape(1, -1)], 1)
if "post" in which:
    with torch.inference_mode():
        y = DecStage(6)(dec_in)
    post_in = torch.cat([y, prompt.transpose(0, 1).reshape(1, -1), prompt_mask], 1)
    for mode in ["dot", "anchor", "invsig", "boxes", "all"]:
        check(PostStage(mode), post_in, "post_" + mode)

class EncDec(nn.Module):
    """[fpn72 | prompt(33*256) | pm(33)] -> dec outputs (composed encoder + decoder)"""
    def forward(self, f):
        feat = f[:, :256 * 5184].reshape(1, 256, 72, 72).flatten(2).permute(2, 0, 1)
        pr = f[:, 256 * 5184:256 * 5184 + 33 * 256].reshape(1, 33, 256)
        pm = f[:, 256 * 5184 + 33 * 256:]
        zero = torch.clamp(f[:, :1], min=0.0, max=0.0)
        enc = det.transformer.encoder; dec = det.transformer.decoder
        m = enc(src=[feat], src_key_padding_mask=None, src_pos=[img_pos[-1]], prompt=pr.transpose(0, 1),
                prompt_pos=torch.zeros_like(pr.transpose(0, 1)), prompt_key_padding_mask=pm, feat_sizes=[(72, 72)],
                encoder_extra_kwargs=None)
        e = m["memory"].transpose(0, 1); pos_bf = m["pos_embed"].transpose(0, 1)
        hs, rb, pres = G.decoder_forward_bf(dec, head.query_embed.unsqueeze(0) + zero, e, pos_bf,
                                            head.ref_boxes0.unsqueeze(0) + zero, pr, pm)
        return torch.cat([hs.reshape(1, -1), rb.reshape(1, -1), pres.reshape(1, -1)], 1)
class GeoEnc(nn.Module):
    """[fpn72 | text(32*256) | pad(32)] -> enc_hs"""
    def forward(self, f):
        feat = f[:, :256 * 5184].reshape(1, 256, 72, 72).flatten(2).permute(2, 0, 1)
        tm = f[:, 256 * 5184:256 * 5184 + 32 * 256].reshape(1, 32, 256)
        tp = f[:, 256 * 5184 + 32 * 256:]
        zero = torch.clamp(f[:, :1], min=0.0, max=0.0)
        g = head.geo_pre + zero; gm = torch.zeros(1, 1) + zero
        for lay in ge.encode:
            g = lay(tgt=g, memory=feat, tgt_key_padding_mask=gm, pos=img_pos[-1])
        g = ge.encode_norm(g)
        pr = torch.cat([tm, g], 1); pm = torch.cat([tp, torch.zeros(1, 1)], 1)
        enc = det.transformer.encoder
        m = enc(src=[feat], src_key_padding_mask=None, src_pos=[img_pos[-1]], prompt=pr.transpose(0, 1),
                prompt_pos=torch.zeros_like(pr.transpose(0, 1)), prompt_key_padding_mask=pm, feat_sizes=[(72, 72)],
                encoder_extra_kwargs=None)
        return m["memory"].reshape(1, -1)
if "comp" in which:
    check(EncDec(), enc_in, "encdec")
    ge_in = torch.cat([fpn72, text_mem.transpose(0, 1).reshape(1, -1), text_pad], 1)
    check(GeoEnc(), ge_in, "geoenc")

class DecL0(nn.Module):
    """decoder layer-0 internals, batch-first (fp16 error bisection)"""
    def __init__(self, mode): super().__init__(); self.mode = mode
    def forward(self, f):
        e = f[:, :256 * 5184].reshape(1, 5184, 256)
        pr = f[:, 256 * 5184:256 * 5184 + 33 * 256].reshape(1, 33, 256)
        pm = f[:, 256 * 5184 + 33 * 256:]
        zero = torch.clamp(f[:, :1], min=0.0, max=0.0)
        dec = det.transformer.decoder; layer = dec.layers[0]
        pos = img_pos[-1].transpose(0, 1)
        tgt = head.query_embed.unsqueeze(0) + zero; ref = head.ref_boxes0.unsqueeze(0) + zero
        presence = dec.presence_token.weight[None]
        sine = G.gen_sineembed_clean(ref, dec.d_model)
        if self.mode == "sine": return sine.reshape(1, -1)
        qpos = dec.ref_point_head(sine)
        if self.mode == "qpos": return qpos.reshape(1, -1)
        mm = G.rpb_matrix_4d(dec, ref, (72, 72), batch_first=True)
        if self.mode == "rpb": return mm.reshape(1, -1)
        cam = torch.cat([torch.zeros_like(mm[:, :, :1, :]), mm], 2)
        t = torch.cat([presence, tgt], 1); qp = torch.cat([torch.zeros_like(presence), qpos], 1)
        q = t + qp
        t = t + G.mha_core_bf(layer.self_attn, q, q, t); t = layer.norm2(t)
        if self.mode == "sa": return t.reshape(1, -1)
        t = layer.catext_norm(t + G.mha_core_bf(layer.ca_text, t + qp, pr, pr, key_padding_mask=pm))
        if self.mode == "catext": return t.reshape(1, -1)
        if self.mode == "ca_nobias":
            return layer.norm1(t + G.mha_core_bf(layer.cross_attn, t + qp, e + pos, e)).reshape(1, -1)
        t = layer.norm1(t + G.mha_core_bf(layer.cross_attn, t + qp, e + pos, e, attn_bias=cam))
        if self.mode == "ca": return t.reshape(1, -1)
        t = layer.forward_ffn(t)
        if self.mode == "ffn": return t.reshape(1, -1)
        normed = dec.norm(t[:, 1:])
        return torch.sigmoid(dec.bbox_embed(normed) + G.inverse_sigmoid_clean(ref)).reshape(1, -1)
if "decl0" in which:
    for mode in ["sine", "qpos", "rpb", "sa", "catext", "ca_nobias", "ca", "ffn", "box"]:
        check(DecL0(mode), dec_in, "decl0_" + mode)
