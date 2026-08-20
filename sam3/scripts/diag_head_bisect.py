import os, sys, types, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precheck_sam3 as P, gpu_patches as G
from build_sam3 import load_image, parity
det = P.build_detector(os.path.join(P.ROOT, "models", "sam3.1_multiplex.pt"))
sizes = [(288, 288), (144, 144), (72, 72)]
x = load_image(os.path.join(P.ROOT, "vendor_sam3/assets/images/truck.jpg"))
tok = det.backbone.language_backbone.tokenizer(["wheel"], context_length=32)
with torch.inference_mode():
    vis_ref = P.VisionFlat(det)(x); txt_full = P.TextFlat(det)(tok)
    stock_in = torch.cat([vis_ref, txt_full], 1)
    ref = P.HeadFlat(det, sizes)(stock_in)
from sam3.model.model_misc import MultiheadAttention, DotProductScoring
from sam3.model import model_misc, decoder as dm, maskformer_segmentation as ms
import sam3.model.sam3_image as si
def run(tag):
    with torch.inference_mode():
        parity(tag, P.HeadFlat(det, sizes)(stock_in), ref)
for m in det.modules():
    if isinstance(m, MultiheadAttention): m.forward = types.MethodType(G.sam3_mha_forward_4d, m)
run("+mha")
model_misc.inverse_sigmoid = G.inverse_sigmoid_clean; dm.inverse_sigmoid = G.inverse_sigmoid_clean; si.inverse_sigmoid = G.inverse_sigmoid_clean
run("+invsig")
dm.gen_sineembed_for_position = G.gen_sineembed_clean
run("+sine")
dec = det.transformer.decoder
dec.register_buffer("rpb_coords_h", (torch.arange(72.)/72).view(1,72,1), persistent=False)
dec.register_buffer("rpb_coords_w", (torch.arange(72.)/72).view(1,72,1), persistent=False)
dm.TransformerDecoder._get_rpb_matrix = G.rpb_matrix_4d
run("+rpb")
DotProductScoring.mean_pool_text = G.dot_prod_mean_pool_text
run("+dotprod")
ms.MaskPredictor.forward = G.mask_predictor_forward_4d
run("+maskpred")
pd = det.segmentation_head.pixel_decoder
pd.norms = torch.nn.ModuleList([G.GroupNorm4d(g) for g in pd.norms])
run("+gn")
with torch.inference_mode():
    hin = torch.cat([vis_ref, txt_full], 1)
    parity("HeadFlat4d", G.HeadFlat4d(det, sizes)(hin), ref)
with torch.inference_mode():
    y = G.HeadFlat4d(det, sizes)(hin)
for tag, sl in [("logits", slice(0,200)), ("boxes", slice(200,1000)), ("presence", slice(1000,1001)), ("masks", slice(1001,None))]:
    parity(tag, y[:, sl], ref[:, sl])
# hypothesis: float vs bool pad mask
hf = G.HeadFlat4d(det, sizes)
orig_fwd = hf.forward
import gpu_patches
src = open(gpu_patches.__file__).read()
# quick hack: run forward with the pad converted to bool by monkeypatching torch.cat? simpler: subclass
class HB(G.HeadFlat4d):
    def forward(self, flat):
        L = self.L
        n = sum(self.n)
        pad = flat[:, n + L*256:]
        # rebuild flat with the same values (no-op) but keep a bool view for a manual run
        return super().forward(flat)
with torch.inference_mode():
    # direct experiment: compare encoder outputs stock vs mine
    from sam3.model.data_misc import FindStage
    fpn = [vis_ref[:, sum(256*h*w for h,w in sizes[:i]):sum(256*h*w for h,w in sizes[:i+1])].reshape(1,256,*sizes[i]) for i in range(3)]
    tm = txt_full[:, :32*256].reshape(1,32,256).transpose(0,1)
    tp = txt_full[:, 32*256:]
    pos = [det.backbone.vision_backbone.position_encoding(torch.zeros(1,256,h,w)) for h,w in sizes]
    bo = {"backbone_fpn": fpn, "vision_pos_enc": pos, "language_features": tm, "language_mask": tp > 0.5}
    fi = FindStage(img_ids=torch.tensor([0]), text_ids=torch.tensor([0]), input_boxes=None, input_boxes_mask=None, input_boxes_label=None, input_points=None, input_points_mask=None)
    from sam3.model.geometry_encoders import Prompt
    geo = Prompt(box_embeddings=torch.zeros(0,1,4), box_mask=torch.zeros(1,0,dtype=torch.bool))
    prompt_s, pmask_s, bo2 = det._encode_prompt(bo, fi, geo)
    bo3, enc_s, _ = det._run_encoder(bo2, fi, prompt_s, pmask_s)
    # mine
    img_feats = [fpn[-1].flatten(2).permute(2,0,1)]; img_pos = [pos[-1].flatten(2).permute(2,0,1)]
    geo2 = Prompt(box_embeddings=torch.zeros(0,1,4), box_mask=torch.zeros(1,0,dtype=torch.bool))
    gf, gm = det.geometry_encoder(geo_prompt=geo2, img_feats=img_feats, img_sizes=[(72,72)], img_pos_embeds=img_pos)
    prompt_m = torch.cat([tm, gf], 0); pmask_m = torch.cat([tp, torch.zeros(1, gf.shape[0])], 1)
    parity("prompt", prompt_m, prompt_s); print("masks", pmask_m, pmask_s.float())
    mem = det.transformer.encoder(src=img_feats.copy(), src_key_padding_mask=None, src_pos=img_pos.copy(), prompt=prompt_m, prompt_pos=torch.zeros_like(prompt_m), prompt_key_padding_mask=pmask_m, feat_sizes=[(72,72)], encoder_extra_kwargs=None)
    parity("encoder", mem["memory"], enc_s["encoder_hidden_states"])
    mem2 = det.transformer.encoder(src=img_feats.copy(), src_key_padding_mask=None, src_pos=img_pos.copy(), prompt=prompt_m, prompt_pos=torch.zeros_like(prompt_m), prompt_key_padding_mask=pmask_s, feat_sizes=[(72,72)], encoder_extra_kwargs=None)
    parity("encoder(boolmask)", mem2["memory"], enc_s["encoder_hidden_states"])
with torch.inference_mode():
    out = {"encoder_hidden_states": enc_s["encoder_hidden_states"]}
    out_s, hs_s = det._run_decoder(memory=out["encoder_hidden_states"], pos_embed=enc_s["pos_embed"], src_mask=enc_s["padding_mask"], out=out, prompt=prompt_s, prompt_mask=pmask_s, encoder_out=enc_s)
    dec = det.transformer.decoder
    tgt = dec.query_embed.weight.unsqueeze(1)
    hs, rb, pres, _ = dec(tgt=tgt, memory=mem["memory"], memory_key_padding_mask=mem["padding_mask"], pos=mem["pos_embed"], reference_boxes=None, level_start_index=mem["level_start_index"], spatial_shapes=mem["spatial_shapes"], valid_ratios=mem["valid_ratios"], tgt_mask=None, memory_text=prompt_m, text_attention_mask=pmask_m, apply_dac=False)
    parity("hs", hs.transpose(1,2), hs_s)
    print("valid_ratios", mem["valid_ratios"], enc_s["valid_ratios"], "spatial", mem["spatial_shapes"], enc_s["spatial_shapes"], "lsi", mem["level_start_index"], enc_s["level_start_index"])
    hs2, rb2, pres2, _ = dec(tgt=tgt, memory=mem["memory"], memory_key_padding_mask=mem["padding_mask"], pos=mem["pos_embed"], reference_boxes=None, level_start_index=mem["level_start_index"], spatial_shapes=mem["spatial_shapes"], valid_ratios=mem["valid_ratios"], tgt_mask=None, memory_text=prompt_m, text_attention_mask=pmask_s, apply_dac=False)
    parity("hs(boolmask)", hs2.transpose(1,2), hs_s)
