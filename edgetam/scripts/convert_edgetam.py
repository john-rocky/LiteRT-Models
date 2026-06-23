#!/usr/bin/env python3
"""
EdgeTAM (on-device SAM2, CVPR 2025, Apache-2.0) -> LiteRT CompiledModel GPU (ML Drift).

Produces the three assets the Android app bundles:
  edgetam_encoder.tflite  image (1,3,1024,1024) -> flat [ie | fpn0 | fpn1]   (run once per image)
  edgetam_decoder.tflite  flat [ie | sparse | fpn0 | fpn1] -> masks (1,3,256,256)  (run per tap)
  edgetam_prompt.bin      prompt-encoder constants for the Kotlin point encoder

Both graphs use a single concatenated input/output so CompiledModel never has to map
same-sized tensors by order (ie & fpn1 are both 1048576 floats -> by-order mapping would swap them).

GPU-correctness patches (all verified numerically identical to the PyTorch model, and
verified to run correctly on a Pixel 8a ML Drift GPU -> mask_fg ~= 11264 on the circle self-test):

  * SE global average pool  (THE encoder GPU fix)
      RepViT SqueezeExcite does `x.mean((2,3))`, which litert-torch lowers to a single
      multi-axis SUM reducing ~65k spatial elements. The Mali ML Drift delegate mis-computes
      that reduction and returns NaN (FP32 too -> not an FP16 overflow), so every mask came
      out empty. Splitting it into two single-axis means (mean(3) then mean(2)) is numerically
      identical and the GPU computes it correctly. (Exact erf-GELU is left untouched -- it is
      GPU-correct here; the sigmoid-GELU approximation noticeably hurt mask quality.)

  * ZeroStuffConvT          decoder upscale ConvTranspose2d -> interpolate+mask+conv2d
                            (TRANSPOSE_CONV is rejected by the Pixel 8a delegate). The stuffing
                            mask MUST be a constant buffer built in __init__ (building it with
                            .repeat() at runtime lowers to BROADCAST_TO + 8-D tensors).
  * _dec4d                  4-D mask decoder (collapse the point-batch dim; keep all tensors <=4D).

Env: litert-torch (main), transformers>=5.12 (has EdgeTam natively -- do NOT add the
`all_tied_weights_keys` bridge, it breaks EdgeTam's post_init). If scipy's native extensions
are unavailable, stub the leaf modules as below.
"""
import sys, types, collections, math, os, numpy as np


# --- optional: stub scipy native leaves if the conversion env has a broken scipy build ---
class _Dummy:
    def __getattr__(s, n):
        if n.startswith("__") and n.endswith("__"): raise AttributeError(n)
        return _Dummy()
    def __call__(s, *a, **k): return _Dummy()
class _Leaf(types.ModuleType):
    def __getattr__(s, n):
        if n.startswith("__") and n.endswith("__"): raise AttributeError(n)
        return _Dummy()
for _nm in ["scipy.sparse.linalg._propack", "scipy.optimize._cobyla", "scipy.optimize._slsqp",
            "scipy.optimize._minpack", "scipy.optimize._lbfgsb", "scipy.optimize._zeros",
            "scipy.optimize._highs", "scipy.optimize._direct", "scipy.optimize._trlib",
            "scipy.optimize._group_columns", "scipy.optimize._bglu_dense"]:
    sys.modules[_nm] = _Leaf(_nm)

import torch, torch.nn as nn, torch.nn.functional as F, litert_torch
from transformers import EdgeTamModel
import transformers.models.edgetam.modeling_edgetam as M
import timm.layers.squeeze_excite as SQ
from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer import quantizer
from PIL import Image, ImageDraw

OUT = os.environ.get("EDGETAM_OUT", os.path.dirname(os.path.abspath(__file__)) + "/../app/src/main/assets")
OUT = os.path.abspath(OUT); os.makedirs(OUT, exist_ok=True)
IE, SP, F0, F1 = 1048576, 512, 2097152, 1048576           # ie(256*64*64) sparse(2*256) fpn0(32*256*256) fpn1(64*128*128)
GPU_BAD = {'GATHER_ND', 'GATHER', 'SELECT_V2', 'SELECT', 'PACK', 'SPLIT', 'CAST', 'TOPK_V2', 'BROADCAST_TO', 'WHILE', 'TRANSPOSE_CONV'}
RECIPE = [{"regex": ".*", "operation": "*", "algorithm_key": "float_casting",
           "op_config": {"weight_tensor_config": {"num_bits": 16, "dtype": "FLOAT"}}}]


# --- SE GPU fix: split the multi-axis mean into two single-axis means ---
def _se_forward_gpu(self, x):
    x_se = x.mean(3, keepdim=True).mean(2, keepdim=True)
    if self.add_maxpool:
        x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
    x_se = self.fc1(x_se); x_se = self.act(self.bn(x_se)); x_se = self.fc2(x_se)
    return x * self.gate(x_se)
SQ.SEModule.forward = _se_forward_gpu


# --- decoder patches ---
class ZeroStuffConvT(nn.Module):
    def __init__(s, ct, in_hw):
        super().__init__(); s.st = ct.stride[0]; s.k = ct.kernel_size[0]; s.oh = in_hw * s.st
        s.register_buffer("w", ct.weight.flip(2, 3).transpose(0, 1).contiguous()); s.bias = ct.bias
        mask = torch.zeros(1, 1, s.oh, s.oh); mask[:, :, ::s.st, ::s.st] = 1.0; s.register_buffer("mask", mask)
    def forward(s, x):
        return F.conv2d(F.interpolate(x, size=(s.oh, s.oh), mode="nearest") * s.mask, s.w, s.bias, padding=s.k - 1)[:, :, :s.oh, :s.oh]

def _dec4d(self, image_embeddings, image_positional_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings,
           multimask_output, high_resolution_features, attention_similarity=None, target_embedding=None, **kw):
    bs, nc, h, w = image_embeddings.shape; pb = sparse_prompt_embeddings.shape[1]
    ot = torch.cat([self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], 0).repeat(bs, pb, 1, 1)
    pe = torch.cat((ot, sparse_prompt_embeddings), 2).to(self.iou_token.weight.dtype)
    ie = (image_embeddings + dense_prompt_embeddings).repeat_interleave(pb, 0); ipe = image_positional_embeddings.repeat_interleave(pb, 0)
    pe, ie = self.transformer(point_embeddings=pe, image_embeddings=ie, image_positional_embeddings=ipe,
                              attention_similarity=attention_similarity, target_embedding=target_embedding, **kw)
    it = pe[:, :, 1, :]; mt = pe[:, :, 2:(2 + self.num_mask_tokens), :]; ie = ie.transpose(2, 3).view(bs * pb, nc, h, w)
    f0, f1 = high_resolution_features; f0 = f0.repeat_interleave(pb, 0); f1 = f1.repeat_interleave(pb, 0)
    ue = self.activation(self.upscale_layer_norm(self.upscale_conv1(ie) + f1)); ue = self.activation(self.upscale_conv2(ue) + f0)
    hyper = torch.stack([self.output_hypernetworks_mlps[i](mt[:, :, i, :]) for i in range(self.num_mask_tokens)], 2)
    _, nc2, h2, w2 = ue.shape; B = bs * pb
    masks = (hyper.view(B, self.num_mask_tokens, nc2) @ ue.view(B, nc2, h2 * w2)).view(B, self.num_mask_tokens, h2, w2)
    return masks[:, 1:], None, None, None   # multimask: drop the single-mask token, keep 3
M.EdgeTamMaskDecoder.forward = _dec4d


def opcheck(path, tag):
    it = Interpreter(model_path=path); it.allocate_tensors()
    ops = collections.Counter(d.get('op_name', '?') for d in it._get_ops_details())
    bad = {k: v for k, v in ops.items() if k in GPU_BAD}
    over = sum(1 for d in it.get_tensor_details() if len(d.get('shape', [])) > 4)
    print(f"{tag}: GPU_BAD={bad or 'NONE'} >4D={over} SUM={ops.get('SUM', 0)}")


def fp16(src, dst):
    if os.path.exists(dst): os.remove(dst)
    q = quantizer.Quantizer(src); q.load_quantization_recipe(RECIPE); q.quantize().export_model(dst)
    return os.path.getsize(dst) / 1e6


def main():
    m = EdgeTamModel.from_pretrained("yonigozlan/EdgeTAM-hf").eval()
    md = m.mask_decoder
    md.upscale_conv1 = ZeroStuffConvT(md.upscale_conv1, 64)
    md.upscale_conv2 = ZeroStuffConvT(md.upscale_conv2, 128)
    px = torch.randn(1, 3, 1024, 1024)

    # ---- ENCODER ----
    class EncFlat(nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, x):
            f = s.m.get_image_embeddings(x)                    # [fpn0(32), fpn1(64), image_emb(256)]
            return torch.cat([f[-1].reshape(-1), f[0].reshape(-1), f[1].reshape(-1)])[None]
    enc_fp32 = f"{OUT}/edgetam_encoder_fp32.tflite"
    litert_torch.convert(EncFlat(m).eval(), (px,)).export(enc_fp32)
    opcheck(enc_fp32, "encoder")
    print("encoder FP16 %.1fMB" % fp16(enc_fp32, f"{OUT}/edgetam_encoder.tflite")); os.remove(enc_fp32)

    # ---- DECODER (ipe + dense baked as constants) ----
    cap = {}
    hk = md.register_forward_pre_hook(lambda mod, a, kw: cap.update(kw), with_kwargs=True)
    with torch.no_grad():
        m(pixel_values=px, input_points=torch.tensor([[[[512., 512.]]]]), input_labels=torch.tensor([[[1]]]), multimask_output=True)
    hk.remove()
    ipe = cap['image_positional_embeddings'].detach().clone(); dn = cap['dense_prompt_embeddings'].detach().clone()

    class DecFlat(nn.Module):
        def __init__(s, d, ipe, dn): super().__init__(); s.d = d; s.register_buffer("ipe", ipe); s.register_buffer("dn", dn)
        def forward(s, flat):
            f = flat[0]
            ie = f[:IE].reshape(1, 256, 64, 64); sp = f[IE:IE + SP].reshape(1, 1, 2, 256)
            f0 = f[IE + SP:IE + SP + F0].reshape(1, 32, 256, 256); f1 = f[IE + SP + F0:].reshape(1, 64, 128, 128)
            return s.d(ie, s.ipe, sp, s.dn, multimask_output=True, high_resolution_features=[f0, f1])[0]
    dec_fp32 = f"{OUT}/edgetam_decoder_fp32.tflite"
    litert_torch.convert(DecFlat(md, ipe, dn).eval(), (torch.randn(1, IE + SP + F0 + F1),)).export(dec_fp32)
    opcheck(dec_fp32, "decoder")
    print("decoder FP16 %.1fMB" % fp16(dec_fp32, f"{OUT}/edgetam_decoder.tflite")); os.remove(dec_fp32)

    # ---- PROMPT constants for the Kotlin point encoder: [Gaussian(2x128) | point_embed[1] | not_a_point] ----
    pe_mod = m.prompt_encoder
    G = pe_mod.shared_embedding.positional_embedding                      # (2,128)
    buf = np.concatenate([G.detach().numpy().flatten(),
                          pe_mod.point_embed.weight[1].detach().numpy(),
                          pe_mod.not_a_point_embed.weight[0].detach().numpy()]).astype(np.float32)
    open(f"{OUT}/edgetam_prompt.bin", "wb").write(buf.tobytes())
    print("prompt.bin floats:", buf.size)

    # ---- end-to-end self-test (desktop interpreter, FP16) on a synthetic circle ----
    enc = Interpreter(model_path=f"{OUT}/edgetam_encoder.tflite"); enc.allocate_tensors()
    dec = Interpreter(model_path=f"{OUT}/edgetam_decoder.tflite"); dec.allocate_tensors()
    Gp = buf[:256]; pe1 = buf[256:512]; nap = buf[512:768]
    MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
    img = Image.new("RGB", (512, 512), "black"); ImageDraw.Draw(img).ellipse([136, 136, 376, 376], fill="white")
    a = (np.asarray(img.resize((1024, 1024), Image.BILINEAR)).astype(np.float32) / 255. - MEAN) / STD
    enc.set_tensor(enc.get_input_details()[0]['index'], a.transpose(2, 0, 1)[None].astype(np.float32)); enc.invoke()
    ef = enc.get_tensor(enc.get_output_details()[0]['index']).flatten()
    ie, f0, f1 = ef[:IE], ef[IE:IE + F0], ef[IE + F0:]
    cc = 2 * ((512.0 + 0.5) / 1024) - 1; sp = np.zeros(512, np.float32)
    for k in range(128):
        pr = 2 * math.pi * (cc * Gp[k] + cc * Gp[128 + k]); sp[k] = math.sin(pr) + pe1[k]; sp[128 + k] = math.cos(pr) + pe1[128 + k]
    sp[256:] = nap
    flat_in = np.concatenate([ie, sp, f0, f1]).astype(np.float32)[None]
    dec.set_tensor(dec.get_input_details()[0]['index'], flat_in); dec.invoke()
    mk = dec.get_tensor(dec.get_output_details()[0]['index'])[0]
    print("END-TO-END (circle) mask[0] fg=%d/%d  (expect ~11264)" % ((mk[0] > 0).sum(), mk[0].size))


if __name__ == "__main__":
    main()
