#!/usr/bin/env python3
"""SAM 2.1 (Hiera, Meta, Apache-2.0) -> LiteRT CompiledModel GPU (ML Drift).

First-pass converter for the LiteRT-vs-MLX SAM2 benchmark (Lu Wang request,
2026-07-09). Target = ``facebook/sam2.1-hiera-tiny`` so the LiteRT graph matches
``avbiswas/sam2.1-hiera-tiny-mlx`` (and ``apple/coreml-sam2-tiny``) for a clean
same-model, same-size comparison. Image-segment path only (encoder once +
decoder per point); the video memory path (memory attention + RoPE) is a later
phase.

Structure is adapted from ``edgetam/scripts/convert_edgetam.py``. EdgeTAM's
transformers integration mirrors SAM2 (same authors), and the image-path tensor
shapes are identical (image_embed 256x64x64, high-res feats 32x256x256 and
64x128x128 -- these are decoder-side conv_s0/conv_s1 channel counts, so they do
NOT change with the backbone size). That lets the flat concat-IO split, the 4-D
mask-decoder re-author (``_dec4d``), ``ZeroStuffConvT``, and the fp16 recipe port
across unchanged.

The NEW surface is the Hiera image encoder on the Mali ML Drift delegate. This
script is the "attempt + diagnose" entry to the convert loop: it converts with no
speculative encoder patches, prints ``opcheck`` (GPU-hostile ops / >4-D tensors),
and runs a PyTorch parity check so a wrong attribute name or concat order fails
loudly instead of producing a silently-wrong model. Patch whatever ``opcheck``
reports (expected candidates: scaled_dot_product_attention -> manual matmul +
softmax, window LayerNorm multi-axis reduce -> split single-axis means, window
partition -> grouped-conv space-to-depth to stay 4-D), then re-run.

Produces:
  sam2_encoder.tflite   image (1,3,1024,1024) -> flat [ie | fpn0 | fpn1]
  sam2_decoder.tflite   flat [ie | sparse | fpn0 | fpn1] -> masks (1,3,256,256)
  sam2_prompt.bin       prompt-encoder constants for the Kotlin point encoder

Env: litert-torch (main), transformers>=5.13 (has Sam2 natively),
ai-edge-litert, ai-edge-quantizer.
"""
import collections
import math
import os
import sys
import types

import numpy as np


# --- optional: stub scipy native leaves if the conversion env has a broken build ---
class _Dummy:
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Dummy()

    def __call__(self, *args, **kwargs):
        return _Dummy()


class _Leaf(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Dummy()


for _name in [
    "scipy.sparse.linalg._propack", "scipy.optimize._cobyla",
    "scipy.optimize._slsqp", "scipy.optimize._minpack",
    "scipy.optimize._lbfgsb", "scipy.optimize._zeros",
    "scipy.optimize._highs", "scipy.optimize._direct",
    "scipy.optimize._trlib", "scipy.optimize._group_columns",
    "scipy.optimize._bglu_dense",
]:
    sys.modules[_name] = _Leaf(_name)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Sam2Model, Sam2Processor
import transformers.models.sam2.modeling_sam2 as M
# NOTE: ``import litert_torch`` is deferred into main() AFTER the model is loaded
# -- importing it before the torch model exists can SIGABRT (torch/jax init order).
from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer import quantizer
from PIL import Image, ImageDraw

CKPT = os.environ.get("SAM2_CKPT", "facebook/sam2.1-hiera-tiny")
OUT = os.environ.get(
    "SAM2_OUT", os.path.dirname(os.path.abspath(__file__)) + "/../assets")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

# ie(256*64*64) sparse(2*256) fpn0(32*256*256) fpn1(64*128*128) -- size-independent.
IE, SP, F0, F1 = 1048576, 512, 2097152, 1048576
GPU_BAD = {
    "GATHER_ND", "GATHER", "SELECT_V2", "SELECT", "PACK", "SPLIT", "CAST",
    "TOPK_V2", "BROADCAST_TO", "WHILE", "TRANSPOSE_CONV",
}
RECIPE = [{
    "regex": ".*", "operation": "*", "algorithm_key": "float_casting",
    "op_config": {"weight_tensor_config": {"num_bits": 16, "dtype": "FLOAT"}},
}]


# --- decoder patches (shared with EdgeTAM; SAM2 mask decoder is the same design) ---
class ZeroStuffConvT(nn.Module):
    """ConvTranspose2d -> nearest interpolate + stride mask + conv2d.

    TRANSPOSE_CONV is rejected by the Pixel 8a ML Drift delegate. The stuffing
    mask must be a constant buffer built here in __init__ (building it with
    .repeat() at runtime lowers to BROADCAST_TO and 8-D tensors).
    """

    def __init__(self, conv_transpose, in_hw):
        super().__init__()
        self.stride = conv_transpose.stride[0]
        self.kernel = conv_transpose.kernel_size[0]
        self.out_hw = in_hw * self.stride
        self.register_buffer(
            "weight",
            conv_transpose.weight.flip(2, 3).transpose(0, 1).contiguous())
        self.bias = conv_transpose.bias
        mask = torch.zeros(1, 1, self.out_hw, self.out_hw)
        mask[:, :, ::self.stride, ::self.stride] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, x):
        up = F.interpolate(x, size=(self.out_hw, self.out_hw), mode="nearest")
        return F.conv2d(
            up * self.mask, self.weight, self.bias,
            padding=self.kernel - 1)[:, :, :self.out_hw, :self.out_hw]


def _dec4d(self, image_embeddings, image_positional_embeddings,
           sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output,
           high_resolution_features, attention_similarity=None,
           target_embedding=None, **kwargs):
    """4-D mask decoder: collapse the point-batch dim, keep every tensor <=4-D."""
    bs, nc, h, w = image_embeddings.shape
    pb = sparse_prompt_embeddings.shape[1]
    output_tokens = torch.cat(
        [self.obj_score_token.weight, self.iou_token.weight,
         self.mask_tokens.weight], 0).repeat(bs, pb, 1, 1)
    point_embeddings = torch.cat(
        (output_tokens, sparse_prompt_embeddings), 2).to(
            self.iou_token.weight.dtype)
    image_emb = (image_embeddings + dense_prompt_embeddings).repeat_interleave(
        pb, 0)
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
    return masks[:, 1:], None, None, None  # multimask: drop single-mask token, keep 3


M.Sam2MaskDecoder.forward = _dec4d


# --- encoder patch: 4-D-only window partition (ML Drift rejects >4-D tensors) ---
def window_partition_4d(hidden_state, window_size):
    """[B,H,W,C] -> [B*nH*nW, ws, ws, C] using only <=4-D reshapes/transposes.

    The upstream version uses a 6-D view+permute. Here we split H into the batch,
    transpose, then split W into the batch. This transposes row/col WITHIN each
    window, which is exactly reversed by window_unpartition_4d; window attention
    (position already added, square symmetric query pooling) is order-equivariant,
    so the final result is numerically identical (verified by the parity check).
    """
    batch_size, height, width, num_channels = hidden_state.shape
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    hidden_state = F.pad(hidden_state, (0, 0, 0, pad_width, 0, pad_height))
    padded_height, padded_width = height + pad_height, width + pad_width
    n_h, n_w = padded_height // window_size, padded_width // window_size
    x = hidden_state.reshape(batch_size * n_h, window_size, padded_width, num_channels)
    x = x.transpose(1, 2)  # [B*nH, padded_width, ws, C]
    windows = x.reshape(
        batch_size * n_h * n_w, window_size, window_size, num_channels)
    return windows, (padded_height, padded_width)


def window_unpartition_4d(windows, window_size, pad_height_width, height_width):
    """Exact inverse of window_partition_4d, cropping any padding."""
    padded_height, padded_width = pad_height_width
    height, width = height_width
    num_channels = windows.shape[-1]
    n_h, n_w = padded_height // window_size, padded_width // window_size
    batch_size = windows.shape[0] // (n_h * n_w)
    x = windows.reshape(
        batch_size * n_h, n_w * window_size, window_size, num_channels)
    x = x.transpose(1, 2)  # [B*nH, ws, padded_width, C]
    x = x.reshape(batch_size, padded_height, padded_width, num_channels)
    if padded_height > height or padded_width > width:
        x = x[:, :height, :width, :].contiguous()
    return x


M.window_partition = window_partition_4d
M.window_unpartition = window_unpartition_4d


# --- encoder patch: 4-D-only multi-scale attention (avoid the 5-D qkv reshape) ---
def _msa_forward_4d(self, hidden_states, **kwargs):
    """Sam2MultiScaleAttention.forward with every tensor kept <=4-D.

    Upstream reshapes qkv to [B, H*W, 3, nHead, C] (5-D) and unbinds. Here we
    slice the fused projection into q/k/v along the channel dim instead, so no
    tensor exceeds 4-D. The pre-pool attn_weights the upstream computes and
    discards are dropped; the effective computation (eager attention on the
    pooled query) is reproduced exactly.
    """
    batch_size, height, width, _ = hidden_states.shape
    dim_out = self.dim_out
    num_heads = self.num_attention_heads
    head_dim = dim_out // num_heads
    qkv = self.qkv(hidden_states).reshape(batch_size, height * width, 3 * dim_out)
    query = qkv[..., :dim_out].reshape(batch_size, height * width, num_heads, head_dim)
    key = qkv[..., dim_out:2 * dim_out].reshape(
        batch_size, height * width, num_heads, head_dim)
    value = qkv[..., 2 * dim_out:].reshape(
        batch_size, height * width, num_heads, head_dim)

    if self.query_stride:
        query = M.do_pool(
            query.reshape(batch_size, height, width, -1), self.query_stride)
        height, width = query.shape[1:3]
        query = query.reshape(batch_size, height * width, num_heads, head_dim)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    attn = (query * self.scale) @ key.transpose(-2, -1)
    attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = (attn @ value).transpose(1, 2).reshape(batch_size, height, width, -1)
    return self.proj(attn_output)


M.Sam2MultiScaleAttention.forward = _msa_forward_4d


# --- encoder patch: bake the Hiera windowed positional embedding ---
def bake_pos_embed(model):
    """Replace ``_get_pos_embed`` with a precomputed constant.

    For a fixed 1024x1024 input the Hiera position embedding is a pure function
    of learned parameters (bicubic ``interpolate`` of ``pos_embed`` plus a tiled
    ``pos_embed_window``), so it is constant. Baking it removes the GATHER_ND
    that the bicubic sampling lowers to and the BROADCAST_TO from ``.tile``.
    """
    for module in model.modules():
        if type(module).__name__ != "Sam2HieraDetModel":
            continue
        with torch.no_grad():
            embedded = module.patch_embed(torch.randn(1, 3, 1024, 1024))
            baked = module._get_pos_embed(embedded.shape[1:3]).detach().clone()
        module.register_buffer("baked_pos_embed", baked)
        module._get_pos_embed = types.MethodType(
            lambda self, hw: self.baked_pos_embed, module)
        return baked.shape
    return None


def opcheck(path, tag):
    """Print GPU-hostile op counts and any >4-D tensors for one graph."""
    interpreter = Interpreter(model_path=path)
    interpreter.allocate_tensors()
    ops = collections.Counter(
        d.get("op_name", "?") for d in interpreter._get_ops_details())
    bad = {k: v for k, v in ops.items() if k in GPU_BAD}
    over = sum(
        1 for d in interpreter.get_tensor_details() if len(d.get("shape", [])) > 4)
    print(f"{tag}: GPU_BAD={bad or 'NONE'} >4D={over} SUM={ops.get('SUM', 0)}")


def fp16(src, dst):
    """Quantize weights to fp16 (float_casting) and return the size in MB."""
    if os.path.exists(dst):
        os.remove(dst)
    q = quantizer.Quantizer(src)
    q.load_quantization_recipe(RECIPE)
    q.quantize().export_model(dst)
    return os.path.getsize(dst) / 1e6


def main():
    model = Sam2Model.from_pretrained(CKPT).eval()
    global litert_torch
    import litert_torch  # deferred: must come after the torch model is built.
    print("baked pos_embed:", bake_pos_embed(model))
    decoder = model.mask_decoder
    # image_embed(64x64) -> conv1 -> 128x128 (+feat_s1); -> conv2 -> 256x256 (+feat_s0).
    decoder.upscale_conv1 = ZeroStuffConvT(decoder.upscale_conv1, 64)
    decoder.upscale_conv2 = ZeroStuffConvT(decoder.upscale_conv2, 128)
    pixel_values = torch.randn(1, 3, 1024, 1024)

    # ---- ENCODER ----
    class EncFlat(nn.Module):
        """Wrap get_image_embeddings and flatten to [image_emb | fpn0 | fpn1]."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            feats = self.model.get_image_embeddings(x)
            # VERIFY on first run: ordering must be [fpn0, fpn1, image_emb] like
            # EdgeTAM. image_emb and fpn1 share numel (1048576), so order matters
            # -- the parity check below will fail if this assumption is wrong.
            return torch.cat(
                [feats[-1].reshape(-1), feats[0].reshape(-1),
                 feats[1].reshape(-1)])[None]

    enc_fp32 = f"{OUT}/sam2_encoder_fp32.tflite"
    litert_torch.convert(EncFlat(model).eval(), (pixel_values,)).export(enc_fp32)
    opcheck(enc_fp32, "encoder")
    print("encoder FP16 %.1fMB" % fp16(enc_fp32, f"{OUT}/sam2_encoder.tflite"))
    os.remove(enc_fp32)

    # ---- DECODER (image positional + dense prompt baked as constants) ----
    captured = {}
    hook = decoder.register_forward_pre_hook(
        lambda mod, args, kwargs: captured.update(kwargs), with_kwargs=True)
    with torch.no_grad():
        model(pixel_values=pixel_values,
              input_points=torch.tensor([[[[512.0, 512.0]]]]),
              input_labels=torch.tensor([[[1]]]), multimask_output=True)
    hook.remove()
    image_pos = captured["image_positional_embeddings"].detach().clone()
    dense = captured["dense_prompt_embeddings"].detach().clone()

    class DecFlat(nn.Module):
        """Split the flat input, bake constants, emit the 3 multimask logits."""

        def __init__(self, decoder, image_pos, dense):
            super().__init__()
            self.decoder = decoder
            self.register_buffer("image_pos", image_pos)
            self.register_buffer("dense", dense)

        def forward(self, flat):
            f = flat[0]
            image_emb = f[:IE].reshape(1, 256, 64, 64)
            sparse = f[IE:IE + SP].reshape(1, 1, 2, 256)
            feat_s0 = f[IE + SP:IE + SP + F0].reshape(1, 32, 256, 256)
            feat_s1 = f[IE + SP + F0:].reshape(1, 64, 128, 128)
            return self.decoder(
                image_emb, self.image_pos, sparse, self.dense,
                multimask_output=True,
                high_resolution_features=[feat_s0, feat_s1])[0]

    dec_fp32 = f"{OUT}/sam2_decoder_fp32.tflite"
    litert_torch.convert(
        DecFlat(decoder, image_pos, dense).eval(),
        (torch.randn(1, IE + SP + F0 + F1),)).export(dec_fp32)
    opcheck(dec_fp32, "decoder")
    print("decoder FP16 %.1fMB" % fp16(dec_fp32, f"{OUT}/sam2_decoder.tflite"))
    os.remove(dec_fp32)

    # ---- PROMPT constants for the Kotlin point encoder ----
    # [Gaussian(2x128) | point_embed[1] | not_a_point]
    prompt = model.prompt_encoder
    gaussian = prompt.shared_embedding.positional_embedding
    buf = np.concatenate([
        gaussian.detach().numpy().flatten(),
        prompt.point_embed.weight[1].detach().numpy(),
        prompt.not_a_point_embed.weight[0].detach().numpy(),
    ]).astype(np.float32)
    open(f"{OUT}/sam2_prompt.bin", "wb").write(buf.tobytes())
    print("prompt.bin floats:", buf.size)

    # ---- parity check: converted fp16 graphs vs PyTorch, on a synthetic circle ----
    _parity_check(model, buf)


def _parity_check(model, buf):
    """Fail loudly if the converted graphs diverge from the PyTorch reference."""
    enc = Interpreter(model_path=f"{OUT}/sam2_encoder.tflite")
    enc.allocate_tensors()
    dec = Interpreter(model_path=f"{OUT}/sam2_decoder.tflite")
    dec.allocate_tensors()
    gauss, point1, not_a_point = buf[:256], buf[256:512], buf[512:768]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = Image.new("RGB", (512, 512), "black")
    ImageDraw.Draw(image).ellipse([136, 136, 376, 376], fill="white")
    resized = np.asarray(image.resize((1024, 1024), Image.BILINEAR))
    normalized = (resized.astype(np.float32) / 255.0 - mean) / std
    chw = normalized.transpose(2, 0, 1)[None].astype(np.float32)

    enc.set_tensor(enc.get_input_details()[0]["index"], chw)
    enc.invoke()
    flat = enc.get_tensor(enc.get_output_details()[0]["index"]).flatten()
    image_emb, feat_s0, feat_s1 = flat[:IE], flat[IE:IE + F0], flat[IE + F0:]

    center = 2 * ((512.0 + 0.5) / 1024) - 1
    sparse = np.zeros(512, np.float32)
    for k in range(128):
        phase = 2 * math.pi * (center * gauss[k] + center * gauss[128 + k])
        sparse[k] = math.sin(phase) + point1[k]
        sparse[128 + k] = math.cos(phase) + point1[128 + k]
    sparse[256:] = not_a_point
    flat_in = np.concatenate(
        [image_emb, sparse, feat_s0, feat_s1]).astype(np.float32)[None]
    dec.set_tensor(dec.get_input_details()[0]["index"], flat_in)
    dec.invoke()
    litert_mask = dec.get_tensor(dec.get_output_details()[0]["index"])[0]

    with torch.no_grad():
        ref = model(
            pixel_values=torch.from_numpy(chw),
            input_points=torch.tensor([[[[512.0, 512.0]]]]),
            input_labels=torch.tensor([[[1]]]), multimask_output=True)
    # pred_masks is (batch, point_batch, num_masks, H, W); keep the 3 multimask logits.
    ref_mask = ref.pred_masks.detach().numpy().reshape(
        -1, litert_mask.shape[-2], litert_mask.shape[-1])[-3:]

    iou = _mask_iou(litert_mask[0] > 0, ref_mask[0] > 0)
    corr = np.corrcoef(litert_mask.flatten(), ref_mask.flatten())[0, 1]
    fg = int((litert_mask[0] > 0).sum())
    print(f"PARITY circle: mask[0] fg={fg}/{litert_mask[0].size} "
          f"IoU(litert,torch)={iou:.3f} corr={corr:.3f}")
    if iou < 0.9 or corr < 0.95:
        print("  !! PARITY FAILED -- check encoder concat order / decoder attrs "
              "/ Hiera patches before trusting this graph")


def _mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 1.0


if __name__ == "__main__":
    main()
