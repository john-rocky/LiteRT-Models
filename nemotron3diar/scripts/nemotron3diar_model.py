"""Plain-PyTorch re-authoring of nvidia/Nemotron-3-Diarization for LiteRT export (no transformers).

graph A  N3DFrontend: mel [1,104,128] -> 8-frame stacking [1,13,1024] -> Linear 1024->512 (no bias)
         -> chunk_embeds [1,13,512]. The speaker cache and the FIFO hold these rows (pre input-LN).
graph B  N3DEncoder (fixed T; low_latency T=541 = 264 cache + 264 FIFO + 9 chunk + 4 look-ahead):
         packed_embeds [1,T,512] (L real rows + zero rows), attn_bias [1,1,1,T] (0 valid key,
         -3e4 pad key, added to the scores), rope_cos / rope_sin [1,1,T,64] (host table, positions
         0..T-1) -> input LN -> 31 x [LN -> RoPE MHA (rank-4 matmul/softmax) -> +res, LN -> fc1 ->
         GELU(erf) -> fc2 -> +res] -> LN -> proj 512->192 -> row mask -> sub-pixel Conv1d 192->1536
         (k=3, pad=1) -> [1,8T,192] -> ReLU -> Linear 192->192 -> ReLU -> Linear 192->8
         -> logits [1,8T,8] (sigmoid on the host).

The row mask zeroes the proj output of the pad rows so the k=3 convolution sees, at the last real
row, the same zero padding the reference sees at the end of its length-L sequence. It is derived
from attn_bias in-graph, so the I/O stays the four inputs above:
  row_mask="relu" (low_latency)   relu(bias + 1): 1 on valid rows (bias 0), 0 on pad rows (bias -3e4).
  row_mask="two_level" (offline)  bias 0 = valid, -16384 = a real row whose key is masked (the offline pass
         masks the frame after the last full hop, which still runs through the head), -32768 = pad row;
         y = bias * 2^-14 + 2 in {2, 1, 0} -> relu(y) - relu(y - 1) = {1, 1, 0}. Both forms are exact in fp32
         and fp16 (powers of two), and neither lowers to RELU_0_TO_1.

LayerNorm modes (ln_mode, all 64 LayerNorms at once; the parameters and keys are the same):
  plain  nn.LayerNorm.
  safe   SafeLayerNorm v2 (the down-scaled LayerNorm of the Parakeet LiteRT conversion): per row
         S = clamp(amax/8, min 1), normalize x/S and never multiply the variance back by S^2. The LN inputs
         reach |x| ~ 956, so the plain (x - mu)^2 (~9e5) overflows fp16 (65504); every safe-mode intermediate
         stays within O(amax). The eps is divided by S^2 so fp32 equals the plain LN to rounding.
  safe_guide  the guide's formula verbatim (eps added in the down-scaled domain = eps * S^2 in the
         original units); shifts the fp32 logits by up to ~3e-3 here, kept only as the measured record.

Checkpoint key map (model.safetensors, transformers layout -> this file):
  model.audio_tower.embedder.projection.weight              -> N3DFrontend.proj.weight      [512,1024]
  model.audio_tower.input_layer_norm.{weight,bias}          -> N3DEncoder.ln_in             [512]
  model.audio_tower.layers.{i}.layer_norm1.{weight,bias}    -> N3DEncoder.layers[i].ln1     [512]
  model.audio_tower.layers.{i}.self_attn.q_proj.weight      -> N3DEncoder.layers[i].q       [512,512]
  model.audio_tower.layers.{i}.self_attn.k_proj.weight      -> N3DEncoder.layers[i].k       [512,512]
  model.audio_tower.layers.{i}.self_attn.v_proj.weight      -> N3DEncoder.layers[i].v       [512,512]
  model.audio_tower.layers.{i}.self_attn.o_proj.{weight,bias} -> N3DEncoder.layers[i].o     [512,512],[512]
  model.audio_tower.layers.{i}.layer_norm2.{weight,bias}    -> N3DEncoder.layers[i].ln2     [512]
  model.audio_tower.layers.{i}.mlp.fc1.{weight,bias}        -> N3DEncoder.layers[i].fc1     [2048,512],[2048]
  model.audio_tower.layers.{i}.mlp.fc2.{weight,bias}        -> N3DEncoder.layers[i].fc2     [512,2048],[512]
  model.audio_tower.layer_norm.{weight,bias}                -> N3DEncoder.ln_out            [512]
  model.proj.{weight,bias}                                  -> N3DEncoder.proj              [192,512],[192]
  model.upsampler.conv.{weight,bias}                        -> N3DEncoder.up                [1536,192,3],[1536]
  classifier.dense.{weight,bias}                            -> N3DEncoder.dense             [192,192],[192]
  classifier.out_proj.{weight,bias}                         -> N3DEncoder.out               [8,192],[8]
  silence_embeds                                            -> host only (speaker-cache padding) [512]
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file

HIDDEN = 512
HEADS = 8
HEAD_DIM = 64
FFN = 2048
LAYERS = 31
MEL = 128
STACK = 8
HEAD_HIDDEN = 192
SPEAKERS = 8
LN_EPS = 1e-5
ROPE_THETA = 10000.0
PAD_BIAS = -3.0e4
PAD_BIAS_TWO_LEVEL = -32768.0
MASKED_BIAS_TWO_LEVEL = -16384.0
T_LOW_LATENCY = 541
T_OFFLINE = 684
MEL_FRAMES_LOW_LATENCY = 104


class N3DFrontend(nn.Module):
  """mel [1,F,128] (F a multiple of 8) -> chunk_embeds [1,F/8,512]."""

  def __init__(self):
    super().__init__()
    self.proj = nn.Linear(STACK * MEL, HIDDEN, bias=False)

  def forward(self, mel):
    b, f, m = mel.shape
    return self.proj(mel.reshape(b, f // STACK, STACK * m))


def rotate_half(x):
  x1 = x[..., : HEAD_DIM // 2]
  x2 = x[..., HEAD_DIM // 2 :]
  return torch.cat((-x2, x1), dim=-1)


class SafeLayerNormGuide(nn.LayerNorm):
  """SafeLayerNorm v2 as written in the guide: eps is added in the down-scaled domain, so the
  effective eps is eps * s^2 (a 1e-3-scale logit shift on this model, kept for the record)."""

  def forward(self, x):
    amax = x.abs().amax(-1, keepdim=True)
    s = (amax * 0.125).clamp(min=1.0)
    xs = x / s
    d = xs - xs.mean(-1, keepdim=True)
    var = (d * d).mean(-1, keepdim=True)  # down-scaled variance, never multiplied by s^2
    return d * torch.rsqrt(var + self.eps) * self.weight + self.bias


class SafeLayerNorm(nn.LayerNorm):
  """SafeLayerNorm v2 with the eps scaled into the down-scaled domain (eps / s^2), so fp32 matches
  nn.LayerNorm to rounding. eps / s^2 is O(1e-5) or smaller and never overflows fp16."""

  def forward(self, x):
    amax = x.abs().amax(-1, keepdim=True)
    s = (amax * 0.125).clamp(min=1.0)
    xs = x / s
    d = xs - xs.mean(-1, keepdim=True)
    var = (d * d).mean(-1, keepdim=True)  # down-scaled variance, never multiplied by s^2
    return d * torch.rsqrt(var + self.eps / (s * s)) * self.weight + self.bias


LN_MODES = {"plain": nn.LayerNorm, "safe": SafeLayerNorm, "safe_guide": SafeLayerNormGuide}


class N3DLayer(nn.Module):

  def __init__(self, ln_mode="plain"):
    super().__init__()
    ln = LN_MODES[ln_mode]
    self.ln1 = ln(HIDDEN, eps=LN_EPS)
    self.q = nn.Linear(HIDDEN, HIDDEN, bias=False)
    self.k = nn.Linear(HIDDEN, HIDDEN, bias=False)
    self.v = nn.Linear(HIDDEN, HIDDEN, bias=False)
    self.o = nn.Linear(HIDDEN, HIDDEN, bias=True)
    self.ln2 = ln(HIDDEN, eps=LN_EPS)
    self.fc1 = nn.Linear(HIDDEN, FFN)
    self.fc2 = nn.Linear(FFN, HIDDEN)
    self.scaling = HEAD_DIM**-0.5

  def forward(self, x, attn_bias, cos, sin):
    b, t, _ = x.shape
    h = self.ln1(x)
    q = self.q(h).reshape(b, t, HEADS, HEAD_DIM).transpose(1, 2)  # [1,8,T,64]
    k = self.k(h).reshape(b, t, HEADS, HEAD_DIM).transpose(1, 2)
    v = self.v(h).reshape(b, t, HEADS, HEAD_DIM).transpose(1, 2)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling + attn_bias  # [1,8,T,T]
    probs = torch.softmax(scores, dim=-1)
    a = torch.matmul(probs, v).transpose(1, 2).reshape(b, t, HIDDEN)
    x = x + self.o(a)
    return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class N3DEncoder(nn.Module):

  def __init__(self, ln_mode="plain", row_mask="relu"):
    super().__init__()
    assert row_mask in ("relu", "two_level"), row_mask
    self.row_mask = row_mask
    ln = LN_MODES[ln_mode]
    self.ln_in = ln(HIDDEN, eps=LN_EPS)
    self.layers = nn.ModuleList([N3DLayer(ln_mode) for _ in range(LAYERS)])
    self.ln_out = ln(HIDDEN, eps=LN_EPS)
    self.proj = nn.Linear(HIDDEN, HEAD_HIDDEN)
    self.up = nn.Conv1d(HEAD_HIDDEN, HEAD_HIDDEN * STACK, kernel_size=3, padding=1)
    self.dense = nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN)
    self.out = nn.Linear(HEAD_HIDDEN, SPEAKERS)

  def forward(self, packed_embeds, attn_bias, rope_cos, rope_sin):
    b, t, _ = packed_embeds.shape
    x = self.ln_in(packed_embeds)
    for layer in self.layers:
      x = layer(x, attn_bias, rope_cos, rope_sin)
    y = self.proj(self.ln_out(x))  # [1,T,192]
    if self.row_mask == "relu":
      mask = torch.relu(attn_bias + 1.0)
    else:
      z = attn_bias * (2.0**-14) + 2.0
      mask = torch.relu(z) - torch.relu(z - 1.0)
    y = y * mask.reshape(b, t, 1)
    y = self.up(y.transpose(1, 2)).transpose(1, 2)  # [1,T,1536]
    y = y.reshape(b, t * STACK, HEAD_HIDDEN)
    return self.out(torch.relu(self.dense(torch.relu(y))))


class FrontendIO(nn.Module):
  """Export wrapper: the input name comes from sample_kwargs, the output name from the dict key."""

  def __init__(self, frontend):
    super().__init__()
    self.frontend = frontend

  def forward(self, mel):
    return {"chunk_embeds": self.frontend(mel)}


class EncoderIO(nn.Module):

  def __init__(self, encoder):
    super().__init__()
    self.encoder = encoder

  def forward(self, packed_embeds, attn_bias, rope_cos, rope_sin):
    return {"logits": self.encoder(packed_embeds, attn_bias, rope_cos, rope_sin)}


# ---------------------------------------------------------------------------- weights


def _key_map():
  m = {
      "model.audio_tower.embedder.projection.weight": ("frontend", "proj.weight"),
      "model.audio_tower.input_layer_norm.weight": ("encoder", "ln_in.weight"),
      "model.audio_tower.input_layer_norm.bias": ("encoder", "ln_in.bias"),
      "model.audio_tower.layer_norm.weight": ("encoder", "ln_out.weight"),
      "model.audio_tower.layer_norm.bias": ("encoder", "ln_out.bias"),
      "model.proj.weight": ("encoder", "proj.weight"),
      "model.proj.bias": ("encoder", "proj.bias"),
      "model.upsampler.conv.weight": ("encoder", "up.weight"),
      "model.upsampler.conv.bias": ("encoder", "up.bias"),
      "classifier.dense.weight": ("encoder", "dense.weight"),
      "classifier.dense.bias": ("encoder", "dense.bias"),
      "classifier.out_proj.weight": ("encoder", "out.weight"),
      "classifier.out_proj.bias": ("encoder", "out.bias"),
      "silence_embeds": ("host", "silence_embeds"),
  }
  per_layer = {
      "layer_norm1.weight": "ln1.weight",
      "layer_norm1.bias": "ln1.bias",
      "self_attn.q_proj.weight": "q.weight",
      "self_attn.k_proj.weight": "k.weight",
      "self_attn.v_proj.weight": "v.weight",
      "self_attn.o_proj.weight": "o.weight",
      "self_attn.o_proj.bias": "o.bias",
      "layer_norm2.weight": "ln2.weight",
      "layer_norm2.bias": "ln2.bias",
      "mlp.fc1.weight": "fc1.weight",
      "mlp.fc1.bias": "fc1.bias",
      "mlp.fc2.weight": "fc2.weight",
      "mlp.fc2.bias": "fc2.bias",
  }
  for i in range(LAYERS):
    for src, dst in per_layer.items():
      m[f"model.audio_tower.layers.{i}.{src}"] = ("encoder", f"layers.{i}.{dst}")
  return m


def load_models(safetensors_path, ln_mode="plain", row_mask="relu"):
  """Strict load: every checkpoint tensor lands in exactly one parameter, and vice versa."""
  ckpt = load_file(safetensors_path)
  key_map = _key_map()
  assert set(ckpt) == set(key_map), (
      sorted(set(ckpt) - set(key_map))[:10], sorted(set(key_map) - set(ckpt))[:10])
  frontend, encoder = N3DFrontend(), N3DEncoder(ln_mode, row_mask)
  sds = {"frontend": {}, "encoder": {}, "host": {}}
  for src, (dst_mod, dst_key) in key_map.items():
    sds[dst_mod][dst_key] = ckpt[src].to(torch.float32)
  frontend.load_state_dict(sds["frontend"], strict=True)
  encoder.load_state_dict(sds["encoder"], strict=True)
  silence = sds["host"]["silence_embeds"]
  n_ckpt = sum(t.numel() for t in ckpt.values())
  n_model = (sum(p.numel() for p in frontend.parameters())
             + sum(p.numel() for p in encoder.parameters()) + silence.numel())
  assert n_ckpt == n_model, (n_ckpt, n_model)
  return frontend.eval(), encoder.eval(), silence, n_ckpt


# ---------------------------------------------------------------------------- host-side tables


def rope_tables(t=T_LOW_LATENCY):
  """cos/sin [1,1,T,64] for positions 0..T-1, computed as transformers does (fp32 outer product)."""
  inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.int64).float() / HEAD_DIM))
  pos = torch.arange(t, dtype=torch.int64)[None, :].float()
  freqs = (inv_freq[None, :, None] @ pos[:, None, :]).transpose(1, 2)  # [1,T,32]
  emb = torch.cat((freqs, freqs), dim=-1)
  return emb.cos()[:, None], emb.sin()[:, None]


def attn_bias_for(length, t=T_LOW_LATENCY, valid=None, two_level=False):
  """0 on the first `length` rows, the pad bias after them; with `valid` ([length] bools, two_level only) the
  masked real rows get MASKED_BIAS_TWO_LEVEL."""
  bias = np.full((1, 1, 1, t), PAD_BIAS_TWO_LEVEL if two_level else PAD_BIAS, np.float32)
  bias[..., :length] = 0.0
  if valid is not None:
    assert two_level, "masked real rows need the two_level row mask"
    bias[0, 0, 0, :length][~np.asarray(valid, bool)] = MASKED_BIAS_TWO_LEVEL
  return bias


def pack(rows, t=T_LOW_LATENCY):
  """rows [L,512] -> packed_embeds [1,T,512] (zero tail)."""
  out = np.zeros((1, t, HIDDEN), np.float32)
  out[0, : rows.shape[0]] = rows
  return out
