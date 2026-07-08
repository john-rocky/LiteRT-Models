"""Vision-RWKV (VRWKV-S) -> LiteRT CompiledModel GPU: re-author + convert + parity.

VRWKV is an RWKV-style vision backbone (ICLR 2025). Its core op is a bidirectional
WKV (a CUDA kernel), which we re-author GPU-clean. Because the token count is FIXED
(T = 14*14 = 196), the bidirectional WKV is exactly a per-channel decay-biased
attention over the token sequence:

  L[c,t,i] = k[c,i] - (spatial_decay[c]/T) * |t-i| + (spatial_first[c]/T) * delta(t,i)
  y[c,t]   = sum_i softmax_i(L[c,t,:]) * v[c,i]

i.e. C independent [T,T] attention matrices -> plain 4D softmax + matmul, no scan.
Q-Shift becomes pad+slice+concat (<=4D); LayerScale (gamma) is baked into the
output/value projections; GELU-free (RWKV uses square-relu + sigmoid gating).

Run:  python build_vrwkv.py [parity|convert|fp16|all]
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "vrwkv_s_in1k_224.pth")
FP32 = os.path.join(HERE, "vrwkv_s.tflite")
FP16 = os.path.join(HERE, "vrwkv_s_fp16.tflite")

IMG_SIZE = 224
PATCH = 16
GRID = IMG_SIZE // PATCH          # 14
T = GRID * GRID                   # 196 tokens
C = 384                           # embed dims (VRWKV-S)
DEPTH = 12
HIDDEN = C * 4
GAMMA = 1 / 4                     # q-shift channel fraction per direction
IN_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
IN_STD = np.array([58.395, 57.12, 57.375], np.float32)


def load_state():
    sd = torch.load(CKPT, map_location="cpu")["state_dict"]
    return {k: v.float() for k, v in sd.items()}


# ---------------------------------------------------------------- Bi-WKV --------
_IDX = torch.arange(T, dtype=torch.float32)
DIST_HOST = (_IDX.view(T, 1) - _IDX.view(1, T)).abs().view(1, 1, T, T)   # |t-i|
# The token-distance geometry is fed as a RUNTIME input so the per-channel
# [C,T,T] decay bias (w * dist) is computed live instead of being const-folded
# into a ~59 MB-per-block flatbuffer constant. eye = relu(1 - dist) (0 off the
# diagonal, 1 on it) avoids a second input.
_RT = {}


def bi_wkv_matrix(k, v, decay, first):
    """GPU-clean bidirectional WKV via a fixed-size [B,C,T,T] softmax."""
    b = k.shape[0]
    dist = _RT["dist"]                             # [1,1,T,T] runtime input
    eye = torch.relu(1.0 - dist)
    w = (decay / T).view(1, C, 1, 1)               # per-channel decay
    u = (first / T).view(1, C, 1, 1)               # per-channel self weight
    kc = k.transpose(1, 2).reshape(b, C, 1, T)     # [B,C,1,T] over i
    vc = v.transpose(1, 2).reshape(b, C, T, 1)     # [B,C,T,1] over i
    logits = kc - w * dist + u * eye               # [B,C,T,T]
    attn = F.softmax(logits, dim=-1)               # over i
    y = torch.matmul(attn, vc)                     # [B,C,T,1]
    return y.reshape(b, C, T).transpose(1, 2)      # [B,T,C]


def bi_wkv_reference(k, v, decay, first):
    """Explicit double-loop reference of the same math (correctness oracle)."""
    b = k.shape[0]
    w = (decay / T)
    u = (first / T)
    out = torch.zeros(b, T, C)
    for t in range(T):
        num = torch.zeros(b, C)
        den = torch.zeros(b, C)
        for i in range(T):
            if i == t:
                weight = torch.exp(u + k[:, t])
            else:
                weight = torch.exp(k[:, i] - w * abs(t - i))
            num = num + weight * v[:, i]
            den = den + weight
        out[:, t] = num / (den + 1e-6)
    return out


# ------------------------------------------------------------- q-shift ----------
def q_shift(x):
    """Spatial q-shift: shift channel quarters left/right/up/down by 1 px.

    x: [B, T, C] -> [B, T, C], token grid GRID*GRID, via pad+slice+concat (<=4D).
    """
    b = x.shape[0]
    g = x.transpose(1, 2).reshape(b, C, GRID, GRID)   # [B,C,H,W]
    q = C // 4
    # quarter 0: shift right (take cols 0..W-2 into 1..W-1), col 0 -> 0
    q0 = F.pad(g[:, 0:q, :, 0:GRID - 1], (1, 0, 0, 0))
    # quarter 1: shift left
    q1 = F.pad(g[:, q:2 * q, :, 1:GRID], (0, 1, 0, 0))
    # quarter 2: shift down
    q2 = F.pad(g[:, 2 * q:3 * q, 0:GRID - 1, :], (0, 0, 1, 0))
    # quarter 3: shift up
    q3 = F.pad(g[:, 3 * q:4 * q, 1:GRID, :], (0, 0, 0, 1))
    rest = g[:, 4 * q:, :, :]
    out = torch.cat([q0, q1, q2, q3, rest], dim=1)
    return out.reshape(b, C, T).transpose(1, 2)


# ------------------------------------------------------------- blocks -----------
class SpatialMix(nn.Module):
    def __init__(self, p, pre):
        super().__init__()
        self.decay = nn.Parameter(p[pre + "spatial_decay"], requires_grad=False)
        self.first = nn.Parameter(p[pre + "spatial_first"], requires_grad=False)
        self.mix_k = nn.Parameter(p[pre + "spatial_mix_k"].view(C), requires_grad=False)
        self.mix_v = nn.Parameter(p[pre + "spatial_mix_v"].view(C), requires_grad=False)
        self.mix_r = nn.Parameter(p[pre + "spatial_mix_r"].view(C), requires_grad=False)
        self.key = nn.Parameter(p[pre + "key.weight"], requires_grad=False)
        self.value = nn.Parameter(p[pre + "value.weight"], requires_grad=False)
        self.recept = nn.Parameter(p[pre + "receptance.weight"], requires_grad=False)
        self.output = nn.Parameter(p[pre + "output.weight"], requires_grad=False)

    def forward(self, x):
        xx = q_shift(x)
        xk = x * self.mix_k + xx * (1 - self.mix_k)
        xv = x * self.mix_v + xx * (1 - self.mix_v)
        xr = x * self.mix_r + xx * (1 - self.mix_r)
        k = xk @ self.key.t()
        v = xv @ self.value.t()
        sr = torch.sigmoid(xr @ self.recept.t())
        y = bi_wkv_matrix(k, v, self.decay, self.first)
        return (sr * y) @ self.output.t()


class ChannelMix(nn.Module):
    def __init__(self, p, pre):
        super().__init__()
        self.mix_k = nn.Parameter(p[pre + "spatial_mix_k"].view(C), requires_grad=False)
        self.mix_r = nn.Parameter(p[pre + "spatial_mix_r"].view(C), requires_grad=False)
        self.key = nn.Parameter(p[pre + "key.weight"], requires_grad=False)
        self.recept = nn.Parameter(p[pre + "receptance.weight"], requires_grad=False)
        self.value = nn.Parameter(p[pre + "value.weight"], requires_grad=False)

    def forward(self, x):
        xx = q_shift(x)
        xk = x * self.mix_k + xx * (1 - self.mix_k)
        xr = x * self.mix_r + xx * (1 - self.mix_r)
        k = torch.square(torch.relu(xk @ self.key.t()))
        kv = k @ self.value.t()
        return torch.sigmoid(xr @ self.recept.t()) * kv


def layer_norm(x, w, b):
    return F.layer_norm(x, (C,), w, b)


class Block(nn.Module):
    """VRWKV block in POST-norm mode (config post_norm=True): the norm follows
    the mixer and the LayerScale gamma is baked into that norm's affine params."""

    def __init__(self, p, i):
        super().__init__()
        pre = f"backbone.layers.{i}."
        self.i = i
        g1 = p[pre + "gamma1"]
        g2 = p[pre + "gamma2"]
        self.ln1w = nn.Parameter(g1 * p[pre + "ln1.weight"], requires_grad=False)
        self.ln1b = nn.Parameter(g1 * p[pre + "ln1.bias"], requires_grad=False)
        self.ln2w = nn.Parameter(g2 * p[pre + "ln2.weight"], requires_grad=False)
        self.ln2b = nn.Parameter(g2 * p[pre + "ln2.bias"], requires_grad=False)
        if i == 0:
            self.ln0w = nn.Parameter(p[pre + "ln0.weight"], requires_grad=False)
            self.ln0b = nn.Parameter(p[pre + "ln0.bias"], requires_grad=False)
        self.att = SpatialMix(p, pre + "att.")
        self.ffn = ChannelMix(p, pre + "ffn.")

    def forward(self, x):
        if self.i == 0:
            x = layer_norm(x, self.ln0w, self.ln0b)
        x = x + layer_norm(self.att(x), self.ln1w, self.ln1b)
        x = x + layer_norm(self.ffn(x), self.ln2w, self.ln2b)
        return x


class VRWKV(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.patch_w = nn.Parameter(p["backbone.patch_embed.projection.weight"], requires_grad=False)
        self.patch_b = nn.Parameter(p["backbone.patch_embed.projection.bias"], requires_grad=False)
        self.pos = nn.Parameter(p["backbone.pos_embed"], requires_grad=False)
        self.blocks = nn.ModuleList([Block(p, i) for i in range(DEPTH)])
        self.lnw = nn.Parameter(p["backbone.ln1.weight"], requires_grad=False)
        self.lnb = nn.Parameter(p["backbone.ln1.bias"], requires_grad=False)
        self.head_w = nn.Parameter(p["head.fc.weight"], requires_grad=False)
        self.head_b = nn.Parameter(p["head.fc.bias"], requires_grad=False)

    def forward(self, img, dist):
        _RT["dist"] = dist                                           # runtime token geometry
        x = F.conv2d(img, self.patch_w, self.patch_b, stride=PATCH)   # [B,C,14,14]
        x = x.flatten(2).transpose(1, 2)                              # [B,T,C]
        x = x + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = layer_norm(x, self.lnw, self.lnb)
        x = x.mean(dim=1)                                             # global avg pool
        return x @ self.head_w.t() + self.head_b                     # [B,1000]


# ------------------------------------------------------------- driver -----------
def preprocess(path):
    im = Image.open(path).convert("RGB")
    short, long = 256, None
    w, h = im.size
    scale = short / min(w, h)
    im = im.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h = im.size
    left, top = (w - IMG_SIZE) // 2, (h - IMG_SIZE) // 2
    im = im.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    arr = np.asarray(im, np.float32)
    arr = (arr - IN_MEAN) / IN_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)[None])            # [1,3,224,224]


def labels():
    with open(os.path.join(HERE, "imagenet_classes.txt")) as f:
        return [l.strip() for l in f]


def stage_parity(model):
    # (1) matrix Bi-WKV == explicit reference on random small input
    torch.manual_seed(0)
    _RT["dist"] = DIST_HOST
    k = torch.randn(1, T, C) * 0.5
    v = torch.randn(1, T, C) * 0.5
    dec = torch.randn(C)
    fst = torch.randn(C)
    a = bi_wkv_matrix(k, v, dec, fst)
    b = bi_wkv_reference(k, v, dec, fst)
    print("Bi-WKV matrix vs explicit: max|d| %.3e  corr %.7f" %
          ((a - b).abs().max(), np.corrcoef(a.flatten(), b.flatten())[0, 1]))

    # (2) full model on a real image -> top-5 must be sensible
    x = preprocess(os.path.join(HERE, "dog.jpg"))
    with torch.no_grad():
        logits = model(x, DIST_HOST)[0]
    names = labels()
    top = torch.topk(logits, 5)
    print("dog.jpg top-5:")
    for s, idx in zip(top.values, top.indices):
        print("   %6.2f  %s" % (s.item(), names[idx]))


def stage_convert(model):
    ex = (torch.zeros(1, 3, IMG_SIZE, IMG_SIZE), DIST_HOST.clone())
    import litert_torch
    litert_torch.convert(model.eval(), ex).export(FP32)
    print("convert: %.1f MB -> %s" % (os.path.getsize(FP32) / 1e6, FP32))


def stage_fp16():
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
    if os.path.exists(FP16):
        os.remove(FP16)
    qt = quantizer.Quantizer(float_model=FP32)
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(FP16)
    print("fp16: %.1f MB -> %s" % (os.path.getsize(FP16) / 1e6, FP16))


def stage_device(model):
    """Drive the fp16 tflite on the Pixel 8a GPU and check top-1 vs fp32."""
    import subprocess
    dev = "/data/local/tmp"
    img = preprocess(os.path.join(HERE, "dog.jpg"))
    with torch.no_grad():
        ref = model(img, DIST_HOST)[0]
    names = labels()
    img.numpy().astype(np.float32).tofile(os.path.join(HERE, "vin.bin.0"))
    DIST_HOST.numpy().astype(np.float32).tofile(os.path.join(HERE, "vin.bin.1"))
    for f in ("vin.bin.0", "vin.bin.1", "vrwkv_s_fp16.tflite"):
        subprocess.run(["adb", "push", os.path.join(HERE, f), "%s/%s" % (dev, f)],
                       capture_output=True)
    r = subprocess.run(
        ["adb", "shell", "cd %s && LD_LIBRARY_PATH=. ./gpu_test_bin "
         "vrwkv_s_fp16.tflite 5 vin.bin vout.bin" % dev],
        capture_output=True, text=True)
    print([l for l in r.stderr.splitlines() if "RUN OK" in l or "Replacing" in l])
    subprocess.run(["adb", "pull", "%s/vout.bin.0" % dev, os.path.join(HERE, "vout.bin.0")],
                   capture_output=True)
    dev_logits = np.fromfile(os.path.join(HERE, "vout.bin.0"), np.float32)
    corr = np.corrcoef(dev_logits, ref.numpy())[0, 1]
    d_top = np.argsort(dev_logits)[::-1][:5]
    r_top = int(ref.argmax())
    print("device fp16 top-1: %s   (fp32 top-1: %s)   logits corr %.5f" %
          (names[d_top[0]], names[r_top], corr))
    print("device top-5:", [names[i] for i in d_top])
    print("top-1 match:", "YES" if d_top[0] == r_top else "NO")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    p = load_state()
    model = VRWKV(p).eval()
    if stage in ("parity", "all"):
        stage_parity(model)
    if stage in ("convert", "all"):
        stage_convert(model)
    if stage in ("fp16", "all"):
        stage_fp16()
    if stage in ("device", "all"):
        stage_device(model)


if __name__ == "__main__":
    main()
