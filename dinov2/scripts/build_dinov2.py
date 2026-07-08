"""DINOv2 ViT-S/14 dense features -> LiteRT CompiledModel GPU (feature viz).

Outputs the 256 patch tokens (16x16 grid at 224) of DINOv2-small; a host-side
PCA of those tokens -> RGB gives the classic "what the backbone sees" overlay.

Re-authored GPU-clean with the proven ViT recipes: fused-qkv attention decomposed
to 4D (C12), LayerScale gamma baked into the projections, SafeLayerNorm (fp16
variance-overflow safe), GELU -> x*sigmoid(1.702x). The pos_embed is baked at a
fixed 224 grid by timm at model creation, so there is no runtime interpolation
(no GATHER_ND).

Run:  python build_dinov2.py [parity|convert|fp16|device|viz|all]
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "dinov2_s.pth")
FP32 = os.path.join(HERE, "dinov2_s.tflite")
FP16 = os.path.join(HERE, "dinov2_s_fp16.tflite")
IMG_SIZE = 448
PATCH = 14
GRID = IMG_SIZE // PATCH            # 32
N_PATCH = GRID * GRID              # 256
N_TOK = N_PATCH + 1               # + cls
C = 384
DEPTH = 12
HEADS = 6
HEAD_DIM = C // HEADS
SCALE = HEAD_DIM ** -0.5
LN_S = 64.0                       # SafeLayerNorm pre-square scale
IN_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IN_STD = np.array([0.229, 0.224, 0.225], np.float32)


def safe_layer_norm(x, w, b):
    """LayerNorm over C with an fp16-overflow-safe variance.

    Scales the deviation by 1/LN_S before squaring so the per-token sum of
    squares stays within fp16 range even for DINOv2's massive activations, then
    rescales — algebraically identical to the plain variance.
    """
    mean = x.mean(-1, keepdim=True)
    d = x - mean
    var = (d * (1.0 / LN_S)).pow(2).mean(-1, keepdim=True) * (LN_S * LN_S)
    return (d * torch.rsqrt(var + 1e-6)) * w + b


def gelu(x):
    """tanh approximation of GELU (delegate-friendly, near-exact)."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


class Block(nn.Module):
    """One DINOv2 transformer block, 4D attention + baked LayerScale."""

    def __init__(self, p, i):
        super().__init__()
        pre = f"blocks.{i}."
        self.n1w = nn.Parameter(p[pre + "norm1.weight"], requires_grad=False)
        self.n1b = nn.Parameter(p[pre + "norm1.bias"], requires_grad=False)
        self.n2w = nn.Parameter(p[pre + "norm2.weight"], requires_grad=False)
        self.n2b = nn.Parameter(p[pre + "norm2.bias"], requires_grad=False)
        self.qkv_w = nn.Parameter(p[pre + "attn.qkv.weight"], requires_grad=False)
        self.qkv_b = nn.Parameter(p[pre + "attn.qkv.bias"], requires_grad=False)
        g1 = p[pre + "ls1.gamma"]
        g2 = p[pre + "ls2.gamma"]
        # LayerScale gamma baked into the output projections.
        self.proj_w = nn.Parameter(g1.view(C, 1) * p[pre + "attn.proj.weight"],
                                   requires_grad=False)
        self.proj_b = nn.Parameter(g1 * p[pre + "attn.proj.bias"],
                                   requires_grad=False)
        self.fc1_w = nn.Parameter(p[pre + "mlp.fc1.weight"], requires_grad=False)
        self.fc1_b = nn.Parameter(p[pre + "mlp.fc1.bias"], requires_grad=False)
        self.fc2_w = nn.Parameter(g2.view(C, 1) * p[pre + "mlp.fc2.weight"],
                                  requires_grad=False)
        self.fc2_b = nn.Parameter(g2 * p[pre + "mlp.fc2.bias"], requires_grad=False)

    def forward(self, x):
        h = safe_layer_norm(x, self.n1w, self.n1b)
        qkv = h @ self.qkv_w.t() + self.qkv_b            # [1,N,3C]
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(1, N_TOK, HEADS, HEAD_DIM).transpose(1, 2)   # [1,H,N,d]
        k = k.view(1, N_TOK, HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(1, N_TOK, HEADS, HEAD_DIM).transpose(1, 2)
        attn = (q * SCALE) @ k.transpose(-2, -1)          # [1,H,N,N]
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(1, N_TOK, C)
        x = x + (out @ self.proj_w.t() + self.proj_b)
        h2 = safe_layer_norm(x, self.n2w, self.n2b)
        h2 = gelu(h2 @ self.fc1_w.t() + self.fc1_b)
        x = x + (h2 @ self.fc2_w.t() + self.fc2_b)
        return x


class DINOv2(nn.Module):
    """DINOv2 ViT-S/14 emitting the 256 patch tokens at a fixed 224 input."""

    def __init__(self, p):
        super().__init__()
        self.patch_w = nn.Parameter(p["patch_embed.proj.weight"], requires_grad=False)
        self.patch_b = nn.Parameter(p["patch_embed.proj.bias"], requires_grad=False)
        self.cls = nn.Parameter(p["cls_token"], requires_grad=False)
        self.pos = nn.Parameter(p["pos_embed"], requires_grad=False)
        self.blocks = nn.ModuleList([Block(p, i) for i in range(DEPTH)])
        self.nw = nn.Parameter(p["norm.weight"], requires_grad=False)
        self.nb = nn.Parameter(p["norm.bias"], requires_grad=False)

    def forward(self, img):
        x = F.conv2d(img, self.patch_w, self.patch_b, stride=PATCH)   # [1,C,16,16]
        x = x.flatten(2).transpose(1, 2)                             # [1,256,C]
        x = torch.cat([self.cls.expand(1, 1, C), x], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = safe_layer_norm(x, self.nw, self.nb)
        return x[:, 1:]                                              # [1,256,C] patches


def load():
    return {k: v.float() for k, v in torch.load(WEIGHTS, map_location="cpu").items()}


def preprocess(path):
    im = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    arr = (np.asarray(im, np.float32) / 255.0 - IN_MEAN) / IN_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)[None])


def pca_rgb(feats):
    """Top-3 PCA of [256,C] patch features -> [16,16,3] in 0..1."""
    x = feats - feats.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    proj = x @ vt[:3].T                       # [256,3]
    proj = (proj - proj.min(0)) / (np.ptp(proj, axis=0) + 1e-6)
    return proj.reshape(GRID, GRID, 3)


def stage_parity(model):
    import timm
    ref = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                            img_size=IMG_SIZE, num_classes=0).eval()
    x = preprocess(os.path.join(HERE, "test.jpg"))
    with torch.no_grad():
        mine = model(x)[0].numpy()
        gold = ref.forward_features(x)[0, 1:].numpy()
    corr = np.corrcoef(mine.flatten(), gold.flatten())[0, 1]
    print("re-authored vs timm patch features: corr %.6f  max|d| %.4f" %
          (corr, np.abs(mine - gold).max()))


def stage_convert(model):
    import litert_torch
    litert_torch.convert(model.eval(), (torch.zeros(1, 3, IMG_SIZE, IMG_SIZE),)).export(FP32)
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
    import subprocess
    dev = "/data/local/tmp"
    x = preprocess(os.path.join(HERE, "test.jpg"))
    with torch.no_grad():
        ref = model(x)[0].numpy()
    x.numpy().astype(np.float32).tofile(os.path.join(HERE, "din.bin"))
    for f in ("din.bin", "dinov2_s_fp16.tflite"):
        subprocess.run(["adb", "push", os.path.join(HERE, f), "%s/%s" % (dev, f)],
                       capture_output=True)
    r = subprocess.run(
        ["adb", "shell", "cd %s && LD_LIBRARY_PATH=. ./gpu_test_bin "
         "dinov2_s_fp16.tflite 5 din.bin dout.bin" % dev],
        capture_output=True, text=True)
    print([l for l in r.stderr.splitlines() if "RUN OK" in l or "Replacing" in l])
    subprocess.run(["adb", "pull", "%s/dout.bin.0" % dev,
                    os.path.join(HERE, "dout.bin.0")], capture_output=True)
    dev_feats = np.fromfile(os.path.join(HERE, "dout.bin.0"), np.float32)
    corr = np.corrcoef(dev_feats, ref.flatten())[0, 1]
    print("device fp16 patch features vs fp32: corr %.5f  NaN %s" %
          (corr, np.isnan(dev_feats).any()))


def stage_viz(model):
    x = preprocess(os.path.join(HERE, "test.jpg"))
    with torch.no_grad():
        feats = model(x)[0].numpy()
    rgb = pca_rgb(feats)
    big = np.array(Image.fromarray((rgb * 255).astype(np.uint8)).resize(
        (IMG_SIZE, IMG_SIZE), Image.NEAREST))
    src = np.array(Image.open(os.path.join(HERE, "test.jpg")).convert("RGB").resize(
        (IMG_SIZE, IMG_SIZE)))
    Image.fromarray(np.concatenate([src, big], 1)).save(os.path.join(HERE, "viz.png"))
    print("wrote viz.png")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    model = DINOv2(load()).eval()
    if stage in ("parity", "all"):
        stage_parity(model)
    if stage in ("convert", "all"):
        stage_convert(model)
    if stage in ("fp16", "all"):
        stage_fp16()
    if stage in ("device", "all"):
        stage_device(model)
    if stage in ("viz", "all"):
        stage_viz(model)


if __name__ == "__main__":
    main()
