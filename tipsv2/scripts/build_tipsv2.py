"""TIPSv2-B/14 + DPT heads (depth / surface normals / ADE20K seg) -> LiteRT CompiledModel GPU.

Google DeepMind TIPSv2 (CVPR 2026, Apache-2.0): a DINOv2-style ViT-B/14 backbone with one
register token, plus three DPT heads trained on the frozen backbone. This script re-authors
the whole thing GPU-clean (every rewrite exact unless noted), checks parity against the
official HF `google/tipsv2-b14-dpt` model, converts with litert-torch, fp16-quantizes and
runs the on-device GPU harness.

Rewrites (all proven recipes from this zoo):
  backbone  fused-qkv attention -> 4D per-head matmuls (C12)
            LayerScale gamma baked into attn.proj / mlp.fc2
            LayerNorm -> SafeLayerNorm (fp16 variance-overflow safe)
            exact GELU -> tanh GELU (ERF has no GPU kernel; ~1e-3 abs)
            pos_embed used at its native 32x32 grid (448 input) -> no interpolation
  DPT       readout `cat(patch, cls.expand) @ W` -> `patch @ W_a + cls @ W_b` (exact, no BROADCAST_TO)
            ConvTranspose2d(k=s) -> zero-stuff + Conv2d (exact)
            bilinear x2 align_corners=True -> two constant-RHS matmuls (exact; GPU bans that resize)
            seg logits emitted at the head's native 256x256 (argmax host-side) instead of 448x448
  depth     the depth decoder's activations reach ~1e8 (fp16 max 65504 -> the GPU returned a
            constant); it is a ReLU/affine chain ending in a scale-invariant normalization, so
            power-of-2 scales are folded into its weights/biases (bit-exact in fp32) to keep
            every stage <~100

Needs a `test.jpg` next to this script (any photo) and `python dump_ref.py` run once for the
parity reference.

Run:  python build_tipsv2.py [parity|convert|fp16|device|opcheck|all]
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FP32 = os.path.join(HERE, "tipsv2_b14_dpt.tflite")
FP16 = os.path.join(HERE, "tipsv2_b14_dpt_fp16.tflite")
IMG = 448
PATCH = 14
GRID = IMG // PATCH              # 32
N_PATCH = GRID * GRID            # 1024
N_REG = 1
N_TOK = 1 + N_REG + N_PATCH      # cls + register + patches = 1026
C = 768
DEPTH = 12
HEADS = 12
HD = C // HEADS
SCALE = HD ** -0.5
LN_EPS = 1e-6
LN_S = 64.0                      # SafeLayerNorm pre-square scale
TAPS = (2, 5, 8, 11)             # out_indices (3,6,9,12) - 1
NECK = (96, 192, 384, 768)
FUSE = 256
N_BINS = 256
MIN_D, MAX_D = 1e-3, 10.0
N_CLS = 150
HEAD_RES = GRID * 8              # 256 — DPT output resolution before the final resize


def safe_layer_norm(x, w, b, eps=LN_EPS):
    mean = x.mean(-1, keepdim=True)
    d = x - mean
    var = (d * (1.0 / LN_S)).pow(2).mean(-1, keepdim=True) * (LN_S * LN_S)
    return (d * torch.rsqrt(var + eps)) * w + b


def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def P(t):
    return nn.Parameter(t.detach().clone().float(), requires_grad=False)


# ---------------------------------------------------------------- backbone
class Block(nn.Module):
    def __init__(self, sd, i):
        super().__init__()
        pre = f"vision_encoder.blocks.{i}."
        self.n1w, self.n1b = P(sd[pre + "norm1.weight"]), P(sd[pre + "norm1.bias"])
        self.n2w, self.n2b = P(sd[pre + "norm2.weight"]), P(sd[pre + "norm2.bias"])
        self.qkv_w, self.qkv_b = P(sd[pre + "attn.qkv.weight"]), P(sd[pre + "attn.qkv.bias"])
        g1, g2 = sd[pre + "ls1.gamma"], sd[pre + "ls2.gamma"]
        self.proj_w = P(g1.view(C, 1) * sd[pre + "attn.proj.weight"])
        self.proj_b = P(g1 * sd[pre + "attn.proj.bias"])
        self.fc1_w, self.fc1_b = P(sd[pre + "mlp.fc1.weight"]), P(sd[pre + "mlp.fc1.bias"])
        self.fc2_w = P(g2.view(C, 1) * sd[pre + "mlp.fc2.weight"])
        self.fc2_b = P(g2 * sd[pre + "mlp.fc2.bias"])

    def forward(self, x):
        h = safe_layer_norm(x, self.n1w, self.n1b)
        qkv = h @ self.qkv_w.t() + self.qkv_b
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(1, N_TOK, HEADS, HD).transpose(1, 2)
        k = k.view(1, N_TOK, HEADS, HD).transpose(1, 2)
        v = v.view(1, N_TOK, HEADS, HD).transpose(1, 2)
        attn = ((q * SCALE) @ k.transpose(-2, -1)).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(1, N_TOK, C)
        x = x + (out @ self.proj_w.t() + self.proj_b)
        h2 = safe_layer_norm(x, self.n2w, self.n2b)
        h2 = gelu(h2 @ self.fc1_w.t() + self.fc1_b)
        return x + (h2 @ self.fc2_w.t() + self.fc2_b)


class Backbone(nn.Module):
    """TIPSv2 ViT-B/14 at 448: returns [(cls [1,C], patches [1,C,32,32])] for the 4 taps."""

    def __init__(self, sd):
        super().__init__()
        self.patch_w = P(sd["vision_encoder.patch_embed.proj.weight"])
        self.patch_b = P(sd["vision_encoder.patch_embed.proj.bias"])
        pos = sd["vision_encoder.pos_embed"].float()           # [1,1025,C] native 32x32 grid
        assert pos.shape[1] == 1 + N_PATCH, pos.shape
        # pre-assemble the constant token prefix: cls+pos[0], register (no pos), and patch pos rows
        self.cls_pos = P(sd["vision_encoder.cls_token"].float() + pos[:, :1])   # [1,1,C]
        self.reg = P(sd["vision_encoder.register_tokens"])                       # [1,1,C]
        self.patch_pos = P(pos[:, 1:])                                           # [1,1024,C]
        self.blocks = nn.ModuleList([Block(sd, i) for i in range(DEPTH)])
        self.nw, self.nb = P(sd["vision_encoder.norm.weight"]), P(sd["vision_encoder.norm.bias"])

    def forward(self, img):
        x = F.conv2d(img, self.patch_w, self.patch_b, stride=PATCH)   # [1,C,32,32]
        x = x.flatten(2).transpose(1, 2) + self.patch_pos             # [1,1024,C]
        x = torch.cat([self.cls_pos, self.reg, x], dim=1)             # [1,1026,C]
        taps = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in TAPS:
                y = safe_layer_norm(x, self.nw, self.nb)
                cls = y[:, 0]                                                     # [1,C]
                patch = y[:, 1 + N_REG:].transpose(1, 2).reshape(1, C, GRID, GRID)
                taps.append((cls, patch))
        return taps


# ---------------------------------------------------------------- DPT pieces
def up2_matrix(n):
    """Exact bilinear x2 upsample (align_corners=True) as a constant [2n, n] matrix."""
    m = np.zeros((2 * n, n), np.float32)
    for i in range(2 * n):
        s = i * (n - 1) / (2 * n - 1)
        i0 = int(np.floor(s))
        i1 = min(i0 + 1, n - 1)
        w = s - i0
        m[i, i0] += 1.0 - w
        m[i, i1] += w
    return torch.from_numpy(m)


class Up2(nn.Module):
    """y = U @ x @ U^T done as two constant-RHS matmuls (+ transposes) — GPU-clean and exact."""

    def __init__(self, n):
        super().__init__()
        self.ut = P(up2_matrix(n).t().contiguous())   # [n, 2n]

    def forward(self, x):                              # [1,C,n,n]
        y = x @ self.ut                                # [1,C,n,2n]
        y = y.transpose(-1, -2) @ self.ut              # [1,C,2n,2n] (rows = W, cols = H)
        return y.transpose(-1, -2)


class ZeroStuffConvT(nn.Module):
    """ConvTranspose2d(k=s, stride=s) == nearest-up x mask (zero-stuff) + Conv2d(flipped) — exact."""

    def __init__(self, w, b, n_in):
        super().__init__()
        self.s = self.k = w.shape[-1]
        self.w = P(w.flip(2, 3).transpose(0, 1).contiguous())
        self.b = P(b)
        s = self.s
        mk = np.zeros((n_in * s, n_in * s), np.float32)
        mk[::s, ::s] = 1.0
        self.mask = P(torch.from_numpy(mk)[None, None])
        self.n_out = n_in * s

    def forward(self, x):
        xn = F.interpolate(x, size=(self.n_out, self.n_out), mode="nearest")
        y = F.conv2d(xn * self.mask, self.w, bias=self.b, padding=self.k - 1)
        return y[:, :, :self.n_out, :self.n_out]


class ResUnit(nn.Module):
    def __init__(self, sd, pre):
        super().__init__()
        self.w1, self.w2 = P(sd[pre + "conv1.weight"]), P(sd[pre + "conv2.weight"])

    def forward(self, x):
        h = F.conv2d(F.relu(x), self.w1, padding=1)
        h = F.conv2d(F.relu(h), self.w2, padding=1)
        return h + x


class Fusion(nn.Module):
    def __init__(self, sd, pre, n_in, has_res, w_scale=1.0, lam_out=1.0):
        super().__init__()
        self.res = ResUnit(sd, pre + "residual_unit.") if has_res else None
        self.main = ResUnit(sd, pre + "main_unit.")
        self.up = Up2(n_in)
        # range fold: weights carry the local factor, the bias the ABSOLUTE running scale lam_out
        self.ow = P(sd[pre + "out_conv.weight"] * w_scale)
        self.ob = P(sd[pre + "out_conv.bias"] * lam_out)

    def forward(self, x, residual=None):
        if self.res is not None:
            x = x + self.res(residual)
        x = self.main(x)
        x = self.up(x)
        return F.conv2d(x, self.ow, self.ob)


class DPTHead(nn.Module):
    """Reassemble + fusion + project for one task; `kind` in {depth, normals, seg}."""

    def __init__(self, sd, name, kind):
        super().__init__()
        self.kind = kind
        pre = name + "."
        # fp16 range fold (depth only). lambda = running scale of the activations at each point:
        #   convs[i] (bias-free) out: 2^-12, 2^-10, 2^-8, 2^-6 (level 0..3)
        #   fusion k: weights x1/4 -> out lambda 2^-8, 2^-10, 2^-12, 2^-14 (= the level it adds to)
        #   project: weights x1/32 -> 2^-19; head linear: weights x1/16 -> 2^-23.
        # Rule: a layer's WEIGHTS carry the local factor, its BIAS carries the absolute lambda_out,
        # and both operands of every residual add sit at the same lambda -> bit-exact in fp32.
        if kind == "depth":
            conv_s = (2.0 ** -12, 2.0 ** -10, 2.0 ** -8, 2.0 ** -6)
            fus_w, fus_lam = 0.25, (2.0 ** -8, 2.0 ** -10, 2.0 ** -12, 2.0 ** -14)
            proj_w, proj_lam = 2.0 ** -5, 2.0 ** -19
            head_w, head_lam = 2.0 ** -4, 2.0 ** -23
        else:
            conv_s, fus_w, fus_lam = (1.0,) * 4, 1.0, (1.0,) * 4
            proj_w = proj_lam = head_w = head_lam = 1.0
        self.lam = head_lam
        self.ro_a = nn.ParameterList()   # patch half of readout_projects
        self.ro_b = nn.ParameterList()   # cls half
        self.ro_bias = nn.ParameterList()
        self.pw = nn.ParameterList()
        self.pb = nn.ParameterList()
        self.cw = nn.ParameterList()
        for i in range(4):
            w = sd[pre + f"reassemble.readout_projects.{i}.weight"]   # [C, 2C]
            self.ro_a.append(P(w[:, :C]))
            self.ro_b.append(P(w[:, C:]))
            self.ro_bias.append(P(sd[pre + f"reassemble.readout_projects.{i}.bias"]))
            self.pw.append(P(sd[pre + f"reassemble.out_projections.{i}.weight"]))
            self.pb.append(P(sd[pre + f"reassemble.out_projections.{i}.bias"]))
            self.cw.append(P(sd[pre + f"convs.{i}.weight"] * conv_s[i]))
        self.rs0 = ZeroStuffConvT(sd[pre + "reassemble.resize_layers.0.weight"],
                                  sd[pre + "reassemble.resize_layers.0.bias"], GRID)
        self.rs1 = ZeroStuffConvT(sd[pre + "reassemble.resize_layers.1.weight"],
                                  sd[pre + "reassemble.resize_layers.1.bias"], GRID)
        self.rs3w = P(sd[pre + "reassemble.resize_layers.3.weight"])
        self.rs3b = P(sd[pre + "reassemble.resize_layers.3.bias"])
        self.fus = nn.ModuleList([
            Fusion(sd, pre + "fusion_blocks.0.", GRID // 2, False, fus_w, fus_lam[0]),
            Fusion(sd, pre + "fusion_blocks.1.", GRID, True, fus_w, fus_lam[1]),
            Fusion(sd, pre + "fusion_blocks.2.", GRID * 2, True, fus_w, fus_lam[2]),
            Fusion(sd, pre + "fusion_blocks.3.", GRID * 4, True, fus_w, fus_lam[3]),
        ])
        self.prw = P(sd[pre + "project.weight"] * proj_w)
        self.prb = P(sd[pre + "project.bias"] * proj_lam)
        last = {"depth": "depth_head", "normals": "normals_head", "seg": "segmentation_head"}[kind]
        self.hw = P(sd[pre + last + ".weight"] * head_w)
        self.hb = P(sd[pre + last + ".bias"] * head_lam)
        if kind == "depth":
            self.bins = P(torch.linspace(MIN_D, MAX_D, N_BINS).view(N_BINS, 1))
            self.eps = MIN_D * self.lam      # (relu(l) + MIN_D) * lam, exactly

    def forward(self, taps):
        feats = []
        for i, (cls, x) in enumerate(taps):
            xf = x.flatten(2).transpose(1, 2)                                   # [1,1024,C]
            ro = cls @ self.ro_b[i].t() + self.ro_bias[i]                       # [1,C]
            y = gelu(xf @ self.ro_a[i].t() + ro.unsqueeze(1))                   # [1,1024,C]
            y = y.transpose(1, 2).reshape(1, C, GRID, GRID)
            y = F.conv2d(y, self.pw[i], self.pb[i])
            if i == 0:
                y = self.rs0(y)
            elif i == 1:
                y = self.rs1(y)
            elif i == 3:
                y = F.conv2d(y, self.rs3w, self.rs3b, stride=2, padding=1)
            feats.append(F.conv2d(y, self.cw[i], padding=1))
        out = self.fus[0](feats[3])
        out = self.fus[1](out, feats[2])
        out = self.fus[2](out, feats[1])
        out = self.fus[3](out, feats[0])                                         # [1,256,256,256]
        out = F.conv2d(out, self.prw, self.prb, padding=1)
        if self.kind == "depth":
            out = F.relu(out)
        out = out.permute(0, 2, 3, 1) @ self.hw.t() + self.hb                   # [1,256,256,K]
        if self.kind == "depth":
            out = F.relu(out) + self.eps
            out = out / out.sum(dim=-1, keepdim=True)     # scale-invariant -> lam cancels
            depth = (out @ self.bins).permute(0, 3, 1, 2)                        # [1,1,256,256]
            return F.interpolate(depth, size=(IMG, IMG), mode="bilinear", align_corners=False)
        if self.kind == "normals":
            out = out / torch.clamp(out.norm(dim=-1, keepdim=True), min=1e-12)
            out = out.permute(0, 3, 1, 2)                                        # [1,3,256,256]
            return F.interpolate(out, size=(IMG, IMG), mode="bilinear", align_corners=False)
        return out.permute(0, 3, 1, 2)                                           # [1,150,256,256]


class TIPSv2DPT(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.backbone = Backbone(sd)
        self.depth = DPTHead(sd, "depth_head", "depth")
        self.normals = DPTHead(sd, "normals_head", "normals")
        self.seg = DPTHead(sd, "segmentation_head", "seg")

    def forward(self, img):
        taps = self.backbone(img)
        return self.depth(taps), self.normals(taps), self.seg(taps)


# ---------------------------------------------------------------- stages
def load_sd():
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    return load_file(hf_hub_download("google/tipsv2-b14-dpt", "model.safetensors"))


def preprocess(path):
    im = Image.open(path).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
    return torch.from_numpy((np.asarray(im, np.float32) / 255.0).transpose(2, 0, 1)[None])


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def stage_parity(model):
    x = preprocess(os.path.join(HERE, "test.jpg"))
    with torch.no_grad():
        d, n, s = [t.numpy() for t in model(x)]
        s448 = F.interpolate(torch.from_numpy(s), size=(IMG, IMG), mode="bilinear",
                             align_corners=False).numpy()
    rd = np.load(os.path.join(HERE, "ref_depth.npy"))
    rn = np.load(os.path.join(HERE, "ref_normals.npy"))
    rs = np.load(os.path.join(HERE, "ref_seg.npy"))
    print("depth   corr %.6f  max|d| %.4f m  (ref range %.2f..%.2f)" %
          (corr(d, rd), np.abs(d - rd).max(), rd.min(), rd.max()))
    print("normals corr %.6f  max|d| %.4f  mean angle err %.3f deg" %
          (corr(n, rn), np.abs(n - rn).max(),
           np.degrees(np.arccos(np.clip((n * rn).sum(1), -1, 1))).mean()))
    print("seg     corr %.6f  argmax agree %.4f  (256->448 bilinear vs official 448)" %
          (corr(s448, rs), (s448.argmax(1) == rs.argmax(1)).mean()))


def stage_convert(model):
    import litert_torch
    litert_torch.convert(model.eval(), (torch.zeros(1, 3, IMG, IMG),)).export(FP32)
    print("convert: %.1f MB -> %s" % (os.path.getsize(FP32) / 1e6, FP32))
    opcheck(FP32)


def opcheck(path):
    import collections
    from ai_edge_litert.interpreter import Interpreter
    BANNED = {"GATHER_ND", "GATHER", "TOPK_V2", "PACK", "SPLIT", "FLEX_ERF", "ERF",
              "TRANSPOSE_CONV", "BROADCAST_TO", "SELECT", "SELECT_V2", "CAST"}
    it = Interpreter(model_path=path)
    it.allocate_tensors()
    ops = collections.Counter(d.get("op_name", "?") for d in it._get_ops_details())
    bad = {k: v for k, v in ops.items() if k in BANNED or k.startswith("FLEX")}
    over = sum(1 for d in it.get_tensor_details() if len(d.get("shape", [])) > 4)
    print("ops:", dict(ops))
    print("op-check: banned %s | >4D tensors %d" % (bad or "NONE", over))
    print("VERDICT:", "GPU-CLEAN" if not bad and not over else "BLOCKERS REMAIN")


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
    """Push fp16 model + input to the Pixel, run the CompiledModel GPU harness, compare."""
    import subprocess
    dev = "/data/local/tmp"
    x = preprocess(os.path.join(HERE, "test.jpg"))
    with torch.no_grad():
        ref = [t.numpy() for t in model(x)]
    x.numpy().astype(np.float32).tofile(os.path.join(HERE, "tin.bin"))
    for f in ("tin.bin", os.path.basename(FP16)):
        subprocess.run(["adb", "push", os.path.join(HERE, f), "%s/%s" % (dev, f)],
                       capture_output=True)
    r = subprocess.run(
        ["adb", "shell", "cd %s && LD_LIBRARY_PATH=. ./gpu_test_bin %s 3 tin.bin tout.bin"
         % (dev, os.path.basename(FP16))], capture_output=True, text=True)
    for l in r.stderr.splitlines() + r.stdout.splitlines():
        if any(k in l for k in ("RUN OK", "Replacing", "ms", "rror", "fail")):
            print("  ", l)
    names = ("depth", "normals", "seg")
    for i, (nm, rf) in enumerate(zip(names, ref)):
        subprocess.run(["adb", "pull", "%s/tout.bin.%d" % (dev, i),
                        os.path.join(HERE, "tout.bin.%d" % i)], capture_output=True)
        dv = np.fromfile(os.path.join(HERE, "tout.bin.%d" % i), np.float32)
        if dv.size != rf.size:
            print("%s: device size %d != ref %d (output order?)" % (nm, dv.size, rf.size))
            continue
        dv = dv.reshape(rf.shape)
        extra = ""
        if nm == "seg":
            extra = "  argmax agree %.4f" % (dv.argmax(1) == rf.argmax(1)).mean()
        print("device fp16 %s vs fp32 torch: corr %.5f  max|d| %.4f  NaN %s%s" %
              (nm, corr(dv, rf), np.abs(dv - rf).max(), np.isnan(dv).any(), extra))


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"   # parity / convert / fp16 / device / opcheck / all
    model = TIPSv2DPT(load_sd()).eval()
    if stage in ("parity", "all"):
        stage_parity(model)
    if stage in ("convert", "all"):
        stage_convert(model)
    if stage in ("fp16", "all"):
        stage_fp16()
    if stage in ("device", "all"):
        stage_device(model)
    if stage == "opcheck":
        opcheck(FP32)


if __name__ == "__main__":
    main()
