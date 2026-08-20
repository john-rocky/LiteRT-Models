#!/usr/bin/env python3
"""Isolate the Metal-GPU mismatch of the text encoder: export sub-blocks, compare CPU vs GPU."""
import os, sys, time, types, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precheck_sam3 as P, gpu_patches as G
from ai_edge_litert.interpreter import Interpreter
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator

det = P.build_detector(os.path.join(P.ROOT, "models", "sam3.1_multiplex.pt"))
G.apply_text_patches(det)
te = det.backbone.language_backbone
tok = te.tokenizer(["wheel"], context_length=32)
emb = te.encoder.token_embedding(tok).detach()
x0 = (emb + te.encoder.positional_embedding[:32]).unsqueeze(1).detach()  # (1,1,32,1024)
out = os.path.join(P.ROOT, "models", "precheck")

def check(mod, xin, name):
    import litert_torch
    with torch.inference_mode():
        ref = mod(xin).reshape(-1).numpy()
    p = os.path.join(out, f"dtxt_{name}.tflite")
    litert_torch.convert(mod.eval(), (xin,)).export(p)
    it = Interpreter(model_path=p); it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], xin.numpy()); it.invoke()
    ycpu = it.get_tensor(it.get_output_details()[0]["index"]).reshape(-1)
    from ai_edge_litert.options import Options
    res = []
    for f32 in (False, True):
        if f32:
            o = Options.create(); o.hardware_accelerators = HardwareAccelerator.GPU; o.gpu_options.enforce_f32 = True
            m = CompiledModel.from_file(p, options=o)
        else:
            m = CompiledModel.from_file(p, HardwareAccelerator.GPU)
        ib = m.create_input_buffers(0); ob = m.create_output_buffers(0); ib[0].write(xin.numpy().ravel())
        m.run_by_index(0, ib, ob); y = np.array(ob[0].read(ref.size, np.float32)); m.close()
        res.append(f"{'f32' if f32 else 'fp16'} corr={np.corrcoef(y, ref)[0,1]:.6f} max|d|={np.abs(y-ref).max():.3g}")
    print(f"[{name}] cpu corr={np.corrcoef(ycpu, ref)[0,1]:.6f} |ref|max={np.abs(ref).max():.3g}  gpu " + " / ".join(res), flush=True)

blk = te.encoder.transformer.resblocks[0]
class Attn(nn.Module):
    def __init__(s): super().__init__(); s.b = blk
    def forward(s, x): return s.b.attention(q_x=s.b.ln_1(x))
class LN(nn.Module):
    def __init__(s): super().__init__(); s.b = blk
    def forward(s, x): return s.b.ln_1(x)
class Mlp(nn.Module):
    def __init__(s): super().__init__(); s.b = blk
    def forward(s, x): return s.b.mlp(s.b.ln_2(x))
class Blk(nn.Module):
    def __init__(s): super().__init__(); s.b = blk
    def forward(s, x): return s.b(x)
class NoMask(nn.Module):
    def __init__(s): super().__init__(); s.b = blk
    def forward(s, x):
        a = s.b.attn; B,_,L,E = x.shape; H=a.num_heads; hd=E//H; w,b=a.in_proj_weight,a.in_proj_bias
        q = torch.nn.functional.linear(x, w[:E], b[:E]).reshape(B,L,H,hd).transpose(1,2)
        k = torch.nn.functional.linear(x, w[E:2*E], b[E:2*E]).reshape(B,L,H,hd).transpose(1,2)
        v = torch.nn.functional.linear(x, w[2*E:], b[2*E:]).reshape(B,L,H,hd).transpose(1,2)
        sc = torch.matmul(q*(1.0/8.0), k.transpose(-2,-1))
        return torch.matmul(torch.softmax(sc,-1), v)
class QK(nn.Module):
    def __init__(s): super().__init__(); s.b = blk
    def forward(s, x):
        a = s.b.attn; B,_,L,E = x.shape; H=a.num_heads; hd=E//H; w,b=a.in_proj_weight,a.in_proj_bias
        q = torch.nn.functional.linear(x, w[:E], b[:E]).reshape(B,L,H,hd).transpose(1,2)
        k = torch.nn.functional.linear(x, w[E:2*E], b[E:2*E]).reshape(B,L,H,hd).transpose(1,2)
        return torch.matmul(q*(1.0/8.0), k.transpose(-2,-1))
for name, M in [("ln", LN), ("qk", QK), ("nomask", NoMask), ("attn", Attn), ("mlp", Mlp), ("block", Blk)]:
    try: check(M(), x0, name)
    except Exception as e: print(f"[{name}] FAILED {type(e).__name__}: {str(e)[:200]}")

class Res1(nn.Module):   # x + attn(ln1(x))
    def __init__(s): super().__init__(); s.b = blk
    def forward(s, x): return x + s.b.attention(q_x=s.b.ln_1(x))
class Blk23(nn.Module):
    def __init__(s, i): super().__init__(); s.b = te.encoder.transformer.resblocks[i]
    def forward(s, x): return s.b(x)
check(Res1(), x0, "res1")
# feed a deep block with its real input
with torch.inference_mode():
    xd = x0
    for b in te.encoder.transformer.resblocks[:12]: xd = b(xd)
for i in (12, 13):
    check(Blk23(i), xd, f"block{i}_realinput")
print("--- safe softmax variants (with LN, f32 tells)")
class SM(nn.Module):
    def __init__(s, mode): super().__init__(); s.b = blk; s.mode = mode
    def forward(s, x):
        a = s.b.attn; B,_,L,E = x.shape; H=a.num_heads; hd=E//H; w,b=a.in_proj_weight,a.in_proj_bias
        x = s.b.ln_1(x)
        q = torch.nn.functional.linear(x, w[:E], b[:E]).reshape(B,L,H,hd).transpose(1,2)
        k = torch.nn.functional.linear(x, w[E:2*E], b[E:2*E]).reshape(B,L,H,hd).transpose(1,2)
        v = torch.nn.functional.linear(x, w[2*E:], b[2*E:]).reshape(B,L,H,hd).transpose(1,2)
        sc = torch.matmul(q*(1.0/8.0), k.transpose(-2,-1)); keep = s.b.causal_keep
        if s.mode == "A_maxall":
            m = sc.max(-1, keepdim=True).values; e = torch.exp(sc - m) * keep
        elif s.mode == "B_maxvalid_noclamp":
            m = (sc*keep + (1-keep)*(-1e4)).max(-1, keepdim=True).values; e = torch.exp(sc - m) * keep
        elif s.mode == "C_maxvalid_clamp":
            m = (sc*keep + (1-keep)*(-1e4)).max(-1, keepdim=True).values; e = torch.exp(torch.clamp(sc - m, max=0.0)) * keep
        elif s.mode == "D_maxvalid_min":
            m = (sc*keep + (1-keep)*(-1e4)).max(-1, keepdim=True).values; e = torch.exp(torch.minimum(sc - m, torch.zeros_like(sc))) * keep
        elif s.mode == "E_maxvalid_relu":
            m = (sc*keep + (1-keep)*(-1e4)).max(-1, keepdim=True).values; d = sc - m; e = torch.exp(d - torch.relu(d)) * keep
        elif s.mode == "G_mulkeep_exp":
            m = (sc*keep + (1-keep)*(-1e4)).max(-1, keepdim=True).values; e = torch.exp((sc - m) * keep) * keep
        elif s.mode == "H_mulkeep_exp2":
            m = (sc*keep + (1-keep)*(-1e4)).max(-1, keepdim=True).values; e = torch.exp((sc - m) * keep) - (1.0 - keep)
        elif s.mode == "F_addneg_maxall":   # additive mask then plain max/exp/sum manual
            sc2 = sc + (1-keep)*(-1e4); m = sc2.max(-1, keepdim=True).values; e = torch.exp(sc2 - m)
        p = e / e.sum(-1, keepdim=True)
        return torch.matmul(p, v)
for mode in ["G_mulkeep_exp","H_mulkeep_exp2"]:
    check(SM(mode), x0, mode)
