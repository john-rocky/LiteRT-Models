#!/usr/bin/env python3
"""Prefix-depth bisection of the text encoder on Metal (with the current gpu_patches)."""
import os, sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precheck_sam3 as P, gpu_patches as G
from ai_edge_litert.interpreter import Interpreter
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.options import Options
det = P.build_detector(os.path.join(P.ROOT, "models", "sam3.1_multiplex.pt"))
G.apply_text_patches(det)
te = det.backbone.language_backbone
tok = te.tokenizer(["wheel"], context_length=32)
emb = te.encoder.token_embedding(tok).detach()
out = os.path.join(P.ROOT, "models", "precheck")
def check(mod, xin, name):
    import litert_torch
    with torch.inference_mode(): ref = mod(xin).reshape(-1).numpy()
    p = os.path.join(out, f"dtxt2_{name}.tflite")
    litert_torch.convert(mod.eval(), (xin,)).export(p)
    it = Interpreter(model_path=p); it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], xin.numpy()); it.invoke()
    ycpu = it.get_tensor(it.get_output_details()[0]["index"]).reshape(-1)
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
class Prefix(nn.Module):
    def __init__(s, n, tail=False): super().__init__(); s.n = n; s.tail = tail
    def forward(s, x):
        enc = te.encoder
        x = (x + enc.positional_embedding[:32]).unsqueeze(1)
        for b in enc.transformer.resblocks[:s.n]: x = b(x)
        if s.tail:
            x = enc.ln_final(x); x = te.resizer(x)
        return x
for n in [1, 2, 4, 8, 12, 16, 20, 24]:
    check(Prefix(n), emb, f"prefix{n}")
check(Prefix(24, tail=True), emb, "full")
