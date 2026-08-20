#!/usr/bin/env python3
"""Per-block activation magnitudes of the SAM3 ViT-L trunk (fp16-wall diagnosis)."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precheck_sam3 as P
det = P.build_detector(os.path.join(P.ROOT, "models", "sam3.1_multiplex.pt"))
vit = det.backbone.vision_backbone.trunk
torch.manual_seed(0)
x = torch.randn(1, 3, 1008, 1008)
stats = []
def hook(name):
    def f(m, i, o):
        t = o if isinstance(o, torch.Tensor) else o[0]
        stats.append((name, t.abs().max().item(), t.abs().mean().item()))
    return f
hs = [vit.patch_embed.register_forward_hook(hook("patch_embed")), vit.ln_pre.register_forward_hook(hook("ln_pre"))]
for i, b in enumerate(vit.blocks):
    hs.append(b.register_forward_hook(hook(f"block{i}{'G' if b.window_size==0 else ''}")))
    hs.append(b.norm1.register_forward_hook(hook(f"  b{i}.norm1")))
    hs.append(b.mlp.fc1.register_forward_hook(hook(f"  b{i}.fc1")))
    hs.append(b.attn.qkv.register_forward_hook(hook(f"  b{i}.qkv")))
with torch.inference_mode():
    det.backbone.vision_backbone(x, need_sam3_out=True, need_interactive_out=False, need_propagation_out=False)
for n, mx, mn in stats:
    if not n.startswith("  ") or mx > 200:
        print(f"{n:14s} max|x|={mx:9.2f}  mean|x|={mn:7.3f}")
# LayerNorm variance-sum estimate: sum over 1024 of (x-mean)^2 for the residual stream
