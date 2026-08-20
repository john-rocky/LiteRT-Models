#!/usr/bin/env python3
"""Isolate which part of the SAM3 ViT-L export mis-lowers (raw graph corr=0.607).

Exports and parity-checks, one at a time, on the loaded SAM 3.1 detector:
  pe      patch_embed + abs-pos (tiled) + ln_pre                (8-D tile broadcast)
  win     one window-attention Block (6-D window partition)     [1,72,72,1024] -> same
  glob    one global-attention Block (5184-token SDPA + RoPE)   [1,72,72,1024] -> same
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precheck_sam3 as P  # noqa: E402  (reuses build_detector + patches)


def parity(path, x, ref, tag):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path, num_threads=8)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    it.set_tensor(inp["index"], x.numpy().astype(np.float32))
    it.invoke()
    y = it.get_tensor(it.get_output_details()[0]["index"]).reshape(-1)
    r = ref.reshape(-1).numpy()
    print(f"[{tag}] corr={np.corrcoef(y, r)[0, 1]:.5f} max|diff|={np.abs(y - r).max():.4g}",
          flush=True)


def run(mod, x, name, out):
    import litert_torch
    with torch.inference_mode():
        ref = mod(x)
    t0 = time.time()
    p = os.path.join(out, f"diag_{name}.tflite")
    litert_torch.convert(mod.eval(), (x,)).export(p)
    print(f"[{name}] exported {time.time()-t0:.0f}s", flush=True)
    P.opcheck(p, name)
    parity(p, x, ref, name)


class PE(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        v = self.vit
        x = v.patch_embed(x)
        if v.pos_embed is not None:
            x = x + v._get_abs_pos(v.pos_embed, x.shape[1:3]) if hasattr(v, "_get_abs_pos") else x
        return x


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = os.path.join(P.ROOT, "models", "precheck")
    det = P.build_detector(os.path.join(P.ROOT, "models", "sam3.1_multiplex.pt"))
    vit = det.backbone.vision_backbone.trunk
    print("[vit] blocks:", len(vit.blocks), "window:", vit.blocks[0].window_size,
          "global idx:", [i for i, b in enumerate(vit.blocks) if b.window_size == 0])
    torch.manual_seed(0)
    x = torch.randn(1, 72, 72, 1024) * 0.5
    if what in ("win", "all"):
        run(vit.blocks[0], x, "win", out)
    if what in ("glob", "all"):
        run(vit.blocks[7], x, "glob", out)
    if what in ("pe", "all"):
        img = torch.randn(1, 3, 1008, 1008)
        # full ViT forward up to (and including) the first block for a real-path check
        class Front(nn.Module):
            def __init__(self, v):
                super().__init__()
                self.v = v

            def forward(self, x):
                from sam3.model.vitdet import get_abs_pos
                v = self.v
                x = v.patch_embed(x)
                h, w = x.shape[1], x.shape[2]
                x = x + get_abs_pos(v.pos_embed, v.pretrain_use_cls_token, (h, w),
                                    v.retain_cls_token, tiling=v.tile_abs_pos)
                return v.ln_pre(x)
        try:
            run(Front(vit), img, "pe", out)
        except Exception as e:  # noqa: BLE001
            print("[pe] skipped:", type(e).__name__, str(e)[:300])
    sys.stdout.flush()


if __name__ == "__main__":
    main()
