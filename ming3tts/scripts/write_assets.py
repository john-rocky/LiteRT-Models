"""Host assets + the fp16 weight-range census (conversion venv).

out/export/temb_epss{5,6,7,10}.bin  TimestepEmbedder(t_k) [steps, 1024] float32 (DiT time MLP at
                                    the EPSS + sway times; the flow graphs take it as an input)
out/export/istft_window.bin         audio.decoder.head.istft.window [3528] float32 (from the checkpoint)
out/export/embed_tokens_f16.bin     LM input embedding table [151936, 896] float16 (host lookup)
out/gates/fp16_weight_census.json   per component: values that change when bf16 -> fp16
"""
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import ming_modules as M  # noqa: E402


def main():
    C.ensure_dirs()
    w = M.load_weights(["flowloss.cfm.model.t_embedder."])
    te = M.TimeEmbed(w).eval()
    out = {}
    with torch.no_grad():
        for steps in (5, 6, 7, 10):
            t = M.epss_times(steps)
            tab = te(t[:steps]).numpy().astype("<f4")
            tab.tofile(C.EXPORT / f"temb_epss{steps}.bin")
            out[f"temb_epss{steps}"] = list(tab.shape)
    census = {}
    with safe_open(str(C.HF_DIR / "model.safetensors"), framework="pt") as f:
        win = f.get_tensor("audio.decoder.head.istft.window").float().numpy().astype("<f4")
        win.tofile(C.EXPORT / "istft_window.bin")
        emb = f.get_tensor("model.model.embed_tokens.weight").float().numpy()
        emb.astype("<f2").tofile(C.EXPORT / "embed_tokens_f16.bin")
        groups = {"lm": "model.model.layers.", "lm_norm": "model.model.norm.", "embed": "model.model.embed_tokens.",
                  "flow": "flowloss.", "agg": "linear_proj_audio.", "decoder": "audio.decoder.",
                  "encoder": "audio.encoder.", "heads": ("stop_head.", "spk_head.")}
        for g, pre in groups.items():
            pre = pre if isinstance(pre, tuple) else (pre,)
            n = changed = sub = over = zero = 0
            worst = 0.0
            for k in f.keys():
                if not k.startswith(pre):
                    continue
                x = f.get_tensor(k).float().numpy().ravel()
                h = x.astype(np.float16).astype(np.float32)
                d = np.abs(h - x)
                n += x.size
                changed += int((d > 0).sum())
                ax = np.abs(x)
                sub += int(((ax > 0) & (ax < 6.103515625e-05)).sum())
                over += int((ax > 65504).sum())
                zero += int(((ax > 0) & (h == 0)).sum())
                worst = max(worst, float(d.max()) if d.size else 0.0)
            census[g] = {"values": n, "changed_by_fp16": changed, "fp16_subnormal_range": sub,
                         "above_fp16_max": over, "flushed_to_zero": zero, "max_abs_change": worst}
    C.write_json(C.GATES / "fp16_weight_census.json", {"note": __doc__, "assets": out, "census": census})
    for g, v in census.items():
        print("CENSUS", g, v)


if __name__ == "__main__":
    main()
