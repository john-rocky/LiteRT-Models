"""
Export the neural English G2P (app/src/main/assets/dp_g2p_litert.tflite) used by NeuralG2p.kt
— a LiteRT (tflite) model, run on the LiteRT CompiledModel CPU accelerator.

Source: DeepPhonemizer `en_us_cmudict_forward` (MIT, https://github.com/as-ideas/DeepPhonemizer)
— a non-autoregressive forward transformer, char -> stress-less ARPABET.

Why this shape: tflite/LiteRT needs static shapes, and the "natural" variable-length export does
NOT convert (a known litert-torch dynamic-shape gap, already reported). Workaround = a single
fixed-length graph [1, 96] with the padding mask computed IN-GRAPH from `id == 0`, so the Kotlin
side passes one 0-padded FLOAT array (the proven CompiledModel writeFloat path) and decodes the
real length back. CPU only: the graph uses EQUAL / SELECT_V2 / 5D attention tensors that the GPU
delegate rejects but the CPU accelerator runs fine.

Usage:
    pip install --no-deps deep-phonemizer tqdm
    # plus a litert-torch env (ai-edge-torch) + ai-edge-litert for the op check
    python convert_dp_g2p_litert.py
"""
import os
import sys
import types

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["JAX_PLATFORMS"] = "cpu"


# litert-torch's min-cut layout partitioner imports scipy.sparse.csgraph.maximum_flow, whose
# transitive _propack import fails to dlopen on macOS. Stub it (SVD is unused by maximum_flow).
class _Any:
    def __getattr__(self, k):
        return None


_stub = types.ModuleType("scipy.sparse.linalg._propack")
for _n in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
    setattr(_stub, _n, _Any())
sys.modules["scipy.sparse.linalg._propack"] = _stub

import numpy as np
import torch
import torch.nn as nn

# torch >=2.6 defaults weights_only=True; dp checkpoints pickle classes.
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
from dp.phonemizer import Phonemizer  # noqa: E402

MAXT = 96
CKPT = "en_us_cmudict_forward.pt"
URL = "https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/" + CKPT
OUT = os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "assets", "dp_g2p_litert.tflite")

if not os.path.exists(CKPT):
    import urllib.request
    print("downloading", URL)
    urllib.request.urlretrieve(URL, CKPT)

phon = Phonemizer.from_checkpoint(CKPT)
m = phon.predictor.model.eval()
tt, pt = phon.predictor.text_tokenizer, phon.predictor.phoneme_tokenizer
CHAR2IDX, REP = dict(tt.token_to_idx), tt.char_repeats
IDX2ARPA = {int(k): v.strip("[]") for k, v in pt.idx_to_token.items()}
SPECIAL = {"_", "<en_us>", "<end>"}


class Wrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, text):                       # [N, T] float32, 0.0 = pad
        ids = text.to(torch.int64)
        pad_mask = ids.eq(0)                        # in-graph padding mask
        x = ids.transpose(0, 1)
        x = self.m.embedding(x)
        x = self.m.pos_encoder(x)
        x = self.m.encoder(x, src_key_padding_mask=pad_mask)
        x = self.m.fc_out(x)
        return x.transpose(0, 1)                    # [N, T, 42]


def tokenize(w):
    ids = [1]
    for c in w.lower():
        if c in CHAR2IDX and c != "_":
            ids += [CHAR2IDX[c]] * REP
    return ids + [2]


def padded(w):
    ids = tokenize(w)
    L = len(ids)
    return np.array([ids + [0] * (MAXT - L)], np.float32), L


def decode(arg):
    d, p = [], None
    for t in arg:
        if t != p:
            d.append(t); p = t
    return "".join(IDX2ARPA[t] for t in d if t != 0 and t in IDX2ARPA and IDX2ARPA[t] not in SPECIAL)


import litert_torch  # noqa: E402  (import AFTER building the torch model)

wrap = Wrap(m).eval()
dummy, _ = padded("hello")
edge = litert_torch.convert(wrap, (torch.from_numpy(dummy),))
edge.export(OUT)
print("wrote", OUT, os.path.getsize(OUT) // 1_000_000, "MB")

# Validate free input on the LiteRT runtime.
from ai_edge_litert.interpreter import Interpreter  # noqa: E402
itp = Interpreter(model_path=OUT); itp.allocate_tensors()
ti, oi = itp.get_input_details()[0], itp.get_output_details()[0]
ok = 0
words = ["hello", "cat", "anthropic", "github", "kubernetes", "daisuke", "nvidia", "supercalifragilistic"]
for w in words:
    t, L = padded(w)
    itp.set_tensor(ti["index"], t); itp.invoke()
    got = decode(itp.get_tensor(oi["index"])[0].argmax(-1).tolist()[:L])
    lib = phon(w, lang="en_us").replace("[", "").replace("]", "")
    ok += got == lib
    print(f"  {'ok ' if got == lib else 'DIFF'} {w}: {got}")
print(f"match {ok}/{len(words)}")
