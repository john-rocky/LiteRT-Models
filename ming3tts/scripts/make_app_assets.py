"""Host files the Android app reads beside the graphs, from the upstream checkpoint.

  python scripts/make_app_assets.py            # writes $MING3TTS_OUT/app_assets/

  tokenizer_vocab.txt   one BPE token per line, id order (tokenizer.json model.vocab)
  tokenizer_merges.txt  "a b" per line, rank order (tokenizer.json model.merges)
  tokenizer_added.txt   "id<TAB>content" (tokenizer.json added_tokens)
  char_classes.bin      int32 n, then n x (int32 first code point, int32 class bits) runs: the
                        character classes of Python's `regex` (\\p{L}, \\p{N}, \\s, used by the Qwen
                        pre-tokenizer) and of Python str (isspace, isalnum, \\d of `re`, used by the
                        upstream text frontend), so the phone needs no Unicode tables of its own
  stop_head.bin         stop_head.weight [2, 896] then stop_head.bias [2], float32

write_assets.py writes the other host files (embedding table, time embeddings, ISTFT window).
The class table depends on the Python and `regex` versions (REPRODUCE.md pins them).
"""
import json
import os
import re
import struct
import sys

import regex
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

B_L, B_N, B_WS, B_PYSPACE, B_PYALNUM, B_PYDIGIT = 1, 2, 4, 8, 16, 32


def write_tokenizer_assets(out):
    t = json.load(open(C.HF_DIR / "tokenizer.json", encoding="utf-8"))
    vocab = t["model"]["vocab"]
    inv = [None] * len(vocab)
    for tok, i in vocab.items():
        assert inv[i] is None
        inv[i] = tok
    assert all(s is not None for s in inv)
    for s in inv:
        assert "\n" not in s and "\r" not in s and "\t" not in s and " " not in s, repr(s)
    (out / "tokenizer_vocab.txt").write_text("\n".join(inv) + "\n", encoding="utf-8")
    merges = []
    for m in t["model"]["merges"]:
        a, b = m if isinstance(m, list) else m.split(" ")
        assert " " not in a and " " not in b
        merges.append(f"{a} {b}")
    (out / "tokenizer_merges.txt").write_text("\n".join(merges) + "\n", encoding="utf-8")
    lines = []
    for a in t["added_tokens"]:
        assert "\t" not in a["content"] and "\n" not in a["content"]
        lines.append(f"{a['id']}\t{a['content']}")
    (out / "tokenizer_added.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"vocab": len(inv), "merges": len(merges), "added": len(lines)}


def write_char_classes(out):
    """Runs of identical class bits over all code points (surrogates included, as Python sees them)."""
    rl = regex.compile(r"\p{L}")
    rn = regex.compile(r"\p{N}")
    rs = regex.compile(r"\s")
    rd = re.compile(r"\d")
    runs, prev = [], None
    for cp in range(0x110000):
        c = chr(cp)
        bits = 0
        if rl.fullmatch(c):
            bits |= B_L
        if rn.fullmatch(c):
            bits |= B_N
        if rs.fullmatch(c):
            bits |= B_WS
        if c.isspace():
            bits |= B_PYSPACE
        if c.isalnum():
            bits |= B_PYALNUM
        if rd.fullmatch(c):
            bits |= B_PYDIGIT
        if bits != prev:
            runs.append((cp, bits))
            prev = bits
    with open(out / "char_classes.bin", "wb") as f:
        f.write(struct.pack("<i", len(runs)))
        for s, b in runs:
            f.write(struct.pack("<ii", s, b))
    return {"runs": len(runs)}


def write_stop_head(out):
    with safe_open(str(C.HF_DIR / "model.safetensors"), framework="pt") as f:
        w = f.get_tensor("stop_head.weight").float().numpy()
        b = f.get_tensor("stop_head.bias").float().numpy()
    with open(out / "stop_head.bin", "wb") as f:
        f.write(w.astype("<f4").tobytes())
        f.write(b.astype("<f4").tobytes())
    return {"weight": list(w.shape), "bias": list(b.shape)}


def main():
    out = C.OUT / "app_assets"
    out.mkdir(parents=True, exist_ok=True)
    res = {"tokenizer": write_tokenizer_assets(out), "char_classes": write_char_classes(out),
           "stop_head": write_stop_head(out)}
    res["sha256"] = {p.name: C.sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.suffix != ".json"}
    C.write_json(out / "app_assets.json", res)
    for name, h in res["sha256"].items():
        print("ASSET", name, os.path.getsize(out / name), h, flush=True)


if __name__ == "__main__":
    main()
