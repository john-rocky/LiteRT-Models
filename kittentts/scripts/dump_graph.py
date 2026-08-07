"""Map the KittenTTS nano ONNX graph into its StyleTTS2 module tree."""
import sys
from collections import Counter, defaultdict

import onnx

PATH = sys.argv[1] if len(sys.argv) > 1 else (
    "models/nano-0.8-fp32/kitten_tts_nano_v0_8.onnx")

model = onnx.load(PATH, load_external_data=False)
g = model.graph

init = {t.name: t for t in g.initializer}


def prefix(name, depth=2):
    parts = [p for p in name.split("/") if p]
    return "/".join(parts[:depth]) if parts else "(root)"


groups = defaultdict(Counter)
for n in g.node:
    groups[prefix(n.name)][n.op_type] += 1

print("== node groups (depth 2) ==")
for k in sorted(groups):
    total = sum(groups[k].values())
    top = ", ".join(f"{o}x{c}" for o, c in groups[k].most_common(6))
    print(f"  {k:42s} {total:4d}  {top}")

print("\n== LSTM nodes ==")
for n in g.node:
    if n.op_type == "LSTM":
        shapes = []
        for i in n.input[1:4]:
            if i in init:
                shapes.append(tuple(init[i].dims))
        attrs = {a.name: (a.i if a.type == 2 else a.s.decode() if a.type == 3 else "?")
                 for a in n.attribute}
        print(f"  {n.name}: W/R/B={shapes} attrs={attrs}")

print("\n== ConvTranspose ==")
for n in g.node:
    if n.op_type == "ConvTranspose":
        w = n.input[1]
        dims = tuple(init[w].dims) if w in init else "?"
        attrs = {a.name: list(a.ints) for a in n.attribute if a.ints}
        print(f"  {n.name}: W={dims} {attrs}")

print("\n== Resize ==")
for n in g.node:
    if n.op_type == "Resize":
        print(f"  {n.name} inputs={list(n.input)}")

print("\n== outputs and their producers ==")
producers = {}
for n in g.node:
    for o in n.output:
        producers[o] = n
for out in g.output:
    p = producers.get(out.name)
    print(f"  {out.name} <- {p.op_type if p else '?'} {p.name if p else ''}")

print("\n== Loop body ==")
for n in g.node:
    if n.op_type == "Loop":
        for a in n.attribute:
            if a.name == "body":
                ops = Counter(x.op_type for x in a.g.node)
                print(f"  {n.name}: body ops {dict(ops)}")
                print(f"  loop inputs: {list(n.input)}")
                print(f"  loop outputs: {list(n.output)}")
