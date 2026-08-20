#!/usr/bin/env python3
"""torch.export the head / text wrappers and list GPU-hostile aten ops with source lines."""
import os, sys, collections, re, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precheck_sam3 as P
what = sys.argv[1] if len(sys.argv) > 1 else "head"
det = P.build_detector(os.path.join(P.ROOT, "models", "sam3.1_multiplex.pt"))
sizes = [(288, 288), (144, 144), (72, 72)]
if what == "head":
    m = P.HeadFlat(det, sizes)
    n_img = sum(256 * h * w for h, w in sizes)
    x = torch.randn(1, n_img + P.CONTEXT * 256 + P.CONTEXT) * 0.5
    x[:, -P.CONTEXT:] = 0.0; x[:, -P.CONTEXT + 5:] = 1.0
else:
    m = P.TextFlat(det)
    x = det.backbone.language_backbone.tokenizer(["a red car"], context_length=P.CONTEXT)
ep = torch.export.export(m.eval(), (x,))
ep = ep.run_decompositions()
BAD = ("index", "gather", "where", "clamp", "scatter", "topk", "argmax", "eq", "ne", "gt", "lt", "ge", "le",
       "logical", "masked_fill", "_to_copy", "cumsum", "sort", "nonzero", "unbind", "select_scatter",
       "slice_scatter", "index_put", "expand", "repeat_interleave", "embedding")
cnt = collections.Counter(); src = collections.defaultdict(collections.Counter); big = collections.Counter()
for n in ep.graph.nodes:
    if n.op != "call_function": continue
    t = str(n.target)
    st = n.meta.get("stack_trace", "")
    lines = [l.strip() for l in st.splitlines() if "vendor_sam3" in l or "precheck_sam3" in l]
    loc = re.sub(r'.*/(sam3/[^"]+)", line (\d+).*', r'\1:\2', lines[-1]) if lines else "?"
    val = n.meta.get("val")
    if hasattr(val, "shape") and len(val.shape) > 4:
        big[(t.split(".")[0], loc)] += 1
    if any(b in t for b in BAD):
        cnt[t] += 1; src[t][loc] += 1
print("== hostile aten ops")
for t, c in cnt.most_common():
    print(f"{t:45s} {c:4d}  ", dict(src[t].most_common(4)))
print("== >4-D producers")
for (t, loc), c in big.most_common(30):
    print(f"{t:35s} {c:4d} {loc}")
