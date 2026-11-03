#!/usr/bin/env python3
"""im2col sizes of every CONV_2D in RE-USE graphs against TFLite's mobile im2col line (r6 step 1).

Reads the flatbuffer only (no runtime). Starts from r4/conv_im2col_r4.py (same reading of shapes / options); adds
the margin to the line, the source block (from the output tensor name), and pairs the k-th CONV_2D of each graph so
the 2 s and 1 s graphs sit side by side.

The check (LiteRT v2.2.0 tflite/kernels/conv.cc 504-536 and 262-274): im2col_bytes = batch x out_h x out_w x c_in x
k_h x k_w x sizeof(input type); on a mobile build (IsMobilePlatform()) a conv that needs im2col with
im2col_bytes >= kMaxIm2colBufferSizeMobile (1024^3 B) fails Prepare with "Dilated im2col buffer size overflowed." when
its dilation is not 1, and silently drops im2col (reference kernel) when the dilation is 1. need_im2col for the
Android kernel (kGenericOptimized, ruy builds) = dilation != 1 or stride != 1 or filter h/w != 1.

  python -I conv_table_r6.py OUT.json NAME=GRAPH.tflite [NAME=GRAPH.tflite ...]
"""
import json
import sys
from collections import Counter

from ai_edge_litert import schema_py_generated as schema

LINE = 1024 ** 3  # kMaxIm2colBufferSizeMobile
CHECKED_OPS = ("CONV_2D", "CONV_3D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV", "CONV_3D_TRANSPOSE")
OPNAME = {v: k for k, v in schema.BuiltinOperator.__dict__.items() if not k.startswith("_")}
TYPE_BYTES = {schema.TensorType.FLOAT32: 4, schema.TensorType.FLOAT16: 2, schema.TensorType.INT8: 1,
              schema.TensorType.UINT8: 1, schema.TensorType.INT16: 2}


def block_of(name):
    """Short source block from a litert-torch tensor name (module path)."""
    n = name.lower()
    for key, label in (("dense_encoder", "encoder"), ("mask_decoder", "mag_decoder"), ("mag_decoder", "mag_decoder"),
                       ("phase_decoder", "phase_decoder")):
        if key in n:
            return label
    return "?"


def scan(path):
    buf = open(path, "rb").read()
    m = schema.ModelT.InitFromObj(schema.Model.GetRootAs(buf, 0))
    g = m.subgraphs[0]
    codes = m.operatorCodes
    counts = Counter()
    convs = []
    for i, op in enumerate(g.operators):
        oc = codes[op.opcodeIndex]
        code = max(oc.builtinCode, oc.deprecatedBuiltinCode)
        name = OPNAME.get(code, str(code))
        counts[name] += 1
        if name != "CONV_2D":
            continue
        o = op.builtinOptions
        tin, tfil = g.tensors[op.inputs[0]], g.tensors[op.inputs[1]]
        tout = g.tensors[op.outputs[0]]
        ins, fil, out = ([int(v) for v in t.shape] for t in (tin, tfil, tout))
        kh, kw, cin = fil[1], fil[2], fil[3]
        dil = (int(o.dilationHFactor), int(o.dilationWFactor))
        stride = (int(o.strideH), int(o.strideW))
        need = dil != (1, 1) or stride != (1, 1) or kh != 1 or kw != 1
        tb = TYPE_BYTES.get(tin.type, 4)
        nbytes = out[0] * out[1] * out[2] * cin * kh * kw * tb if need else 0
        oname = tout.name.decode() if isinstance(tout.name, bytes) else str(tout.name)
        convs.append(dict(
            node=i, input=ins, filter=fil, output=out, stride=list(stride), dilation=list(dil),
            padding="SAME" if o.padding == schema.Padding.SAME else "VALID", need_im2col=need,
            im2col_bytes=nbytes, im2col_gib=round(nbytes / 2 ** 30, 4),
            margin_to_line=(round(LINE / nbytes, 3) if nbytes else None),
            fails_mobile_prepare=bool(need and nbytes >= LINE and dil != (1, 1)),
            reference_fallback_on_mobile=bool(need and nbytes >= LINE and dil == (1, 1)),
            block=block_of(oname), out_tensor=oname[-160:]))
    checked = {k: counts[k] for k in CHECKED_OPS if counts[k]}
    return dict(graph=path, num_ops=len(g.operators), checked_op_counts=checked, conv2d=convs)


def main():
    out = sys.argv[1]
    res = {}
    for spec in sys.argv[2:]:
        name, path = spec.split("=", 1)
        res[name] = scan(path)
    names = list(res)
    # side-by-side: k-th CONV_2D of each graph
    n = max(len(r["conv2d"]) for r in res.values())
    rows = []
    for k in range(n):
        row = dict(k=k)
        for nm in names:
            c = res[nm]["conv2d"][k] if k < len(res[nm]["conv2d"]) else None
            row[nm] = None if c is None else dict(node=c["node"], output=c["output"], filter=c["filter"],
                                                    dilation=c["dilation"], stride=c["stride"],
                                                    im2col_gib=c["im2col_gib"], margin=c["margin_to_line"],
                                                    fails=c["fails_mobile_prepare"], block=c["block"])
        rows.append(row)
    res["_side_by_side"] = rows
    res["_line_bytes"] = LINE
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    for nm in names:
        r = res[nm]
        bad = [c for c in r["conv2d"] if c["fails_mobile_prepare"]]
        top = sorted((c for c in r["conv2d"] if c["need_im2col"]), key=lambda c: -c["im2col_bytes"])[:6]
        print(f"== {nm}: {r['num_ops']} ops, CONV_2D {len(r['conv2d'])}, checked ops {r['checked_op_counts']}, "
              f"fails on mobile: {[c['node'] for c in bad]}")
        for c in top:
            print(f"   node {c['node']:6d} {c['block']:13s} in {c['input']} filt {c['filter']} out {c['output']} "
                  f"s{tuple(c['stride'])} d{tuple(c['dilation'])} {c['padding']} im2col {c['im2col_gib']:.3f} GiB "
                  f"margin {c['margin_to_line']}{'  FAIL' if c['fails_mobile_prepare'] else ''}")


if __name__ == "__main__":
    main()
