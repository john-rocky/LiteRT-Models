"""Truncate a .tflite to its first K operators — the bisect tool for finding
which op range makes a graph slow (memorize, 0.04x, is the target).

The subgraph's outputs are replaced by the output tensors of operator K-1
(plus any original outputs already produced by ops < K). Operators are in
execution order per TFLite convention. Weights past the cut stay in the file
(harmless bloat; only referenced buffers matter to the compiler).

  python cut_prefix.py model.tflite 120 out_k120.tflite
  python cut_prefix.py model.tflite --list          # print op index -> opcode
"""
import sys
import flatbuffers
from ai_edge_litert import schema_py_generated as S


def opcode_name(model, op):
    oc = model.operatorCodes[op.opcodeIndex]
    code = max(oc.builtinCode, oc.deprecatedBuiltinCode)
    for k in dir(S.BuiltinOperator):
        if not k.startswith('_') and getattr(S.BuiltinOperator, k) == code:
            return k
    return f'code_{code}'


def load(src):
    raw = open(src, 'rb').read()
    model = S.ModelT.InitFromPackedBuf(raw, 0)
    import numpy as np
    for bt in model.buffers:
        if bt.data is None or len(bt.data) == 0:
            off = getattr(bt, 'offset', 0) or 0
            size = getattr(bt, 'size', 0) or 0
            if off > 1 and size > 0:
                bt.data = np.frombuffer(raw[off:off + size], dtype=np.uint8)
        if hasattr(bt, 'offset'):
            bt.offset = 0
            bt.size = 0
    return model


def run(src, k, dst):
    model = load(src)
    sg = model.subgraphs[0]
    ops = list(sg.operators)
    assert 0 < k <= len(ops), f'k must be in 1..{len(ops)}'
    kept = ops[:k]
    produced = set()
    for op in kept:
        produced.update(o for o in (op.outputs or []) if o >= 0)
    tail = [o for o in (kept[-1].outputs or []) if o >= 0]
    keep_orig = [o for o in sg.outputs if o in produced and o not in tail]
    sg.operators = kept
    sg.outputs = tail + keep_orig
    b = flatbuffers.Builder(1024 * 1024)
    b.Finish(model.Pack(b), b'TFL3')
    open(dst, 'wb').write(b.Output())
    print(f'{src} -> {dst}: kept {k}/{len(ops)} ops, outputs={list(sg.outputs)}')


if __name__ == '__main__':
    if sys.argv[2] == '--list':
        model = load(sys.argv[1])
        for i, op in enumerate(model.subgraphs[0].operators):
            print(i, opcode_name(model, op))
    else:
        run(sys.argv[1], int(sys.argv[2]), sys.argv[3])
