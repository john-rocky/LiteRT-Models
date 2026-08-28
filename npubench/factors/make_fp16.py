"""Inverse of fold_deq: store large fp32 consts as fp16 + DEQUANTIZE.

Produces the fp16-weights twin of a graph, changing nothing else — the
controlled A/B for the weight-storage-dtype variable.
"""
import sys
import numpy as np
import flatbuffers
from ai_edge_litert import schema_py_generated as S

TT = S.TensorType
MIN_ELEMS = 4096  # only weight-sized consts; leave LN gammas etc. fp32


def buffer_bytes(raw, bt):
    if bt.data is not None and len(bt.data) > 0:
        return bytes(bt.data)
    off = getattr(bt, 'offset', 0) or 0
    size = getattr(bt, 'size', 0) or 0
    return raw[off:off + size] if off > 1 and size > 0 else b''


def run(src, dst):
    raw = open(src, 'rb').read()
    model = S.ModelT.InitFromPackedBuf(raw, 0)
    for bt in model.buffers:
        data = buffer_bytes(raw, bt)
        bt.data = np.frombuffer(data, dtype=np.uint8) if data else None
        if hasattr(bt, 'offset'):
            bt.offset = 0
            bt.size = 0

    deq_code = None
    for i, oc in enumerate(model.operatorCodes):
        if max(oc.builtinCode, oc.deprecatedBuiltinCode) == S.BuiltinOperator.DEQUANTIZE:
            deq_code = i
    if deq_code is None:
        oc = S.OperatorCodeT()
        oc.builtinCode = S.BuiltinOperator.DEQUANTIZE
        oc.deprecatedBuiltinCode = S.BuiltinOperator.DEQUANTIZE
        oc.version = 1
        model.operatorCodes.append(oc)
        deq_code = len(model.operatorCodes) - 1

    converted = 0
    for sg in model.subgraphs:
        io_tensors = set(list(sg.inputs) + list(sg.outputs))
        new_ops = []
        for ti, t in enumerate(sg.tensors):
            if ti in io_tensors or t.type != TT.FLOAT32:
                continue
            bt = model.buffers[t.buffer]
            if bt.data is None or len(bt.data) < MIN_ELEMS * 4:
                continue
            f16 = np.frombuffer(bt.data.tobytes(), dtype=np.float32).astype(np.float16)
            nb = S.BufferT()
            nb.data = np.frombuffer(f16.tobytes(), dtype=np.uint8)
            model.buffers.append(nb)
            # new fp16 const tensor
            t16 = S.TensorT()
            t16.shape = list(t.shape) if t.shape is not None else []
            t16.type = TT.FLOAT16
            t16.buffer = len(model.buffers) - 1
            t16.name = (t.name or b'') + b'_f16'
            sg.tensors.append(t16)
            bt.data = None  # orphan the fp32 payload, or the file keeps both copies
            # original tensor becomes the DEQUANTIZE output (activation)
            empty = S.BufferT()
            model.buffers.append(empty)
            t.buffer = len(model.buffers) - 1
            op = S.OperatorT()
            op.opcodeIndex = deq_code
            op.inputs = [len(sg.tensors) - 1]
            op.outputs = [ti]
            new_ops.append(op)
            converted += 1
        sg.operators = new_ops + list(sg.operators)

    b = flatbuffers.Builder(1024 * 1024)
    b.Finish(model.Pack(b), b'TFL3')
    open(dst, 'wb').write(b.Output())
    print(f'{src} -> {dst}: fp16-ized {converted} consts')


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
