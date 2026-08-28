"""Fold constant fp16 DEQUANTIZE ops into fp32 weights.

Identity transform: the original graph already computes in fp32 downstream of each
DEQUANTIZE, so folding fp16->fp32 into the stored weights must leave outputs
bit-identical. The only variable that changes is weight storage dtype (and the
DEQUANTIZE ops disappearing from the graph).

Handles both inline buffers and the newer offset-buffer format (weights appended
after the flatbuffer); output is written with all buffers inline.
"""
import sys
import numpy as np
import flatbuffers
from ai_edge_litert import schema_py_generated as S

TT = S.TensorType


def buffer_bytes(raw, bt):
    """Materialize a BufferT's bytes whether inline or offset-addressed."""
    if bt.data is not None and len(bt.data) > 0:
        return bytes(bt.data)
    off = getattr(bt, 'offset', 0) or 0
    size = getattr(bt, 'size', 0) or 0
    if off > 1 and size > 0:
        return raw[off:off + size]
    return b''


def fold(src, dst):
    raw = open(src, 'rb').read()
    model = S.ModelT.InitFromPackedBuf(raw, 0)

    # inline every offset buffer so the repacked file is self-contained
    for bt in model.buffers:
        data = buffer_bytes(raw, bt)
        bt.data = np.frombuffer(data, dtype=np.uint8) if data else None
        if hasattr(bt, 'offset'):
            bt.offset = 0
            bt.size = 0

    deq_codes = set()
    for i, oc in enumerate(model.operatorCodes):
        if max(oc.builtinCode, oc.deprecatedBuiltinCode) == S.BuiltinOperator.DEQUANTIZE:
            deq_codes.add(i)

    buf_users = {}
    for sg in model.subgraphs:
        for t in sg.tensors:
            buf_users[t.buffer] = buf_users.get(t.buffer, 0) + 1

    folded = skipped = 0
    for sg in model.subgraphs:
        keep = []
        for op in sg.operators:
            if op.opcodeIndex not in deq_codes:
                keep.append(op)
                continue
            t_in = sg.tensors[op.inputs[0]]
            t_out = sg.tensors[op.outputs[0]]
            data = model.buffers[t_in.buffer].data
            if (t_in.type != TT.FLOAT16 or t_out.type != TT.FLOAT32
                    or data is None or len(data) == 0):
                keep.append(op)
                skipped += 1
                continue
            f32 = np.frombuffer(data.tobytes(), dtype=np.float16).astype(np.float32)
            nb = S.BufferT()
            nb.data = np.frombuffer(f32.tobytes(), dtype=np.uint8)
            model.buffers.append(nb)
            t_out.buffer = len(model.buffers) - 1
            if buf_users.get(t_in.buffer, 0) <= 1:
                model.buffers[t_in.buffer].data = None
            folded += 1
        sg.operators = keep

    b = flatbuffers.Builder(1024 * 1024)
    b.Finish(model.Pack(b), b'TFL3')
    open(dst, 'wb').write(b.Output())
    print(f'{src} -> {dst}: folded={folded} skipped={skipped}')


if __name__ == '__main__':
    fold(sys.argv[1], sys.argv[2])
