"""Store large fp32 consts as int8 (or int4) + DEQUANTIZE — the weight-only
quantized twin of a graph, changing nothing else. Companion to make_fp16.py.

Per-tensor symmetric scale (scale = max|w| / 127, or /7 for int4). Accuracy is
not the point (speed probes); the DEQUANTIZE pattern matches what dynamic-range
quantized files carry.

  python make_int8.py src.tflite dst.tflite         # int8
  python make_int8.py src.tflite dst.tflite --int4  # int4 (2 values per byte)
"""
import sys
import numpy as np
import flatbuffers
from ai_edge_litert import schema_py_generated as S

TT = S.TensorType
MIN_ELEMS = 4096


def buffer_bytes(raw, bt):
    if bt.data is not None and len(bt.data) > 0:
        return bytes(bt.data)
    off = getattr(bt, 'offset', 0) or 0
    size = getattr(bt, 'size', 0) or 0
    return raw[off:off + size] if off > 1 and size > 0 else b''


def run(src, dst, int4=False):
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

    qmax = 7 if int4 else 127
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
            w = np.frombuffer(bt.data.tobytes(), dtype=np.float32)
            scale = float(np.abs(w).max()) / qmax or 1.0
            q = np.clip(np.round(w / scale), -qmax - 1, qmax).astype(np.int8)
            if int4:
                # pack two int4 values per byte, low nibble first
                if q.size % 2:
                    q = np.concatenate([q, np.zeros(1, np.int8)])
                lo = (q[0::2] & 0x0F).astype(np.uint8)
                hi = ((q[1::2] & 0x0F) << 4).astype(np.uint8)
                payload = (lo | hi).tobytes()
            else:
                payload = q.tobytes()
            nb = S.BufferT()
            nb.data = np.frombuffer(payload, dtype=np.uint8)
            model.buffers.append(nb)
            tq = S.TensorT()
            tq.shape = list(t.shape) if t.shape is not None else []
            tq.type = TT.INT4 if int4 else TT.INT8
            tq.buffer = len(model.buffers) - 1
            tq.name = (t.name or b'') + (b'_i4' if int4 else b'_i8')
            bt.data = None  # orphan the fp32 payload, or the file keeps both copies
            qp = S.QuantizationParametersT()
            qp.scale = [scale]
            qp.zeroPoint = [0]
            tq.quantization = qp
            sg.tensors.append(tq)
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
    print(f'{src} -> {dst}: {"int4" if int4 else "int8"}-ized {converted} consts')


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], int4='--int4' in sys.argv[3:])
