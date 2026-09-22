"""Fold constant-only RESHAPE nodes without changing any constant bytes.

Folding before FLOAT_CASTING prevents a delegate from receiving RESHAPE whose
only data input is a dequantized constant. Dynamic reshapes remain unchanged.
"""
import hashlib
from pathlib import Path
import flatbuffers
import numpy as np
from ai_edge_litert import schema_py_generated as schema


def _payload(model, raw, index):
    value=model.buffers[index]
    if value.offset and value.size:
        start,end=int(value.offset),int(value.offset+value.size)
        assert 0 <= start <= end <= len(raw)
        return raw[start:end]
    return bytes(value.data) if value.data is not None else b''


def _operator(model, op):
    code=model.operatorCodes[op.opcodeIndex]
    return max(code.builtinCode, code.deprecatedBuiltinCode)


def constant_reshape_paths(path):
    raw=Path(path).read_bytes();model=schema.ModelT.InitFromPackedBuf(raw,0);rows=[]
    for gi,graph in enumerate(model.subgraphs):
        constants={i for i,t in enumerate(graph.tensors) if _payload(model,raw,int(t.buffer))}
        for oi,op in enumerate(graph.operators):
            code=_operator(model,op);first=int(op.inputs[0]) if len(op.inputs) else -1
            if code==schema.BuiltinOperator.DEQUANTIZE and first in constants:
                constants.update(int(i) for i in op.outputs)
            elif code==schema.BuiltinOperator.RESHAPE and first in constants:
                rows.append({'subgraph':gi,'operator':oi,'input':first,
                    'outputs':[int(i) for i in op.outputs],
                    'input_shape':[int(n) for n in graph.tensors[first].shape]})
                constants.update(int(i) for i in op.outputs)
    return rows


def fold_constant_reshapes(source, target=None):
    source=Path(source);target=Path(target) if target is not None else source
    raw=source.read_bytes();model=schema.ModelT.InitFromPackedBuf(raw,0)
    original=[_payload(model,raw,i) for i in range(len(model.buffers))]
    # Materialize external buffers before repacking; all payload bytes survive.
    for buffer,data in zip(model.buffers,original):
        if buffer.offset and buffer.size:
            buffer.data=np.frombuffer(data,dtype=np.uint8).copy();buffer.offset=buffer.size=0
    changes=[]
    for gi,graph in enumerate(model.subgraphs):
        operators=[]
        for oi,op in enumerate(graph.operators):
            source_index=int(op.inputs[0]) if len(op.inputs) else -1
            if _operator(model,op)!=schema.BuiltinOperator.RESHAPE or source_index<0:
                operators.append(op);continue
            tensor=graph.tensors[source_index];payload=original[int(tensor.buffer)]
            if not payload:
                operators.append(op);continue
            assert len(op.outputs)==1
            output_index=int(op.outputs[0]);output=graph.tensors[output_index]
            assert tensor.type==output.type
            assert int(np.prod(tensor.shape))==int(np.prod(output.shape))
            output.buffer=int(tensor.buffer)
            changes.append({'subgraph':gi,'original_operator':oi,'source_tensor':source_index,
                'output_tensor':output_index,'source_shape':[int(n) for n in tensor.shape],
                'output_shape':[int(n) for n in output.shape],'shared_buffer':int(tensor.buffer),
                'payload_sha256':hashlib.sha256(payload).hexdigest(),'payload_bytes':len(payload)})
        graph.operators=operators
    assert all(original[i]==(bytes(b.data) if b.data is not None else b'') for i,b in enumerate(model.buffers))
    if changes:
        builder=flatbuffers.Builder(0);builder.Finish(model.Pack(builder),file_identifier=b'TFL3')
        target.write_bytes(builder.Output())
    elif source!=target:target.write_bytes(raw)
    remaining=constant_reshape_paths(target)
    assert not remaining,remaining
    return {'source_sha256':hashlib.sha256(raw).hexdigest(),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'folded_reshapes':len(changes),'changes':changes,'all_original_constant_buffers_byte_identical':True,
        'remaining_constant_reshape_paths':remaining,'weight_casting_order':'Fold fp32 graph first; quantize afterward.'}
