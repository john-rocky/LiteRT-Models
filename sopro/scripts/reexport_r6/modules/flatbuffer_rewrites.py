"""Lossless layout canonicalization after LiteRT-Torch's rank-3 dot lowering.

This changes only tensor shapes and adds singleton RESHAPE operations around
BATCH_MATMUL. Original buffers, including every floating constant, stay byte
identical. BMM options (adjX/adjY/asymmetric quantization) are copied verbatim.
"""
import copy
import hashlib
import numpy as np


def promote_batch_matmul_rank4(source, target=None, subgraphs=None):
    import flatbuffers
    from ai_edge_litert import schema_py_generated as schema
    target = target or source
    source_bytes=source.read_bytes()
    model = schema.ModelT.InitFromPackedBuf(source_bytes, 0)
    original_buffers=[]
    external_buffers=[]
    for i,buffer in enumerate(model.buffers):
        if buffer.offset and buffer.size:
            # FLOAT_CASTING can store weights after the flatbuffer root. ModelT
            # repacking alone drops that payload, so preserve it explicitly.
            start,end=int(buffer.offset),int(buffer.offset+buffer.size)
            assert 0<=start<=end<=len(source_bytes),(i,start,end,len(source_bytes))
            data=source_bytes[start:end]
            buffer.data=np.frombuffer(data,dtype=np.uint8).copy()
            buffer.offset=buffer.size=0
            external_buffers.append(i)
        else:
            data=bytes(buffer.data) if buffer.data is not None else b''
        original_buffers.append(data)
    original_sha = hashlib.sha256(source_bytes).hexdigest()
    reshape_opcode = None
    for i,code in enumerate(model.operatorCodes):
        if max(code.builtinCode, code.deprecatedBuiltinCode) == schema.BuiltinOperator.RESHAPE:
            reshape_opcode = i
            break
    if reshape_opcode is None:
        code = schema.OperatorCodeT()
        code.builtinCode = code.deprecatedBuiltinCode = schema.BuiltinOperator.RESHAPE
        code.version = 1
        reshape_opcode = len(model.operatorCodes)
        model.operatorCodes.append(code)
    changes=[]
    for gi,graph in enumerate(model.subgraphs):
        if subgraphs is not None and gi not in subgraphs:
            continue

        def intermediate(old_index, shape, suffix):
            tensor=copy.deepcopy(graph.tensors[old_index])
            tensor.name=tensor.name + suffix.encode()
            tensor.shape=np.asarray(shape,dtype=np.int32)
            if tensor.shapeSignature is not None:
                tensor.shapeSignature=np.asarray(shape,dtype=np.int32)
            tensor.buffer=0
            index=len(graph.tensors); graph.tensors.append(tensor)
            return index

        def reshape(input_index, output_index, shape):
            buf=schema.BufferT(); buf.data=np.frombuffer(np.asarray(shape,dtype='<i4').tobytes(),dtype=np.uint8).copy()
            buffer_index=len(model.buffers); model.buffers.append(buf)
            tensor=schema.TensorT(); tensor.name=f'r6_rank4_shape_{gi}_{len(graph.tensors)}'.encode()
            tensor.shape=np.asarray([len(shape)],dtype=np.int32); tensor.type=schema.TensorType.INT32; tensor.buffer=buffer_index
            shape_index=len(graph.tensors); graph.tensors.append(tensor)
            op=schema.OperatorT(); op.opcodeIndex=reshape_opcode
            op.inputs=np.asarray([input_index,shape_index],dtype=np.int32); op.outputs=np.asarray([output_index],dtype=np.int32)
            op.builtinOptionsType=schema.BuiltinOptions.ReshapeOptions
            op.builtinOptions=schema.ReshapeOptionsT(); op.builtinOptions.newShape=np.asarray(shape,dtype=np.int32)
            return op

        new_ops=[]
        for oi,op in enumerate(graph.operators):
            code=model.operatorCodes[op.opcodeIndex]
            is_bmm=max(code.builtinCode,code.deprecatedBuiltinCode)==schema.BuiltinOperator.BATCH_MATMUL
            inputs=list(op.inputs)
            if not is_bmm or not any(i>=0 and len(graph.tensors[i].shape)==3 for i in inputs):
                new_ops.append(op); continue
            detail=dict(subgraph=gi,original_operator=oi,opcode='BATCH_MATMUL',inputs=[],outputs=[])
            op=copy.deepcopy(op)
            for j,index in enumerate(inputs):
                shape=graph.tensors[index].shape.tolist()
                assert 2<=len(shape)<=4,(gi,oi,shape)
                target_shape=[1]*(4-len(shape))+shape
                if target_shape != shape:
                    promoted=intermediate(index,target_shape,f'_r6_rank4_input_{oi}_{j}')
                    new_ops.append(reshape(index,promoted,target_shape)); op.inputs[j]=promoted
                detail['inputs'].append(dict(original_tensor=int(index),original_shape=shape,new_shape=target_shape))
            after=[]
            for j,index in enumerate(list(op.outputs)):
                shape=graph.tensors[index].shape.tolist(); target_shape=[1]*(4-len(shape))+shape
                if target_shape != shape:
                    promoted=intermediate(index,target_shape,f'_r6_rank4_output_{oi}_{j}')
                    op.outputs[j]=promoted; after.append(reshape(promoted,index,shape))
                detail['outputs'].append(dict(original_tensor=int(index),original_shape=shape,new_shape=target_shape))
            new_ops.append(op); new_ops.extend(after); changes.append(detail)
        graph.operators=new_ops
    for i,data in enumerate(original_buffers):
        assert data == (bytes(model.buffers[i].data) if model.buffers[i].data is not None else b''),i
    if changes:
        builder=flatbuffers.Builder(0); builder.Finish(model.Pack(builder),file_identifier=b'TFL3')
        target.write_bytes(builder.Output())
    elif source != target:
        target.write_bytes(source_bytes)
    return dict(source_sha256=original_sha,target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                original_buffers=len(original_buffers),original_buffer_bytes_unchanged=True,
                materialized_external_buffers=external_buffers if changes else [],
                transformed_batch_matmuls=len(changes),added_shape_constant_dtype='INT32',changes=changes)
