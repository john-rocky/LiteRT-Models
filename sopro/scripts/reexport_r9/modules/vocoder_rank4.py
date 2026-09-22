"""Promote every rank-three streaming-vocoder tensor to a leading-one rank four.

Floating buffers and operation order stay byte-identical. Only tensor shape
metadata, axis options and integer shape/axis/padding constants change. Conv2D
layouts remain unchanged. Reduction axes also include the new singleton axis
when keep_dims=False, retaining the original lower-rank reduction outputs.
"""
import copy
import hashlib
import numpy as np


def promote(source, target):
    import flatbuffers
    from ai_edge_litert import schema_py_generated as s
    raw=source.read_bytes();model=s.ModelT.InitFromPackedBuf(raw,0)
    names={v:k for k,v in vars(s.BuiltinOperator).items() if isinstance(v,int)}
    original_buffers=[]
    for buffer in model.buffers:
        if buffer.offset and buffer.size:
            data=raw[int(buffer.offset):int(buffer.offset+buffer.size)]
            assert len(data)==buffer.size
            buffer.data=np.frombuffer(data,dtype=np.uint8).copy();buffer.offset=buffer.size=0
        original_buffers.append(bytes(buffer.data) if buffer.data is not None else b'')
    changes=[];promoted=[]
    for gi,g in enumerate(model.subgraphs):
        rank3={i for i,t in enumerate(g.tensors) if t.shape is not None and len(t.shape)==3}
        old_shapes={i:g.tensors[i].shape.tolist() for i in rank3}
        for i in rank3:
            t=g.tensors[i];t.shape=np.asarray([1,*old_shapes[i]],dtype=np.int32)
            if t.shapeSignature is not None:t.shapeSignature=t.shape.copy()
            promoted.append(dict(subgraph=gi,tensor=i,name=t.name.decode(),before=old_shapes[i],after=t.shape.tolist()))

        def integer(index):
            t=g.tensors[index];assert t.type==s.TensorType.INT32,(index,t.type)
            buffer=model.buffers[t.buffer];assert buffer.data is not None
            return np.frombuffer(buffer.data,dtype='<i4').reshape(t.shape).copy()

        def constant_like(index,value):
            value=np.asarray(value,dtype='<i4')
            buffer=s.BufferT();buffer.data=np.frombuffer(value.tobytes(),dtype=np.uint8).copy()
            tensor=copy.deepcopy(g.tensors[index]);tensor.name=f'rank4_shape_{gi}_{len(g.tensors)}'.encode()
            tensor.shape=np.asarray(value.shape,dtype=np.int32);tensor.buffer=len(model.buffers)
            if tensor.shapeSignature is not None:tensor.shapeSignature=tensor.shape.copy()
            model.buffers.append(buffer);result=len(g.tensors);g.tensors.append(tensor);return result

        for oi,op in enumerate(g.operators):
            op.inputs=op.inputs.copy()
            code=model.operatorCodes[op.opcodeIndex];name=names[max(code.builtinCode,code.deprecatedBuiltinCode)]
            in3=bool(op.inputs.size and int(op.inputs[0]) in rank3)
            out3=any(int(i) in rank3 for i in op.outputs)
            detail=dict(subgraph=gi,operator=oi,name=name,changes=[])

            def replace_input(position,new_value):
                old=int(op.inputs[position]);before=integer(old).tolist()
                op.inputs[position]=constant_like(old,new_value)
                detail['changes'].append(dict(input=position,before=before,after=np.asarray(new_value).tolist()))

            if name=='RESHAPE' and out3:
                shape=g.tensors[int(op.outputs[0])].shape.tolist()
                if len(op.inputs)>1:replace_input(1,shape)
                if op.builtinOptions is not None:op.builtinOptions.newShape=np.asarray(shape,dtype=np.int32)
            elif name=='TRANSPOSE' and in3:
                perm=integer(int(op.inputs[1]));replace_input(1,np.concatenate(([0],perm+1)))
            elif name=='SLICE' and in3:
                replace_input(1,np.concatenate(([0],integer(int(op.inputs[1])))))
                replace_input(2,np.concatenate(([1],integer(int(op.inputs[2])))))
            elif name in ('PAD','PADV2','MIRROR_PAD') and in3:
                replace_input(1,np.concatenate((np.zeros((1,2),dtype=np.int32),integer(int(op.inputs[1]))),axis=0))
            elif name=='CONCATENATION' and in3:
                axis=int(op.builtinOptions.axis);normalized=axis if axis>=0 else axis+3
                op.builtinOptions.axis=normalized+1
                detail['changes'].append(dict(option='axis',before=axis,after=normalized+1))
                assert all(int(i) in rank3 for i in op.inputs)
            elif name in ('MEAN','SUM','REDUCE_MAX','REDUCE_MIN','REDUCE_PROD','REDUCE_ANY') and in3:
                axes=integer(int(op.inputs[1]));axes=np.where(axes<0,axes+3,axes)+1
                if not op.builtinOptions.keepDims:axes=np.concatenate(([0],axes.ravel()))
                replace_input(1,axes)
            elif in3 or out3:
                assert name in ('ADD','SUB','MUL','DIV','MAXIMUM','MINIMUM','SQUARED_DIFFERENCE','GELU',
                    'FULLY_CONNECTED','RESHAPE','RSQRT','SQRT','EXP','LOG','TANH','LOGISTIC','ABS',
                    'DEQUANTIZE','QUANTIZE','CAST'),(gi,oi,name)
            if detail['changes']:changes.append(detail)
    for i,original in enumerate(original_buffers):
        assert original==(bytes(model.buffers[i].data) if model.buffers[i].data is not None else b''),i
    assert not any(t.shape is not None and len(t.shape)==3 for g in model.subgraphs for t in g.tensors)
    builder=flatbuffers.Builder(0);builder.Finish(model.Pack(builder),file_identifier=b'TFL3');target.write_bytes(builder.Output())
    return dict(source_sha256=hashlib.sha256(raw).hexdigest(),target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        original_buffer_count=len(original_buffers),all_original_buffer_bytes_unchanged=True,
        promoted_tensors=promoted,operator_changes=changes,rank3_tensor_count_after=0)
