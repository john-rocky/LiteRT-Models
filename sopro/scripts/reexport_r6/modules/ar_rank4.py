"""Exact AR attention rank normalization with an unchanged step operator histogram."""
import collections
import copy
import numpy as np
from common import sha256

def promote_existing_reshapes(path):
    """Restore singleton rank discarded by conversion without adding operators.

    Every affected BMM input already has a reshape producer. Its new shape is
    simply [1] + its old shape. BMM output and pointwise intermediates acquire
    the same singleton until an existing rank-four tensor or reshape boundary.
    No activation or learned buffer is changed; all original buffer bytes stay.
    """
    import flatbuffers
    from flatbuffer_rewrites import promote_batch_matmul_rank4
    from ai_edge_litert import schema_py_generated as schema
    prefill_promotion=promote_batch_matmul_rank4(path,subgraphs=[0])
    model=schema.ModelT.InitFromPackedBuf(path.read_bytes(),0)
    opnames={v:k for k,v in vars(schema.BuiltinOperator).items() if isinstance(v,int)}
    name=lambda op:opnames[max(model.operatorCodes[op.opcodeIndex].builtinCode,model.operatorCodes[op.opcodeIndex].deprecatedBuiltinCode)]
    old_buffers=[None if b.data is None else bytes(b.data) for b in model.buffers]
    before=sha256(path);changes=[]
    for gi,g in enumerate(model.subgraphs):
        if gi==0:continue
        producer={int(t):op for op in g.operators for t in op.outputs}
        consumers=collections.defaultdict(list)
        for op in g.operators:
            for ti in op.inputs:
                if ti>=0:consumers[int(ti)].append(op)
        def promote_tensor(ti):
            t=g.tensors[int(ti)]
            assert len(t.shape)==3,(gi,ti,t.shape)
            t.shape=np.array([1,*t.shape],np.int32)
            if t.shapeSignature is not None:t.shapeSignature=np.array([1,*t.shapeSignature],np.int32)
        def propagate(ti):
            for consumer in consumers[int(ti)]:
                kind=name(consumer)
                if kind=='RESHAPE':continue
                assert kind in ('MUL','ADD','SUB','SOFTMAX','BATCH_MATMUL'),(gi,ti,kind)
                for oi in consumer.outputs:
                    if len(g.tensors[int(oi)].shape)==3:
                        promote_tensor(int(oi));propagate(int(oi))
        for oi,op in enumerate(g.operators):
            if name(op)!='BATCH_MATMUL':continue
            if all(len(g.tensors[int(t)].shape)==4 for t in op.inputs):continue
            row={'subgraph':gi,'operator':oi,'inputs_before':[list(map(int,g.tensors[int(t)].shape)) for t in op.inputs]}
            for ti in op.inputs:
                ti=int(ti);t=g.tensors[ti]
                if len(t.shape)==4:continue
                assert len(t.shape)==3,(gi,oi,list(t.shape))
                p=producer[ti];assert name(p)=='RESHAPE',(gi,oi,name(p))
                promote_tensor(ti)
                shape=np.array(t.shape,np.int32)
                if len(p.inputs)==2:
                    original=g.tensors[int(p.inputs[1])]
                    fresh=copy.deepcopy(original);fresh.name=(original.name or b'shape')+b'_rank4_'+str(ti).encode()
                    fresh.shape=np.array([4],np.int32)
                    if fresh.shapeSignature is not None:fresh.shapeSignature=np.array([4],np.int32)
                    buffer=schema.BufferT();buffer.data=np.frombuffer(shape.tobytes(),dtype=np.uint8).copy()
                    fresh.buffer=len(model.buffers);model.buffers.append(buffer)
                    p.inputs=np.array(p.inputs,dtype=np.int32,copy=True)
                    p.inputs[1]=len(g.tensors);g.tensors.append(fresh)
                if p.builtinOptions is not None:p.builtinOptions.newShape=shape
            for ti in op.outputs:
                ti=int(ti)
                if len(g.tensors[ti].shape)==3:promote_tensor(ti);propagate(ti)
            changes.append(row)
    assert all((None if b.data is None else bytes(b.data))==old for b,old in zip(model.buffers,old_buffers))
    builder=flatbuffers.Builder(0);builder.Finish(model.Pack(builder),file_identifier=b'TFL3')
    tmp=path.with_suffix('.rank4.tmp');tmp.write_bytes(builder.Output());tmp.replace(path)
    report={'input_sha256':prefill_promotion['source_sha256'],'output_sha256':sha256(path),
        'prefill_promotion':prefill_promotion,'step_bmm_promotions':len(changes),'changes':changes,
        'step_added_operator_count':0,'all_original_buffers_byte_identical':True,
        'proof':'Only existing reshape shapes and singleton tensor metadata change; all BMM options and buffer bytes retained.'}
    return report
