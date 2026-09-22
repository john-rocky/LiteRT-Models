"""Inspect every operator and tensor in an exact static export."""
import collections
from pathlib import Path
from ai_edge_litert import schema_py_generated as schema
from common import sha256
from litert_utils import inspect_flatbuffer


def scan(path):
    path=Path(path);model=schema.ModelT.InitFromPackedBuf(path.read_bytes(),0)
    names={v:k for k,v in vars(schema.BuiltinOperator).items() if isinstance(v,int)}
    types={v:k for k,v in vars(schema.TensorType).items() if isinstance(v,int)}
    report=inspect_flatbuffer(path);tensors=[];operators=[];forbidden=[]
    for gi,graph in enumerate(model.subgraphs):
        for ti,tensor in enumerate(graph.tensors):
            row={'subgraph':gi,'tensor':ti,'name':tensor.name.decode() if tensor.name else '',
                 'shape':None if tensor.shape is None else tensor.shape.tolist(),'dtype':types[tensor.type]}
            tensors.append(row)
            if tensor.type==schema.TensorType.INT64:forbidden.append({'reason':'INT64 tensor',**row})
        for oi,op in enumerate(graph.operators):
            code=model.operatorCodes[op.opcodeIndex];name=names[max(code.builtinCode,code.deprecatedBuiltinCode)]
            shapes=[graph.tensors[int(i)].shape.tolist() for i in op.inputs if i>=0]
            row={'subgraph':gi,'operator':oi,'name':name,'inputs':op.inputs.tolist(),'outputs':op.outputs.tolist(),'input_shapes':shapes}
            operators.append(row)
            if name in ('GATHER_ND','BROADCAST_TO','CUSTOM') or name=='BATCH_MATMUL' and any(len(s)==3 for s in shapes):
                forbidden.append({'reason':'Unsupported exact-export structure',**row})
    report.update(tensors=tensors,operators=operators,forbidden=forbidden,
                  tensor_dtype_histogram=dict(collections.Counter(t['dtype'] for t in tensors)),
                  sha256=sha256(path),pass_=bool(report['pass'] and not forbidden))
    return report
