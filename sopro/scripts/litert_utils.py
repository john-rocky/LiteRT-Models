"""Local CompiledModel and static flatbuffer inspection helpers; no Interpreter."""
import collections
import numpy as np
from common import sha256


def inspect_flatbuffer(path):
    from ai_edge_litert import schema_py_generated as schema
    m = schema.ModelT.InitFromPackedBuf(path.read_bytes(), 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    types = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    hist, highrank, custom, io, signatures = collections.Counter(), [], [], [], []
    for gi, g in enumerate(m.subgraphs):
        def detail(index):
            t = g.tensors[index]
            return {'index': int(index), 'name': t.name.decode(), 'shape': [int(v) for v in t.shape], 'dtype': types[t.type]}
        for oi, op in enumerate(g.operators):
            code = m.operatorCodes[op.opcodeIndex]
            name = names[max(code.builtinCode, code.deprecatedBuiltinCode)]
            hist[name] += 1
            if name == 'CUSTOM':
                custom.append({'subgraph': gi, 'operator': oi, 'code': str(code.customCode)})
        for ti, t in enumerate(g.tensors):
            if t.shape is not None and len(t.shape) > 4:
                highrank.append({'subgraph': gi, **detail(ti)})
        io.append({'subgraph': gi, 'inputs': [detail(i) for i in g.inputs], 'outputs': [detail(i) for i in g.outputs]})
    for sig in m.signatureDefs or []:
        signatures.append({'key': sig.signatureKey.decode(), 'subgraph': int(sig.subgraphIndex),
            'inputs': [{'name': x.name.decode(), 'tensor_index': int(x.tensorIndex)} for x in sig.inputs],
            'outputs': [{'name': x.name.decode(), 'tensor_index': int(x.tensorIndex)} for x in sig.outputs]})
    return {'sha256': sha256(path), 'size_bytes': path.stat().st_size, 'op_histogram': dict(hist),
            'custom_ops': custom, 'tensors_rank_gt4': highrank, 'io': io, 'signatures': signatures,
            'max_tensor_rank': max(len(t.shape) for g in m.subgraphs for t in g.tensors if t.shape is not None),
            'pass': not custom and not highrank}


class CM:
    def __init__(self, path, threads=4, environment=None):
        from ai_edge_litert.compiled_model import CompiledModel, Options, CpuOptions, HardwareAccelerator
        self.m = CompiledModel.from_file(str(path), environment=environment, options=Options(
            hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads)))
        self.sig = self.m.get_signature_by_index(0)
        self.ins, self.outs = self.m.create_input_buffers(0), self.m.create_output_buffers(0)
        info = inspect_flatbuffer(path)
        sig = info['signatures'][0]
        tensors = info['io'][sig['subgraph']]
        ins = {x['index']: x for x in tensors['inputs']}
        outs = {x['index']: x for x in tensors['outputs']}
        # Buffer indexes are signature order, not subgraph tensor-index order.
        self.inputs = [{**ins[x['tensor_index']], 'signature_name': x['name']} for x in sig['inputs']]
        self.outputs = [{**outs[x['tensor_index']], 'signature_name': x['name']} for x in sig['outputs']]
        print('COMPILEDMODEL_SIGNATURE', self.sig, 'I/O', self.inputs, self.outputs, flush=True)

    @staticmethod
    def _dtype(spec):
        return {'FLOAT32': np.float32, 'FLOAT64': np.float64, 'INT32': np.int32, 'INT64': np.int64, 'BOOL': np.bool_}[spec['dtype']]

    def call_named(self, arrays):
        """Use actual signature names (args_N); return actual output_N names.

        Explicit names disambiguate equally shaped x/mu/prompt or K/V tensors.
        Buffer order is checked against CompiledModel's signature, not guessed
        from graph tensor index. Integer I/O retains its declared type.
        """
        expected = {x['signature_name'] for x in self.inputs}
        assert set(arrays) == expected, (set(arrays), expected)
        for buf, spec in zip(self.ins, self.inputs):
            a = np.asarray(arrays[spec['signature_name']])
            assert list(a.shape) == spec['shape'], (spec, a.shape)
            buf.write(np.ascontiguousarray(a, dtype=self._dtype(spec)).ravel())
        self.m.run_by_index(0, self.ins, self.outs)
        return {spec['signature_name']: np.array(buf.read(int(np.prod(spec['shape'])), self._dtype(spec))).reshape(spec['shape'])
                for buf, spec in zip(self.outs, self.outputs)}

    def call_ordered(self, *arrays):
        """Inputs in export example order; outputs in numbered export order."""
        names = {x['signature_name'] for x in self.inputs}
        assert names == {f'args_{i}' for i in range(len(arrays))}, names
        out = self.call_named({f'args_{i}': a for i, a in enumerate(arrays)})
        assert set(out) == {f'output_{i}' for i in range(len(out))}, set(out)
        return [out[f'output_{i}'] for i in range(len(out))]

    def __call__(self, *arrays):
        # Round-one graph inputs and outputs have unique shapes. Explicit shape
        # mapping also catches accidental signature reordering across exports.
        unused = list(arrays)
        for buf, spec in zip(self.ins, self.inputs):
            matches = [i for i, a in enumerate(unused) if list(a.shape) == spec['shape']]
            assert len(matches) == 1, (spec, [a.shape for a in unused])
            a = unused.pop(matches[0])
            buf.write(np.ascontiguousarray(a, dtype=self._dtype(spec)).ravel())
        assert not unused
        self.m.run_by_index(0, self.ins, self.outs)
        return [np.array(buf.read(int(np.prod(spec['shape'])), self._dtype(spec))).reshape(spec['shape']) for buf, spec in zip(self.outs, self.outputs)]

    def close(self):
        # 2.2.0 calls explicit TensorBuffer release destroy(), not close().
        for b in self.ins + self.outs:
            b.destroy()
        self.ins.clear()
        self.outs.clear()
        self.m.close()


class MergedCM:
    """One CompiledModel instance, two explicitly named signature buffer sets."""
    def __init__(self,path,threads=4,environment=None):
        from ai_edge_litert.compiled_model import CompiledModel,Options,CpuOptions,HardwareAccelerator
        self.model=CompiledModel.from_file(str(path),environment=environment,options=Options(
            hardware_accelerators=HardwareAccelerator.CPU,cpu_options=CpuOptions(num_threads=threads)))
        inspection=inspect_flatbuffer(path)
        self.signatures={}
        for sig in inspection['signatures']:
            index=self.model.get_signature_index(sig['key'])
            assert index>=0
            graph=inspection['io'][sig['subgraph']]
            specs={}
            for direction in ('inputs','outputs'):
                tensors={x['index']:x for x in graph[direction]}
                specs[direction]=[{**tensors[x['tensor_index']],'signature_name':x['name']} for x in sig[direction]]
            self.signatures[sig['key']]={'index':index,**specs,
                'input_buffers':self.model.create_input_buffers(index),
                'output_buffers':self.model.create_output_buffers(index)}
        assert set(self.signatures)=={'prefill','step'}
        print('MERGED_SIGNATURES',self.model.get_signature_list(),flush=True)

    def call(self,key,*arrays):
        sig=self.signatures[key]
        byname={f'args_{i}':a for i,a in enumerate(arrays)}
        assert set(byname)=={s['signature_name'] for s in sig['inputs']}
        dtypes={'FLOAT32':np.float32,'INT32':np.int32}
        for buffer,spec in zip(sig['input_buffers'],sig['inputs']):
            value=byname[spec['signature_name']]
            assert list(value.shape)==spec['shape']
            buffer.write(np.ascontiguousarray(value,dtype=dtypes[spec['dtype']]).ravel())
        self.model.run_by_index(sig['index'],sig['input_buffers'],sig['output_buffers'])
        outputs={spec['signature_name']:np.array(buffer.read(int(np.prod(spec['shape'])),dtypes[spec['dtype']])).reshape(spec['shape'])
                 for buffer,spec in zip(sig['output_buffers'],sig['outputs'])}
        return [outputs[f'output_{i}'] for i in range(len(outputs))]

    def close(self):
        for sig in self.signatures.values():
            for buffer in sig['input_buffers']+sig['output_buffers']:
                buffer.destroy()
        self.signatures.clear()
        self.model.close()
