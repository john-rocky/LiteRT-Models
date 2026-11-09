"""Run a LiteRT graph from Python with the CompiledModel API (CPU, or GPU = Metal on the Mac), inputs and outputs by
signature name, shapes read from the flatbuffer."""
import numpy as np


def sig_io(path):
    """{signature key: {'inputs': [(name, shape)], 'outputs': [(name, shape)]}} from the flatbuffer."""
    from ai_edge_litert import schema_py_generated as schema
    model = schema.ModelT.InitFromPackedBuf(open(path, 'rb').read(), 0)
    out = {}
    for sd in model.signatureDefs:
        g = model.subgraphs[sd.subgraphIndex]
        key = sd.signatureKey.decode() if isinstance(sd.signatureKey, bytes) else sd.signatureKey

        def io(lst):
            res = []
            for t in lst:
                name = t.name.decode() if isinstance(t.name, bytes) else t.name
                res.append((name, [int(s) for s in g.tensors[t.tensorIndex].shape]))
            return res
        out[key] = {'inputs': io(sd.inputs), 'outputs': io(sd.outputs)}
    return out


class Graph:
    def __init__(self, path, accel='gpu', enforce_f32=False):
        from ai_edge_litert.compiled_model import CompiledModel
        from ai_edge_litert.hardware_accelerator import HardwareAccelerator
        from ai_edge_litert.options import GpuOptions, Options
        if accel == 'gpu':
            opts = Options(hardware_accelerators=HardwareAccelerator.GPU, gpu_options=GpuOptions(enforce_f32=enforce_f32))
        else:
            opts = Options(hardware_accelerators=HardwareAccelerator.CPU)
        self.path = path
        self.m = CompiledModel.from_file(path, options=opts)
        self.io = sig_io(path)
        self.keys = list(self.io)
        self.buffers = {}

    def run(self, inputs, key=None):
        """inputs: list in signature order or {name: array}. -> {output name: float32 array}."""
        key = key or self.keys[0]
        idx = self.m.get_signature_index(key)
        sig = self.m.get_signature_by_index(idx)
        names_in = list(sig['inputs']) if isinstance(sig, dict) else [n for n, _ in self.io[key]['inputs']]
        names_out = list(sig['outputs']) if isinstance(sig, dict) else [n for n, _ in self.io[key]['outputs']]
        if key not in self.buffers:
            self.buffers[key] = (self.m.create_input_buffers(idx), self.m.create_output_buffers(idx))
        ins, outs = self.buffers[key]
        if not isinstance(inputs, dict):
            inputs = dict(zip([n for n, _ in self.io[key]['inputs']], inputs))
        shapes_in = dict(self.io[key]['inputs'])
        for name, buf in zip(names_in, ins):
            arr = np.ascontiguousarray(np.asarray(inputs[name], dtype=np.float32))
            assert arr.size == int(np.prod(shapes_in[name])), (name, arr.shape, shapes_in[name])
            buf.write(arr.reshape(-1))
        self.m.run_by_index(idx, ins, outs)
        shapes_out = dict(self.io[key]['outputs'])
        res = {}
        for j, (name, buf) in enumerate(zip(names_out, outs)):
            n = int(np.prod(shapes_out[name]))   # the Python API takes (output_index, signature_index)
            res[name] = buf.read(n, np.float32).reshape(shapes_out[name]).copy()
        return res
