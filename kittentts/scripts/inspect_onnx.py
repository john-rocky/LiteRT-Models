"""Inspect KittenTTS ONNX graphs: inputs/outputs, dynamic axes, op histogram."""
import sys
from collections import Counter

import onnx


def describe(path: str) -> None:
    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    print(f"=== {path} ===")
    print(f"ir_version={model.ir_version} opset={[o.version for o in model.opset_import]}")

    def shape_of(vi):
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            dims.append(d.dim_param if d.dim_param else d.dim_value)
        return dims

    print("-- inputs --")
    for vi in graph.input:
        print(f"  {vi.name}: {onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)} {shape_of(vi)}")
    print("-- outputs --")
    for vi in graph.output:
        print(f"  {vi.name}: {onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)} {shape_of(vi)}")

    ops = Counter(n.op_type for n in graph.node)
    print(f"-- {len(graph.node)} nodes, op histogram --")
    for op, c in ops.most_common():
        print(f"  {op:24s} {c}")

    # LSTM / recurrent details
    for n in graph.node:
        if n.op_type in ("LSTM", "GRU", "RNN", "Scan", "Loop", "If"):
            print(f"  [recurrent/control] {n.op_type} name={n.name} inputs={list(n.input)[:3]}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        describe(p)
