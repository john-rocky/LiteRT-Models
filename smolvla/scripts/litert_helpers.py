"""Small LiteRT helpers vendored for the SmolVLA scripts (no repo-local imports).

- run a .tflite through the LiteRT CompiledModel Python API (CPU),
- scan a .tflite flatbuffer for its op distribution / max tensor rank / banned ops,
- cast float32 weights to float16 with ai_edge_quantizer (FLOAT_CASTING recipe),
- parity metrics.
"""

import collections
import json
import os

import numpy as np

# Ops that the ML Drift GPU path rejects or that this export must not contain.
BANNED_OPS = {
    "GATHER_ND", "GATHER", "SELECT", "SELECT_V2", "NOT_EQUAL", "EQUAL", "GREATER",
    "LESS", "TOPK_V2", "CAST", "PACK", "SPLIT", "BROADCAST_TO", "CUMSUM",
    "TRANSPOSE_CONV", "EMBEDDING_LOOKUP", "CUSTOM", "FLEX", "WHERE",
}


# ----------------------------------------------------------------- metrics
def corr(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def max_abs_diff(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.abs(a - b).max())


def norm_ratio(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(np.linalg.norm(a) / np.linalg.norm(b))


def parity(a, b):
    """corr, max|a-b|, ||a||/||b|| and max|b| (b is the reference)."""
    return {
        "corr": corr(a, b),
        "max_abs_diff": max_abs_diff(a, b),
        "norm_ratio": norm_ratio(a, b),
        "ref_max_abs": float(np.abs(np.asarray(b, np.float64)).max()),
    }


def fmt_parity(p):
    return (f"corr={p['corr']:.9f} max|diff|={p['max_abs_diff']:.3e} "
            f"norm_ratio={p['norm_ratio']:.9f} (ref max|x|={p['ref_max_abs']:.4g})")


# ------------------------------------------------------ CompiledModel (CPU)
_SHARED_ENV = None


def shared_environment():
    """One LiteRT Environment per process, passed to every CompiledModel."""
    global _SHARED_ENV
    if _SHARED_ENV is None:
        import ai_edge_litert.compiled_model as cm
        _SHARED_ENV = cm.Environment.create(options=cm.EnvironmentOptions(
            runtime_path=os.path.dirname(os.path.abspath(cm.__file__))))
    return _SHARED_ENV


class TFLiteRunner:
    """Runs signature 0 of a .tflite with the LiteRT CompiledModel API.

    accelerator="cpu" (XNNPACK, num_threads) or "gpu" (the LiteRT GPU accelerator of
    this host: Metal on macOS); gpu_fp32=True sets GpuOptions(enforce_f32=True),
    otherwise the GPU runs at its default (fp16) precision.
    """

    def __init__(self, path, num_threads=8, accelerator="cpu", gpu_fp32=False):
        from ai_edge_litert.compiled_model import CompiledModel
        from ai_edge_litert.compiled_model import CpuOptions
        from ai_edge_litert.compiled_model import GpuOptions
        from ai_edge_litert.compiled_model import HardwareAccelerator
        from ai_edge_litert.compiled_model import Options

        self.path = path
        if accelerator == "gpu":
            opts = Options(hardware_accelerators=HardwareAccelerator.GPU,
                           gpu_options=GpuOptions(enforce_f32=gpu_fp32))
        else:
            opts = Options(hardware_accelerators=HardwareAccelerator.CPU,
                           cpu_options=CpuOptions(num_threads=num_threads))
        self.model = CompiledModel.from_file(path, options=opts, environment=shared_environment())
        self.fully_accelerated = self.model.is_fully_accelerated()
        sig = self.model.get_signature_by_index(0)
        self.signature = sig
        self.input_names = list(sig["inputs"])
        self.output_names = list(sig["outputs"])
        self.ins = self.model.create_input_buffers(0)
        self.outs = self.model.create_output_buffers(0)
        # get_output_buffer_requirements(output_index, signature_index)
        self.out_sizes = [
            self.model.get_output_buffer_requirements(j, 0)["buffer_size"] // 4
            for j in range(len(self.outs))
        ]

    def run(self, inputs):
        """inputs: list of float32 arrays in signature input order.

        Returns the list of flat float32 outputs in signature output order.
        """
        assert len(inputs) == len(self.ins), (len(inputs), len(self.ins))
        for buf, x in zip(self.ins, inputs):
            buf.write(np.ascontiguousarray(x, np.float32))
        self.model.run_by_index(0, self.ins, self.outs)
        return [buf.read(n, np.float32) for buf, n in zip(self.outs, self.out_sizes)]

    def run_shaped(self, inputs, out_shapes):
        """Like run(), but selects outputs by element count and reshapes them.

        Outputs with distinct sizes are matched by size whatever order out_shapes
        uses; outputs that share a size (prefix k_all / v_all) are taken in
        signature output order (output_0, output_1, ... = the graph's return
        order), so list same-size shapes in return order.
        """
        flat = self.run(inputs)
        used = set()
        res = []
        for shape in out_shapes:
            n = int(np.prod(shape))
            j = next((j for j, f in enumerate(flat) if f.size == n and j not in used), None)
            assert j is not None, f"no output left with shape {shape}"
            used.add(j)
            res.append(flat[j].reshape(shape))
        return res


# -------------------------------------------------------- flatbuffer scan
def _load_model_t(path):
    from ai_edge_litert import schema_py_generated as schema
    with open(path, "rb") as f:
        buf = f.read()
    return schema, schema.ModelT.InitFromPackedBuf(buf, 0)


def op_scan(path):
    """Op histogram, max tensor rank, banned ops and GELU options of subgraph 0."""
    schema, model = _load_model_t(path)
    name_of = {v: k for k, v in schema.BuiltinOperator.__dict__.items()
               if not k.startswith("_")}
    names = []
    for code in model.operatorCodes:
        op = max(code.builtinCode, code.deprecatedBuiltinCode)
        name = name_of.get(op, str(op))
        if name == "CUSTOM":
            cc = code.customCode
            cc = cc.decode() if isinstance(cc, bytes) else cc
            name = f"CUSTOM:{cc}"
        names.append(name)
    hist = collections.Counter()
    gelu_approx = collections.Counter()
    graph = model.subgraphs[0]
    for op in graph.operators:
        n = names[op.opcodeIndex]
        hist[n] += 1
        if n == "GELU":
            opts = op.builtinOptions
            gelu_approx[bool(getattr(opts, "approximate", False))] += 1
    ranks = [len(t.shape) if t.shape is not None else 0 for t in graph.tensors]
    max_rank = max(ranks) if ranks else 0
    over4 = sum(1 for r in ranks if r > 4)
    banned = {}
    for n, c in hist.items():
        base = n.split(":")[0]
        if base in BANNED_OPS or "FLEX" in n.upper():
            banned[n] = c
    io = {
        "inputs": [(_tname(graph.tensors[i]), [int(d) for d in graph.tensors[i].shape])
                   for i in graph.inputs],
        "outputs": [(_tname(graph.tensors[i]), [int(d) for d in graph.tensors[i].shape])
                    for i in graph.outputs],
    }
    return {
        "structural": structural_checks(path),
        "ops": dict(sorted(hist.items(), key=lambda kv: -kv[1])),
        "num_ops": int(sum(hist.values())),
        "max_rank": int(max_rank),
        "tensors_over_4d": int(over4),
        "banned": banned,
        "gelu_approximate": {str(k): v for k, v in gelu_approx.items()},
        "io": io,
        "size_mb": os.path.getsize(path) / 1e6,
    }


WEIGHT_CONSUMERS = {"FULLY_CONNECTED", "CONV_2D", "DEPTHWISE_CONV_2D", "BATCH_MATMUL",
                    "DEQUANTIZE", "RESHAPE", "TRANSPOSE", "SLICE", "STRIDED_SLICE", "SUM",
                    "MEAN", "PAD", "CONCATENATION"}


def _bhwc(shape):
    """TFLite GPU-delegate style BHWC view of a tensor shape of rank <= 4."""
    s = list(shape)
    if len(s) == 0:
        return (1, 1, 1, 1)
    if len(s) == 1:
        return (1, 1, 1, s[0])
    if len(s) == 2:
        return (s[0], 1, 1, s[1])
    if len(s) == 3:
        return (s[0], 1, s[1], s[2])
    return tuple(s)


def _bhwc_broadcastable(shape, out_shape):
    a, o = _bhwc(shape), _bhwc(out_shape)
    return all(x == y or x == 1 for x, y in zip(a, o))


def structural_checks(path):
    """Static checks for ML Drift on-device traps that a name blocklist misses.

    const_only_ops: ops whose every input is a constant (rejected on device; the
      DEQUANTIZE of fp16 weights is the standard weight pattern and listed apart).
    multi_axis_reductions: SUM/MEAN/REDUCE_* over more than one axis.
    outputs_with_consumers: graph outputs that are also consumed inside the graph.
    large_const_operands: constants with > 4096 elements feeding an op that is not
      a weight consumer (e.g. a baked table in an ADD/MUL).
    fc_output_ranks: histogram of FULLY_CONNECTED output ranks.
    mixed_rank_elementwise: binary elementwise ops whose runtime operands differ
      in rank (e.g. an FC output flattened to [N,C] added to a [1,N,C] tensor),
      split into same-size and broadcast cases.
    batch_matmul_forms: operand ranks and adj_x/adj_y of every BATCH_MATMUL.
    """
    schema, model = _load_model_t(path)
    name_of = {v: k for k, v in schema.BuiltinOperator.__dict__.items() if not k.startswith("_")}
    opnames = [name_of.get(max(c.builtinCode, c.deprecatedBuiltinCode), "?") for c in model.operatorCodes]
    g = model.subgraphs[0]

    def is_const(ti):
        b = model.buffers[g.tensors[ti].buffer]
        has_data = b.data is not None and len(b.data) > 0
        return has_data or (b.offset is not None and b.offset > 1 and b.size is not None and b.size > 0)

    consumers = collections.Counter()
    const_only, deq_const, multi_axis, big_const = [], 0, [], collections.Counter()
    fc_ranks = collections.Counter()
    mixed = collections.Counter()
    bmm_forms = collections.Counter()
    for oi, op in enumerate(g.operators):
        name = opnames[op.opcodeIndex]
        ins = [i for i in (op.inputs if op.inputs is not None else []) if i >= 0]
        for i in ins:
            consumers[i] += 1
        if ins and all(is_const(i) for i in ins):
            if name == "DEQUANTIZE":
                deq_const += 1
            else:
                const_only.append((oi, name))
        if name in ("SUM", "MEAN", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_PROD") and len(ins) > 1:
            axes = int(np.prod(g.tensors[ins[1]].shape)) if g.tensors[ins[1]].shape is not None else 1
            if axes > 1:
                multi_axis.append((oi, name, axes))
        if name not in WEIGHT_CONSUMERS:
            for i in ins:
                shp = g.tensors[i].shape
                if is_const(i) and shp is not None and int(np.prod(shp)) > 4096:
                    big_const[name] += 1
        if name == "FULLY_CONNECTED":
            for o in op.outputs:
                fc_ranks[len(g.tensors[o].shape)] += 1
        if name in ("ADD", "MUL", "SUB", "DIV", "SQUARED_DIFFERENCE", "MAXIMUM", "MINIMUM"):
            rt = [i for i in ins if not is_const(i)]
            if len({len(g.tensors[i].shape) for i in rt}) > 1:
                shapes = tuple(tuple(int(d) for d in g.tensors[i].shape) for i in rt)
                out_shape = tuple(int(d) for d in g.tensors[op.outputs[0]].shape)
                if len({int(np.prod(x)) for x in shapes}) == 1:
                    kind = "same-size"
                elif all(_bhwc_broadcastable(x, out_shape) for x in shapes):
                    kind = "broadcast, BHWC-consistent"
                else:
                    kind = "broadcast, BHWC-CONFLICT"
                mixed[f"{name} {list(map(list, shapes))} {kind}"] += 1
        if name == "BATCH_MATMUL":
            o = op.builtinOptions
            ranks = "x".join(str(len(g.tensors[i].shape)) for i in ins)
            bmm_forms[f"rank {ranks} adj_x={bool(o.adjX)} adj_y={bool(o.adjY)}"] += 1
    out_consumed = [(_tname(g.tensors[o]), int(consumers[o])) for o in g.outputs if consumers[o]]
    return {
        "const_only_ops": const_only,
        "dequantize_of_constant_weights": deq_const,
        "multi_axis_reductions": multi_axis,
        "outputs_with_consumers": out_consumed,
        "large_const_operands": dict(big_const),
        "fc_output_ranks": {str(k): v for k, v in sorted(fc_ranks.items())},
        "mixed_rank_elementwise": dict(mixed),
        "batch_matmul_forms": dict(bmm_forms),
    }


def _tname(t):
    n = t.name
    return n.decode() if isinstance(n, bytes) else n


def fmt_op_scan(s):
    lines = [f"  size {s['size_mb']:.1f} MB, {s['num_ops']} ops, max rank {s['max_rank']}, "
             f">4D tensors {s['tensors_over_4d']}, banned {s['banned'] or 'NONE'}",
             f"  ops: {s['ops']}"]
    if s["gelu_approximate"]:
        lines.append(f"  GELU approximate flag counts: {s['gelu_approximate']}")
    st = s["structural"]
    ma = st["multi_axis_reductions"]
    ma_txt = f"{len(ma)} (e.g. op {ma[0][0]} {ma[0][1]} over {ma[0][2]} axes)" if ma else "NONE"
    lines.append(f"  const-only ops: {st['const_only_ops'] or 'NONE'}; DEQUANTIZE(const weight): "
                 f"{st['dequantize_of_constant_weights']}; multi-axis reductions: "
                 f"{ma_txt}; outputs also consumed: "
                 f"{st['outputs_with_consumers'] or 'NONE'}; large const operands: "
                 f"{st['large_const_operands'] or 'NONE'}; FC output ranks: {st['fc_output_ranks']}")
    lines.append(f"  BATCH_MATMUL forms: {st['batch_matmul_forms'] or 'NONE'}; mixed-rank "
                 f"elementwise ops (runtime operands): {st['mixed_rank_elementwise'] or 'NONE'}")
    return "\n".join(lines)


# ----------------------------------------------------------- fp16 weights
def to_fp16(fp32_path, fp16_path):
    """Store float32 weights as float16 (ai_edge_quantizer FLOAT_CASTING)."""
    from ai_edge_quantizer import quantizer
    from ai_edge_quantizer import recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName
    from ai_edge_quantizer.recipe import qtyping

    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT),
        algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16_path):
        os.remove(fp16_path)
    qt = quantizer.Quantizer(float_model=fp32_path)
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(fp16_path)
    return fp16_path


# --------------------------------------------------------------- fixtures
class FixtureWriter:
    """Raw little-endian float32 (or int32) files + a shapes.json index.

    Layout: <root>/<group>/<file>.bin; shapes.json maps group -> list of entries
    {"file", "name", "role" ("input"/"output"/"meta"), "index", "shape", "dtype"}.
    Graph inputs are written with role "input" and index = signature input order.
    """

    def __init__(self, root):
        self.root = root
        self.index = {}
        os.makedirs(root, exist_ok=True)

    def add(self, group, name, array, role="meta", index=None, dtype="float32", note=None):
        arr = np.ascontiguousarray(np.asarray(array).astype("<f4" if dtype == "float32" else "<i4"))
        prefix = ""
        if role == "input":
            prefix = f"in_{index:02d}_"
        elif role == "output":
            prefix = f"out_{index:02d}_"
        rel = os.path.join(group, f"{prefix}{name}.bin")
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arr.tofile(path)
        entry = {"file": rel, "name": name, "role": role, "shape": list(arr.shape), "dtype": dtype}
        if index is not None:
            entry["index"] = index
        if note:
            entry["note"] = note
        self.index.setdefault(group, []).append(entry)
        return rel

    def write_index(self, extra=None):
        doc = {"format": "raw little-endian, C order; float32 unless dtype says int32",
               "groups": self.index}
        if extra:
            doc.update(extra)
        with open(os.path.join(self.root, "shapes.json"), "w") as f:
            json.dump(doc, f, indent=1)


def load_fixture_group(root, group):
    """Returns {name: array} for every entry of group plus the ordered input list."""
    with open(os.path.join(root, "shapes.json")) as f:
        doc = json.load(f)
    arrays, inputs = {}, []
    for e in doc["groups"][group]:
        dt = "<f4" if e["dtype"] == "float32" else "<i4"
        a = np.fromfile(os.path.join(root, e["file"]), dtype=dt).reshape(e["shape"])
        arrays[e["name"]] = a
        if e["role"] == "input":
            inputs.append((e["index"], a))
    inputs = [a for _, a in sorted(inputs, key=lambda t: t[0])]
    return arrays, inputs
