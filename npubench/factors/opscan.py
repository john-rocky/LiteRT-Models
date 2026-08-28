"""Static op-composition scan of tflite files: what distinguishes NPU-losers from NPU-winners."""
import sys, json, math
from collections import Counter, defaultdict
from ai_edge_litert import schema_py_generated as S

BUILTIN = {v: k for k, v in vars(S.BuiltinOperator).items() if isinstance(v, int)}

def tensor_shape(sg, idx):
    if idx < 0:
        return None
    t = sg.Tensors(idx)
    if t.ShapeLength() == 0:
        return []
    return [t.Shape(j) for j in range(t.ShapeLength())]

def is_const(model, sg, idx):
    if idx < 0:
        return False
    t = sg.Tensors(idx)
    b = model.Buffers(t.Buffer())
    return b.DataLength() > 0

def scan(path):
    buf = open(path, 'rb').read()
    model = S.Model.GetRootAsModel(buf, 0)
    opcodes = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
        name = BUILTIN.get(code, f'UNK{code}')
        if name == 'CUSTOM':
            name = 'CUSTOM:' + (oc.CustomCode().decode() if oc.CustomCode() else '?')
        opcodes.append(name)

    hist = Counter()
    rank_hist = Counter()
    bmm_aa = 0           # BATCH_MATMUL with both inputs runtime (act x act)
    bmm_wa = 0           # BATCH_MATMUL with one const input
    bmm_aa_flops = 0.0   # 2*B*M*K*N summed over act-x-act batch matmuls
    fc_flops = 0.0       # FULLY_CONNECTED weight matmul flops
    conv_flops = 0.0     # CONV_2D / DEPTHWISE / TRANSPOSE_CONV flops
    softmax_axes = []    # softmax reduce-dim sizes
    max_seq = 0          # largest T among rank-3 activation tensors [B,T,C]
    bmm_shapes = Counter()
    n_ops = 0

    nsub = model.SubgraphsLength()
    for si in range(nsub):
        sg = model.Subgraphs(si)
        for i in range(sg.OperatorsLength()):
            op = sg.Operators(i)
            name = opcodes[op.OpcodeIndex()]
            hist[name] += 1
            n_ops += 1
            outs = [op.Outputs(j) for j in range(op.OutputsLength())]
            ins = [op.Inputs(j) for j in range(op.InputsLength())]
            oshape = tensor_shape(sg, outs[0]) if outs else None
            if oshape is not None:
                rank_hist[len(oshape)] += 1
                if len(oshape) == 3 and oshape[0] in (1, 2) and oshape[1] > max_seq:
                    max_seq = oshape[1]

            if name == 'BATCH_MATMUL' and len(ins) >= 2:
                a_const = is_const(model, sg, ins[0])
                b_const = is_const(model, sg, ins[1])
                sa = tensor_shape(sg, ins[0]) or []
                sb = tensor_shape(sg, ins[1]) or []
                if a_const or b_const:
                    bmm_wa += 1
                else:
                    bmm_aa += 1
                    if oshape and len(sa) >= 2:
                        batch = math.prod(oshape[:-2]) if len(oshape) > 2 else 1
                        m, n = oshape[-2], oshape[-1]
                        k = sa[-1]
                        bmm_aa_flops += 2.0 * batch * m * k * n
                        bmm_shapes[(batch, m, k, n)] += 1
            elif name == 'FULLY_CONNECTED' and len(ins) >= 2:
                sw = tensor_shape(sg, ins[1]) or []
                if oshape and len(sw) == 2:
                    rows = math.prod(oshape[:-1])
                    fc_flops += 2.0 * rows * sw[0] * sw[1]
            elif name in ('CONV_2D', 'DEPTHWISE_CONV_2D', 'TRANSPOSE_CONV') and len(ins) >= 2:
                sw = tensor_shape(sg, ins[1]) or []
                if oshape and len(oshape) == 4 and len(sw) == 4:
                    out_elems = math.prod(oshape)
                    if name == 'DEPTHWISE_CONV_2D':
                        per = sw[1] * sw[2]
                    else:
                        per = sw[1] * sw[2] * sw[3]
                    conv_flops += 2.0 * out_elems * per
            elif name == 'SOFTMAX':
                ish = tensor_shape(sg, ins[0]) or []
                if ish:
                    softmax_axes.append(ish[-1])

    return {
        'n_ops': n_ops,
        'n_subgraphs': nsub,
        'hist': dict(hist.most_common()),
        'rank_hist': {str(k): v for k, v in sorted(rank_hist.items())},
        'bmm_aa': bmm_aa,
        'bmm_wa': bmm_wa,
        'bmm_aa_gflops': round(bmm_aa_flops / 1e9, 3),
        'fc_gflops': round(fc_flops / 1e9, 3),
        'conv_gflops': round(conv_flops / 1e9, 3),
        'softmax_n': len(softmax_axes),
        'softmax_max_axis': max(softmax_axes) if softmax_axes else 0,
        'max_seq_rank3': max_seq,
        'top_bmm_shapes': [f'{k}x{v}' for k, v in bmm_shapes.most_common(5)],
    }

if __name__ == '__main__':
    out = {}
    for path in sys.argv[1:]:
        name = path.split('/')[-1].replace('.tflite', '')
        try:
            out[name] = scan(path)
        except Exception as e:
            out[name] = {'error': repr(e)}
        print(f'-- {name} done', file=sys.stderr)
    json.dump(out, open('opscan_results.json', 'w'), indent=1)
    print('wrote opscan_results.json')
