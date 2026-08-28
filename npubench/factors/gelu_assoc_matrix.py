"""Which tanh-GELU spellings does the converter already fuse to the GELU op?

Five spellings of 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) around one
Linear(64,64), converted with litert-torch 0.9.3 (ai-edge-litert 2.1.6),
op histogram read back from the flatbuffer (opscan.py). Result 2026-08-28:

  pow        0.044715 * torch.pow(x, 3)   -> FULLY_CONNECTED + GELU   (fused)
  mull       0.044715 * (x * x * x)       -> FULLY_CONNECTED + GELU   (fused)
  mulr       0.044715 * (x * (x * x))     -> FULLY_CONNECTED + GELU   (fused)
  xhalf      x * 0.5 ordering (pow cube)  -> FULLY_CONNECTED + GELU   (fused)
  coefffirst 0.044715 * x * x * x         -> MUL x6 + TANH + ADD x2   (NOT fused)

The unfused spelling is Python's natural left association
((0.044715*x)*x)*x — the coefficient folds into the innermost MUL, so no
x^3 subterm exists and no MatchGeluApproximate* variant in
tensorflow/compiler/mlir/lite/transforms/optimize_patterns.td matches.
It is the exact spelling in the zoo's DINOv2/TIPSv2 builders
(~/Downloads/meeting/dinov2-work/build_dinov2.py:57) and in the phase-3/6
probes — which is why those files ship TANH chains.

  python gelu_assoc_matrix.py   # rebuilds the 5 files and prints the table
"""
import math

import torch
import torch.nn as nn

C = math.sqrt(2.0 / math.pi)

FORMS = {
    'pow': lambda x: 0.5 * x * (1.0 + torch.tanh(C * (x + 0.044715 * torch.pow(x, 3.0)))),
    'mull': lambda x: 0.5 * x * (1.0 + torch.tanh(C * (x + 0.044715 * (x * x * x)))),
    'mulr': lambda x: 0.5 * x * (1.0 + torch.tanh(C * (x + 0.044715 * (x * (x * x))))),
    'xhalf': lambda x: x * 0.5 * (1.0 + torch.tanh(C * (x + 0.044715 * torch.pow(x, 3.0)))),
    'coefffirst': lambda x: 0.5 * x * (1.0 + torch.tanh(C * (x + 0.044715 * x * x * x))),
}


class M(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.fc = nn.Linear(64, 64)
        self.f = f

    def forward(self, x):
        return self.f(self.fc(x))


if __name__ == '__main__':
    import litert_torch
    from ai_edge_litert import schema_py_generated as S

    def ops(path):
        model = S.ModelT.InitFromPackedBuf(open(path, 'rb').read(), 0)
        names = {getattr(S.BuiltinOperator, k): k for k in dir(S.BuiltinOperator) if not k.startswith('_')}
        hist = {}
        for op in model.subgraphs[0].operators:
            oc = model.operatorCodes[op.opcodeIndex]
            n = names[max(oc.builtinCode, oc.deprecatedBuiltinCode)]
            hist[n] = hist.get(n, 0) + 1
        return hist

    for tag, f in FORMS.items():
        torch.manual_seed(0)
        litert_torch.convert(M(f).eval(), (torch.randn(1, 8, 64),)).export(f'v_{tag}.tflite')
        print(f'{tag:<11} {ops(f"v_{tag}.tflite")}')
