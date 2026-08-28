"""Fifth probe batch — decode-GEMV weight-dtype (bandwidth-bound region).

The fp16-vs-fp32 null result (experiment 1) was measured on compute-bound
graphs. A T=1 GEMV chain is bandwidth-bound: latency ~ bytes-of-weights /
memory-bandwidth, so storage dtype SHOULD matter here if the NPU streams
weights. Arms (same seed, same math, only storage differs):

  gemv_fp32  - 8x Linear(4096,4096)+ReLU chain, fp32 consts     (537 MB)
  gemv_fp16  - same file through make_fp16.py (fp16+DEQUANTIZE) (268 MB)
  gemv_int8  - same file through make_int8.py (int8+DEQUANTIZE) (134 MB)
  gemv_int4  - same file through make_int8.py --int4            ( 67 MB)

Compute: 2*134M = 0.27 GF per inference — negligible; any latency difference
between arms is weight-streaming. Input rank kept at 2 ([1,4096]): DRQ-int8
had a rank-3 wall on earlier models (parakeet-ja).

  python probe_batch5.py        # builds gemv_fp32.tflite
"""
import torch
import torch.nn as nn

D, LAYERS = 4096, 8


class Gemv(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcs = nn.ModuleList(nn.Linear(D, D) for _ in range(LAYERS))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < LAYERS - 1:
                x = torch.relu(x)
        return x


if __name__ == '__main__':
    torch.manual_seed(0)
    m = Gemv().eval()
    import litert_torch
    litert_torch.convert(m, (torch.randn(1, D),)).export('gemv_fp32.tflite')
    print('wrote gemv_fp32.tflite')
