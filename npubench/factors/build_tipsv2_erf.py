"""TIPSv2-B14-DPT with builtin GELU (exact erf) instead of the tanh decomposition.
Same flip experiment as DINOv2: one variable, fp32 export, host parity vs the
shipped fp16 file.
"""
import importlib.util
import os

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = '/Users/majimadaisuke/Downloads/depthanything-android/tipsv2/scripts/build_tipsv2.py'
SHIPPED = '/Users/majimadaisuke/Downloads/depthanything-android/tipsv2/scripts/tipsv2_b14_dpt_fp16.tflite'

spec = importlib.util.spec_from_file_location('btips', SRC)
btips = importlib.util.module_from_spec(spec)
spec.loader.exec_module(btips)

from safetensors.torch import load_file as _lf
btips.load_sd = lambda: _lf(os.path.join(HERE, 'tipsv2_model.safetensors'))
btips.gelu = lambda x: F.gelu(x)   # the one variable

model = btips.TIPSv2DPT(btips.load_sd()).eval()

import litert_torch
OUT = os.path.join(HERE, 'tipsv2_erf_fp32.tflite')
litert_torch.convert(model, (torch.zeros(1, 3, btips.IMG, btips.IMG),)).export(OUT)
print('wrote %s (%.1f MB)' % (OUT, os.path.getsize(OUT) / 1e6))

from ai_edge_litert.interpreter import Interpreter


def run(path, x):
    it = Interpreter(model_path=path, num_threads=4)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]['index'], x)
    it.invoke()
    return {o['name']: it.get_tensor(o['index']) for o in it.get_output_details()}


rng = np.random.default_rng(11)
x = rng.standard_normal((1, 3, btips.IMG, btips.IMG)).astype(np.float32)
a, b = run(SHIPPED, x), run(OUT, x)
for (ka, va), (kb, vb) in zip(sorted(a.items()), sorted(b.items())):
    corr = np.corrcoef(va.ravel(), vb.ravel())[0, 1]
    print('out %-40s corr %.6f max_abs %.4f' % (ka[:40], corr, np.abs(va - vb).max()))
