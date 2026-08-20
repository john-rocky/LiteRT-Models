#!/usr/bin/env python3
"""Run a .tflite on the host-Mac GPU (Metal) via CompiledModel: fp16 default vs enforce_f32,
parity vs the XNNPACK CPU interpreter, wall time. Usage: gpu_mac_run.py model.tflite [seed]"""
import sys, time
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.options import Options, GpuOptions

path = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
it = Interpreter(model_path=path, num_threads=8); it.allocate_tensors()
inp = it.get_input_details()[0]; out = it.get_output_details()[0]
rng = np.random.default_rng(seed)
x = rng.standard_normal(inp["shape"]).astype(np.float32)
it.set_tensor(inp["index"], x); t0 = time.time(); it.invoke(); tcpu = time.time() - t0
ref = it.get_tensor(out["index"]).reshape(-1).astype(np.float32)
print(f"[cpu] {tcpu*1000:.0f} ms  out={ref.size}  |ref|max={np.abs(ref).max():.3g}")

def run(tag, opts):
    t0 = time.time()
    m = (CompiledModel.from_file(path, options=opts) if opts is not None
         else CompiledModel.from_file(path, HardwareAccelerator.GPU))
    tc = time.time() - t0
    ib = m.create_input_buffers(0); ob = m.create_output_buffers(0)
    ib[0].write(x.ravel())
    ts = []
    for _ in range(4):
        t1 = time.time(); m.run_by_index(0, ib, ob)
        y = np.array(ob[0].read(int(ref.size), np.float32)); ts.append(time.time() - t1)
    d = np.abs(y - ref)
    print(f"[gpu {tag}] compile {tc:.1f}s fully={m.is_fully_accelerated()} best {min(ts)*1000:.0f} ms "
          f"corr={np.corrcoef(y, ref)[0,1]:.6f} max|diff|={d.max():.4g} mean|diff|={d.mean():.4g} "
          f"rel-rms={np.sqrt((d**2).mean())/np.sqrt((ref**2).mean()):.4g}")
    m.close()

run("fp16", None)
o = Options.create(); o.hardware_accelerators = HardwareAccelerator.GPU; o.gpu_options.enforce_f32 = True
run("f32 ", o)
