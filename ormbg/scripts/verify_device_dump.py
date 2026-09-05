"""Compare a device run of ormbg.tflite against the host CPU on the identical input.

The instrumented test (app/src/androidTest/java/com/ormbg/BgRemoverTest.kt) writes the
preprocessed input tensor and the raw model output into the app's filesDir. Pull them and run:

  adb exec-out run-as com.ormbg cat files/ormbg_test/input_nchw.f32 > input_nchw.f32
  adb exec-out run-as com.ormbg cat files/ormbg_test/raw_out.f32   > raw_out.f32
  pip install ai-edge-litert numpy
  python scripts/verify_device_dump.py ormbg.tflite input_nchw.f32 raw_out.f32

The host CPU run is the fp32 reference; the device GPU computes in fp16, so expect a
correlation of 0.999x and a max |diff| in the 1e-2 range, not zero. Exact equality would mean
the device did not run on the GPU.
"""

import sys

import numpy as np
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator

SIZE = 1024


def summary(name, m):
  norm = (m - m.min()) / (m.max() - m.min() + 1e-6)
  p = 64
  corners = np.mean(
      [norm[:p, :p].mean(), norm[:p, -p:].mean(), norm[-p:, :p].mean(), norm[-p:, -p:].mean()]
  )
  c = SIZE // 2 - p // 2
  center = norm[c : c + p, c : c + p].mean()
  print(
      f"{name:6s} raw_min={m.min():.4f} raw_max={m.max():.4f} mean={norm.mean():.4f} "
      f"fg_frac={(norm > 0.5).mean():.4f} center={center:.4f} corners={corners:.4f}"
  )
  return norm


def main(model_path, input_path, device_output_path):
  x = np.fromfile(input_path, dtype="<f4").reshape(1, 3, SIZE, SIZE)
  device = np.fromfile(device_output_path, dtype="<f4").reshape(SIZE, SIZE)

  model = CompiledModel.from_file(model_path, hardware_accel=HardwareAccelerator.CPU)
  inputs = model.create_input_buffers(0)
  outputs = model.create_output_buffers(0)
  inputs[0].write(x.ravel())
  model.run_by_index(0, inputs, outputs)
  host = np.asarray(outputs[0].read(SIZE * SIZE, np.float32)).reshape(SIZE, SIZE)

  host_n = summary("host", host)
  device_n = summary("device", device)
  corr = np.corrcoef(host.ravel(), device.ravel())[0, 1]
  print(f"raw:        corr={corr:.6f} max|diff|={np.abs(host - device).max():.5f}")
  print(
      f"normalized: max|diff|={np.abs(host_n - device_n).max():.5f} "
      f"fg mask IoU={((host_n > 0.5) & (device_n > 0.5)).sum() / ((host_n > 0.5) | (device_n > 0.5)).sum():.5f}"
  )


if __name__ == "__main__":
  if len(sys.argv) != 4:
    sys.exit(__doc__)
  main(*sys.argv[1:4])
