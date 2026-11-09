import sys

import numpy as np
from PIL import Image
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.options import Options

path, photo = sys.argv[1], sys.argv[2]   # bir.tflite from this repo, any photo
model = CompiledModel.from_file(path, options=Options(hardware_accelerators=HardwareAccelerator.CPU))
img = Image.open(photo).convert("RGB")
x = np.asarray(img.resize((1024, 1024), Image.BILINEAR), np.float32) / 255
x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
inputs, outputs = model.create_input_buffers(0), model.create_output_buffers(0)
inputs[0].write(x.reshape(-1))                       # [1, 3, 1024, 1024]
model.run_by_index(0, inputs, outputs)
alpha = outputs[0].read(1024 * 1024, np.float32).reshape(1024, 1024)   # 0..1, 1 = the object
mask = Image.fromarray((alpha * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
img.putalpha(mask)
img.save("cutout.png")
print(alpha.min(), alpha.max(), (alpha > 0.5).mean())
