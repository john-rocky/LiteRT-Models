import numpy as np
from PIL import Image
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.options import Options

gpu = Options(hardware_accelerators=HardwareAccelerator.GPU)
img = Image.open("photo.png").convert("RGB").resize((1024, 1024), Image.BICUBIC)   # Idol feeds the cropped object
x = (np.asarray(img, np.float32) / 255).transpose(2, 0, 1)[None]                  # [1, 3, 1024, 1024] in [0, 1]
rope = {"args_1": np.fromfile("dino_rope_cos.f32", "<f4"), "args_2": np.fromfile("dino_rope_sin.f32", "<f4")}
for i in range(4):                                                                  # 8 layers per graph
    model = CompiledModel.from_file(f"dino_{i}.tflite", options=gpu)
    names = model.get_signature_by_index(0)["inputs"]                               # args_0, args_1, args_2
    inputs, outputs = model.create_input_buffers(0), model.create_output_buffers(0)
    for name, buf in zip(names, inputs):
        buf.write(np.ascontiguousarray(x if name == "args_0" else rope[name], np.float32).reshape(-1))
    model.run_by_index(0, inputs, outputs)
    x = outputs[0].read(4101 * 1280, np.float32).reshape(1, 4101, 1280)
print(x.shape, x[0, 0, :4])   # the image feature Idol's denoiser reads: 1 class, 4 register, 64 x 64 patch tokens
