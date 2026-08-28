"""Rebuild DINOv2-S from the official checkpoint twice: tanh-GELU (control,
must reproduce the shipped file) and sigmoid-GELU (the one-variable variant).
Both fp32. pos_embed interpolated 37x37 -> 32x32 and baked, as in the original
conversion.
"""
import importlib.util
import os

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = '/Users/majimadaisuke/Downloads/depthanything-android/dinov2/scripts/build_dinov2.py'
CKPT = os.path.join(HERE, 'dinov2_vits14_pretrain.pth')
SHIPPED = os.path.join(HERE, 'dinov2_s_fp16.tflite')

spec = importlib.util.spec_from_file_location('bdv2', SRC)
bdv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bdv2)

sd = {k: v.float() for k, v in torch.load(CKPT, map_location='cpu').items()}

# interpolate pos_embed: [1, 1+37*37, C] -> [1, 1+32*32, C]
pos = sd['pos_embed']
cls_pos, patch_pos = pos[:, :1], pos[:, 1:]
g = int(patch_pos.shape[1] ** 0.5)                       # 37
patch_pos = patch_pos.reshape(1, g, g, -1).permute(0, 3, 1, 2)
patch_pos = F.interpolate(patch_pos, size=(bdv2.GRID, bdv2.GRID),
                          mode='bicubic', antialias=True, align_corners=False)
patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, bdv2.GRID * bdv2.GRID, -1)
sd['pos_embed'] = torch.cat([cls_pos, patch_pos], dim=1)
print('pos_embed ->', tuple(sd['pos_embed'].shape))

import litert_torch
from ai_edge_litert.interpreter import Interpreter


def run(path, x):
    it = Interpreter(model_path=path, num_threads=4)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]['index'], x)
    it.invoke()
    return it.get_tensor(it.get_output_details()[0]['index'])


def build(tag):
    out = os.path.join(HERE, f'dinov2_ab_{tag}_fp32.tflite')
    model = bdv2.DINOv2(sd).eval()
    litert_torch.convert(model, (torch.zeros(1, 3, bdv2.IMG_SIZE, bdv2.IMG_SIZE),)).export(out)
    print('wrote %s (%.1f MB)' % (out, os.path.getsize(out) / 1e6))
    return out


rng = np.random.default_rng(11)
x = rng.standard_normal((1, 3, bdv2.IMG_SIZE, bdv2.IMG_SIZE)).astype(np.float32)

p_tanh = build('tanh')
a_ship = run(SHIPPED, x)
a_tanh = run(p_tanh, x)
print('rebuild-tanh vs shipped-fp16: corr %.6f max_abs %.4f'
      % (np.corrcoef(a_ship.ravel(), a_tanh.ravel())[0, 1], np.abs(a_ship - a_tanh).max()))

bdv2.gelu = lambda t: t * torch.sigmoid(1.702 * t)
p_sig = build('sig')
a_sig = run(p_sig, x)
print('sig vs tanh (approximation delta): corr %.6f max_abs %.4f'
      % (np.corrcoef(a_tanh.ravel(), a_sig.ravel())[0, 1], np.abs(a_tanh - a_sig).max()))
