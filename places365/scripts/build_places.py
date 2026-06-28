#!/usr/bin/env python3
"""Places365 scene recognition (ResNet18, MIT) -> LiteRT CompiledModel GPU.

CSAILVision/places365 resnet18 (365 scene categories). Pure CNN. One re-authoring: the global
AdaptiveAvgPool2d(1) -> mean(3).mean(2) (the Mali multi-axis-pool fix). Output = 365-class scene logits;
softmax + top-k in the app. Run: ~/clipconv/bin/python build_places.py [forward|all]
"""
import sys, os, collections, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); SIZE = 224
BANNED = {"GATHER","GATHER_ND","TOPK_V2","GELU","ERF","WHERE","SELECT","SELECT_V2","BROADCAST_TO","POW",
          "TRANSPOSE_CONV","CAST","EMBEDDING_LOOKUP","RFFT2D","FFT","STFT","COMPLEX","RFFT","IRFFT","CUMSUM"}


class ZeroPadMaxPool(nn.Module):
    """ResNet maxpool(3,s2,p1) pads with -inf -> PADV2 (Mali won't delegate). The pool follows a ReLU so
    inputs are >=0, making a 0-pad EXACTLY equivalent -> emits a delegatable PAD instead."""
    def forward(s, x):
        return F.max_pool2d(F.pad(x, (1, 1, 1, 1), value=0.0), kernel_size=3, stride=2, padding=0)


class GAP(nn.Module):
    """AdaptiveAvgPool2d(1) -> two single-axis means (Mali multi-axis-pool fix)."""
    def forward(s, x): return x.mean(3, keepdim=True).mean(2, keepdim=True)


def build():
    from torchvision.models import resnet18
    m = resnet18(num_classes=365)
    ck = torch.load(os.path.join(HERE, "resnet18_places365.pth.tar"), map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    miss, unexp = m.load_state_dict(sd, strict=False); m.eval()
    n = 0
    for nm, ch in list(m.named_children()):
        if isinstance(ch, nn.AdaptiveAvgPool2d):
            setattr(m, nm, GAP()); n += 1
        if isinstance(ch, nn.MaxPool2d):
            setattr(m, nm, ZeroPadMaxPool())
    cats = [l.split(" ")[0][3:] for l in open(os.path.join(HERE, "categories_places365.txt")).read().splitlines()]
    json.dump(cats, open(os.path.join(HERE, "places365_classes.json"), "w"))
    print(f"  loaded places365 resnet18; missing {len(miss)} unexpected {len(unexp)}; swapped {n} avgpool; "
          f"{len(cats)} classes; params {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    return m


def opcheck(p, l):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=p); it.allocate_tensors()
    ops = collections.Counter(d.get("op_name", "?") for d in it._get_ops_details())
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    over = sum(1 for d in it.get_tensor_details() if len(d.get("shape", [])) > 4)
    print(f"[{l}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"[{l}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(p)/1e6:.1f}MB",
          "GPU-CLEAN" if not bad and not over else "BLOCKERS")
    return it


def to_fp16(fp32, fp16):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT), algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16): os.remove(fp16)
    q = quantizer.Quantizer(float_model=fp32); q.load_quantization_recipe(rm.get_quantization_recipe())
    q.quantize().export_model(fp16); return fp16


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    m = build(); x = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad(): ref = m(x)
    print(f"forward: logits {tuple(ref.shape)}")
    if stage == "forward": sys.exit()
    import litert_torch
    fp32 = os.path.join(HERE, "places.tflite"); litert_torch.convert(m, (x,)).export(fp32)
    it = opcheck(fp32, "places")
    d = it.get_input_details()[0]; it.set_tensor(d["index"], x.numpy().astype(d["dtype"])); it.invoke()
    o = it.get_tensor(it.get_output_details()[0]["index"])
    print(f"tflite vs torch corr {np.corrcoef(o.ravel(), ref.numpy().ravel())[0,1]:.6f}")
    if stage == "all":
        to_fp16(fp32, os.path.join(HERE, "places_fp16.tflite")); opcheck(os.path.join(HERE, "places_fp16.tflite"), "places_fp16")
