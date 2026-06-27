#!/usr/bin/env python3
"""MobileNetV3-Large ImageNet classifier -> LiteRT CompiledModel GPU.
torchvision mobilenet_v3_large (IMAGENET1K_V2, 1000 classes). Pure CNN. One re-authoring: the SE-block
AdaptiveAvgPool2d(1) + the final classifier pool -> mean(3).mean(2). Hardswish -> native HARD_SWISH."""
import sys, os, collections, json
import numpy as np, torch, torch.nn as nn
HERE=os.path.dirname(os.path.abspath(__file__)); SIZE=224
BANNED={"GATHER","GATHER_ND","TOPK_V2","GELU","ERF","WHERE","SELECT","SELECT_V2","BROADCAST_TO","POW","TRANSPOSE_CONV","CAST","EMBEDDING_LOOKUP","RFFT2D","FFT","STFT","COMPLEX","RFFT","IRFFT","CUMSUM"}
class GAP(nn.Module):
    def forward(s,x): return x.mean(3,keepdim=True).mean(2,keepdim=True)
def build():
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    w=MobileNet_V3_Large_Weights.IMAGENET1K_V2
    m=mobilenet_v3_large(weights=w).eval()
    n=0
    for mod in m.modules():
        for cn,ch in list(mod.named_children()):
            if isinstance(ch,nn.AdaptiveAvgPool2d): setattr(mod,cn,GAP()); n+=1
    json.dump(list(w.meta["categories"]), open(os.path.join(HERE,"imagenet_classes.json"),"w"))
    print(f"  swapped {n} AdaptiveAvgPool2d -> mean(3).mean(2); {len(w.meta['categories'])} classes")
    return m
def opcheck(path,label):
    from ai_edge_litert.interpreter import Interpreter
    it=Interpreter(model_path=path); it.allocate_tensors()
    ops=collections.Counter(d.get("op_name","?") for d in it._get_ops_details())
    bad={k:v for k,v in ops.items() if k.upper() in BANNED}
    over=sum(1 for d in it.get_tensor_details() if len(d.get("shape",[]))>4)
    print(f"[{label}] ops:",dict(sorted(ops.items(),key=lambda kv:-kv[1])))
    print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(path)/1e6:.1f}MB","GPU-CLEAN" if not bad and not over else "BLOCKERS")
    return it
def to_fp16(fp32,fp16):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm=recipe_manager.RecipeManager()
    rm.add_quantization_config(regex=".*",operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,op_config=qtyping.OpQuantizationConfig(weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16,dtype=qtyping.TensorDataType.FLOAT),compute_precision=qtyping.ComputePrecision.FLOAT),algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16): os.remove(fp16)
    q=quantizer.Quantizer(float_model=fp32); q.load_quantization_recipe(rm.get_quantization_recipe()); q.quantize().export_model(fp16); return fp16
def main():
    stage=sys.argv[1] if len(sys.argv)>1 else "all"
    m=build(); img=torch.randn(1,3,SIZE,SIZE)
    with torch.no_grad(): ref=m(img)
    print(f"forward: logits {tuple(ref.shape)}")
    if stage=="forward": return
    import litert_torch
    fp32=os.path.join(HERE,"mnv3.tflite"); litert_torch.convert(m,(img,)).export(fp32)
    it=opcheck(fp32,"mnv3"); d=it.get_input_details()[0]; it.set_tensor(d["index"],img.numpy().astype(d["dtype"])); it.invoke()
    o=it.get_tensor(it.get_output_details()[0]["index"])
    print(f"tflite vs torch: corr {np.corrcoef(o.ravel(),ref.numpy().ravel())[0,1]:.6f}")
    if stage=="all": to_fp16(fp32,os.path.join(HERE,"mnv3_fp16.tflite")); opcheck(os.path.join(HERE,"mnv3_fp16.tflite"),"mnv3_fp16")
if __name__=="__main__": main()
