import sys, os, collections, json
import numpy as np, torch, torch.nn as nn
HERE=os.path.dirname(os.path.abspath(__file__)); SIZE=256
BANNED={"GATHER","GATHER_ND","TOPK_V2","GELU","ERF","WHERE","SELECT","SELECT_V2","BROADCAST_TO","POW","TRANSPOSE_CONV","CAST","EMBEDDING_LOOKUP","RFFT2D","FFT","STFT","COMPLEX","RFFT","IRFFT","CUMSUM","MIRROR_PAD"}
class GAP(nn.Module):
    def forward(s,x): return x.mean(3,keepdim=True).mean(2,keepdim=True)
def build():
    import timm
    m=timm.create_model("mobilenetv4_conv_medium.e500_r256_in1k", pretrained=True).eval()
    n=0
    for mod in m.modules():
        for cn,ch in list(mod.named_children()):
            if isinstance(ch,nn.AdaptiveAvgPool2d): setattr(mod,cn,GAP()); n+=1
    print(f"  loaded mobilenetv4_conv_medium; swapped {n} avgpool; params {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    return m
def opcheck(p,l):
    from ai_edge_litert.interpreter import Interpreter
    it=Interpreter(model_path=p); it.allocate_tensors()
    ops=collections.Counter(d.get("op_name","?") for d in it._get_ops_details())
    bad={k:v for k,v in ops.items() if k.upper() in BANNED}; over=sum(1 for d in it.get_tensor_details() if len(d.get("shape",[]))>4)
    print(f"[{l}] ops:",dict(sorted(ops.items(),key=lambda kv:-kv[1])))
    print(f"[{l}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(p)/1e6:.1f}MB","GPU-CLEAN" if not bad and not over else "BLOCKERS")
    return it
def to_fp16(fp32,fp16):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm=recipe_manager.RecipeManager()
    rm.add_quantization_config(regex=".*",operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,op_config=qtyping.OpQuantizationConfig(weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16,dtype=qtyping.TensorDataType.FLOAT),compute_precision=qtyping.ComputePrecision.FLOAT),algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16): os.remove(fp16)
    q=quantizer.Quantizer(float_model=fp32); q.load_quantization_recipe(rm.get_quantization_recipe()); q.quantize().export_model(fp16); return fp16
if __name__=="__main__":
    m=build(); x=torch.randn(1,3,SIZE,SIZE)
    with torch.no_grad(): ref=m(x)
    print(f"forward: logits {tuple(ref.shape)}")
    if (sys.argv[1] if len(sys.argv)>1 else "all")=="forward": sys.exit()
    import litert_torch
    fp32=f"{HERE}/mnv4.tflite"; litert_torch.convert(m,(x,)).export(fp32)
    it=opcheck(fp32,"mnv4"); d=it.get_input_details()[0]; it.set_tensor(d["index"],x.numpy().astype(d["dtype"])); it.invoke()
    o=it.get_tensor(it.get_output_details()[0]["index"])
    print(f"tflite vs torch corr {np.corrcoef(o.ravel(),ref.numpy().ravel())[0,1]:.6f}")
    to_fp16(fp32,f"{HERE}/mnv4_fp16.tflite"); opcheck(f"{HERE}/mnv4_fp16.tflite","mnv4_fp16")
