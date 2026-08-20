#!/usr/bin/env python3
"""MambaVision-T-1K classifier (logits 1x1000) GPU-clean for LiteRT. Same 4 patches as make_mambavision_gpu.py."""
import sys, types, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
class _Dummy:
    def __getattr__(self,n):
        if n.startswith("__") and n.endswith("__"): raise AttributeError(n)
        return _Dummy()
    def __call__(self,*a,**k): return _Dummy()
class _Leaf(types.ModuleType):
    def __getattr__(self,n):
        if n.startswith("__") and n.endswith("__"): raise AttributeError(n)
        return _Dummy()
for name in ["scipy.sparse.linalg._propack","scipy.optimize._cobyla","scipy.optimize._slsqp",
             "scipy.optimize._minpack","scipy.optimize._lbfgsb","scipy.optimize._zeros",
             "scipy.optimize._highs","scipy.optimize._direct","scipy.optimize._trlib",
             "scipy.optimize._group_columns","scipy.optimize._bglu_dense"]:
    sys.modules[name]=_Leaf(name)
def selective_scan_fn(u,delta,A,B,C,D=None,z=None,delta_bias=None,delta_softplus=False,return_last_state=None):
    dtype=u.dtype; u=u.float(); delta=delta.float()
    if delta_bias is not None: delta=delta+delta_bias.float()[None,:,None]
    if delta_softplus: delta=F.relu(delta)+torch.log1p(torch.exp(-torch.abs(delta)))
    b,d,L=u.shape
    dp=delta.permute(0,2,1); up=u.permute(0,2,1); Bp=B.float().permute(0,2,1); Cp=C.float().permute(0,2,1)
    Aacc=torch.exp(dp.unsqueeze(-1)*A.float()[None,None]); h=(dp*up).unsqueeze(-1)*Bp.unsqueeze(2)
    step=1
    while step<L:
        A_sh=torch.cat([torch.ones_like(Aacc[:,:step]),Aacc[:,:L-step]],dim=1)
        h_sh=torch.cat([torch.zeros_like(h[:,:step]),h[:,:L-step]],dim=1)
        Aacc,h=Aacc*A_sh,Aacc*h_sh+h; step*=2
    y=(h*Cp.unsqueeze(2)).sum(-1)
    if D is not None: y=y+up*D.float()[None,None]
    return y.permute(0,2,1).to(dtype)
mm=types.ModuleType("mamba_ssm"); mo=types.ModuleType("mamba_ssm.ops"); mi=types.ModuleType("mamba_ssm.ops.selective_scan_interface")
mi.selective_scan_fn=selective_scan_fn
sys.modules.update({"mamba_ssm":mm,"mamba_ssm.ops":mo,"mamba_ssm.ops.selective_scan_interface":mi})
import transformers as _tf
if not hasattr(_tf.PreTrainedModel,"all_tied_weights_keys"):
    _tf.PreTrainedModel.all_tied_weights_keys=property(lambda s:{})
from transformers import AutoModelForImageClassification
# Fine-tune override: MAMBAVISION_MODEL_ID=<hf-repo-or-local-dir> — a fine-tuned
# MambaVision-T classifier saved with save_pretrained() (labels flow from config).
MID=os.environ.get("MAMBAVISION_MODEL_ID","nvidia/MambaVision-T-1K")
model=AutoModelForImageClassification.from_pretrained(MID,trust_remote_code=True).eval()
print("loaded classifier; type:",type(model).__name__)
_mvmod=sys.modules[type(model).__module__]
def _wp(x,window_size):
    B,C,H,W=x.shape; assert window_size==H==W; return x.flatten(2).transpose(1,2)
def _wr(windows,window_size,H,W):
    B=windows.shape[0]; C=windows.shape[2]; return windows.transpose(1,2).reshape(B,C,H,W)
_mvmod.window_partition=_wp; _mvmod.window_reverse=_wr
AttnCls=None
for m in model.modules():
    if hasattr(m,"qkv") and hasattr(m,"num_heads") and hasattr(m,"head_dim") and hasattr(m,"proj"): AttnCls=type(m); break
def _attn_fwd(self,x):
    B,N,C=x.shape; H=self.num_heads; Hd=self.head_dim
    q=self.q_lin(x).reshape(B,N,H,Hd).permute(0,2,1,3); k=self.k_lin(x).reshape(B,N,H,Hd).permute(0,2,1,3)
    v=self.v_lin(x).reshape(B,N,H,Hd).permute(0,2,1,3); q,k=self.q_norm(q),self.k_norm(k)
    q=q*self.scale; attn=(q@k.transpose(-2,-1)).softmax(-1)
    x=(attn@v).transpose(1,2).reshape(B,N,C); return self.proj_drop(self.proj(x))
AttnCls.forward=_attn_fwd
for m in model.modules():
    if isinstance(m,AttnCls):
        Cf=m.qkv.in_features; w=m.qkv.weight; bsd=m.qkv.bias
        for nm,sl in (("q_lin",slice(0,Cf)),("k_lin",slice(Cf,2*Cf)),("v_lin",slice(2*Cf,3*Cf))):
            lin=nn.Linear(Cf,Cf,bias=bsd is not None)
            with torch.no_grad():
                lin.weight.copy_(w[sl])
                if bsd is not None: lin.bias.copy_(bsd[sl])
            setattr(m,nm,lin)
print("patched all 4")
class Logits(nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,x):
        o=s.m(x)
        if isinstance(o,dict): return o.get("logits", next(iter(o.values())))
        if hasattr(o,"logits"): return o.logits
        return o[0] if isinstance(o,(tuple,list)) else o
wrap=Logits(model).eval()
# labels from config
id2label=model.config.id2label
labels=[id2label[i] for i in range(len(id2label))]
open("imagenet_labels.txt","w").write("\n".join(labels))
print("labels:",len(labels),"e.g.",labels[:2])
import litert_torch, collections
from ai_edge_litert.interpreter import Interpreter
dummy=torch.randn(1,3,224,224)
with torch.no_grad():
    lg=wrap(dummy); print("logits shape:",tuple(lg.shape))
litert_torch.convert(wrap,(dummy,)).export("mambavision_t_cls_gpu.tflite")
it=Interpreter(model_path="mambavision_t_cls_gpu.tflite"); it.allocate_tensors()
ops=collections.Counter(d.get("op_name","?") for d in it._get_ops_details())
GPU_BAD={"GATHER_ND","GATHER","SELECT_V2","SELECT","PACK","SPLIT","CAST","TOPK_V2","BROADCAST_TO","WHILE","TRANSPOSE_CONV"}
bad={k:v for k,v in ops.items() if k in GPU_BAD}; over=sum(1 for d in it.get_tensor_details() if len(d.get("shape",[]))>4)
print(f"op-check: TOTAL={sum(ops.values())} GPU_BAD={bad or 'NONE'} >4D={over} GATHER_ND={ops.get('GATHER_ND',0)}")
# sanity: top-1 on a real photo (ImageNet preprocessing)
def preprocess(path):
    im=Image.open(path).convert("RGB").resize((224,224),Image.BICUBIC)
    a=np.asarray(im).astype(np.float32)/255.0
    a=(a-np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
    return a.transpose(2,0,1)[None].astype(np.float32)
for img in ["/Users/majimadaisuke/Downloads/pexels-goksel-37932522.jpg","/Users/majimadaisuke/Downloads/pexels-mikkel-kvist-2722911-29122139.jpg"]:
    if os.path.exists(img):
        x=preprocess(img); it.set_tensor(it.get_input_details()[0]["index"],x); it.invoke()
        lg=it.get_tensor(it.get_output_details()[0]["index"]).flatten()
        top=lg.argsort()[-3:][::-1]
        print("  ",os.path.basename(img)[:22],"-> top3:",[labels[i] for i in top])
from ai_edge_quantizer import quantizer
RECIPE=[{"regex":".*","operation":"*","algorithm_key":"float_casting","op_config":{"weight_tensor_config":{"num_bits":16,"dtype":"FLOAT"}}}]
q=quantizer.Quantizer("mambavision_t_cls_gpu.tflite"); q.load_quantization_recipe(RECIPE)
q.quantize().export_model("mambavision_t_cls_gpu_fp16.tflite")
print("FP32 %.1f MB -> FP16 %.1f MB"%(os.path.getsize("mambavision_t_cls_gpu.tflite")/1e6,os.path.getsize("mambavision_t_cls_gpu_fp16.tflite")/1e6))
print("VERDICT:", "GPU-CLEAN" if not bad and not over else "BLOCKERS")
