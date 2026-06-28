import warnings; warnings.filterwarnings("ignore")
import sys, types, inspect
_orig_gsf=inspect.getsourcefile
inspect.getsourcefile=lambda o:(_orig_gsf(o) if True else None) if False else (lambda: (_orig_gsf(o)))()
def _safe_gsf(o):
    try: return _orig_gsf(o)
    except Exception: return None
inspect.getsourcefile=_safe_gsf
class _Stub(types.ModuleType):
    __file__="<stub>"; __spec__=None; __path__=[]
    def __getattr__(s,n):
        if n.startswith("__"): raise AttributeError(n)
        return lambda *a,**k: None
def _mk(name, **attrs):
    m=_Stub(name)
    for k,v in attrs.items(): setattr(m,k,v)
    sys.modules[name]=m; return m
for n in ["mmdet","mmdet.models","mmdet.models.utils","mmdet.structures","mmdet.structures.bbox","mmcv.ops"]: _mk(n)
_mk("mmdet.utils", ConfigType=dict, OptConfigType=dict, MultiConfig=dict, reduce_mean=(lambda x:x), InstanceList=list, OptInstanceList=list)
for n in ["chumpy"]:
    try: __import__(n)
    except Exception: _mk(n)
import os, mmpose, torch, torch.nn as nn, collections, numpy as np
from mmpose.apis import init_model

# --- ScaleNorm: torch.norm lowers to an ABS/MAXIMUM/DIV path Mali mis-computes (x/inf=0); use manual sum-of-squares ---
from mmpose.models.utils.transformer import ScaleNorm as _SN
def _sn_forward(self, x):
    # SafeRMSNorm: ScaleNorm input reaches ~|274| so sum(x^2)~3.6M OVERFLOWS fp16 (65504) on Mali -> norm=inf
    # -> x/inf=0 (all-zero head). Scale x down by S before squaring so the sum stays fp16-safe (exact).
    xs = x * (1.0 / 64.0)
    norm = torch.sqrt((xs * xs).sum(dim=-1, keepdim=True) + 1e-12) * (64.0 * self.scale)
    return x / norm.clamp(min=self.eps) * self.g
_SN.forward = _sn_forward


# --- GAU act@act BMM -> broadcast-reduce (Mali mis-computes act@act BMM -> all-zero output) ---
import torch.nn.functional as _F
from mmpose.models.utils.rtmcc_block import RTMCCBlock as _RTMCC, rope as _rope
def _gau_forward(self, inputs):
    x = inputs
    x = self.ln(x)
    uv = self.act_fn(self.uv(x))
    u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
    base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta
    if self.pos_enc: base = _rope(base, dim=1)
    q, k = torch.unbind(base, dim=2)                      # [B,K,s] each
    qk = (q.unsqueeze(2) * k.unsqueeze(1)).sum(-1)        # [B,K,K]  (== bmm(q,k^T), broadcast-reduce)
    if self.use_rel_bias:
        bias = self.rel_pos_bias(q.size(1))
        qk = qk + bias[:, :q.size(1), :k.size(1)]
    kernel = torch.square(_F.relu(qk / self.sqrt_s))     # [B,K,K]
    x = u * (kernel.unsqueeze(-1) * v.unsqueeze(1)).sum(2)  # [B,K,e]  (== bmm(kernel,v))
    return self.o(x)
_RTMCC._forward = _gau_forward

HERE=os.path.dirname(os.path.abspath(__file__))
BANNED={"GATHER","GATHER_ND","TOPK_V2","GELU","ERF","WHERE","SELECT","SELECT_V2","BROADCAST_TO","POW","TRANSPOSE_CONV","CAST","EMBEDDING_LOOKUP","RFFT2D","FFT","STFT","COMPLEX","RFFT","IRFFT","CUMSUM"}
root=os.path.dirname(mmpose.__file__)
cfg=os.path.join(root, ".mim/configs", os.environ["RTM_CFG"])
ckpt=os.environ["RTM_CKPT"]
H=int(os.environ.get("RTM_H","256")); W=int(os.environ.get("RTM_W","192"))
NAME=os.environ.get("RTM_NAME","rtm")
class Wrap(nn.Module):
    def __init__(s,m):
        super().__init__(); s.b=m.backbone; s.h=m.head
        s.neck=m.neck if (hasattr(m,"neck") and m.neck is not None) else None
    def forward(s,x):
        f=s.b(x)
        if s.neck is not None: f=s.neck(f)
        out=s.h(f if isinstance(f,(list,tuple)) else (f,))
        return out if isinstance(out,(list,tuple)) else (out,)

def to_fp16(fp32,fp16):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm=recipe_manager.RecipeManager()
    rm.add_quantization_config(regex=".*",operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,op_config=qtyping.OpQuantizationConfig(weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16,dtype=qtyping.TensorDataType.FLOAT),compute_precision=qtyping.ComputePrecision.FLOAT),algorithm_key=AlgorithmName.FLOAT_CASTING)
    import os
    if os.path.exists(fp16): os.remove(fp16)
    q=quantizer.Quantizer(float_model=fp32); q.load_quantization_recipe(rm.get_quantization_recipe()); q.quantize().export_model(fp16); return fp16


import torch.nn.functional as _Fps
class _ZeroStuffConvT2d(nn.Module):
    """Exact GPU-clean ConvTranspose2d (RESIZE_NEAREST + MUL + CONV_2D, no TRANSPOSE_CONV)."""
    def __init__(s, ct, Hin, Win):
        super().__init__()
        s.s=ct.stride[0]; s.k=ct.kernel_size[0]; s.p=ct.padding[0]; s.op=ct.output_padding[0]; s.Hin=Hin; s.Win=Win
        w=ct.weight.detach().flip(2).flip(3).permute(1,0,2,3).contiguous(); s.register_buffer("w",w)
        s.register_buffer("b", ct.bias.detach().clone() if ct.bias is not None else torch.zeros(ct.out_channels))
        import numpy as _np; mh=_np.zeros((Hin*s.s,Win*s.s),_np.float32); mh[::s.s,::s.s]=1.0
        s.register_buffer("mask", torch.from_numpy(mh)[None,None])
    def forward(s,x):
        xn=_Fps.interpolate(x, size=(s.Hin*s.s, s.Win*s.s), mode="nearest")*s.mask
        y=_Fps.conv2d(xn, s.w, bias=s.b, padding=s.k-1)
        olH=(s.Hin-1)*s.s+s.k-2*s.p+s.op; olW=(s.Win-1)*s.s+s.k-2*s.p+s.op
        return y[:,:,s.p:s.p+olH, s.p:s.p+olW]
class _PixelShuffleD2S(nn.Module):
    """nn.PixelShuffle(r) -> depth-to-space ConvTranspose2d (fixed perm weight) wrapped in ZeroStuffConvT2d
    (litert-torch lowers native PixelShuffle to a 6D tensor; this stays 4D)."""
    def __init__(s, r, Cin, Hin, Win):
        super().__init__()
        Cout=Cin//(r*r)
        ct=nn.ConvTranspose2d(Cin, Cout, kernel_size=r, stride=r, bias=False)
        w=torch.zeros(Cin, Cout, r, r)
        for co in range(Cout):
            for i in range(r):
                for j in range(r):
                    w[co*r*r + i*r + j, co, i, j]=1.0
        ct.weight.data.copy_(w)
        s.impl=_ZeroStuffConvT2d(ct, Hin, Win)
    def forward(s,x): return s.impl(x)
def patch_pixelshuffle(m, H, W):
    """Replace RTMWHead's nn.PixelShuffle with the 4D depth-to-space (probes the input spatial size)."""
    head=m.head
    if not hasattr(head,"ps") or not isinstance(head.ps, nn.PixelShuffle): return 0
    shp={}
    h=head.ps.register_forward_pre_hook(lambda mod,i: shp.update(s=tuple(i[0].shape)))
    with torch.no_grad(): Wrap(m)(torch.zeros(1,3,H,W))
    h.remove()
    _,Cin,Hin,Win=shp["s"]; r=head.ps.upscale_factor
    head.ps=_PixelShuffleD2S(r, int(Cin), int(Hin), int(Win))
    print(f"  PixelShuffle(r={r}) -> depth-to-space ZeroStuffConvT2d (Cin={Cin}, {Hin}x{Win})")
    return 1

def opcheck(p,l):
    from ai_edge_litert.interpreter import Interpreter
    it=Interpreter(model_path=p); it.allocate_tensors()
    ops=collections.Counter(d.get("op_name","?") for d in it._get_ops_details())
    bad={k:v for k,v in ops.items() if k.upper() in BANNED}
    over=sum(1 for d in it.get_tensor_details() if len(d.get("shape",[]))>4)
    print(f"[{l}] ops:",dict(sorted(ops.items(),key=lambda kv:-kv[1])))
    print(f"[{l}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(p)/1e6:.1f}MB","GPU-CLEAN" if not bad and not over else "BLOCKERS")
    return it
if __name__=="__main__":
    m=init_model(cfg,ckpt,device="cpu").eval(); patch_pixelshuffle(m,H,W); w=Wrap(m).eval()
    img=torch.randn(1,3,H,W)
    with torch.no_grad(): _o=w(img); sx,sy=_o[0],_o[1]
    print("out simcc_x",tuple(sx.shape),"simcc_y",tuple(sy.shape))
    import litert_torch
    fp32=os.path.join(HERE,f"{NAME}.tflite"); litert_torch.convert(w,(img,)).export(fp32)
    to_fp16(fp32, os.path.join(HERE,f"{NAME}_fp16.tflite")); opcheck(os.path.join(HERE,f"{NAME}_fp16.tflite"),f"{NAME}_fp16")
    it=opcheck(fp32,NAME)
    d=it.get_input_details()[0]; it.set_tensor(d["index"],img.numpy().astype(d["dtype"])); it.invoke()
    od=it.get_output_details()
    o0=it.get_tensor(od[0]["index"]); o1=it.get_tensor(od[1]["index"])
    refs=[sx.numpy(),sy.numpy()]
    # match outputs by shape
    for o in (o0,o1):
        best=max(refs,key=lambda r: -abs(r.size-o.size))
        c=np.corrcoef(o.ravel(),best.ravel())[0,1] if o.size==best.size else float("nan")
        print(f"  tflite-vs-torch out{tuple(o.shape)} corr {c:.5f}")
