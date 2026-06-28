import sys, os, collections
import numpy as np, torch, torch.nn as nn
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
SIZE=int(os.environ.get("AN_SIZE","256")); STYLE=os.environ.get("AN_STYLE","paprika")
BANNED={"GATHER","GATHER_ND","TOPK_V2","GELU","ERF","WHERE","SELECT","SELECT_V2","BROADCAST_TO","POW","TRANSPOSE_CONV","CAST","EMBEDDING_LOOKUP","RFFT2D","FFT","STFT","COMPLEX","RFFT","IRFFT","CUMSUM","MIRROR_PAD"}
class SafeGroupNorm1(nn.Module):
    """GroupNorm(num_groups=1) re-authored 4D (native GroupNorm -> GATHER_ND): reduce over (C,H,W) via three
    single-axis means in a down-scaled domain (fp16-safe, exact). num_groups=1 is scale-invariant."""
    def __init__(s, ref, S=4.0):
        super().__init__(); s.S=float(S); s.eps=ref.eps
        s.register_buffer("w", ref.weight.detach().clone().view(1,-1,1,1))
        s.register_buffer("b", ref.bias.detach().clone().view(1,-1,1,1))
    def forward(s, x):
        xs=x*(1.0/s.S)
        m=xs.mean(3,keepdim=True).mean(2,keepdim=True).mean(1,keepdim=True)
        d=xs-m
        var=(d*d).mean(3,keepdim=True).mean(2,keepdim=True).mean(1,keepdim=True)*(s.S*s.S)
        d=d*s.S
        return d*torch.rsqrt(var+s.eps)*s.w+s.b

class Wrap(nn.Module):
    def __init__(s,g): super().__init__(); s.g=g
    def forward(s,x): return s.g(x, align_corners=False)
def build():
    from model import Generator
    g=Generator().eval()
    g.load_state_dict(torch.load(f"{HERE}/{STYLE}.pt",map_location="cpu"))
    # ReflectionPad2d -> ZeroPad2d (reflect lowers to GATHER_ND + SELECT; zero-pad = PAD)
    for mod in g.modules():
        for cn,ch in list(mod.named_children()):
            if isinstance(ch,nn.ReflectionPad2d):
                setattr(mod,cn,nn.ZeroPad2d(ch.padding))

    print(f"  loaded {STYLE}; params {sum(p.numel() for p in g.parameters())/1e6:.2f}M")
    # conv-weight scaling (GroupNorm is scale-invariant -> exact; keeps Mali fp16 conv accumulation precise)
    from PIL import Image as _I
    _cm={}
    def _mk(n):
        def h(mod,i,o): _cm[n]=max(_cm.get(n,0.0),float(o.abs().max()))
        return h
    _hs=[mo.register_forward_hook(_mk(n)) for n,mo in g.named_modules() if isinstance(mo,nn.Conv2d)]
    _im=_I.open(f"{HERE}/content.jpg").convert("RGB"); _sz=min(_im.size)
    _im=_im.crop(((_im.width-_sz)//2,(_im.height-_sz)//2,(_im.width+_sz)//2,(_im.height+_sz)//2)).resize((SIZE,SIZE),_I.BILINEAR)
    _xx=(np.asarray(_im).astype(np.float32)/127.5-1.0).transpose(2,0,1)[None]
    with torch.no_grad(): g(torch.from_numpy(_xx), align_corners=False)
    for h in _hs: h.remove()
    _ns=0
    for n,mo in g.named_modules():
        if isinstance(mo,nn.Conv2d) and mo.out_channels!=3:
            k=_cm.get(n,10.0)/10.0
            if k>1e-6:
                mo.weight.data.div_(k)
                if mo.bias is not None: mo.bias.data.div_(k)
                _ns+=1
    _ni=0
    for mod in g.modules():
        for cn,ch in list(mod.named_children()):
            if isinstance(ch,nn.GroupNorm): setattr(mod,cn,SafeGroupNorm1(ch,4.0)); _ni+=1
    print(f"  scaled {_ns} convs + {_ni} SafeGroupNorm1")
    return Wrap(g).eval()
def to_fp16(fp32,fp16):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm=recipe_manager.RecipeManager()
    rm.add_quantization_config(regex=".*",operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,op_config=qtyping.OpQuantizationConfig(weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16,dtype=qtyping.TensorDataType.FLOAT),compute_precision=qtyping.ComputePrecision.FLOAT),algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16): os.remove(fp16)
    q=quantizer.Quantizer(float_model=fp32); q.load_quantization_recipe(rm.get_quantization_recipe()); q.quantize().export_model(fp16); return fp16

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
    m=build(); x=torch.rand(1,3,SIZE,SIZE)*2-1
    with torch.no_grad(): ref=m(x)
    print(f"forward: out {tuple(ref.shape)} range [{ref.min():.2f},{ref.max():.2f}]")
    if (sys.argv[1] if len(sys.argv)>1 else "all")=="forward": sys.exit()
    import litert_torch
    fp32=f"{HERE}/anime.tflite"
    try:
        litert_torch.convert(m,(x,)).export(fp32); it=opcheck(fp32,"anime")
        d=it.get_input_details()[0]; it.set_tensor(d["index"],x.numpy().astype(d["dtype"])); it.invoke()
        o=it.get_tensor(it.get_output_details()[0]["index"])
        print(f"tflite vs torch corr {np.corrcoef(o.ravel(),ref.numpy().ravel())[0,1]:.6f}")
        to_fp16(fp32,f"{HERE}/anime_{STYLE}_fp16.tflite"); opcheck(f"{HERE}/anime_{STYLE}_fp16.tflite","anime_fp16")
    except Exception as e: print("CONVERT FAIL:",repr(e)[:300])
