import sys, os, re, collections
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
SIZE=int(os.environ.get("ST_SIZE","256")); STYLE=os.environ.get("ST_STYLE","candy")
BANNED={"GATHER","GATHER_ND","TOPK_V2","GELU","ERF","WHERE","SELECT","SELECT_V2","BROADCAST_TO","POW","TRANSPOSE_CONV","CAST","EMBEDDING_LOOKUP","RFFT2D","FFT","STFT","COMPLEX","RFFT","IRFFT","CUMSUM","MIRROR_PAD"}
import torch.nn as _nn
class SafeInstanceNorm(_nn.Module):
    def __init__(s, ref, S):
        super().__init__(); s.S=float(S); s.eps=1e-5
        s.register_buffer("w", ref.weight.detach().clone().view(1,-1,1,1))
        s.register_buffer("b", ref.bias.detach().clone().view(1,-1,1,1))
    def forward(s, x):
        xs=x*(1.0/s.S)
        m=xs.mean(3,keepdim=True).mean(2,keepdim=True)
        d=xs-m
        var=(d*d).mean(3,keepdim=True).mean(2,keepdim=True)*(s.S*s.S)
        d=d*s.S
        return d*torch.rsqrt(var+s.eps)*s.w+s.b

def find_ckpt():
    for p in [f"{HERE}/saved_models/{STYLE}.pth", f"{HERE}/{STYLE}.pth"]:
        if os.path.exists(p): return p
    raise FileNotFoundError(STYLE)
def build():
    from transformer_net import TransformerNet
    import transformer_net as TN, torch.nn.functional as _F
    def _conv_fwd(self, x):
        p=self.reflection_pad.padding[0]
        return self.conv2d(_F.pad(x,(p,p,p,p),value=0.0))
    def _up_fwd(self, x):
        if self.upsample: x=_F.interpolate(x,mode="nearest",scale_factor=self.upsample)
        p=self.reflection_pad.padding[0]
        return self.conv2d(_F.pad(x,(p,p,p,p),value=0.0))
    TN.ConvLayer.forward=_conv_fwd; TN.UpsampleConvLayer.forward=_up_fwd
    m=TransformerNet().eval()
    sd=torch.load(find_ckpt(),map_location="cpu",weights_only=False)
    # old saved models have deprecated running stats keys for InstanceNorm
    sd={k:v for k,v in sd.items() if not re.search(r'in\d+\.running_(mean|var)|\.running_(mean|var)',k)}
    miss,unexp=m.load_state_dict(sd,strict=False)
    print(f"  loaded {STYLE}; missing {len(miss)} unexpected {len(unexp)}; params {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    import torch.nn as nn
    from PIL import Image as _I
    # 1) measure each Conv2d output magnitude on the content image
    _cm={}
    def _mk(n):
        def h(mod,i,o): _cm[n]=max(_cm.get(n,0.0),float(o.abs().max()))
        return h
    _hs=[mod.register_forward_hook(_mk(n)) for n,mod in m.named_modules() if isinstance(mod,nn.Conv2d)]
    _im=_I.open(f"{HERE}/content.jpg").convert("RGB"); _sz=min(_im.size)
    _im=_im.crop(((_im.width-_sz)//2,(_im.height-_sz)//2,(_im.width+_sz)//2,(_im.height+_sz)//2)).resize((SIZE,SIZE),_I.BILINEAR)
    _xx=np.asarray(_im).astype(np.float32).transpose(2,0,1)[None]
    with torch.no_grad(): m(torch.from_numpy(_xx))
    for h in _hs: h.remove()
    # 2) scale each Conv2d (except the final 3-channel output conv) so its output ~10 (InstanceNorm is
    #    scale-invariant -> exact; keeps the fp16 conv accumulation precise on Mali)
    _ns=0
    for n,mod in m.named_modules():
        if isinstance(mod,nn.Conv2d) and mod.out_channels!=3:
            k=_cm.get(n,10.0)/10.0
            if k>1e-6:
                mod.weight.data.div_(k)
                if mod.bias is not None: mod.bias.data.div_(k)
                _ns+=1
    # 3) SafeInstanceNorm (two-step mean + small fixed down-scale) for the reduction
    _ni=0
    for name,mod in list(m.named_modules()):
        for cn,ch in list(mod.named_children()):
            if isinstance(ch,nn.InstanceNorm2d): setattr(mod,cn,SafeInstanceNorm(ch,4.0)); _ni+=1
    print(f"  scaled {_ns} convs (~10, IN scale-invariant) + {_ni} SafeInstanceNorm(S=4)")
    return m
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
    m=build(); x=torch.rand(1,3,SIZE,SIZE)*255
    with torch.no_grad(): ref=m(x)
    print(f"forward: out {tuple(ref.shape)} range [{ref.min():.1f},{ref.max():.1f}]")
    if (sys.argv[1] if len(sys.argv)>1 else "all")=="forward": sys.exit()
    import litert_torch
    fp32=f"{HERE}/style.tflite"
    try:
        litert_torch.convert(m,(x,)).export(fp32); it=opcheck(fp32,"style")
        d=it.get_input_details()[0]; it.set_tensor(d["index"],x.numpy().astype(d["dtype"])); it.invoke()
        o=it.get_tensor(it.get_output_details()[0]["index"])
        print(f"tflite vs torch corr {np.corrcoef(o.ravel(),ref.numpy().ravel())[0,1]:.6f}")
        to_fp16(fp32,f"{HERE}/style_{STYLE}_fp16.tflite"); opcheck(f"{HERE}/style_{STYLE}_fp16.tflite","style_fp16")
    except Exception as e:
        import traceback; print("CONVERT FAIL:",repr(e)[:300])
