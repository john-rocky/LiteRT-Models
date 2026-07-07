import sys, os, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
sys.path.insert(0, os.path.expanduser("~/Downloads/meeting/sinet-src"))
_o=F.interpolate; F.interpolate=lambda *a,**k:_o(*a,**{**k,**({'align_corners':False} if k.get('align_corners') is True else {})})
from lib.Network_Res2Net_GRA_NCD import Network
# ZeroPadMaxPool for res2net stem
class ZeroPadMaxPool(nn.Module):
    def forward(s,x): x=F.pad(x,(1,1,1,1),value=0.0); return F.max_pool2d(x,3,stride=2,padding=0)
net = Network(channel=32, imagenet_pretrained=False).eval()
sd=torch.load(os.path.expanduser("~/Downloads/meeting/sinet-src/Net_epoch_best.pth"), map_location="cpu")
sd=sd.get('model_state_dict', sd.get('state_dict', sd)) if isinstance(sd,dict) else sd
sd={k[7:] if k.startswith('module.') else k:v for k,v in sd.items()}
print("load:", net.load_state_dict(sd, strict=False))
for name,m in list(net.named_modules()):
    if isinstance(m, nn.MaxPool2d):
        p=net; *path,last=name.split('.')
        for q in path: p=getattr(p,q)
        setattr(p,last,ZeroPadMaxPool())
class Wrap(nn.Module):
    def __init__(s,n): super().__init__(); s.n=n
    def forward(s,x):
        out=s.n(x)                 # (S_g, S_5, S_4, S_3)
        return torch.sigmoid(out[-1])   # final camouflage map [1,1,352,352]
w=Wrap(net).eval()
dummy=torch.rand(1,3,352,352)
with torch.no_grad(): o=w(dummy)
print("out:", tuple(o.shape), "range", round(float(o.min()),3), round(float(o.max()),3))
np.save("ref_in.npy", dummy.numpy()); np.save("ref_out.npy", o.numpy())
import litert_torch
litert_torch.convert(w,(dummy,)).export("sinet.tflite")
print("saved %.1f MB"%(os.path.getsize("sinet.tflite")/1e6))
