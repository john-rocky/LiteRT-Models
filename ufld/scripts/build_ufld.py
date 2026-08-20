import sys, os, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/Ultra-Fast-Lane-Detection")
from model.model import parsingNet
from huggingface_hub import hf_hub_download

# Fine-tune overrides (defaults reproduce the official CULane ship exactly):
#   UFLD_CKPT=w.pth      your own UFLD trainer checkpoint ({'model': ...} or raw;
#                        'module.' prefixes stripped)
#   UFLD_GRIDING=200     griding_num (cls dim becomes griding+1)
#   UFLD_ROWS=18         row anchors (18 CULane / 56 TuSimple)
#   UFLD_LANES=4         max lanes
#   UFLD_BACKBONE=18     resnet backbone variant
GRID = int(os.environ.get("UFLD_GRIDING", "200"))
ROWS = int(os.environ.get("UFLD_ROWS", "18"))
LANES = int(os.environ.get("UFLD_LANES", "4"))
# CULane default: griding_num=200 -> cls_dim=(201,18,4)
net = parsingNet(pretrained=False, backbone=os.environ.get("UFLD_BACKBONE", "18"),
                 cls_dim=(GRID + 1, ROWS, LANES), use_aux=False).eval()
pth = os.environ.get("UFLD_CKPT", "culane_official.pth")
sd = torch.load(pth, map_location="cpu")
sd = sd.get('model', sd)
sd = { k[len('module.'):] if k.startswith('module.') else k: v for k,v in sd.items() }
missing, unexpected = net.load_state_dict(sd, strict=False)
print("missing:", len(missing), "unexpected:", len(unexpected))

class ZeroPadMaxPool(nn.Module):
    def forward(self, x):
        x = F.pad(x,(1,1,1,1),value=0.0); return F.max_pool2d(x,3,stride=2,padding=0)
for name,m in list(net.named_modules()):
    if isinstance(m, nn.MaxPool2d):
        p=net; *path,last=name.split('.')
        for q in path: p=getattr(p,q)
        setattr(p,last,ZeroPadMaxPool())

dummy = torch.randn(1,3,288,800)
with torch.no_grad(): o = net(dummy)
print("out:", tuple(o.shape))   # expect [1,GRID+1,ROWS,LANES] (CULane: [1,201,18,4])
np.save("ref_in.npy", dummy.numpy()); np.save("ref_out.npy", o.numpy())
import litert_torch
litert_torch.convert(net,(dummy,)).export("ufld.tflite")
print("saved %.1f MB"%(os.path.getsize("ufld.tflite")/1e6))
