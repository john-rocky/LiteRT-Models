import sys, os, torch, torch.nn as nn, numpy as np
# Fine-tune overrides (defaults reproduce the official ship exactly):
#   SILENTFACE_SRC=dir       Silent-Face-Anti-Spoofing checkout (model code)
#   SILENTFACE_CKPT=w.pth    your own MiniFASNetV2 checkpoint
#   SILENTFACE_NUM_CLASSES=N class count (default 3: live/print/replay)
#   SILENTFACE_RES=N         square input (default 80; drives conv6 kernel)
SRC = os.environ.get("SILENTFACE_SRC", os.path.expanduser("~/Downloads/meeting/silentface-src"))
sys.path.insert(0, SRC)
from src.model_lib.MiniFASNet import MiniFASNetV2
from src.utility import get_kernel
R = int(os.environ.get("SILENTFACE_RES", "80"))
kernel = get_kernel(R, R)   # (5,5) at 80
net = MiniFASNetV2(conv6_kernel=kernel, num_classes=int(os.environ.get("SILENTFACE_NUM_CLASSES", "3"))).eval()
# load weights (repo strips 'module.' prefix)
sd = torch.load(os.environ.get("SILENTFACE_CKPT",
    os.path.join(SRC, "resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth")), map_location="cpu")
if next(iter(sd)).startswith('module.'):
    sd = {k[7:]: v for k, v in sd.items()}
print("load:", net.load_state_dict(sd, strict=False))
class Wrap(nn.Module):
    def __init__(s, n): super().__init__(); s.n = n
    def forward(s, x): return torch.softmax(s.n(x), dim=1)   # [1,3] live/print/replay probs
w = Wrap(net).eval()
dummy = torch.rand(1, 3, R, R)
with torch.no_grad(): o = w(dummy)
print("out:", tuple(o.shape), "sum", round(float(o.sum()),3))
np.save("ref_in.npy", dummy.numpy()); np.save("ref_out.npy", o.numpy())
import litert_torch
litert_torch.convert(w, (dummy,)).export("silentface.tflite")
print("saved %.2f MB" % (os.path.getsize("silentface.tflite")/1e6))
