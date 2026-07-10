import os, numpy as np, torch
from PIL import Image
import build_rtm_multi as B
from mmpose.apis import init_model
HERE=os.path.dirname(os.path.abspath(__file__))
H=int(os.environ["RTM_H"]); W=int(os.environ["RTM_W"]); NAME=os.environ["RTM_NAME"]; IMG=os.environ["RTM_IMG"]
MEAN=np.array([123.675,116.28,103.53],np.float32); STD=np.array([58.395,57.12,57.375],np.float32)
im=Image.open(IMG).convert("RGB"); ar=W/H; w,h=im.size
if w/h>ar: nw=int(h*ar); im=im.crop(((w-nw)//2,0,(w+nw)//2,h))
else: nh=int(w/ar); im=im.crop((0,(h-nh)//2,w,(h+nh)//2))
im=im.resize((W,H),Image.BILINEAR)
img=((np.asarray(im).astype(np.float32)-MEAN)/STD).transpose(2,0,1)[None].copy()
m=init_model(B.cfg,B.ckpt,device="cpu").eval(); B.patch_pixelshuffle(m,H,W); wn=B.Wrap(m).eval()
with torch.no_grad(): o=wn(torch.from_numpy(img)); sx,sy=o[0].numpy()[0],o[1].numpy()[0]
np.save(f"{HERE}/{NAME}_sx.npy",sx); np.save(f"{HERE}/{NAME}_sy.npy",sy); img.tofile(f"{HERE}/{NAME}_input.bin")
print(f"{NAME}: torch out sx{sx.shape} sy{sy.shape}")
