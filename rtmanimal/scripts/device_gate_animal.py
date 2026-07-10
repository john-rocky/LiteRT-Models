import build_rtm_animal as B, numpy as np, torch, os
from PIL import Image, ImageDraw
HERE=os.path.dirname(os.path.abspath(__file__)); S=256
MEAN=np.array([123.675,116.28,103.53],np.float32); STD=np.array([58.395,57.12,57.375],np.float32)
EDGES=[(0,1),(0,2),(1,2),(2,3),(3,4),(3,5),(5,6),(6,7),(3,8),(8,9),(9,10),(4,11),(11,12),(12,13),(4,14),(14,15),(15,16)]
im=Image.open(f"{HERE}/animal.jpg").convert("RGB"); s=min(im.size)
im=im.crop(((im.width-s)//2,(im.height-s)//2,(im.width+s)//2,(im.height+s)//2)).resize((S,S),Image.BILINEAR)
arr=np.asarray(im).astype(np.float32)
x=((arr-MEAN)/STD).transpose(2,0,1)[None].copy()
from mmpose.apis import init_model
m=init_model(B.cfg,B.ckpt,device="cpu").eval(); w=B.Wrap(m).eval()
with torch.no_grad(): sx,sy=w(torch.from_numpy(x)); sx,sy=sx.numpy()[0],sy.numpy()[0]
np.save(f"{HERE}/an_sx.npy",sx); np.save(f"{HERE}/an_sy.npy",sy); x.tofile(f"{HERE}/an_input.bin")
def decode(sx,sy):
    xb=sx.argmax(1)/2.0; yb=sy.argmax(1)/2.0; conf=(sx.max(1)+sy.max(1))/2
    return np.stack([xb,yb,conf],1)
def draw(kp,name):
    d=im.copy(); dr=ImageDraw.Draw(d)
    for a,b in EDGES:
        if kp[a,2]>0.3 and kp[b,2]>0.3: dr.line([kp[a,0],kp[a,1],kp[b,0],kp[b,1]],fill=(0,230,0),width=4)
    for i in range(17):
        if kp[i,2]>0.3: dr.ellipse([kp[i,0]-4,kp[i,1]-4,kp[i,0]+4,kp[i,1]+4],fill=(255,40,40))
    d.save(name)
kp=decode(sx,sy); draw(kp,f"{HERE}/an_torch.png"); im.save(f"{HERE}/an_in.png")
print(f"input {x.shape}; torch visible {(kp[:,2]>0.3).sum()}/17")
from ai_edge_litert.interpreter import Interpreter
it=Interpreter(model_path=f"{HERE}/rtm_animal_fp16.tflite"); it.allocate_tensors()
d=it.get_input_details()[0]; it.set_tensor(d["index"],x.astype(d["dtype"])); it.invoke()
od=it.get_output_details(); o0=it.get_tensor(od[0]["index"])[0]; o1=it.get_tensor(od[1]["index"])[0]
# match by which equals sx
csx=np.corrcoef(o0.ravel(),sx.ravel())[0,1]
ox,oy=(o0,o1) if csx>0.9 else (o1,o0)
print(f"desktop-fp16 corr sx {np.corrcoef(ox.ravel(),sx.ravel())[0,1]:.5f} sy {np.corrcoef(oy.ravel(),sy.ravel())[0,1]:.5f}")
