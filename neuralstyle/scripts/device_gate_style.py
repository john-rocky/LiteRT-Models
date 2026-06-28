import build_style as B, numpy as np, torch, os
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__)); S=B.SIZE; STYLE=B.STYLE
im=Image.open(f"{HERE}/content.jpg").convert("RGB"); s=min(im.size)
im=im.crop(((im.width-s)//2,(im.height-s)//2,(im.width+s)//2,(im.height+s)//2)).resize((S,S),Image.BILINEAR)
x=np.asarray(im).astype(np.float32).transpose(2,0,1)[None].copy()
m=B.build()
with torch.no_grad(): ref=m(torch.from_numpy(x)).numpy()[0]
np.save(f"{HERE}/st_ref.npy",ref); x.tofile(f"{HERE}/st_input.bin")
def save(arr,p): Image.fromarray(np.clip(arr.transpose(1,2,0),0,255).astype(np.uint8)).save(p)
save(ref,f"{HERE}/st_torch.png"); im.save(f"{HERE}/st_content.png")
print(f"input {x.shape}; torch out range [{ref.min():.0f},{ref.max():.0f}]")
from ai_edge_litert.interpreter import Interpreter
it=Interpreter(model_path=f"{HERE}/style_{STYLE}_fp16.tflite"); it.allocate_tensors()
d=it.get_input_details()[0]; it.set_tensor(d["index"],x.astype(d["dtype"])); it.invoke()
o=it.get_tensor(it.get_output_details()[0]["index"])[0]
print(f"desktop-fp16 vs torch corr {np.corrcoef(o.ravel(),ref.ravel())[0,1]:.6f}")
