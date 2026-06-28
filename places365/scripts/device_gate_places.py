import build_places as B, numpy as np, torch, os, json
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__))
cls=json.load(open(HERE+"/places365_classes.json"))
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)
im=Image.open(HERE+"/scene.jpg").convert("RGB"); s=min(im.size)
im=im.crop(((im.width-s)//2,(im.height-s)//2,(im.width+s)//2,(im.height+s)//2)).resize((224,224),Image.BILINEAR)
x=(((np.asarray(im).astype(np.float32)/255-MEAN)/STD).transpose(2,0,1)[None]).copy()
m=B.build()
with torch.no_grad(): ref=m(torch.from_numpy(x)).numpy()[0]
np.save(HERE+"/pl_ref.npy",ref); x.tofile(HERE+"/pl_input.bin")
top=ref.argsort()[-5:][::-1]
print("torch top-5 scenes:", [(cls[i],round(float(ref[i]),1)) for i in top])
from ai_edge_litert.interpreter import Interpreter
it=Interpreter(model_path=HERE+"/places_fp16.tflite"); it.allocate_tensors()
d=it.get_input_details()[0]; it.set_tensor(d["index"],x.astype(d["dtype"])); it.invoke()
o=it.get_tensor(it.get_output_details()[0]["index"])[0]
print(f"desktop-fp16 vs torch corr {np.corrcoef(o,ref)[0,1]:.6f} top1-match {cls[o.argmax()]==cls[ref.argmax()]}")
