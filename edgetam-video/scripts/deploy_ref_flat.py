#!/usr/bin/env python3
"""Flat-I/O deployment reference = exact Kotlin spec. 4 fp16 graphs, single flat in/out each."""
import sys, types, numpy as np
class _Dummy:
    def __getattr__(s,n):
        if n.startswith("__") and n.endswith("__"): raise AttributeError(n)
        return _Dummy()
    def __call__(s,*a,**k): return _Dummy()
class _Leaf(types.ModuleType):
    def __getattr__(s,n):
        if n.startswith("__") and n.endswith("__"): raise AttributeError(n)
        return _Dummy()
for nm in ["scipy.sparse.linalg._propack","scipy.optimize._cobyla","scipy.optimize._slsqp","scipy.optimize._minpack",
           "scipy.optimize._lbfgsb","scipy.optimize._zeros","scipy.optimize._highs","scipy.optimize._direct",
           "scipy.optimize._trlib","scipy.optimize._group_columns","scipy.optimize._bglu_dense"]:
    sys.modules[nm]=_Leaf(nm)
import torch, torch.nn.functional as F
from transformers import EdgeTamVideoModel, EdgeTamVideoInferenceSession
import transformers.models.edgetam_video.modeling_edgetam_video as MV
from ai_edge_litert.interpreter import Interpreter
GD="/Users/majimadaisuke/Downloads/litert-upstream/edgetam/video_spike/graphs"
IE=1048576; H0=2097152; H1=1048576; NMM=7; MAXP=16; MEMCH=64; MEM=NMM*512+MAXP*4; MC=MEM*MEMCH
NO_OBJ=-1024.0; SCALE=20.0; BIAS=-10.0
def load(n): it=Interpreter(model_path=f"{GD}/{n}_fp16.tflite"); it.allocate_tensors(); return it
enc,mc,dec,mem=load("encode"),load("memcond"),load("decode"),load("memorize")
def run1(it, flat):  # single input, single output
    i=it.get_input_details()[0]; o=it.get_output_details()[0]
    it.set_tensor(i['index'], flat.reshape(i['shape']).astype(np.float32)); it.invoke()
    return it.get_tensor(o['index']).flatten()
m=EdgeTamVideoModel.from_pretrained("yonigozlan/EdgeTAM-hf").eval(); pe=m.prompt_encoder
no_mem=m.no_memory_embedding.detach().numpy().reshape(256)        # added per-channel
no_objptr=m.no_object_pointer.detach().numpy().reshape(256)
mtpe=m.memory_temporal_positional_encoding.detach().numpy().reshape(7,64)
def get_sparse(p,l):
    with torch.no_grad(): sp,_=pe(input_points=p,input_labels=l,input_boxes=None,input_masks=None)
    return sp.numpy().reshape(512)
TRACK_SPARSE=get_sparse(torch.zeros(1,1,1,2),-torch.ones(1,1,1,dtype=torch.int32))
def sine_pe(off,dim): return MV.get_1d_sine_pe(torch.tensor([off/15.0]),dim=dim).numpy().reshape(-1)

torch.manual_seed(0); T=6; vid=torch.randn(T,3,1024,1024)
def hf_run():
    sess=EdgeTamVideoInferenceSession(video=vid,video_height=1024,video_width=1024,dtype=torch.float32)
    oi=sess.obj_id_to_idx(1); sess.add_point_inputs(oi,0,{"point_coords":torch.tensor([[[[512.,512.]]]]),"point_labels":torch.tensor([[[1]]])}); sess.obj_with_new_inputs=[1]
    out=[]
    with torch.no_grad():
        for fi in range(T): out.append(m(inference_session=sess,frame_idx=fi).pred_masks.clone().numpy().reshape(256,256))
    return out
hf=hf_run()

spatial_bank=[]; objptr_bank=[]
def assemble(fi):
    real=[(spatial_bank[0][1], spatial_bank[0][2]+np.tile(mtpe[6],512).reshape(512,64))]
    for off in range(NMM-1,0,-1):
        pf=fi-off
        if pf==spatial_bank[0][0]: continue
        hit=[b for b in spatial_bank if b[0]==pf]
        if hit: real.append((hit[0][1], hit[0][2]+np.tile(mtpe[off-1],512).reshape(512,64)))
    ptrs=sorted(objptr_bank,key=lambda b:-b[0])[:MAXP]; rp=[]
    for frm,ptr in ptrs:
        pp=sine_pe(fi-frm,MEMCH); sub=ptr.reshape(4,64)
        for t in range(4): rp.append((sub[t],pp))
    spat=np.zeros((NMM*512,64),np.float32); spatp=np.zeros_like(spat)
    for i,(mm,ps) in enumerate(real): spat[i*512:(i+1)*512]=mm; spatp[i*512:(i+1)*512]=ps
    ptrm=np.zeros((MAXP*4,64),np.float32); ptrp=np.zeros_like(ptrm)
    for i,(pt,pp) in enumerate(rp): ptrm[i]=pt; ptrp[i]=pp
    memory=np.concatenate([spat,ptrm],0); mpos=np.concatenate([spatp,ptrp],0)  # (3648,64)
    km=np.full(MEM,-1e9,np.float32); km[:len(real)*512]=0; km[NMM*512:NMM*512+len(rp)]=0
    return memory.reshape(-1), mpos.reshape(-1), km

masks_out=[]
for fi in range(T):
    img=vid[fi:fi+1].numpy().reshape(-1)
    eo=run1(enc,img); pix_raw=eo[:IE]; hi0=eo[IE:IE+H0]; hi1=eo[IE+H0:]
    if fi==0:
        pix_feat=(pix_raw.reshape(256,4096)+no_mem.reshape(256,1)).reshape(-1)  # +no_memory per channel
        sparse=get_sparse(torch.tensor([[[[512.,512.]]]]),torch.tensor([[[1]]]))
    else:
        memory,mpos,km=assemble(fi)
        pix_feat=run1(mc, np.concatenate([pix_raw,memory,mpos,km]))
        sparse=TRACK_SPARSE
    do=run1(dec, np.concatenate([pix_feat,hi0,hi1,sparse]))
    masks=do[:196608].reshape(3,256,256); iou=do[196608:196611]; objptr=do[196611:196611+768].reshape(3,256); objscore=do[-1]
    best=int(np.argmax(iou)); mb=masks[best]; ptr=objptr[best]
    app=objscore>0
    mg = mb if app else np.full_like(mb,NO_OBJ); pg = ptr if app else no_objptr
    hr=F.interpolate(torch.tensor(mg)[None,None],size=(1024,1024),mode="bilinear",align_corners=False).numpy().reshape(-1)
    mfm=((hr>0).astype(np.float32))*SCALE+BIAS
    mo=run1(mem, np.concatenate([pix_raw,mfm])); sm=mo[:32768].reshape(512,64); sp=mo[32768:].reshape(512,64)
    spatial_bank.append((fi,sm,sp)); objptr_bank.append((fi,pg)); masks_out.append(mg.reshape(256,256))

print("=== FLAT deployment ref (fp16, Kotlin spec) vs HF ===")
for fi in range(T):
    r=hf[fi]; p=masks_out[fi]; iou=((r>0)&(p>0)).sum()/max(1,((r>0)|(p>0)).sum())
    print("frame %d: IoU=%.4f hf_fg=%d ref_fg=%d"%(fi,iou,(r>0).sum(),(p>0).sum()))
