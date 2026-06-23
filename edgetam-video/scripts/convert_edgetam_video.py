#!/usr/bin/env python3
"""
EdgeTAM VIDEO tracking -> LiteRT CompiledModel GPU: export the 4 stateless per-frame sub-graphs.
The rolling memory bank (7 spatial frames + up to 16 object pointers) + orchestration live in Kotlin.
All GPU-clean patches verified corr 1.0 vs PyTorch (see video_spike/*.py); full pipeline IoU 1.0.

Graphs:
  encode    frame(1,3,1024,1024) -> pix_feat_raw(1,256,64,64), hi0(1,32,256,256), hi1(1,64,128,128)
  memcond   memory_attention: pix_feat_raw + fixed-7 memory bank -> pix_feat(1,256,64,64)
  decode    mask_decoder: pix_feat + hi0 + hi1 + sparse -> masks(1,3,256,256), iou(1,3),
            object_pointers(1,3,256), object_score(1,1)   (Kotlin: argmax iou, gate, pick pointer)
  memorize  memory_encoder + spatial_perceiver: pix_feat_raw + mask_for_mem(1,1,1024,1024)
            -> spatial_mem(1,512,64), spatial_mem_pos(1,512,64)
"""
import sys, types, os, collections, numpy as np
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
import torch, torch.nn as nn, torch.nn.functional as F, litert_torch
import timm.layers.squeeze_excite as SQ
import transformers.models.edgetam_video.modeling_edgetam_video as MV
from transformers import EdgeTamVideoModel
from ai_edge_litert.interpreter import Interpreter
OUT=os.path.expanduser("~/Downloads/litert-upstream/edgetam/video_spike/graphs"); os.makedirs(OUT,exist_ok=True)
GPU_BAD={'GATHER_ND','GATHER','SELECT_V2','SELECT','PACK','UNPACK','SPLIT','SPLIT_V','CAST','TOPK_V2','BROADCAST_TO','WHILE','TRANSPOSE_CONV'}
HD=256; NMM=7; NPTR=16*4   # fixed bank: 7 spatial frames, 16 obj-ptr frames x4 tokens
SPATIAL=512                # perceiver tokens per frame
MEMCH=64                   # memory channel dim

# ---------- GPU-clean patches (all verified corr 1.0) ----------
def se_fwd(self,x):
    xse=x.mean(3,keepdim=True).mean(2,keepdim=True)
    if self.add_maxpool: xse=0.5*xse+0.5*x.amax((2,3),keepdim=True)
    xse=self.fc1(xse); xse=self.act(self.bn(xse)); xse=self.fc2(xse); return x*self.gate(xse)
SQ.SEModule.forward=se_fwd

PERM=torch.tensor([2*i for i in range(HD//2)]+[2*i+1 for i in range(HD//2)])
def rotate_half(x):
    d=x.shape[-1]//2; return torch.cat([-x[...,d:],x[...,:d]],dim=-1)
def _half(c):
    cd=c[...,0::2]; return torch.cat([cd,cd],-1)
def _4d(c): return c[None,None]
def self_rope_fixed(q,k,cos,sin):
    ch,sh=_4d(_half(cos)),_4d(_half(sin)); return q*ch+rotate_half(q)*sh, k*ch+rotate_half(k)*sh
def cross_rope_fixed(q,k,cos,sin,cos_k,sin_k,num_k_exclude_rope=0,repeat_freqs_k=1):
    ch,sh=_4d(_half(cos)),_4d(_half(sin)); q_e=q*ch+rotate_half(q)*sh
    Lk=k.shape[-2]; Lr=Lk-num_k_exclude_rope; k_rope=k[...,:Lr,:]; k_exc=k[...,Lr:,:]
    G=repeat_freqs_k; tpg=Lr//G; sp=cos_k.shape[-2]; tmp=tpg-sp
    ckh,skh=_half(cos_k),_half(sin_k)
    cos_grp=torch.cat([torch.ones(tmp,HD),ckh],0); sin_grp=torch.cat([torch.zeros(tmp,HD),skh],0)
    cos_kf=_4d(torch.cat([cos_grp]*G,0)); sin_kf=_4d(torch.cat([sin_grp]*G,0))
    k_e=k_rope*cos_kf+rotate_half(k_rope)*sin_kf
    return q_e, torch.cat([k_e,k_exc],-2)
MV.apply_rotary_pos_emb_2d_self_attn=self_rope_fixed
MV.apply_rotary_pos_emb_2d_cross_attn=cross_rope_fixed

# Masked cross-attention: thread an additive key_mask (1,1,1,Lk) so a FIXED-size memory bank with
# padded (repeat/zero) slots is numerically EXACT vs HF's variable-length memory (masked slots -> 0 weight).
def _cross_fwd_masked(self, query, key, value, position_embeddings, position_embeddings_k,
                      num_k_exclude_rope=0, rope_k_repeat=0, key_mask=None, **kw):
    bs,pb=query.shape[:2]; ns=(bs*pb,-1,self.num_attention_heads,self.head_dim)
    q=self.q_proj(query).view(*ns).transpose(1,2); k=self.k_proj(key).view(*ns).transpose(1,2); v=self.v_proj(value).view(*ns).transpose(1,2)
    cos,sin=position_embeddings; cos_k,sin_k=position_embeddings_k
    q,k=cross_rope_fixed(q,k,cos,sin,cos_k,sin_k,repeat_freqs_k=rope_k_repeat,num_k_exclude_rope=num_k_exclude_rope)
    aw=torch.matmul(q,k.transpose(2,3))*self.scaling
    if key_mask is not None: aw=aw+key_mask
    aw=torch.softmax(aw,dim=-1); o=torch.matmul(aw,v).transpose(1,2).reshape(bs,pb,-1,self.num_attention_heads*self.head_dim)
    return self.o_proj(o), None
MV.EdgeTamVideoRoPECrossAttention.forward=_cross_fwd_masked
def _layer_fwd(self, queries, keys, key_point_embedding, rope_position_embeddings,
               rope_position_embeddings_k=None, num_k_exclude_rope=0, rope_k_repeat=0, key_mask=None):
    q=self.layer_norm1(queries); a,_=self.self_attn(query=q,key=q,value=q,position_embeddings=rope_position_embeddings); queries=queries+a
    q=self.layer_norm2(queries); a,_=self.cross_attn_image(query=q,key=keys+key_point_embedding,value=keys,
        position_embeddings=rope_position_embeddings,position_embeddings_k=rope_position_embeddings_k,
        num_k_exclude_rope=num_k_exclude_rope,rope_k_repeat=rope_k_repeat,key_mask=key_mask); queries=queries+a
    q=self.layer_norm3(queries); return queries+self.mlp(q)
MV.EdgeTamVideoMemoryAttentionLayer.forward=_layer_fwd
def _memattn_fwd(self, current_vision_features, memory, current_vision_position_embeddings=None,
                 memory_posision_embeddings=None, num_object_pointer_tokens=0, num_spatial_memory_tokens=-1, key_mask=None):
    output=current_vision_features
    if current_vision_position_embeddings is not None: output=output+0.1*current_vision_position_embeddings
    output=output.transpose(0,1); memory=memory.transpose(0,1).unsqueeze(1); memory_posision_embeddings=memory_posision_embeddings.transpose(0,1).unsqueeze(1)
    rpe=self.rotary_emb(); rpek=self.rotary_emb_k()
    for layer in self.layers:
        output=layer(queries=output.unsqueeze(1) if output.ndim==3 else output, keys=memory, key_point_embedding=memory_posision_embeddings,
                     rope_position_embeddings=rpe, rope_position_embeddings_k=rpek, num_k_exclude_rope=num_object_pointer_tokens,
                     rope_k_repeat=num_spatial_memory_tokens, key_mask=key_mask)
    return self.layer_norm(output).transpose(0,1)
MV.EdgeTamVideoMemoryAttention.forward=_memattn_fwd

_wc={}
def _s2d_w(C,w):
    if (C,w) not in _wc:
        W=torch.zeros(C*w*w,1,w,w)
        for c in range(C):
            for i in range(w):
                for j in range(w): W[c*w*w+i*w+j,0,i,j]=1.0
        _wc[(C,w)]=W
    return _wc[(C,w)]
def wp_conv(hidden_state,window_size):
    B,H,W,C=hidden_state.shape; w=window_size; Hg,Wg=H//w,W//w
    x=hidden_state.permute(0,3,1,2)
    sd=F.conv2d(x,_s2d_w(C,w),stride=w,groups=C)
    return sd.view(C,w*w,Hg,Wg).permute(2,3,1,0).reshape(Hg*Wg,w,w,C),(H,W)
MV.window_partition=wp_conv

# Perceiver attention: softmax over a single key (2d self-attn = 1 latent/window) -> exp(0)/exp(0)=x/x
# (DIV identical, ML Drift rejects). With 1 key the weight is identically 1.0 -> attn_output == value.
def _perc_attn_fwd(self, query, key, value, positional_encoding=None, **kw):
    bsz, sq = query.shape[:2]; skv = key.shape[1]
    q=self.q_proj(query).view(bsz,sq,self.num_attention_heads,self.head_dim).transpose(1,2)
    k=self.k_proj(key).view(bsz,skv,self.num_attention_heads,self.head_dim).transpose(1,2)
    v=self.v_proj(value).view(bsz,skv,self.num_attention_heads,self.head_dim).transpose(1,2)
    if positional_encoding is not None:
        p=positional_encoding.view(bsz,skv,self.num_attention_heads,self.head_dim).transpose(1,2); k=k+p; v=v+p
    if skv==1:
        attn=v.transpose(1,2).contiguous().view(bsz,sq,self.inner_dim)
    else:
        aw=torch.softmax(torch.matmul(q,k.transpose(2,3))*self.scaling, dim=-1)
        attn=torch.matmul(aw,v).transpose(1,2).contiguous().view(bsz,sq,self.inner_dim)
    return self.o_proj(attn)
MV.EdgeTamVideoPerceiverAttention.forward=_perc_attn_fwd

class ZeroStuffConvT(nn.Module):
    def __init__(s, ct, in_hw):
        super().__init__(); s.st=ct.stride[0]; s.k=ct.kernel_size[0]; s.oh=in_hw*s.st
        s.register_buffer("w", ct.weight.flip(2,3).transpose(0,1).contiguous()); s.bias=ct.bias
        mask=torch.zeros(1,1,s.oh,s.oh); mask[:,:,::s.st,::s.st]=1.0; s.register_buffer("mask",mask)
    def forward(s,x):
        return F.conv2d(F.interpolate(x,size=(s.oh,s.oh),mode="nearest")*s.mask, s.w, s.bias, padding=s.k-1)[:,:,:s.oh,:s.oh]

m=EdgeTamVideoModel.from_pretrained("yonigozlan/EdgeTAM-hf").eval()
md=m.mask_decoder
md.upscale_conv1=ZeroStuffConvT(md.upscale_conv1,64); md.upscale_conv2=ZeroStuffConvT(md.upscale_conv2,128)

# Bake the memory_encoder sine position-encoding (constant for fixed 64x64; runtime form emits GATHER_ND/>4D)
class ConstPos(nn.Module):
    def __init__(s, const): super().__init__(); s.register_buffer("c", const)
    def forward(s, *a, **k): return s.c
with torch.no_grad():
    _pe=m.memory_encoder.position_encoding((1,64,64,64), torch.device("cpu"), torch.float32).detach().clone()
    _pp=m.spatial_perceiver.positional_encoding((1,64,16,16), torch.device("cpu"), torch.float32).detach().clone()
m.memory_encoder.position_encoding=ConstPos(_pe)
m.spatial_perceiver.positional_encoding=ConstPos(_pp)

# Taint the perceiver latents -> runtime, so layer_norm_latents (layer 0) isn't a MEAN over a const
# parameter (ML Drift rejects MEAN with 2 const inputs). 1e-9*mean is non-folding and numerically negligible.
import types as _types, math as _math
def _fwd1d(self, hidden_states, positional_encoding=None):
    b=hidden_states.shape[0]; taint=1e-9*hidden_states.mean()
    latents=self.latents_1d.unsqueeze(0).expand(b,-1,-1)+taint
    ff=hidden_states.permute(0,2,3,1).flatten(1,2)
    pf=positional_encoding.permute(0,2,3,1).flatten(1,2) if positional_encoding is not None else None
    for layer in self.layers: latents=layer(latents,ff,pf)
    latents=self.layer_norm(latents)
    return latents, (torch.zeros_like(latents) if positional_encoding is not None else None)
def _fwd2d(self, hidden_states):
    b,ch,h,w=hidden_states.shape; taint=1e-9*hidden_states.mean()
    latents=self.latents_2d.unsqueeze(0).expand(b,-1,-1).view(-1,1,ch)+taint
    npd=int(_math.sqrt(self.num_latents_2d)); ws=h//npd
    wf,_=MV.window_partition(hidden_states.permute(0,2,3,1),ws); wf=wf.flatten(1,2)
    for layer in self.layers: latents=layer(latents,wf,positional_encoding=None)
    latents=latents.view(b,npd,npd,ch).permute(0,3,1,2)
    pe2=self.positional_encoding(latents.shape,latents.device,latents.dtype).to(dtype=hidden_states.dtype)
    pe2=pe2.permute(0,2,3,1).flatten(1,2)
    latents=latents.permute(0,2,3,1).flatten(1,2); latents=self.layer_norm(latents)
    return latents, pe2
m.spatial_perceiver._forward_1d=_types.MethodType(_fwd1d, m.spatial_perceiver)
m.spatial_perceiver._forward_2d=_types.MethodType(_fwd2d, m.spatial_perceiver)

for layer in m.memory_attention.layers:
    for attn in (layer.self_attn, layer.cross_attn_image):
        for proj in (attn.q_proj, attn.k_proj):
            proj.weight.data=proj.weight.data[PERM].contiguous(); proj.bias.data=proj.bias.data[PERM].contiguous()

# video mask decoder: 4D, all <=4D, returns 3 masks + iou + sam_tokens + obj_score
def _dec4d(self, image_embeddings, image_positional_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings,
           multimask_output, high_resolution_features, attention_similarity=None, target_embedding=None, **kw):
    bs,nc,h,w=image_embeddings.shape; pb=sparse_prompt_embeddings.shape[1]
    ot=torch.cat([self.obj_score_token.weight,self.iou_token.weight,self.mask_tokens.weight],0).repeat(bs,pb,1,1)
    pe=torch.cat((ot,sparse_prompt_embeddings),2).to(self.iou_token.weight.dtype)
    ie=(image_embeddings+dense_prompt_embeddings).repeat_interleave(pb,0); ipe=image_positional_embeddings.repeat_interleave(pb,0)
    pe,ie=self.transformer(point_embeddings=pe,image_embeddings=ie,image_positional_embeddings=ipe,attention_similarity=attention_similarity,target_embedding=target_embedding,**kw)
    it=pe[:,:,1,:]; mt=pe[:,:,2:(2+self.num_mask_tokens),:]; ie=ie.transpose(2,3).view(bs*pb,nc,h,w)
    f0,f1=high_resolution_features; f0=f0.repeat_interleave(pb,0); f1=f1.repeat_interleave(pb,0)
    ue=self.activation(self.upscale_layer_norm(self.upscale_conv1(ie)+f1)); ue=self.activation(self.upscale_conv2(ue)+f0)
    hyper=torch.stack([self.output_hypernetworks_mlps[i](mt[:,:,i,:]) for i in range(self.num_mask_tokens)],2)
    _,nc2,h2,w2=ue.shape; B=bs*pb
    masks=(hyper.view(B,self.num_mask_tokens,nc2)@ue.view(B,nc2,h2*w2)).view(B,self.num_mask_tokens,h2,w2)
    iou=self.iou_prediction_head(it); obj=self.pred_obj_score_head(pe[:,:,0,:])
    return masks[:,1:], iou[:,:,1:], mt[:,:,1:], obj
MV.EdgeTamVideoMaskDecoder.forward=_dec4d

# constants
img_pos=m.get_image_wide_positional_embeddings().detach()                       # (1,256,64,64)
no_mem=m.no_memory_embedding.detach().reshape(1,-1,1,1)                          # (1,256,1,1)
# lowres vision pos (sine, constant): grab from a dummy vision_encoder run
with torch.no_grad():
    ve=m.vision_encoder(torch.randn(1,3,1024,1024),return_dict=True)
    low_pos=ve.fpn_position_encoding[-1].detach()                               # whatever the model uses
print("low_pos shape", tuple(low_pos.shape))

# ---------- graph modules (single flat concatenated I/O to avoid CompiledModel index ambiguity) ----------
IE=256*64*64; H0=32*256*256; H1=64*128*128; MEM_=NMM*512+NPTR; MC=MEM_*MEMCH
class Encode(nn.Module):
    def __init__(s): super().__init__(); s.ve=m.vision_encoder; s.cs0=md.conv_s0; s.cs1=md.conv_s1
    def forward(s,x):
        fpn=s.ve(x,return_dict=True).fpn_hidden_states
        return torch.cat([fpn[2].reshape(-1), s.cs0(fpn[0]).reshape(-1), s.cs1(fpn[1]).reshape(-1)])[None]  # [1, IE+H0+H1]

class MemCond(nn.Module):  # in: [pix_raw(IE) | memory(MC) | mem_pos(MC) | key_mask(MEM_)] -> pix_feat(IE)
    def __init__(s):
        super().__init__(); s.a=m.memory_attention
        s.register_buffer("pos", low_pos.reshape(1,256,-1).permute(2,0,1).contiguous())
    def forward(s, flat):
        f=flat[0]
        pix=f[:IE].reshape(1,256,64,64)
        memory=f[IE:IE+MC].reshape(MEM_,1,MEMCH)
        mpos=f[IE+MC:IE+2*MC].reshape(MEM_,1,MEMCH)
        km=f[IE+2*MC:].reshape(1,1,1,MEM_)
        cvf=pix.reshape(1,256,4096).permute(2,0,1)
        out=s.a(current_vision_features=cvf, memory=memory, current_vision_position_embeddings=s.pos,
                memory_posision_embeddings=mpos, num_object_pointer_tokens=NPTR, num_spatial_memory_tokens=NMM, key_mask=km)
        return out.squeeze(1).transpose(1,2).reshape(1,IE)

class Decode(nn.Module):  # in: [pix_feat(IE) | hi0(H0) | hi1(H1) | sparse(512)] -> [masks(196608)|iou(3)|objptr(768)|objscore(1)]
    def __init__(s):
        super().__init__(); s.d=md; s.optr=m.object_pointer_proj
        s.register_buffer("ipe", img_pos.clone())
        s.register_buffer("dn", m.prompt_encoder.no_mask_embed.weight.reshape(1,-1,1,1).expand(1,256,64,64).contiguous())
    def forward(s, flat):
        f=flat[0]
        pix=f[:IE].reshape(1,256,64,64); h0=f[IE:IE+H0].reshape(1,32,256,256)
        h1=f[IE+H0:IE+H0+H1].reshape(1,64,128,128); sparse=f[IE+H0+H1:].reshape(1,1,2,256)
        masks,iou,mt,obj=s.d(pix,s.ipe,sparse,s.dn,multimask_output=True,high_resolution_features=[h0,h1])
        ptr=s.optr(mt)
        return torch.cat([masks[0].reshape(-1), iou[0].reshape(-1), ptr[0,0].reshape(-1), obj[0].reshape(-1)])[None]

class Memorize(nn.Module):  # in: [pix_raw(IE) | mask_for_mem(IE)] -> [spatial_mem(32768)|spatial_pos(32768)]
    def __init__(s): super().__init__(); s.e=m.memory_encoder; s.p=m.spatial_perceiver
    def forward(s, flat):
        f=flat[0]; pix=f[:IE].reshape(1,256,64,64); mfm=f[IE:].reshape(1,1,1024,1024)
        mf,pos=s.e(pix, mfm); sm,sp=s.p(mf, pos)
        return torch.cat([sm.reshape(-1), sp.reshape(-1)])[None]

from ai_edge_quantizer import quantizer
RECIPE=[{"regex":".*","operation":"*","algorithm_key":"float_casting","op_config":{"weight_tensor_config":{"num_bits":16,"dtype":"FLOAT"}}}]
def conv_check(mod, ins, name):
    p=f"{OUT}/{name}.tflite"
    if os.path.exists(p): os.remove(p)
    try:
        with torch.no_grad(): mod(*ins)
        litert_torch.convert(mod, tuple(t.detach().clone() for t in ins)).export(p)
    except Exception as e:
        print(f"{name}: FAIL {repr(e)[:200]}"); return
    it=Interpreter(model_path=p); it.allocate_tensors()
    ops=collections.Counter(d.get('op_name','?') for d in it._get_ops_details())
    bad={k:v for k,v in ops.items() if k in GPU_BAD}
    over=sum(1 for d in it.get_tensor_details() if len(d.get('shape',[]))>4)
    fp16=f"{OUT}/{name}_fp16.tflite"
    if os.path.exists(fp16): os.remove(fp16)
    q=quantizer.Quantizer(p); q.load_quantization_recipe(RECIPE); q.quantize().export_model(fp16)
    print(f"{name}: GPU_BAD={bad or 'NONE'}  >4D={over}  fp32={os.path.getsize(p)/1e6:.1f}MB fp16={os.path.getsize(fp16)/1e6:.1f}MB")

# ---- Kotlin constants ----
import numpy as _np
def _save(name, arr): arr.astype(_np.float32).tofile(f"{OUT}/{name}.bin")
_save("no_memory", m.no_memory_embedding.detach().numpy())                              # 256
_save("mtpe", m.memory_temporal_positional_encoding.detach().numpy())                   # 7*64=448
_save("no_objptr", m.no_object_pointer.detach().numpy())                                # 256
with torch.no_grad():
    _ts,_=m.prompt_encoder(input_points=torch.zeros(1,1,1,2),input_labels=-torch.ones(1,1,1,dtype=torch.int32),input_boxes=None,input_masks=None)
_save("track_sparse", _ts.numpy())                                                      # 1*1*2*256=512
_pe=m.prompt_encoder
_G=_pe.shared_embedding.positional_embedding
_save("video_prompt", _np.concatenate([_G.detach().numpy().flatten(), _pe.point_embed.weight[1].detach().numpy(), _pe.not_a_point_embed.weight[0].detach().numpy()]))  # 768
print("constants saved: no_memory(256) mtpe(448) no_objptr(256) track_sparse(512) video_prompt(768)")

MEM=NMM*SPATIAL+NPTR   # 7*512 + 64 = 3648
conv_check(Encode().eval(), (torch.randn(1,3,1024,1024),), "encode")
conv_check(MemCond().eval(), (torch.randn(1, IE+2*MC+MEM),), "memcond")
conv_check(Decode().eval(), (torch.randn(1, IE+H0+H1+512),), "decode")
conv_check(Memorize().eval(), (torch.randn(1, 2*IE),), "memorize")
print("FLAT IO  enc_in=3x1024x1024  enc_out=%d | memcond_in=%d out=%d | decode_in=%d out=%d | memorize_in=%d out=%d"%(
      IE+H0+H1, IE+2*MC+MEM, IE, IE+H0+H1+512, 196608+3+768+1, 2*IE, 65536))
