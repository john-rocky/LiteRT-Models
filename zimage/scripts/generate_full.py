"""FULL on-device pipeline reference: prompt -> image using ONLY the LiteRT deploy
graphs (Qwen3 encoder int8 + S3-DiT int8 + VAE int8). Tokenizer, embed_tokens,
scheduler and host-prep (patchify / RoPE / pad-mask / unpatchify) stay as host
reference code (trivial to port to Kotlin/Swift). Every heavy compute runs through
the int8 tflite graphs. This is the runnable sample the native app ports.
"""
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/Users/majimadaisuke/Downloads/depthanything-android")
from types import SimpleNamespace
from ai_edge_litert.interpreter import Interpreter

PROMPT = "a red apple on a wooden table, studio lighting"
STEPS, SIZE, SEED = 8, 256, 1234
L_ENC, N_IMG, N_CAP = 64, 256, 32
ENC_T, DIT_T, VAE_T = "qwen_enc_L-1.tflite", "zdit_int8integer_L30.tflite", "zvae_int8_256.tflite"


def loader(path):
    it = Interpreter(model_path=path); it.allocate_tensors()
    ins = sorted(it.get_input_details(), key=lambda d: d["index"])
    out = it.get_output_details()[0]

    def run(*arrs):
        for d, a in zip(ins, arrs):
            if hasattr(a, "detach"):
                a = a.detach().cpu().numpy()
            it.set_tensor(d["index"], np.ascontiguousarray(a, dtype="<f4"))
        it.invoke()
        return it.get_tensor(out["index"])
    return run


def main():
    from diffusers import ZImagePipeline
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.models.autoencoders.vae import DecoderOutput

    print("[load] pipeline + 3 int8 graphs ...")
    pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.float32).to("cpu")
    rm = pipe.transformer
    enc_run, dit_run, vae_run = loader(ENC_T), loader(DIT_T), loader(VAE_T)
    embed = pipe.text_encoder.get_input_embeddings()

    # --- text encoder graph swap ---
    def enc_fwd(input_ids=None, attention_mask=None, output_hidden_states=None, **kw):
        seq = input_ids.shape[1]
        ids = input_ids[:, :L_ENC]
        if ids.shape[1] < L_ENC:
            ids = F.pad(ids, (0, L_ENC - ids.shape[1]), value=int(input_ids[0, -1]))
        embs = embed(ids)
        hs = torch.cat([torch.from_numpy(enc_run(embs[i:i + 1])) for i in range(embs.shape[0])], 0)
        full = torch.zeros(input_ids.shape[0], seq, hs.shape[-1])
        full[:, :L_ENC] = hs                       # valid positions (<=L_ENC) match; mask picks them
        return SimpleNamespace(hidden_states=(full, full))  # [-2] = full

    # --- DiT graph swap (per-batch, pad-token masks; from generate_litert) ---
    def dit_fwd(x, t, cap_feats, return_dict=True, **kw):
        outs = []
        for i in range(len(x)):
            adaln = rm.t_embedder(t[i:i + 1] * rm.t_scale).type_as(x[0])
            (xt, capf, x_size, x_pos, cap_pos, x_pad, cap_pad) = rm.patchify_and_embed(
                [x[i]], [cap_feats[i][:N_CAP]], 2, 1)
            xf = rm.rope_embedder(x_pos[0])[:N_IMG]; cf = rm.rope_embedder(cap_pos[0])[:N_CAP]
            u = dit_run(
                xt[0][None].detach().numpy(), capf[0][None].detach().numpy(), adaln.detach().numpy(),
                xf.real[None, None].detach().numpy(), xf.imag[None, None].detach().numpy(),
                cf.real[None, None].detach().numpy(), cf.imag[None, None].detach().numpy(),
                x_pad[0].float()[None, :, None].detach().numpy(),
                cap_pad[0].float()[None, :, None].detach().numpy())
            outs.extend(rm.unpatchify([torch.from_numpy(u)[0]], x_size, 2, 1, None))
        return Transformer2DModelOutput(sample=outs) if return_dict else (outs,)

    # --- VAE graph swap ---
    def vae_dec(z, return_dict=True, **kw):
        img = torch.cat([torch.from_numpy(vae_run(z[i:i + 1].detach().numpy())) for i in range(z.shape[0])], 0)
        return DecoderOutput(sample=img) if return_dict else (img,)

    pipe.text_encoder.forward = enc_fwd
    rm.forward = dit_fwd
    pipe.vae.decode = vae_dec

    g = torch.Generator("cpu").manual_seed(SEED)
    img = pipe(PROMPT, height=SIZE, width=SIZE, num_inference_steps=STEPS,
               guidance_scale=1.0, generator=g).images[0]
    img.save("zfull_int8.png")
    print("[gen] zfull_int8.png  (all-int8-graph pipeline)")


if __name__ == "__main__":
    main()
