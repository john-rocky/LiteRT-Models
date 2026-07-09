"""Precompute every fixed / per-step input the on-device generation loop needs,
and render the desktop int8 reference image with the IDENTICAL algorithm the
device will run (token-space flow-matching Euler over the chunked DiT + VAE).

Device loop (token space [1,256,64], patchify + Euler update are both linear so
the whole loop stays in token space, unpatchify once at the end):
    x = x_tok0
    for s in 0..7:  u = DiT(x, cap, adaln[s], RoPE, pad); x = x + dsigma[s]*u[:, :256]
    latents = unpatchify(x); image = VAE(latents)

Saved to gen_bins/: cap[1,32,2560], xc/xs[1,1,256,64], cc/cs[1,1,32,64],
uc/us[1,1,288,64], xpad[1,256,1], cpad[1,32,1], xpt/cpt[1,1,3840],
adaln[8,256], dsigma[8], x_tok0[1,256,64], and per-step xt_s[1,256,64] +
ref_latents[1,16,32,32] for device-vs-desktop verification.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, "/Users/majimadaisuke/Downloads/depthanything-android")
sys.path.insert(0, ".")
from types import SimpleNamespace
from deploy_dit import DeployDiT

PROMPT = "a red apple on a wooden table, studio lighting"
STEPS, SIZE, SEED = 8, 256, 1234
L_ENC, N_IMG, N_CAP = 64, 256, 32
OUT = "gen_bins"


def save(t, name):
    (t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)).astype(
        "<f4").tofile(f"{OUT}/{name}.bin")


def main():
    from diffusers import ZImagePipeline
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    os.makedirs(OUT, exist_ok=True)

    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.float32).to("cpu")
    rm = pipe.transformer
    dit = DeployDiT(rm).eval()          # fp32 reference == the chunked device algorithm
    rec = {"adaln": [], "xt": [], "fixed": None}

    def dit_fwd(x, t, cap_feats, return_dict=True, **kw):
        outs = []
        for i in range(len(x)):
            adaln = rm.t_embedder(t[i:i + 1] * rm.t_scale).type_as(x[0])
            (xt, capf, x_size, x_pos, cap_pos, x_pad, cap_pad) = rm.patchify_and_embed(
                [x[i]], [cap_feats[i][:N_CAP]], 2, 1)
            xf = rm.rope_embedder(x_pos[0])[:N_IMG]; cf = rm.rope_embedder(cap_pos[0])[:N_CAP]
            xc, xs_, cc, cs = xf.real[None, None], xf.imag[None, None], cf.real[None, None], cf.imag[None, None]
            xpad = x_pad[0].float()[None, :, None]; cpad = cap_pad[0].float()[None, :, None]
            with torch.no_grad():
                u = dit(xt[0][None], capf[0][None], adaln, xc, xs_, cc, cs, xpad, cpad)
            rec["adaln"].append(adaln.detach().clone())
            rec["xt"].append(xt[0][None].detach().clone())
            rec.setdefault("cap_branch", []).append(capf[0][None].detach().clone())
            if rec["fixed"] is None:
                rec["fixed"] = dict(
                    xc=xc.detach().clone(),
                    xs=xs_.detach().clone(), cc=cc.detach().clone(), cs=cs.detach().clone(),
                    xpad=xpad.detach().clone(), cpad=cpad.detach().clone(),
                    xpt=rm.x_pad_token.detach().clone(), cpt=rm.cap_pad_token.detach().clone())
                # patch perm: tokens_flat[j] = latent_flat[patch_perm[j]] (real latent shape)
                lat_probe = torch.arange(x[i].numel()).float().reshape(x[i].shape)
                xt_p, *_ = rm.patchify_and_embed([lat_probe], [cap_feats[i][:N_CAP]], 2, 1)
                rec["patch"] = xt_p[0].flatten().round().long()   # [256*64]
                rec["lat_shape"] = tuple(x[i].shape)
                # unpatch perm from a full [288,64] token probe (real DiT output width)
                n_tok = xt[0].shape[0] + N_CAP
                tok_probe = torch.arange(n_tok * 64).float().reshape(n_tok, 64)
                pl = rm.unpatchify([tok_probe], x_size, 2, 1, None)[0]
                rec["x_size"] = x_size
                rec["perm"] = pl.flatten().round().long()          # -> token-flat idx
            outs.extend(rm.unpatchify([u[0]], x_size, 2, 1, None))
        return Transformer2DModelOutput(sample=outs) if return_dict else (outs,)

    def vae_dec(z, return_dict=True, **kw):
        rec["ref_latents"] = z.detach().clone()
        return _real_vae(z, return_dict=return_dict, **kw)

    _real_vae = pipe.vae.decode
    rm.forward = dit_fwd
    pipe.vae.decode = vae_dec

    # record the exact velocity + sample the scheduler consumes each step
    _real_step = pipe.scheduler.step
    rec["step"] = []

    def step_hook(model_output, timestep, sample, *a, **k):
        rec["step"].append((model_output.detach().clone(), sample.detach().clone(),
                            float(timestep) if not torch.is_tensor(timestep) else float(timestep.reshape(-1)[0])))
        return _real_step(model_output, timestep, sample, *a, **k)
    pipe.scheduler.step = step_hook

    g = torch.Generator("cpu").manual_seed(SEED)
    img = pipe(PROMPT, height=SIZE, width=SIZE, num_inference_steps=STEPS,
               guidance_scale=1.0, generator=g).images[0]
    img.save(f"{OUT}/ref_desktop.png")

    # scheduler deltas for the token-space Euler update
    sig = pipe.scheduler.sigmas.detach().cpu().numpy()
    dsigma = np.array([sig[i + 1] - sig[i] for i in range(STEPS)], dtype="<f4")

    f = rec["fixed"]
    for k, v in f.items():
        save(v, k)
    uc = torch.cat([f["xc"], f["cc"]], dim=2); us = torch.cat([f["xs"], f["cs"]], dim=2)
    save(uc, "uc"); save(us, "us")
    # CFG runs batch=2 (cond+uncond, identical adaln per step) -> dedupe to STEPS
    branches = 2 if len(rec["adaln"]) == 2 * STEPS else 1
    adaln8 = torch.cat(rec["adaln"][::branches], 0)
    save(adaln8, "adaln")                                # [8,256]
    save(rec["cap_branch"][0], "cap_b0")                 # first branch cap
    save(rec["cap_branch"][1 if branches == 2 else 0], "cap_b1")  # second branch cap
    save(dsigma, "dsigma")                               # [8]
    save(rec["xt"][0], "x_tok0")                         # [1,256,64]
    for s in range(STEPS):                               # pipeline per-step input latents
        save(rec["xt"][s * branches], f"xt_{s}")
    save(rec["ref_latents"], "ref_latents")             # [1,16,32,32]
    rec["perm"].numpy().astype("<i4").tofile(f"{OUT}/unpatch_perm.bin")   # [16*32*32]
    rec["patch"].numpy().astype("<i4").tofile(f"{OUT}/patch_perm.bin")    # [256*64]
    pok = int(np.array_equal(np.sort(rec["patch"].numpy()), np.arange(16 * 32 * 32)))
    print(f"[gen_prep] patch_perm pure-perm={pok}; unpatch_perm range "
          f"{int(rec['perm'].min())}..{int(rec['perm'].max())}")
    for s, (mo, samp, tt) in enumerate(rec["step"]):
        save(mo, f"stepv_{s}"); save(samp, f"steps_{s}")
    print(f"[gen_prep] recorded {len(rec['step'])} scheduler steps; "
          f"timesteps={[round(t[2],2) for t in rec['step']]}")
    print(f"[gen_prep] STEPS={STEPS} SIZE={SIZE} SEED={SEED}")
    print(f"[gen_prep] sigmas={[round(float(s),4) for s in sig]}")
    print(f"[gen_prep] saved bins to {OUT}/ + ref_desktop.png "
          f"(latents {tuple(rec['ref_latents'].shape)})")


if __name__ == "__main__":
    main()
