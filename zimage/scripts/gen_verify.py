"""Confirm the full device algorithm in Python (fp32 DeployDiT): latent-space
flow-matching loop with the exact Z-Image CFG (pred = pos + g*(pos-neg);
noise_pred = -pred; latent += dsigma*noise_pred), decode with the real VAE, and
compare to the pipeline reference. This is the exact loop the Kotlin app ports.
"""
import sys
import numpy as np
import torch
sys.path.insert(0, "/Users/majimadaisuke/Downloads/depthanything-android")
sys.path.insert(0, ".")
from deploy_dit import DeployDiT

STEPS, N_IMG, GUID = 8, 256, 1.0
OUT = "gen_bins"


def load(name, shape):
    return torch.from_numpy(np.fromfile(f"{OUT}/{name}.bin", dtype="<f4")).reshape(shape)


def corr(a, b):
    return float(np.corrcoef(a.flatten().numpy(), b.flatten().numpy())[0, 1])


def main():
    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.float32).to("cpu")
    dit = DeployDiT(pipe.transformer).eval()

    adaln = load("adaln", (STEPS, 256))
    xc = load("xc", (1, 1, 256, 64)); xs = load("xs", (1, 1, 256, 64))
    cc = load("cc", (1, 1, 32, 64)); cs = load("cs", (1, 1, 32, 64))
    xpad = load("xpad", (1, 256, 1)); cpad = load("cpad", (1, 32, 1))
    cc_u = load("cc_unc", (1, 1, 32, 64)); cs_u = load("cs_unc", (1, 1, 32, 64))
    cpad_u = load("cpad_unc", (1, 32, 1))
    dsigma = load("dsigma", (STEPS,))
    cap_cond = load("cap_b0", (1, 32, 2560)); cap_unc = load("cap_b1", (1, 32, 2560))
    uperm = torch.from_numpy(np.fromfile(f"{OUT}/unpatch_perm.bin", dtype="<i4")).long()
    pperm = torch.from_numpy(np.fromfile(f"{OUT}/patch_perm.bin", dtype="<i4")).long()
    latent = load("steps_0", (1, 16, 32, 32)).clone()   # initial noise (scheduler sample 0)
    ref_latents = load("ref_latents", (1, 16, 32, 32))

    def patch(lat):
        return lat.reshape(-1)[pperm].reshape(1, N_IMG, 64)

    def unpatch(u):
        return u.reshape(-1)[uperm].reshape(1, 16, 32, 32)

    def dit_v(xt, cap, s, ccx, csx, cpadx):
        return unpatch(dit(xt, cap, adaln[s:s + 1], xc, xs, ccx, csx, xpad, cpadx))

    with torch.no_grad():
        for s in range(STEPS):
            xt = patch(latent)
            pos = dit_v(xt, cap_cond, s, cc, cs, cpad)          # cond context
            neg = dit_v(xt, cap_unc, s, cc_u, cs_u, cpad_u)     # uncond context
            noise_pred = -(pos + GUID * (pos - neg))
            latent = latent + dsigma[s] * noise_pred

    # VAE latent denormalization (pipeline: latents / scaling_factor + shift_factor)
    latent = latent / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    print(f"[verify] final latents vs pipeline ref: corr {corr(latent, ref_latents):.6f}")
    img = pipe.vae.decode(latent).sample
    im = ((img[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().byte().numpy()
    from PIL import Image
    Image.fromarray(im).save(f"{OUT}/selfcheck_loop.png")
    print(f"[verify] saved selfcheck_loop.png")


if __name__ == "__main__":
    main()
