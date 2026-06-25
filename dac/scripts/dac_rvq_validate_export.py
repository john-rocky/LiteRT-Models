"""Validate a numpy reimplementation of DAC's RVQ (the app-side glue) against torch, and export
the RVQ weights for the Kotlin app. RVQ = the only non-GPU part of the codec:
  encode: z[1,1024,T] -> codes[12,T]   (in_proj 1x1 -> L2-normalize -> cosine-argmax -> out_proj, residual)
  decode: codes[12,T] -> z_q[1,1024,T] (codebook lookup -> out_proj -> sum;  == DacRVQ.from_codes)

    ~/clipconv/bin/python scripts/dac_rvq_validate_export.py
"""
import _stub  # noqa: F401
import os
import numpy as np
import torch
from transformers.models.dac.modeling_dac import DacModel

OUT = "out/dac"


def l2norm(x, axis):                      # match F.normalize (eps 1e-12)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def rvq_encode(z, W_in, b_in, CB, W_out, b_out):
    """z [1,1024,T] -> codes [12,T] (+ z_q). Mirrors DacResidualVectorQuantizer.forward (eval)."""
    n_q = len(CB); T = z.shape[-1]
    residual = z[0]                       # [1024,T]
    z_q = np.zeros_like(residual)
    codes = np.zeros((n_q, T), np.int64)
    for i in range(n_q):
        proj = W_in[i] @ residual + b_in[i][:, None]          # [d,T]
        enc = l2norm(proj.T, axis=1)                           # [T,d]
        cbn = l2norm(CB[i], axis=1)                            # [1024,d]
        code = (enc @ cbn.T).argmax(1)                         # cosine nearest
        quant = CB[i][code].T                                  # [d,T] raw lookup
        qrep = W_out[i] @ quant + b_out[i][:, None]            # [1024,T]
        z_q += qrep
        residual = residual - qrep
        codes[i] = code
    return codes, z_q[None]


def rvq_decode(codes, CB, W_out, b_out):
    """codes [12,T] -> z_q [1,1024,T]. Mirrors from_codes."""
    n_q, T = codes.shape
    z_q = np.zeros((W_out[0].shape[0], T), np.float32)
    for i in range(n_q):
        quant = CB[i][codes[i]].T                              # [d,T]
        z_q += W_out[i] @ quant + b_out[i][:, None]
    return z_q[None]


def main():
    os.makedirs(OUT, exist_ok=True)
    m = DacModel.from_pretrained("descript/dac_16khz").eval()
    qs = m.quantizer.quantizers
    W_in = [q.in_proj.weight.detach().numpy()[:, :, 0] for q in qs]    # [d,1024]
    b_in = [q.in_proj.bias.detach().numpy() for q in qs]
    W_out = [q.out_proj.weight.detach().numpy()[:, :, 0] for q in qs]  # [1024,d]
    b_out = [q.out_proj.bias.detach().numpy() for q in qs]
    CB = [q.codebook.weight.detach().numpy() for q in qs]             # [1024,d]
    n_q, d = len(qs), CB[0].shape[1]
    print(f"n_codebooks={n_q} codebook_dim={d} codebook_size={CB[0].shape[0]}")

    audio = torch.randn(1, 1, 16000)
    with torch.no_grad():
        z = m.encoder(audio)                                          # [1,1024,T]
        zq_ref, codes_ref, _, _, _ = m.quantizer(z)

    codes_mine, zq_enc = rvq_encode(z.numpy(), W_in, b_in, CB, W_out, b_out)
    code_match = float((codes_mine == codes_ref[0].numpy()).mean())
    zq_dec = rvq_decode(codes_mine, CB, W_out, b_out)
    print(f">>> RVQ encode codes match torch: {code_match*100:.2f}%")
    print(f">>> RVQ z_q (decode) vs torch corr: "
          f"{np.corrcoef(zq_dec.flatten(), zq_ref.numpy().flatten())[0,1]:.6f} "
          f"max|diff|={np.abs(zq_dec-zq_ref.numpy()).max():.2e}")

    # full round-trip audio vs torch reconstruction
    with torch.no_grad():
        audio_ref = m(audio).audio_values.numpy()
        audio_mine = m.decoder(torch.from_numpy(zq_dec)).numpy()
    print(f">>> full audio round-trip vs torch corr: "
          f"{np.corrcoef(audio_mine.flatten(), audio_ref.flatten())[0,1]:.6f}")

    # export weights for Kotlin: contiguous float32 LE.  layout per codebook i:
    #   CB[i] (1024*d) , W_in[i] (d*1024), b_in[i] (d), W_out[i] (1024*d), b_out[i] (1024)
    blobs = []
    for i in range(n_q):
        blobs += [CB[i].astype("<f4").tobytes(), W_in[i].astype("<f4").tobytes(),
                  b_in[i].astype("<f4").tobytes(), W_out[i].astype("<f4").tobytes(),
                  b_out[i].astype("<f4").tobytes()]
    path = f"{OUT}/dac_rvq.bin"
    with open(path, "wb") as f:
        f.write(b"".join(blobs))
    print(f"exported RVQ weights -> {path} ({os.path.getsize(path)} bytes; "
          f"header-less, n_q={n_q} dim={d} size={CB[0].shape[0]})")


if __name__ == "__main__":
    main()
