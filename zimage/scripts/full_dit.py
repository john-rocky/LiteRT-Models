"""Full GPU-clean Z-Image S3-DiT and its real-weight parity vs diffusers.

Wraps EVERY compute block (noise_refiner mod=True, context_refiner mod=False,
main layers mod=True) and the final layer in GPU-clean reimplementations, reusing
the real model's host-side prep (patchify / position-ids / RoPE / unified / unpatchify)
verbatim. Comparing FullDiT(...) against the stock diffusers forward on identical
inputs proves the entire transformer graph (not just one block) is a correct
GPU-clean drop-in. Random cap features are fine — this tests DiT math, not text.
"""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/Users/majimadaisuke/Downloads/depthanything-android")
from build_zimage import (
    DIM, N_HEADS, HEAD_DIM, HALF, ADALN_DIM, safe_rms, apply_rope_halfsplit,
    _even_odd_perm,
)


def _permuted_linear(lin, row_perm):
    out_f, in_f = lin.weight.shape
    new = nn.Linear(in_f, out_f, bias=lin.bias is not None)
    new.weight = nn.Parameter(lin.weight.detach()[row_perm].clone())
    if lin.bias is not None:
        new.bias = nn.Parameter(lin.bias.detach()[row_perm].clone())
    return new


def safe_ln_noaffine(x, eps=1e-6):
    """fp16-safe LayerNorm without affine (final_layer norm), max-normalized."""
    m = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    xs = x / m
    mu = xs.mean(dim=-1, keepdim=True)
    d = xs - mu
    var = (d * d).mean(dim=-1, keepdim=True)
    return d * torch.rsqrt(var + eps)


class GpuBlock(nn.Module):
    """GPU-clean ZImageTransformerBlock (handles modulation True/False)."""

    def __init__(self, block):
        super().__init__()
        self.modulation = block.modulation
        attn = block.attention
        perm = _even_odd_perm()
        full_perm = torch.tensor(
            [h * HEAD_DIM + i for h in range(N_HEADS) for i in perm], dtype=torch.long
        )
        self.to_q = _permuted_linear(attn.to_q, full_perm)
        self.to_k = _permuted_linear(attn.to_k, full_perm)
        self.to_v = attn.to_v
        self.to_out = attn.to_out[0]
        perm_t = torch.tensor(perm, dtype=torch.long)
        self.register_buffer("nq_w", attn.norm_q.weight.detach()[perm_t].clone())
        self.register_buffer("nk_w", attn.norm_k.weight.detach()[perm_t].clone())
        self.register_buffer("an1_w", block.attention_norm1.weight.detach().clone())
        self.register_buffer("an2_w", block.attention_norm2.weight.detach().clone())
        self.register_buffer("fn1_w", block.ffn_norm1.weight.detach().clone())
        self.register_buffer("fn2_w", block.ffn_norm2.weight.detach().clone())
        self.w1 = block.feed_forward.w1
        self.w2 = block.feed_forward.w2
        self.w3 = block.feed_forward.w3
        self.adaln = block.adaLN_modulation[0] if self.modulation else None
        self.scale = HEAD_DIM ** -0.5

    def _attn(self, x, cos, sin):
        b, n, _ = x.shape
        q = self.to_q(x).view(b, n, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.to_k(x).view(b, n, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.to_v(x).view(b, n, N_HEADS, HEAD_DIM).transpose(1, 2)
        q = apply_rope_halfsplit(safe_rms(q, self.nq_w), cos, sin)
        k = apply_rope_halfsplit(safe_rms(k, self.nk_w), cos, sin)
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        out = torch.matmul(a, v).transpose(1, 2).reshape(b, n, DIM)
        return self.to_out(out)

    def _ffn(self, x):
        return self.w2((self.w1(x) * torch.sigmoid(self.w1(x))) * self.w3(x))

    def forward(self, x, cos, sin, adaln=None):
        if self.modulation:
            mod = self.adaln(adaln).unsqueeze(1)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=2)
            gate_msa, gate_mlp = torch.tanh(gate_msa), torch.tanh(gate_mlp)
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
            attn_out = self._attn(safe_rms(x, self.an1_w) * scale_msa, cos, sin)
            x = x + gate_msa * safe_rms(attn_out, self.an2_w)
            x = x + gate_mlp * safe_rms(self._ffn(safe_rms(x, self.fn1_w) * scale_mlp), self.fn2_w)
        else:
            attn_out = self._attn(safe_rms(x, self.an1_w), cos, sin)
            x = x + safe_rms(attn_out, self.an2_w)
            x = x + safe_rms(self._ffn(safe_rms(x, self.fn1_w)), self.fn2_w)
        return x


class GpuFinal(nn.Module):
    def __init__(self, final):
        super().__init__()
        self.linear = final.linear
        self.adaln_silu_linear = final.adaLN_modulation[1]  # [0]=SiLU, [1]=Linear

    def forward(self, x, c):
        scale = 1.0 + self.adaln_silu_linear(c * torch.sigmoid(c))  # SiLU(c) then Linear
        x = safe_ln_noaffine(x) * scale.unsqueeze(1)
        return self.linear(x)


class FullDiT(nn.Module):
    """GPU-clean forward over the real model's host-prep (basic mode, batch=1)."""

    def __init__(self, rm):
        super().__init__()
        self.rm = rm
        self.noise_refiner = nn.ModuleList([GpuBlock(b) for b in rm.noise_refiner])
        self.context_refiner = nn.ModuleList([GpuBlock(b) for b in rm.context_refiner])
        self.layers = nn.ModuleList([GpuBlock(b) for b in rm.layers])
        self.final = GpuFinal(rm.all_final_layer["2-1"])

    def forward(self, latent_list, t, cap_list):
        rm = self.rm
        device = latent_list[0].device
        adaln = rm.t_embedder(t * rm.t_scale).type_as(latent_list[0])
        (x, cap, x_size, x_pos, cap_pos, x_pad, cap_pad) = rm.patchify_and_embed(
            latent_list, cap_list, 2, 1)
        x_seqlens = [len(xi) for xi in x]
        x = rm.all_x_embedder["2-1"](torch.cat(x, dim=0))
        x, x_freqs, _, _, _ = rm._prepare_sequence(
            list(x.split(x_seqlens, 0)), x_pos, x_pad, rm.x_pad_token, None, device)
        xc, xs = x_freqs.real.unsqueeze(1), x_freqs.imag.unsqueeze(1)
        for layer in self.noise_refiner:
            x = layer(x, xc, xs, adaln)
        cap_seqlens = [len(ci) for ci in cap]
        cap = rm.cap_embedder(torch.cat(cap, dim=0))
        cap, cap_freqs, _, _, _ = rm._prepare_sequence(
            list(cap.split(cap_seqlens, 0)), cap_pos, cap_pad, rm.cap_pad_token, None, device)
        cc, cs = cap_freqs.real.unsqueeze(1), cap_freqs.imag.unsqueeze(1)
        for layer in self.context_refiner:
            cap = layer(cap, cc, cs)
        unified, unified_freqs, _, _ = rm._build_unified_sequence(
            x, x_freqs, x_seqlens, None, cap, cap_freqs, cap_seqlens, None,
            None, None, None, None, False, device)
        uc, us = unified_freqs.real.unsqueeze(1), unified_freqs.imag.unsqueeze(1)
        for layer in self.layers:
            unified = layer(unified, uc, us, adaln)
        unified = self.final(unified, adaln)
        return rm.unpatchify(list(unified.unbind(0)), x_size, 2, 1, None)


def main(latent_hw=32, cap_len=24):
    from diffusers import ZImageTransformer2DModel

    print("[load] real transformer (fp32) ...")
    rm = ZImageTransformer2DModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", subfolder="transformer", torch_dtype=torch.float32).eval()
    full = FullDiT(rm).eval()

    latent = [torch.randn(rm.config.in_channels, 1, latent_hw, latent_hw)]
    cap = [torch.randn(cap_len, rm.config.cap_feat_dim)]
    t = torch.rand(1)

    with torch.no_grad():
        ref = rm(latent, t, cap, return_dict=False)[0][0]   # [C,F,H,W]
        mine = full(latent, t, cap)[0]

    diff = (ref - mine).abs()
    corr = torch.corrcoef(torch.stack([ref.flatten(), mine.flatten()]))[0, 1]
    print(f"ref  {tuple(ref.shape)} range [{ref.min():.4f}, {ref.max():.4f}]")
    print(f"mine {tuple(mine.shape)} range [{mine.min():.4f}, {mine.max():.4f}]")
    print(f"max|diff| = {diff.max().item():.3e}   mean|diff| = {diff.mean().item():.3e}")
    print(f"corr = {corr.item():.8f}")
    ok = corr.item() > 0.9999 and diff.max().item() < 1e-2
    print(f"\nFULL-DiT REAL-WEIGHT PARITY: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
