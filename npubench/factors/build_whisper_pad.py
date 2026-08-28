"""Whisper-tiny encoder rebuilt at T=1536 (mel 3072) — the pad-flip experiment.

The shipped whisper_encoder.tflite (T=1500, sigmoid-GELU, fp32) runs at
0.97 GF/ms on the Hexagon NPU purely because 1500 is an awkward number
(experiment 2). This build changes ONE variable — the sequence length — by:
  * zero-padding the mel input 3000 -> 3072 and re-zeroing the conv1 output
    at positions >= 3000 with a constant mask (one MUL): conv2's last real
    output token (1499) reads conv1 position 3000, which the original graph
    sees as its own zero padding but a naive pad build fills with
    gelu(conv1 bias + boundary taps) — without the mask the pair is only
    corr 0.9999995 / max_abs 2e-2; with it, exact,
  * extending the positional embedding 1500 -> 1536: the checkpoint's stored
    table for real positions (the released weights differ from freshly
    computed sinusoids by up to 3e-4 — formula recompute alone perturbs every
    token and costs 2e-2 at the output), formula sinusoids for the pad tail
    (those rows only feed masked keys and sliced-off outputs),
  * masking keys >= 1500 with -1e4 before softmax (exp underflows to exactly
    0.0 in fp32, so padded keys contribute nothing and the softmax
    normalization is unchanged),
  * slicing the output back to [1, 1500, 384] — drop-in for the decoder.
GELU stays sigmoid (the shipped flavor); attention mirrors whisper's
qkv_attention line-for-line so the exporter lowers it identically.

  python build_whisper_pad.py          # writes whisper_enc_pad1536.tflite
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

T_PAD, T_REAL, MEL_PAD = 1536, 1500, 3072


def gelu_sig(x):
    return x * torch.sigmoid(1.702 * x)


class PaddedEncoder(nn.Module):
    def __init__(self, enc, n_state, n_head):
        super().__init__()
        self.enc = enc
        self.n_head = n_head
        from whisper.model import sinusoids
        ext = sinusoids(T_PAD, n_state)[T_REAL:]
        self.register_buffer('pos', torch.cat([enc.positional_embedding.detach(), ext], dim=0))
        kmask = torch.zeros(1, 1, 1, T_PAD)
        kmask[..., T_REAL:] = -1e4
        self.register_buffer('kmask', kmask)
        cmask = torch.ones(1, 1, MEL_PAD)
        cmask[..., 2 * T_REAL:] = 0.0
        self.register_buffer('cmask', cmask)

    def attn(self, block, x):
        h = block.attn_ln(x)
        q = block.attn.query(h)
        k = block.attn.key(h)
        v = block.attn.value(h)
        B, T, D = q.shape
        scale = (D // self.n_head) ** -0.25
        q = q.view(B, T, self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(B, T, self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(B, T, self.n_head, -1).permute(0, 2, 1, 3)
        qk = q @ k + self.kmask
        w = qk.softmax(dim=-1)
        o = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        return block.attn.out(o)

    def mlp(self, block, x):
        h = block.mlp_ln(x)
        h = gelu_sig(block.mlp[0](h))
        return block.mlp[2](h)

    def forward(self, mel):
        x = gelu_sig(self.enc.conv1(mel)) * self.cmask
        x = gelu_sig(self.enc.conv2(x))
        x = x.permute(0, 2, 1) + self.pos
        for block in self.enc.blocks:
            x = x + self.attn(block, x)
            x = x + self.mlp(block, x)
        x = self.enc.ln_post(x)
        return x[:, :T_REAL]


def main():
    import whisper
    from whisper.model import MultiHeadAttention
    MultiHeadAttention.use_sdpa = False
    model = whisper.load_model('tiny', device='cpu')
    model.eval()
    enc = model.encoder
    padded = PaddedEncoder(enc, model.dims.n_audio_state, model.dims.n_audio_head).eval()

    # reference = original encoder with the shipped sigmoid-GELU math
    orig_gelu = F.gelu
    F.gelu = lambda x, approximate='none': x * torch.sigmoid(1.702 * x)
    torch.manual_seed(42)
    mel = torch.randn(1, model.dims.n_mels, 3000)
    with torch.no_grad():
        ref = enc(mel)
        out = padded(F.pad(mel, (0, MEL_PAD - 3000)))
    F.gelu = orig_gelu
    d = (out - ref).abs().max().item()
    c = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
    print(f'padded vs original (host, sigmoid-GELU both): max_abs_diff={d:.3e} corr={c:.9f}')

    F.gelu = lambda x, approximate='none': x * torch.sigmoid(1.702 * x)
    import litert_torch
    litert_torch.convert(padded, (torch.randn(1, model.dims.n_mels, MEL_PAD),)) \
        .export('whisper_enc_pad1536.tflite')
    F.gelu = orig_gelu
    print('wrote whisper_enc_pad1536.tflite')


if __name__ == '__main__':
    main()
