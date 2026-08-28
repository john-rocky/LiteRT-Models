"""Whisper-tiny encoder at the shipped T=1500 — same-day control arm for the
pad-flip pair (build_whisper_pad.py). Rebuilt from the official checkpoint
with the shipped math (sigmoid-GELU, fp32, manual rank-4 qkv attention
identical to the pad build); no mask, no pad — sequence length is the only
variable between probe_wh_ctrl and probe_wh_pad1536. The device's shipped
whisper_encoder.tflite stays the anchor for "vs shipped" latency claims; this
arm exists so the flip pair shares one builder.

  python build_whisper_ctrl.py   # writes probe_wh_ctrl.tflite
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu_sig(x):
    return x * torch.sigmoid(1.702 * x)


class CtrlEncoder(nn.Module):
    def __init__(self, enc, n_head):
        super().__init__()
        self.enc = enc
        self.n_head = n_head

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
        w = (q @ k).softmax(dim=-1)
        o = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        return block.attn.out(o)

    def mlp(self, block, x):
        h = block.mlp_ln(x)
        return block.mlp[2](gelu_sig(block.mlp[0](h)))

    def forward(self, mel):
        x = gelu_sig(self.enc.conv1(mel))
        x = gelu_sig(self.enc.conv2(x))
        x = x.permute(0, 2, 1) + self.enc.positional_embedding
        for block in self.enc.blocks:
            x = x + self.attn(block, x)
            x = x + self.mlp(block, x)
        return self.enc.ln_post(x)


def main():
    import whisper
    from whisper.model import MultiHeadAttention
    MultiHeadAttention.use_sdpa = False
    model = whisper.load_model('tiny', device='cpu')
    model.eval()
    ctrl = CtrlEncoder(model.encoder, model.dims.n_audio_head).eval()

    # host equivalence vs the whisper-package encoder under the shipped GELU
    orig_gelu = F.gelu
    F.gelu = lambda x, approximate='none': x * torch.sigmoid(1.702 * x)
    torch.manual_seed(42)
    mel = torch.randn(1, model.dims.n_mels, 3000)
    with torch.no_grad():
        ref = model.encoder(mel)
        out = ctrl(mel)
    d = (out - ref).abs().max().item()
    c = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
    print(f'ctrl vs whisper-package encoder (host): max_abs_diff={d:.3e} corr={c:.9f}')

    import litert_torch
    litert_torch.convert(ctrl, (torch.randn(1, model.dims.n_mels, 3000),)) \
        .export('probe_wh_ctrl.tflite')
    F.gelu = orig_gelu
    print('wrote probe_wh_ctrl.tflite')


if __name__ == '__main__':
    main()
