"""Extract Inflect-Nano-v2 weights + golden intermediates for the Keras port.

Runs in the torch venv. Produces:
    out/inflect_weights.npz   plain (weight-norm-collapsed) state dict
    out/inflect_golden.npz    tokens, per-stage reference tensors, final wav
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoint"
sys.path.insert(0, str(CKPT / "runtime"))
sys.path.insert(0, str(CKPT))

import commons  # noqa: E402
import utils  # noqa: E402
from inflect_vits_frontend import run_vits_frontend  # noqa: E402
from models import SynthesizerTrn  # noqa: E402
from text import cleaned_text_to_sequence  # noqa: E402
from text.symbols import symbols  # noqa: E402

OUT = ROOT / "out"
TEXT = "Hello, this is a test of the Inflect text to speech model running on device."
NOISE_SCALE = 0.667


def build_model():
    hps = utils.get_hparams_from_file(str(CKPT / "config.json"))
    model = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).eval()
    utils.load_checkpoint(str(CKPT / "model.pth"), model, None)
    model.dec.remove_weight_norm()
    for flow in model.flow.flows:
        enc = getattr(flow, "enc", None)
        if enc is not None and hasattr(enc, "remove_weight_norm"):
            enc.remove_weight_norm()
    return model


def tokens_for(text):
    phonemes = run_vits_frontend(text).phoneme_text
    seq = cleaned_text_to_sequence(phonemes)
    seq = commons.intersperse(seq, 0)
    return torch.LongTensor(seq).unsqueeze(0)


def main():
    OUT.mkdir(exist_ok=True)
    model = build_model()

    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    np.savez(OUT / "inflect_weights.npz", **sd)
    print(f"weights: {len(sd)} tensors -> inflect_weights.npz")

    tokens = tokens_for(TEXT)
    n = tokens.shape[1]
    x_lengths = torch.LongTensor([n])
    with torch.no_grad():
        x, m_p, logs_p, x_mask = model.enc_p(tokens, x_lengths)
        logw = model.dp(x, x_mask)
        w = torch.exp(logw) * x_mask  # length_scale = 1
        w_ceil = torch.ceil(w)
        durations = w_ceil[0, 0].long()
        t_frames = int(durations.sum())
        m_p_exp = torch.repeat_interleave(m_p, durations, dim=2)
        logs_p_exp = torch.repeat_interleave(logs_p, durations, dim=2)
        torch.manual_seed(7)
        noise = torch.randn_like(m_p_exp)
        z_p = m_p_exp + noise * torch.exp(logs_p_exp) * NOISE_SCALE
        y_mask = torch.ones(1, 1, t_frames)
        z = model.flow(z_p, y_mask, g=None, reverse=True)
        wav = model.dec(z)

    np.savez(
        OUT / "inflect_golden.npz",
        tokens=tokens.numpy(),
        enc_hidden=x.numpy(),
        m_p=m_p.numpy(),
        logs_p=logs_p.numpy(),
        logw=logw.numpy(),
        durations=durations.numpy(),
        z_p=z_p.numpy(),
        z_flow=z.numpy(),
        wav=wav.numpy(),
    )
    print(f"golden: N={n} T={t_frames} wav={wav.shape} -> inflect_golden.npz")


if __name__ == "__main__":
    main()
