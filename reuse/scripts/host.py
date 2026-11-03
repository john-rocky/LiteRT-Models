#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1 (mirrors nvidia/NVIDIA-NemotronLabs-RE-USE models/stfts.py + inference.py)
"""Host side of the RE-USE LiteRT graph (what a phone app does around it), in torch fp32 like the reference:

  spec_in(chunk, sr)       wav [1, N] -> spec [1, 2, T, F]: log1p(|STFT|), angle(STFT), time-major
                           (torch.stft, Hann win, center=True, reflect; n_fft / hop / win scaled from 8 kHz like
                           inference.py's make_even)
  wav_out(out, n, sr)      graph output [1, 3, T, F] (compressed magnitude, x_r, x_i) -> wav [1, n]: the zero-frame
                           rule (a frame whose expm1(relu(mag)) == 0 on more than half the bins is zeroed), cos / sin =
                           x_r / hypot, x_i / hypot (= cos / sin of atan2(x_i, x_r); 0/0 -> angle 0), expm1(relu(mag)),
                           torch.istft (center=True), then pad (1e-8) or trim to n like pad_or_trim_to_match.
"""
import torch

import reuse_ref as R


def spec_in(wav, sr, cfg):
    n_fft, hop, win = R.stft_params(cfg, sr)
    st = torch.stft(wav, n_fft, hop_length=hop, win_length=win, window=torch.hann_window(win), center=True,
                    pad_mode="reflect", normalized=False, return_complex=True)          # [1, F, T]
    mag = torch.log1p(torch.abs(st))
    pha = torch.angle(st)
    return torch.stack([mag, pha], dim=1).transpose(2, 3).contiguous()                # [1, 2, T, F]


def wav_out(out, n, sr, cfg):
    n_fft, hop, win = R.stft_params(cfg, sr)
    out = torch.as_tensor(out, dtype=torch.float32)
    amp = out[:, 0].transpose(1, 2).clone()                                            # [1, F, T]
    xr = out[:, 1].transpose(1, 2)
    xi = out[:, 2].transpose(1, 2)
    mag = torch.expm1(torch.relu(amp))
    zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
    amp[:, :, (zero_portion > 0.5)[0]] = 0
    mag = torch.expm1(torch.relu(amp))
    hyp = torch.sqrt(xr * xr + xi * xi)
    safe = torch.where(hyp > 0, hyp, torch.ones_like(hyp))
    cos = torch.where(hyp > 0, xr / safe, torch.ones_like(hyp))
    sin = torch.where(hyp > 0, xi / safe, torch.zeros_like(hyp))
    com = torch.complex(mag * cos, mag * sin)
    w = torch.istft(com, n_fft, hop_length=hop, win_length=win, window=torch.hann_window(win), center=True)
    if w.shape[1] >= n:
        return w[:, :n]
    pad = torch.full((1, n), 1e-8, dtype=w.dtype)
    pad[:, :w.shape[1]] = w
    return pad
