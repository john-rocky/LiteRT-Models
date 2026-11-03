# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch
import torch.nn as nn

def decompress_signed_log1p(y):
   return torch.sign(y) * (torch.expm1(torch.abs(y)))

RELU = nn.ReLU()

def mag_phase_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True, addeps=False):
    """
    Compute magnitude and phase using STFT.

    Args:
        y (torch.Tensor): Input audio signal.
        n_fft (int): FFT size.
        hop_size (int): Hop size.
        win_size (int): Window size.
        compress_factor (float, optional): Magnitude compression factor. Defaults to 1.0.
        center (bool, optional): Whether to center the signal before padding. Defaults to True.
        eps (bool, optional): Whether adding epsilon to magnitude and phase or not. Defaults to False. 

    Returns:
        tuple: Magnitude, phase, and complex representation of the STFT.
    """
    eps = 1e-10
    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(
                    y, n_fft, 
                    hop_length=hop_size, 
                    win_length=win_size, 
                    window=hann_window,
                    center=center, 
                    pad_mode='reflect', 
                    normalized=False, 
                    return_complex=True)

    if addeps==False:
        mag = torch.abs(stft_spec)
        pha = torch.angle(stft_spec)
    else:
        real_part = stft_spec.real
        imag_part = stft_spec.imag
        mag = torch.sqrt(real_part.pow(2) + imag_part.pow(2) + eps)
        pha = torch.atan2(imag_part + eps, real_part + eps)
    # Compress the magnitude
    if compress_factor in ['log1p','relu_log1p', 'signed_log1p']:
        mag = torch.log1p(mag)
    else:
        mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com


def mag_phase_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    """
    Inverse STFT to reconstruct the audio signal from magnitude and phase.

    Args:
        mag (torch.Tensor): Magnitude of the STFT.
        pha (torch.Tensor): Phase of the STFT.
        n_fft (int): FFT size.
        hop_size (int): Hop size.
        win_size (int): Window size.
        compress_factor (float, optional): Magnitude compression factor. Defaults to 1.0.
        center (bool, optional): Whether to center the signal before padding. Defaults to True.

    Returns:
        torch.Tensor: Reconstructed audio signal.
    """
    if compress_factor == 'log1p':
        mag = torch.expm1(mag)
    elif compress_factor == 'signed_log1p':
        mag = decompress_signed_log1p(mag) 
    elif compress_factor == 'relu_log1p':
        mag = torch.expm1(RELU(mag))
    else:
        mag = torch.pow(RELU(mag), 1.0 / compress_factor)
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(
                    com, 
                    n_fft, 
                    hop_length=hop_size, 
                    win_length=win_size, 
                    window=hann_window, 
                    center=center)
    return wav
