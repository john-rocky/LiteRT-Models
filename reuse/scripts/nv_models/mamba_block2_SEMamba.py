# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from functools import partial
from einops import rearrange
from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    def __init__(self, d_model, cfg):
        super(MambaBlock, self).__init__()
        
        d_state = cfg['model_cfg']['d_state'] # 16
        d_conv = cfg['model_cfg']['d_conv'] # 4
        expand = cfg['model_cfg']['expand'] # 4

        self.forward_blocks  = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_blocks = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) 
        self.output_proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, T, D]
        out_fw = self.forward_blocks(x) + x

        out_bw = self.backward_blocks(torch.flip(x, dims=[1])) + torch.flip(x, dims=[1])
        out_bw = torch.flip(out_bw, dims=[1])
        
        out = torch.cat([out_fw, out_bw], dim=-1)
        out = self.output_proj(out)

        # LayerNorm
        return self.norm(out)


class TFMambaBlock(nn.Module):
    """
    Temporal-Frequency Mamba block for sequence modeling.
    
    Attributes:
    cfg (Config): Configuration for the block.
    time_mamba (MambaBlock): Mamba block for temporal dimension.
    freq_mamba (MambaBlock): Mamba block for frequency dimension.
    tlinear (ConvTranspose1d): ConvTranspose1d layer for temporal dimension.
    flinear (ConvTranspose1d): ConvTranspose1d layer for frequency dimension.
    """
    def __init__(self, cfg):
        super(TFMambaBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = cfg['model_cfg']['hid_feature']
        
        # Initialize Mamba blocks
        self.time_mamba = MambaBlock(d_model=self.hid_feature, cfg=cfg)
        self.freq_mamba = MambaBlock(d_model=self.hid_feature, cfg=cfg)
    
    def forward(self, x):
        """
        Forward pass of the TFMamba block.
        
        Parameters:
        x (Tensor): Input tensor with shape (batch, channels, time, freq).
        
        Returns:
        Tensor: Output tensor after applying temporal and frequency Mamba blocks.
        """
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_mamba(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_mamba(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x
