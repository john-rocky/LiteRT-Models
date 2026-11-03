# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import yaml
import torch
import os
import shutil
import torch.nn.functional as F

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def pad_or_trim_to_match(reference: torch.Tensor, target: torch.Tensor, pad_value: float = 1e-6) -> torch.Tensor:
    """
    Extends the target tensor to match the reference tensor along dim=1
    without breaking autograd, by creating a new tensor and copying data in.
    """
    B, ref_len = reference.shape
    _, tgt_len = target.shape

    if tgt_len == ref_len:
        return target
    elif tgt_len > ref_len:
        return target[:, :ref_len]
    
    # Allocate padded tensor with grad support
    padded = torch.full((B, ref_len), pad_value, dtype=target.dtype, device=target.device)
    padded[:, :tgt_len] = target  # This preserves gradient tracking

    return padded