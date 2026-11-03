# Copyright (c) 2023, Tri Dao, Albert Gu.
# SPDX-License-Identifier: Apache-2.0
#
# Pure-torch copy of two pieces of mamba_ssm (https://github.com/state-spaces/mamba), pinned at
# tag v2.3.2.post1 = commit a14b1dff0454a3bc27d9eb31355dc01e4b2490ec, licensed under the Apache
# License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0):
#   - mamba_ssm/modules/mamba_simple.py   Mamba.__init__ (parameter shapes) and the slow path of
#                                          Mamba.forward (the branch taken when causal_conv1d_fn is None)
#   - mamba_ssm/ops/selective_scan_interface.py   selective_scan_ref (verbatim, real-valued A)
# Changes from the original: the fused CUDA fast path, inference_params / step() caching and the
# complex-A branch are removed (RE-USE never takes them); selective_scan_fn is replaced by
# selective_scan_ref, evaluated over slices of the batch axis so a long clip stays inside the
# memory budget (the scan is independent per batch row, so the slicing does not change the math).
# The mamba_ssm package itself is a CUDA build and is not installed; install_stub() registers this
# module as `mamba_ssm` so NVIDIA's `from mamba_ssm import Mamba` resolves to the class below.
import math
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

SCAN_BATCH_ELEMS = 64 * 1024 * 1024  # max elements of one deltaA slice (B_slice * D * L * N)


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    assert not A.is_complex()
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


def selective_scan_sliced(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    """selective_scan_ref over slices of the batch axis (memory bound only; same math per row)."""
    batch, dim, L = u.shape
    step = max(1, SCAN_BATCH_ELEMS // max(1, dim * L * A.shape[1]))
    if step >= batch:
        return selective_scan_ref(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
    outs = []
    for s in range(0, batch, step):
        e = min(batch, s + step)
        outs.append(selective_scan_ref(u[s:e], delta[s:e], A, B[s:e], C[s:e], D,
                                       None if z is None else z[s:e], delta_bias, delta_softplus))
    return torch.cat(outs, dim=0)


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # (dt_proj / A_log / D initialisation of the original omitted: every value is loaded
        # from the checkpoint with strict=True.)
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        assert inference_params is None
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # slow path (causal_conv1d_fn is None)
        x, z = xz.chunk(2, dim=1)
        # Compute short convolution
        x = self.act(self.conv1d(x)[..., :seqlen])

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]
        if _TAP is not None:
            _TAP(self, x=x, dt=dt, B=B, C=C, z=z)
        y = selective_scan_sliced(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


_TAP = None  # optional callback(module, **scan_inputs) for activation statistics


def set_tap(fn):
    global _TAP
    _TAP = fn


def install_stub():
    """Register this module as `mamba_ssm` (NVIDIA's code does `from mamba_ssm import Mamba`)."""
    if "mamba_ssm" in sys.modules and getattr(sys.modules["mamba_ssm"], "Mamba", None) is Mamba:
        return
    m = types.ModuleType("mamba_ssm")
    m.Mamba = Mamba
    m.__file__ = __file__
    sys.modules["mamba_ssm"] = m
