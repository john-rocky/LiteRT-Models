#!/usr/bin/env python3
"""Make MambaVision-T-1K GPU-clean for LiteRT CompiledModel (ML Drift) — the first on-device GPU Mamba-vision.

Four patches turn the stock (CUDA-only) model into an all-GPU-compatible graph:
  1. selective_scan_fn (CUDA mamba_ssm) -> pure-torch Hillis-Steele PARALLEL scan with CONTIGUOUS shift-slices
     => the SSM scan lowers GPU-clean (GATHER_ND=0, no O(L) unroll). This is the C24/C5 finding made real.
  2. Attention qkv (B,N,3,H,Hd)->5D + SDPA  ->  3 Linears + 4D manual attention (C12, like DA3).
  3. delta_softplus F.softplus (where(x>20,...) -> SELECT)  ->  relu(x)+log1p(exp(-|x|)) stable form (no SELECT).
  4. window_partition / window_reverse 6D view  ->  flatten (MambaVision-T windows == full resolution = 1 window).

Env: macOS, ~/clipconv (torch 2.12, litert-torch main HEAD). mamba_ssm/causal_conv1d are CUDA-only (stubbed);
this box's scipy natives are broken (leaf-stubbed); transformers 5.12 needs an all_tied_weights_keys bridge.
"""
import sys, types, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

# --- stubs for this box: broken scipy native leaves + CUDA-only mamba_ssm ---
class _Dummy:
    def __getattr__(self, n):
        if n.startswith("__") and n.endswith("__"): raise AttributeError(n)
        return _Dummy()
    def __call__(self, *a, **k): return _Dummy()
class _Leaf(types.ModuleType):
    def __getattr__(self, n):
        if n.startswith("__") and n.endswith("__"): raise AttributeError(n)
        return _Dummy()
for name in ["scipy.sparse.linalg._propack","scipy.optimize._cobyla","scipy.optimize._slsqp",
             "scipy.optimize._minpack","scipy.optimize._lbfgsb","scipy.optimize._zeros",
             "scipy.optimize._highs","scipy.optimize._direct","scipy.optimize._trlib",
             "scipy.optimize._group_columns","scipy.optimize._bglu_dense"]:
    sys.modules[name] = _Leaf(name)

# --- patch 1: pure-torch Hillis-Steele selective scan (GPU-clean), drop-in for selective_scan_fn ---
def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=None):
    dtype = u.dtype; u = u.float(); delta = delta.float()
    if delta_bias is not None: delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus: delta = F.relu(delta) + torch.log1p(torch.exp(-torch.abs(delta)))  # patch 3: SELECT-free softplus
    b, d, L = u.shape
    dp = delta.permute(0, 2, 1); up = u.permute(0, 2, 1)               # (b,L,d)
    Bp = B.float().permute(0, 2, 1); Cp = C.float().permute(0, 2, 1)   # (b,L,N)
    Aacc = torch.exp(dp.unsqueeze(-1) * A.float()[None, None])         # (b,L,d,N)
    h = (dp * up).unsqueeze(-1) * Bp.unsqueeze(2)                      # (b,L,d,N)
    step = 1
    while step < L:                                                   # log2(L) levels, contiguous shift-slices
        A_sh = torch.cat([torch.ones_like(Aacc[:, :step]),  Aacc[:, :L-step]], dim=1)
        h_sh = torch.cat([torch.zeros_like(h[:, :step]),    h[:, :L-step]], dim=1)
        Aacc, h = Aacc * A_sh, Aacc * h_sh + h
        step *= 2
    y = (h * Cp.unsqueeze(2)).sum(-1)                                 # (b,L,d)
    if D is not None: y = y + up * D.float()[None, None]
    return y.permute(0, 2, 1).to(dtype)                              # (b,d,L)
mm = types.ModuleType("mamba_ssm"); mo = types.ModuleType("mamba_ssm.ops")
mi = types.ModuleType("mamba_ssm.ops.selective_scan_interface"); mi.selective_scan_fn = selective_scan_fn
sys.modules.update({"mamba_ssm": mm, "mamba_ssm.ops": mo, "mamba_ssm.ops.selective_scan_interface": mi})

import transformers as _tf
if not hasattr(_tf.PreTrainedModel, "all_tied_weights_keys"):
    _tf.PreTrainedModel.all_tied_weights_keys = property(lambda self: {})   # transformers 5.12 bridge
from transformers import AutoModel
# Fine-tune override: MAMBAVISION_MODEL_ID=<hf-repo-or-local-dir> (same as the cls build).
MID = os.environ.get("MAMBAVISION_MODEL_ID", "nvidia/MambaVision-T-1K")
model = AutoModel.from_pretrained(MID, trust_remote_code=True).eval()
print("loaded MambaVision-T-1K")

# --- patch 4: window_partition/reverse 6D -> flatten (single window for MambaVision-T at 224) ---
_mvmod = sys.modules[type(model).__module__]
def _wp(x, window_size):
    B, C, H, W = x.shape
    assert window_size == H == W, "flatten patch assumes a single window"
    return x.flatten(2).transpose(1, 2)                              # [B, H*W, C]
def _wr(windows, window_size, H, W):
    B = windows.shape[0]; C = windows.shape[2]
    return windows.transpose(1, 2).reshape(B, C, H, W)
_mvmod.window_partition = _wp; _mvmod.window_reverse = _wr

# --- patch 2: Attention qkv 5D + SDPA -> 3 Linears + 4D manual attention ---
AttnCls = None
for m in model.modules():
    if hasattr(m, "qkv") and hasattr(m, "num_heads") and hasattr(m, "head_dim") and hasattr(m, "proj"):
        AttnCls = type(m); break
def _attn_fwd(self, x):
    B, N, C = x.shape; H = self.num_heads; Hd = self.head_dim
    q = self.q_lin(x).reshape(B, N, H, Hd).permute(0, 2, 1, 3); k = self.k_lin(x).reshape(B, N, H, Hd).permute(0, 2, 1, 3)
    v = self.v_lin(x).reshape(B, N, H, Hd).permute(0, 2, 1, 3); q, k = self.q_norm(q), self.k_norm(k)
    q = q * self.scale; attn = (q @ k.transpose(-2, -1)).softmax(-1)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C); return self.proj_drop(self.proj(x))
AttnCls.forward = _attn_fwd
for m in model.modules():
    if isinstance(m, AttnCls):
        Cf = m.qkv.in_features; w = m.qkv.weight; bsd = m.qkv.bias
        for nm, sl in (("q_lin", slice(0, Cf)), ("k_lin", slice(Cf, 2*Cf)), ("v_lin", slice(2*Cf, 3*Cf))):
            lin = nn.Linear(Cf, Cf, bias=bsd is not None)
            with torch.no_grad():
                lin.weight.copy_(w[sl])
                if bsd is not None: lin.bias.copy_(bsd[sl])
            setattr(m, nm, lin)
print("patched: Hillis-Steele scan + 4D attention + stable softplus + window-flatten")

# --- convert (FP32) + op-check ---
import litert_torch, collections
from ai_edge_litert.interpreter import Interpreter
dummy = torch.randn(1, 3, 224, 224)
litert_torch.convert(model, (dummy,)).export("mambavision_t_gpu.tflite")
it = Interpreter(model_path="mambavision_t_gpu.tflite"); it.allocate_tensors()
ops = collections.Counter(d.get("op_name", "?") for d in it._get_ops_details())
GPU_BAD = {"GATHER_ND","GATHER","SELECT_V2","SELECT","PACK","SPLIT","CAST","TOPK_V2","BROADCAST_TO","WHILE","TRANSPOSE_CONV"}
bad = {k: v for k, v in ops.items() if k in GPU_BAD}
over = sum(1 for d in it.get_tensor_details() if len(d.get("shape", [])) > 4)
print(f"op-check: TOTAL={sum(ops.values())}  GPU_BAD={bad or 'NONE'}  >4D={over}  GATHER_ND={ops.get('GATHER_ND',0)}")

# --- fidelity: FP32 tflite vs patched pytorch (conversion faithfulness) ---
with torch.no_grad():
    ro = model(dummy); ro = ro[0] if isinstance(ro, (tuple, list)) else ro
    ref = ro.detach().numpy().flatten()
ind = it.get_input_details()[0]
it.set_tensor(ind["index"], dummy.numpy().astype(np.float32)); it.invoke()
got = None
for od in it.get_output_details():
    g = it.get_tensor(od["index"]).flatten()
    if g.size == ref.size: got = g; break
if got is None: got = it.get_tensor(it.get_output_details()[0]["index"]).flatten()
print(f"fidelity tflite vs torch: corr={np.corrcoef(ref,got)[0,1]:.5f}  maxdiff={np.abs(ref-got).max():.2e}")

# --- FP16 ---
from ai_edge_quantizer import quantizer
RECIPE = [{"regex": ".*", "operation": "*", "algorithm_key": "float_casting",
           "op_config": {"weight_tensor_config": {"num_bits": 16, "dtype": "FLOAT"}}}]
q = quantizer.Quantizer("mambavision_t_gpu.tflite"); q.load_quantization_recipe(RECIPE)
q.quantize().export_model("mambavision_t_gpu_fp16.tflite")
print("FP32 %.1f MB -> FP16 %.1f MB" % (os.path.getsize("mambavision_t_gpu.tflite")/1e6,
                                        os.path.getsize("mambavision_t_gpu_fp16.tflite")/1e6))
print("VERDICT:", "GPU-CLEAN" if not bad and not over else "BLOCKERS REMAIN")
