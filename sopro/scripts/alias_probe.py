"""Instrument litert_torch.backend.inline_consts to detect constant aliasing.

The constant cache is keyed on _ConstantFingerprint (device, shape, stride, data_ptr,
storage_offset). This hook records, for every constant lowered, the fingerprint and a
content digest, and reports every cache key that was seen with two different contents:
that is a false cache hit, i.e. the second constant silently reuses the first one's bytes.
Enable with ALIAS_PROBE=1 in the environment; call install() before converting.
"""
import atexit
import hashlib
import os

import torch
from litert_torch.backend import inline_consts as ic

_seen = {}      # fingerprint -> (digest, shape, sum, order)
_collisions = []
_order = [0]


def _digest(t):
    flat = t.detach().cpu().contiguous().reshape(-1)
    if flat.numel():
        flat = flat.view(torch.uint8)
    return hashlib.blake2b(memoryview(flat.numpy()).cast("B"), digest_size=8).hexdigest()


def install():
    orig = ic._ConstantFingerprint.from_tensor.__func__

    def from_tensor(cls, tensor):
        fp = orig(cls, tensor)
        _order[0] += 1
        d = _digest(tensor)
        s = float(tensor.float().sum()) if tensor.numel() < 50_000_000 else float("nan")
        prev = _seen.get(fp)
        if prev is not None and prev[0] != d:
            _collisions.append((fp.data_ptr, tuple(tensor.shape), prev[2], s, prev[3], _order[0]))
        else:
            _seen[fp] = (d, tuple(tensor.shape), s, _order[0])
        return fp

    ic._ConstantFingerprint.from_tensor = classmethod(from_tensor)
    atexit.register(report)


def report():
    print(f"[alias-probe] constants lowered={_order[0]} distinct keys={len(_seen)} "
          f"false cache hits={len(_collisions)}", flush=True)
    for ptr, shape, s_prev, s_new, o_prev, o_new in _collisions:
        print(f"[alias-probe]   key data_ptr=0x{ptr:x} shape={shape}: constant #{o_new} (sum={s_new:.4f}) "
              f"reused the bytes of constant #{o_prev} (sum={s_prev:.4f})", flush=True)


if os.environ.get("ALIAS_PROBE") == "1":
    install()
