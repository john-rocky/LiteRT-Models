#!/usr/bin/env python3
# SPDX-License-Identifier: OpenMDW-1.1 (the RE-USE code it drives: nvidia/NVIDIA-NemotronLabs-RE-USE)
"""Pure-torch fp32 reference of nvidia/NVIDIA-NemotronLabs-RE-USE @ 31f82f2e on the Mac CPU.

The model code is NVIDIA's own (scripts/nv_models/, unmodified copies, OpenMDW-1.1); the only
substitution is mamba_ssm.Mamba -> the pure-torch copy of its slow path + selective_scan_ref
(scripts/mamba_ref.py, Apache-2.0, state-spaces/mamba @ a14b1dff). Weights load with strict=True.
Memory only: nn.Conv2d runs over output-row tiles when its im2col buffer would pass 768 MB
(set_conv_tiling; each output element is the same sum of products as the untiled conv).

enhance_full()    = inference.py (whole clip in one pass, the zero-frame rule, iSTFT, length match)
enhance_chunked() = inference_chunk.py (chunk_size_in_seconds, hop_length_portion, Hann(>=1e-7)
                    weighted overlap-add divided by the window sum)

CLI:
  python -I reuse_ref.py mic   -> runs noisy_audio/mic_test2.wav at its own rate (inference.py),
                                  writes r1/out/ref/, the activation statistics r1/ref_stats_mic_test2.json
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
K = HERE.parent
SRC = Path(os.path.expanduser("~/models/reuse/src"))
sys.path.insert(0, str(HERE))

import mamba_ref  # noqa: E402

mamba_ref.install_stub()
from nv_models.generator_SEMamba_time_d4 import SEMamba  # noqa: E402
from nv_models.stfts import mag_phase_stft, mag_phase_istft  # noqa: E402
from nv_models.util import pad_or_trim_to_match  # noqa: E402

RELU = nn.ReLU()


def load_cfg():
    return json.loads((SRC / "config.json").read_text())


CONV_TILE_BYTES = 768 * 1024 * 1024
_conv2d_forward = nn.Conv2d.forward


def _tiled_conv2d_forward(self, x):
    """nn.Conv2d over row tiles of the output (memory only: torch's CPU dilated conv builds the whole
    im2col buffer, 5 GB for the 256-channel encoder conv on an 8.5 s 16 kHz clip). Each output element
    is the same sum of products as the untiled conv."""
    if (self.padding_mode != "zeros" or self.groups != 1 or x.dim() != 4 or isinstance(self.padding, str)):
        return _conv2d_forward(self, x)
    _, C, H, W = x.shape
    (kh, kw), (sh, sw), (dh, dw), (ph, pw) = self.kernel_size, self.stride, self.dilation, self.padding
    Ho = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    Wo = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    cols = C * kh * kw * Ho * Wo * 4
    if cols <= CONV_TILE_BYTES or Ho < 2:
        return _conv2d_forward(self, x)
    xp = torch.nn.functional.pad(x, (pw, pw, ph, ph))
    rows = max(1, int(Ho * CONV_TILE_BYTES // cols))
    outs = []
    for a in range(0, Ho, rows):
        b = min(Ho, a + rows)
        i0, i1 = a * sh, (b - 1) * sh + dh * (kh - 1) + 1
        outs.append(torch.nn.functional.conv2d(xp[:, :, i0:i1], self.weight, self.bias, self.stride, 0,
                                               self.dilation, 1))
    return torch.cat(outs, 2)


def set_conv_tiling(on=True):
    nn.Conv2d.forward = _tiled_conv2d_forward if on else _conv2d_forward


def load_model(weights=None, tile_convs=True):
    from safetensors.torch import load_file
    set_conv_tiling(tile_convs)
    cfg = load_cfg()
    model = SEMamba(cfg)
    sd = load_file(str(weights or (SRC / "model.safetensors")))
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    model.eval()
    return model, cfg


def make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1


def stft_params(cfg, sr):
    n_fft, hop, win = cfg["stft_cfg"]["n_fft"], cfg["stft_cfg"]["hop_size"], cfg["stft_cfg"]["win_size"]
    base = cfg["stft_cfg"]["sampling_rate"]
    return make_even(n_fft * sr // base), make_even(hop * sr // base), make_even(win * sr // base)


@torch.no_grad()
def model_spec(model, cfg, wav, sr):
    """wav [1, N] -> (amp_g, pha_g) after the zero-frame rule, plus the STFT input (mag, pha)."""
    n_fft, hop, win = stft_params(cfg, sr)
    cf = cfg["model_cfg"]["compress_factor"]
    noisy_mag, noisy_pha, _ = mag_phase_stft(wav, n_fft=n_fft, hop_size=hop, win_size=win,
                                             compress_factor=cf, center=True, addeps=False)
    amp_g, pha_g, _ = model(noisy_mag, noisy_pha)
    mag = torch.expm1(RELU(amp_g))
    zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
    amp_g[:, :, (zero_portion > 0.5)[0]] = 0
    return amp_g, pha_g, noisy_mag, noisy_pha


@torch.no_grad()
def enhance_full(model, cfg, wav, sr):
    """inference.py: wav [1, N] float32 -> enhanced [1, N]."""
    n_fft, hop, win = stft_params(cfg, sr)
    amp_g, pha_g, _, _ = model_spec(model, cfg, wav, sr)
    audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop, win, cfg["model_cfg"]["compress_factor"])
    audio_g = pad_or_trim_to_match(wav.detach(), audio_g, pad_value=1e-8)
    assert audio_g.shape == wav.shape
    return audio_g


def chunk_plan(n, chunk_size, hop_length):
    """inference_chunk.py's chunk starts (the last chunk is aligned to the end)."""
    num_chunks = max(1, math.ceil((n - chunk_size) / hop_length) + 1)
    starts = []
    for i in range(num_chunks):
        starts.append(i * hop_length if i < num_chunks - 1 else n - chunk_size)
    return starts


@torch.no_grad()
def enhance_chunked(model, cfg, wav, sr, chunk_s, hop_portion, chunk_fn=None):
    """inference_chunk.py (one channel). chunk_fn(chunk [1, chunk_size]) -> enhanced chunk [1, chunk_size];
    default = the torch model (enhance_full on the chunk)."""
    chunk_size = int(chunk_s * sr)
    hop_length = int(hop_portion * chunk_size)
    window = torch.clamp(torch.hann_window(chunk_size), min=1e-7)
    n = wav.shape[1]
    assert n >= chunk_size, "clip shorter than one chunk: pad it before calling"
    out = torch.zeros_like(wav)
    wsum = torch.zeros_like(wav)
    starts = chunk_plan(n, chunk_size, hop_length)
    for i, s in enumerate(starts):
        chunk = wav[:, s:s + chunk_size]
        audio_g = enhance_full(model, cfg, chunk, sr) if chunk_fn is None else chunk_fn(chunk)
        if i < len(starts) - 1:
            out[:, s:s + chunk_size] += audio_g * window[0:audio_g.shape[1]]
            wsum[:, s:s + chunk_size] += window[0:audio_g.shape[1]]
        else:
            out[:, -chunk_size:] += audio_g * window[0:audio_g.shape[1]]
            wsum[:, -chunk_size:] += window[0:audio_g.shape[1]]
    return out / wsum


def official(model, cfg, wav, sr):
    """The official form: whole clip (inference.py); clips > 10 s use inference_chunk 5 s / 0.5."""
    if wav.shape[1] > 10 * sr:
        return enhance_chunked(model, cfg, wav, sr, 5.0, 0.5)
    return enhance_full(model, cfg, wav, sr)


class Stats:
    """Activation statistics: norm inputs (amax), Mamba scan inputs (std / amax), per call."""

    def __init__(self, model):
        self.rows = []
        self.handles = []
        for name, m in model.named_modules():
            if isinstance(m, (nn.InstanceNorm2d, nn.LayerNorm)):
                self.handles.append(m.register_forward_pre_hook(self._norm(name)))
            if isinstance(m, mamba_ref.Mamba):
                m._tap_name = name
        mamba_ref.set_tap(self._scan)

    def _norm(self, name):
        def hook(mod, args):
            x = args[0].detach()
            self.rows.append(dict(kind=type(mod).__name__, name=name, shape=list(x.shape),
                                  amax=float(x.abs().max()), std=float(x.std())))
        return hook

    def _scan(self, mod, **t):
        r = dict(kind="scan", name=getattr(mod, "_tap_name", "?"), L=int(t["x"].shape[-1]), B=int(t["x"].shape[0]))
        for k, v in t.items():
            v = v.detach().float()
            r[k + "_std"] = float(v.std())
            r[k + "_amax"] = float(v.abs().max())
        dtb = mod.dt_proj.bias.detach()
        delta = torch.nn.functional.softplus(t["dt"] + dtb[None, :, None])
        r["delta_min"], r["delta_max"] = float(delta.min()), float(delta.max())
        A = -torch.exp(mod.A_log.detach())
        r["dA_min"] = float((delta.max() * A).min())
        self.rows.append(r)

    def close(self):
        for h in self.handles:
            h.remove()
        mamba_ref.set_tap(None)


def read_wav(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(x.T[:1])), sr


def write_wav(path, x, sr):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, dtype=np.float32).reshape(-1), sr, subtype="FLOAT")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "mic"
    torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
    if cmd == "mic":
        model, cfg = load_model()
        n_params = sum(p.numel() for p in model.parameters())
        wav, sr = read_wav(SRC / "noisy_audio" / "mic_test2.wav")
        st = Stats(model)
        t0 = time.time()
        enh = enhance_full(model, cfg, wav, sr)
        dt = time.time() - t0
        st.close()
        od = K / "r1" / "out" / "ref"
        write_wav(od / "mic_test2_noisy.wav", wav, sr)
        write_wav(od / "mic_test2_ref_full.wav", enh, sr)
        n_fft, hop, win = stft_params(cfg, sr)
        summ = dict(clip="noisy_audio/mic_test2.wav", sr=sr, samples=int(wav.shape[1]), n_fft=n_fft, hop=hop, win=win,
                    params=n_params, seconds=round(dt, 2), torch=torch.__version__, threads=torch.get_num_threads(),
                    out_rms=float(enh.pow(2).mean().sqrt()), in_rms=float(wav.pow(2).mean().sqrt()))
        (K / "r1" / "ref_stats_mic_test2.json").write_text(json.dumps(dict(summary=summ, rows=st.rows), indent=1))
        print(json.dumps(summ))
        sc = [r for r in st.rows if r["kind"] == "scan"]
        nm = [r for r in st.rows if r["kind"] != "scan"]
        for k in ("x_std", "dt_std", "B_std", "C_std", "z_std", "x_amax", "dt_amax", "B_amax", "C_amax", "z_amax",
                  "delta_min", "delta_max", "dA_min"):
            v = np.array([r[k] for r in sc])
            print(f"scan {k}: min {v.min():.4g} median {np.median(v):.4g} max {v.max():.4g}")
        for kind in ("InstanceNorm2d", "LayerNorm"):
            v = np.array([r["amax"] for r in nm if r["kind"] == kind])
            print(f"{kind} input amax: n {len(v)} min {v.min():.4g} median {np.median(v):.4g} max {v.max():.4g}")
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
