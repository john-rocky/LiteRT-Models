#!/usr/bin/env python3
"""Compare Pixel 8a GPU outputs (real/imag spectrograms) against torch reference."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stub_tiger  # noqa: F401
import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "TIGER"))
from gpu_tiger import host_istft  # noqa: E402

SUB = sys.argv[1] if len(sys.argv) > 1 else "dialog"
SR, WIN, HOP, ENC, K = 44100, 2048, 512, 1025, 3
S = (1040 - 1) * HOP           # keep in sync with build_tiger.py
T = S // HOP + 1

r = np.fromfile(os.path.join(HERE, "dev_out_r.bin"), np.float32).reshape(1, K, ENC, T)
i = np.fromfile(os.path.join(HERE, "dev_out_i.bin"), np.float32).reshape(1, K, ENC, T)
print("device spec absmax", np.abs(r).max(), np.abs(i).max(),
      "NaN:", np.isnan(r).sum() + np.isnan(i).sum())

est = host_istft(torch.from_numpy(r).view(K, ENC, T), torch.from_numpy(i).view(K, ENC, T),
                 WIN, HOP, S)  # [K, S]
ref = np.load(os.path.join(HERE, f"tiger_{SUB}_ref.npy"))  # [K, S] torch original
for k in range(K):
    c = np.corrcoef(est[k].numpy(), ref[k])[0, 1]
    print(f"src{k}: device-GPU vs torch corr {c:.6f}")
c_all = np.corrcoef(est.numpy().ravel(), ref.ravel())[0, 1]
print(f"ALL: device-GPU vs torch wav corr {c_all:.6f}")

# save audition wavs (mixture chunk + separated)
wav, _ = torchaudio.load(os.path.join(HERE, "TIGER", "test", "test_mixture_466.wav"))
torchaudio.save(os.path.join(HERE, "dev_mix.wav"), wav[:, :S], SR)
names = {0: "music", 1: "effect", 2: "dialog"}  # DnR track order per inference_dnr.py
for k in range(K):
    torchaudio.save(os.path.join(HERE, f"dev_{SUB}_src{k}_{names.get(k,'')}.wav"),
                    est[k:k + 1], SR)
print("saved dev_*.wav for audition")
