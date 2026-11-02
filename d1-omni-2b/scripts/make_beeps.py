"""Writes app/src/main/assets/beeps.wav: the model card's recording (three 880 Hz beeps at 0.5, 1.5 and 2.5 s over
quiet noise, 4 s at 16 kHz, made by the card's own numpy code) as a 16 kHz mono 16-bit PCM WAV file, the samples
scaled as the model repository's test sounds are (np.clip(x, -1, 1) * 32767, truncated to int16).

    pip install numpy
    python -B scripts/make_beeps.py [--out app/src/main/assets/beeps.wav]
"""
from __future__ import annotations

import argparse
import os
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def recording(seconds: float = 4.0) -> tuple[np.ndarray, int]:
    rate = 16000
    clip = np.random.default_rng(0).normal(0, 0.003, int(seconds * rate)).astype(np.float32)
    for start in (0.5, 1.5, 2.5):
        clip[int(start * rate):int(start * rate) + 3200] += 0.3 * np.sin(2 * np.pi * 880 * np.arange(3200) / rate)
    return clip, rate


def write_wav(path: str, samples: np.ndarray, rate: int) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.astype("<i2").tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "app", "src", "main", "assets", "beeps.wav"))
    a = ap.parse_args()
    clip, rate = recording()
    write_wav(a.out, clip, rate)
    print(f"{a.out}: {len(clip)} samples at {rate} Hz")


if __name__ == "__main__":
    main()
