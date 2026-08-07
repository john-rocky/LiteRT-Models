"""Create scripts/bench_inputs.npz for bench.py (run on a machine with espeak).

Pre-tokenizes the bench sentences and records the fp32 LiteRT reference output
(wav + durations) so the benchmark itself needs only numpy + ai-edge-litert —
no phonemizer/espeak on the target device.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from verify_kitten_litert import tokens_for, synthesize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VOICES = ROOT / "models" / "nano-0.8-fp32" / "voices.npz"
VOICE = "expr-voice-2-m"

SENTENCES = [
    "Hello! How can I help you today?",
    "The weather looks great for a walk in the park this afternoon.",
    "Streaming text to speech now runs entirely on the LiteRT runtime, "
    "with dynamic sequence lengths and no fixed buckets.",
]


def main():
    voices = np.load(VOICES)
    out = {"n_sentences": np.array(len(SENTENCES))}
    for i, text in enumerate(SENTENCES):
        ids = tokens_for(text)
        ref_id = min(len(text), voices[VOICE].shape[0] - 1)
        style = voices[VOICE][ref_id:ref_id + 1].astype(np.float32)
        wav, st = synthesize(ids, style)
        out[f"ids_{i}"] = ids
        out[f"style_{i}"] = style
        out[f"ref_wav_{i}"] = wav.astype(np.float32)
        out[f"ref_dur_{i}"] = st["dur"]
        print(f"[{i}] N={ids.shape[1]} T={int(st['dur'].sum())} "
              f"audio={len(wav)/24000:.2f}s  {text[:50]}...")
    np.savez(Path(__file__).parent / "bench_inputs.npz", **out)
    print("wrote scripts/bench_inputs.npz")


if __name__ == "__main__":
    main()
