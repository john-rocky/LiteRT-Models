"""Create scripts/bench_inputs.npz for bench.py (run on a machine with espeak).

Pre-tokenizes the bench sentences and records the fp32 LiteRT reference wavs so
the benchmark itself needs only numpy + ai-edge-litert on the target device.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoint"
sys.path.insert(0, str(CKPT / "runtime"))
sys.path.insert(0, str(CKPT))
sys.path.insert(0, str(Path(__file__).parent))

from verify_inflect_litert import synthesize  # noqa: E402

SENTENCES = [
    "Hello! How can I help you today?",
    "The weather looks great for a walk in the park this afternoon.",
    "Streaming text to speech now runs entirely on the LiteRT runtime, "
    "with dynamic sequence lengths and no fixed buckets.",
]


def tokens_for(text):
    import commons
    from inflect_vits_frontend import run_vits_frontend
    from text import cleaned_text_to_sequence
    seq = commons.intersperse(
        cleaned_text_to_sequence(run_vits_frontend(text).phoneme_text), 0)
    return np.array(seq, dtype=np.int32)[None]


def main():
    out = {"n_sentences": np.array(len(SENTENCES))}
    for i, text in enumerate(SENTENCES):
        tokens = tokens_for(text)
        wav, z_p, _ = synthesize(tokens, seed=7)
        out[f"ids_{i}"] = tokens
        out[f"ref_wav_{i}"] = wav.astype(np.float32)
        print(f"[{i}] N={tokens.shape[1]} T={z_p.shape[1]} "
              f"audio={len(wav)/24000:.2f}s  {text[:50]}...")
    np.savez(Path(__file__).parent / "bench_inputs.npz", **out)
    print("wrote scripts/bench_inputs.npz")


if __name__ == "__main__":
    main()
