"""One-command KittenTTS nano LiteRT benchmark (Raspberry Pi friendly).

Dependencies: numpy + ai-edge-litert only (inputs are pre-tokenized in
scripts/bench_inputs.npz; wavs are written with the stdlib wave module).

    python bench.py                       # fp32, 4 threads
    python bench.py --precision fp16
    python bench.py --threads 2 --write-wavs

Reports per-graph latency, per-sentence synthesis latency, RTF, and an output
identity check (durations bit-compare + log-spectrogram correlation) against
the bundled Mac fp32 reference.
"""
import argparse
import time
import wave
from pathlib import Path

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # older package name
    from tflite_runtime.interpreter import Interpreter

SR = 24000
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"


class Graph:
    def __init__(self, path, threads):
        self.it = Interpreter(model_path=str(path), num_threads=threads)
        self.inputs = self.it.get_input_details()
        self.outputs = self.it.get_output_details()

    @staticmethod
    def _canon(name):
        name = name.split(":")[0]
        return name[len("serving_default_"):] if name.startswith("serving_default_") else name

    def __call__(self, feeds):
        for d in self.inputs:
            self.it.resize_tensor_input(d["index"], list(feeds[self._canon(d["name"])].shape))
        self.it.allocate_tensors()
        # the fused LSTM kernels keep hidden state in variable tensors across
        # invocations — reset per utterance or outputs are corrupted
        self.it.reset_all_variables()
        for d in self.inputs:
            self.it.set_tensor(d["index"], feeds[self._canon(d["name"])])
        t0 = time.perf_counter()
        self.it.invoke()
        dt = time.perf_counter() - t0
        return [self.it.get_tensor(o["index"]) for o in self.outputs], dt


def log_spec_corr(a, b, n_fft=1024, hop=256):
    """Phase-invariant similarity: correlation of log magnitude spectrograms."""
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    win = np.hanning(n_fft).astype(np.float32)
    frames = 1 + (m - n_fft) // hop
    idx = np.arange(n_fft)[None] + hop * np.arange(frames)[:, None]
    A = np.log(np.abs(np.fft.rfft(a[idx] * win, axis=1)) + 1e-5)
    B = np.log(np.abs(np.fft.rfft(b[idx] * win, axis=1)) + 1e-5)
    return float(np.corrcoef(A.ravel(), B.ravel())[0, 1])


def write_wav(path, wav):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=OUT)
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--write-wavs", action="store_true")
    args = ap.parse_args()

    suffix = "" if args.precision == "fp32" else "_fp16"
    graphs = {name: Graph(args.models_dir / f"kitten_{name}{suffix}.tflite", args.threads)
              for name in ("predictor", "prosody", "vocoder")}
    data = np.load(HERE / "bench_inputs.npz")
    n = int(data["n_sentences"])
    speed = np.array([1.0], dtype=np.float32)

    print(f"KittenTTS nano LiteRT bench  precision={args.precision} "
          f"threads={args.threads} runs={args.runs}")
    total_audio = total_compute = 0.0
    for i in range(n):
        ids, style = data[f"ids_{i}"], data[f"style_{i}"]
        best = None
        for _ in range(args.runs):
            p_out, tp = graphs["predictor"]({"input_ids": ids, "style": style, "speed": speed})
            d = [o for o in p_out if o.ndim == 3 and o.shape[-1] == 256][0]
            t_en = [o for o in p_out if o.ndim == 3 and o.shape[-1] == 128][0]
            dur = [o for o in p_out if o.dtype in (np.int32, np.int64)][0]
            en = np.repeat(d[0], dur, axis=0)[None].astype(np.float32)
            asr = np.repeat(t_en[0], dur, axis=0)[None].astype(np.float32)
            pr_out, tpr = graphs["prosody"]({"en": en, "style": style})
            har = [o for o in pr_out if o.ndim == 3][0]
            f0, nn = [o for o in pr_out if o.ndim == 2]
            v_out, tv = graphs["vocoder"]({"asr": asr, "f0": f0, "n": nn,
                                           "har": har, "style": style})
            wav = v_out[0][0]
            total = tp + tpr + tv
            if best is None or total < best[0]:
                best = (total, tp, tpr, tv, wav, dur)
        total, tp, tpr, tv, wav, dur = best
        audio_s = len(wav) / SR
        dur_ok = np.array_equal(dur.astype(np.int64), data[f"ref_dur_{i}"].astype(np.int64))
        sim = log_spec_corr(wav, data[f"ref_wav_{i}"])
        total_audio += audio_s
        total_compute += total
        print(f"[{i}] N={ids.shape[1]:3d} audio={audio_s:5.2f}s  "
              f"pred={tp*1e3:6.1f}ms pro={tpr*1e3:6.1f}ms voc={tv*1e3:6.1f}ms  "
              f"sentence={total*1e3:6.1f}ms RTF={total/audio_s:.3f}  "
              f"durations={'OK' if dur_ok else 'MISMATCH'} logspec-corr={sim:.4f}")
        if args.write_wavs:
            write_wav(OUT / f"bench_{args.precision}_{i}.wav", wav)
    print(f"overall RTF={total_compute/total_audio:.3f} "
          f"({total_compute*1e3:.0f}ms compute / {total_audio:.2f}s audio)")


if __name__ == "__main__":
    main()
