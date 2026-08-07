"""One-command Inflect-Nano-v2 LiteRT benchmark (Raspberry Pi friendly).

Dependencies: numpy + ai-edge-litert only (inputs are pre-tokenized in
scripts/bench_inputs.npz; wavs are written with the stdlib wave module).

    python bench.py                       # fp32, 4 threads, full + streaming
    python bench.py --precision fp16
    python bench.py --threads 2 --write-wavs

Reports encoder/decoder latency, RTF, streaming time-to-first-audio, and an
output identity check (waveform correlation, noise is seeded) against the
bundled Mac fp32 reference.
"""
import argparse
import time
import wave
from pathlib import Path

import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

SR = 24000
HOP = 256
NOISE_SCALE = 0.667
CHUNK = 100
OVERLAP = 64
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"


class Graph:
    def __init__(self, path, threads):
        self.it = Interpreter(model_path=str(path), num_threads=threads)
        self.inp = self.it.get_input_details()[0]
        self.outs = self.it.get_output_details()

    def __call__(self, x):
        self.it.resize_tensor_input(self.inp["index"], list(x.shape))
        self.it.allocate_tensors()
        self.it.set_tensor(self.inp["index"], x)
        t0 = time.perf_counter()
        self.it.invoke()
        dt = time.perf_counter() - t0
        return [self.it.get_tensor(o["index"]) for o in self.outs], dt


def write_wav(path, wav):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes())


def prepare_zp(enc_out, seed=7):
    vals = list(enc_out)
    m_p = [v for v in vals if v.shape[-1] == 128][0]
    logs_p = [v for v in vals if v.shape[-1] == 128][1]
    logw = [v for v in vals if v.shape[-1] == 1][0]
    durations = np.ceil(np.exp(logw[0, :, 0])).astype(np.int64)
    m_p_exp = np.repeat(m_p[0], durations, axis=0)[None]
    logs_p_exp = np.repeat(logs_p[0], durations, axis=0)[None]
    noise = np.random.RandomState(seed).randn(*m_p_exp.shape).astype(np.float32)
    return (m_p_exp + noise * np.exp(logs_p_exp) * NOISE_SCALE).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=OUT)
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--write-wavs", action="store_true")
    args = ap.parse_args()

    suffix = "" if args.precision == "fp32" else "_fp16"
    enc = Graph(args.models_dir / f"inflect_text_encoder{suffix}.tflite", args.threads)
    dec = Graph(args.models_dir / f"inflect_decoder{suffix}.tflite", args.threads)
    data = np.load(HERE / "bench_inputs.npz")
    n = int(data["n_sentences"])

    print(f"Inflect-Nano-v2 LiteRT bench  precision={args.precision} "
          f"threads={args.threads} runs={args.runs}")
    total_audio = total_compute = 0.0
    for i in range(n):
        ids = data[f"ids_{i}"]
        best = None
        for _ in range(args.runs):
            e_out, te = enc(ids)
            z_p = prepare_zp(e_out)
            d_out, td = dec(z_p)
            wav = d_out[0][0]
            if best is None or te + td < best[0]:
                best = (te + td, te, td, wav, z_p)
        total, te, td, wav, z_p = best
        audio_s = len(wav) / SR
        m = min(len(wav), len(data[f"ref_wav_{i}"]))
        sim = float(np.corrcoef(wav[:m], data[f"ref_wav_{i}"][:m])[0, 1])
        total_audio += audio_s
        total_compute += total

        # streaming: overlap-discard chunks (exact for this decoder)
        t_frames = z_p.shape[1]
        pieces, first = [], None
        t0 = time.perf_counter()
        start = 0
        while start < t_frames:
            end = min(start + CHUNK, t_frames)
            lo, hi = max(0, start - OVERLAP), min(t_frames, end + OVERLAP)
            c_out, _ = dec(z_p[:, lo:hi])
            cw = c_out[0][0]
            a = (start - lo) * HOP
            pieces.append(cw[a:a + (end - start) * HOP])
            if first is None:
                first = time.perf_counter() - t0
            start = end
        swav = np.concatenate(pieces)
        m2 = min(len(swav), len(wav))
        scorr = float(np.corrcoef(swav[:m2], wav[:m2])[0, 1])

        print(f"[{i}] N={ids.shape[1]:3d} audio={audio_s:5.2f}s  "
              f"enc={te*1e3:5.1f}ms dec={td*1e3:6.1f}ms sentence={total*1e3:6.1f}ms "
              f"RTF={total/audio_s:.3f}  TTFA={(te+first)*1e3:6.1f}ms "
              f"stream-corr={scorr:.6f}  ref-corr={sim:.6f}")
        if args.write_wavs:
            write_wav(OUT / f"bench_{args.precision}_{i}.wav", wav)
    print(f"overall RTF={total_compute/total_audio:.3f} "
          f"({total_compute*1e3:.0f}ms compute / {total_audio:.2f}s audio)")


if __name__ == "__main__":
    main()
