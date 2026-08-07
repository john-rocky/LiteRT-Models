"""Verify the Inflect LiteRT graphs: accuracy, dynamic lengths, streaming, RTF.

Runs in the torch venv (has ai_edge_litert + the espeak frontend).

Checks:
  1. Golden sentence: tflite text_encoder + decoder vs torch goldens (corr).
  2. A second, different-length sentence end-to-end (dynamic shape proof),
     wav written for listening.
  3. Streaming: decoder run on overlapping z_p chunks, stitched output must
     match the full-utterance decode (the decoder is fully convolutional).
  4. Mac CPU timing reference: encoder/decoder wall time, RTF.
"""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoint"
OUT = ROOT / "out"
sys.path.insert(0, str(CKPT / "runtime"))
sys.path.insert(0, str(CKPT))

SR = 24000
HOP = 256
NOISE_SCALE = 0.667
CHUNK = 100          # z_p frames per streaming chunk (~1.07 s of audio)
OVERLAP = 64         # frames of context on each side, discarded after decode

from ai_edge_litert.interpreter import Interpreter  # noqa: E402


def run_graph(path, x):
    it = Interpreter(model_path=str(path), num_threads=4)
    ind = it.get_input_details()[0]
    it.resize_tensor_input(ind["index"], list(x.shape))
    it.allocate_tensors()
    it.set_tensor(ind["index"], x)
    t0 = time.perf_counter()
    it.invoke()
    dt = time.perf_counter() - t0
    outs = {o["name"]: it.get_tensor(o["index"]) for o in it.get_output_details()}
    return outs, dt


def corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    n = min(len(a), len(b))
    return float(np.corrcoef(a[:n], b[:n])[0, 1])


def synthesize(tokens, seed=7):
    """Full host pipeline on the two tflite graphs. Returns wav + timings."""
    enc_out, t_enc = run_graph(OUT / "inflect_text_encoder.tflite",
                               tokens.astype(np.int32))
    vals = list(enc_out.values())
    m_p = [v for v in vals if v.shape[-1] == 128][0]
    logs_p = [v for v in vals if v.shape[-1] == 128][1]
    logw = [v for v in vals if v.shape[-1] == 1][0]

    w = np.exp(logw[0, :, 0])
    durations = np.ceil(w).astype(np.int64)
    m_p_exp = np.repeat(m_p[0], durations, axis=0)[None]      # [1,T,128]
    logs_p_exp = np.repeat(logs_p[0], durations, axis=0)[None]
    rng = np.random.RandomState(seed)
    noise = rng.randn(*m_p_exp.shape).astype(np.float32)
    z_p = m_p_exp + noise * np.exp(logs_p_exp) * NOISE_SCALE

    dec_out, t_dec = run_graph(OUT / "inflect_decoder.tflite",
                               z_p.astype(np.float32))
    wav = list(dec_out.values())[0][0]
    return wav, z_p, (t_enc, t_dec)


def stream_decode(z_p):
    """Overlap-discard chunked decoding; returns stitched wav + per-chunk times."""
    t_frames = z_p.shape[1]
    pieces, times = [], []
    start = 0
    while start < t_frames:
        end = min(start + CHUNK, t_frames)
        lo = max(0, start - OVERLAP)
        hi = min(t_frames, end + OVERLAP)
        out, dt = run_graph(OUT / "inflect_decoder.tflite",
                            z_p[:, lo:hi].astype(np.float32))
        wav = list(out.values())[0][0]
        pieces.append(wav[(start - lo) * HOP:(start - lo + end - start) * HOP])
        times.append(dt)
        start = end
    return np.concatenate(pieces), times


def main():
    gold = dict(np.load(OUT / "inflect_golden.npz"))

    # 1. golden sentence, exact-noise comparison via golden z_p
    enc_out, t_enc = run_graph(OUT / "inflect_text_encoder.tflite",
                               gold["tokens"].astype(np.int32))
    vals = list(enc_out.values())
    m_p = [v for v in vals if v.shape[-1] == 128][0]
    logw = [v for v in vals if v.shape[-1] == 1][0]
    print(f"enc m_p maxerr={np.abs(m_p - gold['m_p'].transpose(0, 2, 1)).max():.2e}  "
          f"logw maxerr={np.abs(logw - gold['logw'].transpose(0, 2, 1)).max():.2e}")

    z_p_gold = gold["z_p"].transpose(0, 2, 1).astype(np.float32)
    dec_out, t_dec = run_graph(OUT / "inflect_decoder.tflite", z_p_gold)
    wav = list(dec_out.values())[0][0]
    ref = gold["wav"][0, 0]
    dur_s = len(ref) / SR
    print(f"golden wav: corr={corr(wav, ref):.6f} maxerr={np.abs(wav - ref).max():.2e}")
    print(f"timing: enc={t_enc*1e3:.0f}ms dec={t_dec*1e3:.0f}ms audio={dur_s:.2f}s "
          f"RTF={(t_enc+t_dec)/dur_s:.3f}")
    sf.write(OUT / "litert_golden.wav", wav, SR)

    # 2. second sentence (different length) through the full host pipeline
    from inflect_vits_frontend import run_vits_frontend
    from text import cleaned_text_to_sequence
    import commons
    for tag, text in [("short", "Good morning everyone."),
                      ("long", "The quick brown fox jumps over the lazy dog, "
                               "while seventy six trombones led the big parade.")]:
        seq = commons.intersperse(
            cleaned_text_to_sequence(run_vits_frontend(text).phoneme_text), 0)
        tokens = np.array(seq, dtype=np.int32)[None]
        wav2, z_p2, (te, td) = synthesize(tokens)
        print(f"[{tag}] N={tokens.shape[1]} T={z_p2.shape[1]} audio={len(wav2)/SR:.2f}s "
              f"enc={te*1e3:.0f}ms dec={td*1e3:.0f}ms RTF={(te+td)/(len(wav2)/SR):.3f}")
        sf.write(OUT / f"litert_{tag}.wav", wav2, SR)

        # 3. streaming on the same z_p
        swav, stimes = stream_decode(z_p2)
        n = min(len(swav), len(wav2))
        print(f"[{tag}] streaming: chunks={len(stimes)} "
              f"first-chunk={stimes[0]*1e3:.0f}ms "
              f"corr(vs full)={corr(swav[:n], wav2[:n]):.6f} "
              f"maxerr={np.abs(swav[:n] - wav2[:n]).max():.2e}")
        sf.write(OUT / f"litert_{tag}_streamed.wav", swav, SR)


if __name__ == "__main__":
    main()
