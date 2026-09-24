"""Gate: the re-authored fixed-T model (nemotron3diar_model.py, fp32) vs transformers, step by step.

For every captured low_latency step (chunk_io_<fixture>.npz from make_reference.py):
  graph B  logits[:L*8] of N3DEncoder(packed [1,541,512], attn_bias, rope_cos, rope_sin)
           vs transformers chunk_logits [L*8,8]          PASS: max|dlogit| <= 1e-4 and corr >= 0.999999
  graph A  N3DFrontend(mel [1,104,128]) rows vs the chunk rows at the tail of the packed input
Negative controls (must go red): RoPE off (cos=1, sin=0), and attn_bias = 0 (pad rows visible).
Row-mask ablation: without the in-graph row mask only the last real row's 8 logit rows move.
Capture self-checks: returned logits == chunk_logits at the cached offset, their concatenation ==
the saved streaming logits, L == cache + FIFO + chunk rows, zero rows beyond L.

Usage: .venv/bin/python gate_reauthor.py --run-dir <run> --model-dir <dir> --fixtures name1 name2
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

import nemotron3diar_model as n3d

MAX_ABS = 1e-4
MIN_CORR = 0.999999


def corr(a, b):
  a = a.astype(np.float64).ravel()
  b = b.astype(np.float64).ravel()
  a -= a.mean()
  b -= b.mean()
  return float((a @ b) / math.sqrt((a @ a) * (b @ b)))


def run_encoder(encoder, packed, bias, cos, sin):
  with torch.no_grad():
    return encoder(
        torch.from_numpy(packed), torch.from_numpy(bias), cos, sin
    )[0].numpy()


def run_encoder_without_row_mask(encoder, packed, bias, cos, sin):
  """N3DEncoder.forward minus the row mask before the sub-pixel conv (ablation only)."""
  t = packed.shape[1]
  with torch.no_grad():
    b = torch.from_numpy(bias)
    x = encoder.ln_in(torch.from_numpy(packed))
    for layer in encoder.layers:
      x = layer(x, b, cos, sin)
    y = encoder.proj(encoder.ln_out(x))
    y = encoder.up(y.transpose(1, 2)).transpose(1, 2).reshape(1, t * n3d.STACK, n3d.HEAD_HIDDEN)
    return encoder.out(torch.relu(encoder.dense(torch.relu(y))))[0].numpy()


def capture_self_checks(d, streaming_logits):
  n = d["length"].shape[0]
  worst, outs = 0.0, []
  for i in range(n):
    c0 = int(d["cache_pre"][i, 0] + d["cache_pre"][i, 1]) * 8
    o = int(d["out_frames"][i])
    worst = max(worst, float(np.abs(d["out_logits"][i, :o] - d["chunk_logits"][i, c0 : c0 + o]).max()))
    outs.append(d["out_logits"][i, :o])
  cat = np.concatenate(outs, 0)
  rows = -(-d["mel_frames"] // 8)
  return dict(
      out_logits_vs_chunk_logits_slice_max_abs=worst,
      concat_equals_streaming_logits=bool(cat.shape == streaming_logits.shape
                                          and np.array_equal(cat, streaming_logits)),
      length_equals_cache_fifo_rows=bool(np.array_equal(
          d["length"], d["cache_pre"][:, 0] + d["cache_pre"][:, 1] + rows)),
      zero_rows_beyond_L=bool(all(np.all(d["packed_embeds"][i, d["length"][i]:] == 0) for i in range(n))),
  )


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--model-dir", required=True)
  ap.add_argument("--fixtures", nargs="+", required=True)
  args = ap.parse_args()
  results = os.path.join(args.run_dir, "results")

  frontend, encoder, _, n_params = n3d.load_models(os.path.join(args.model_dir, "model.safetensors"))
  print(f"strict load OK: {n_params} params", flush=True)
  cos, sin = n3d.rope_tables()
  ones, zeros = torch.ones_like(cos), torch.zeros_like(sin)
  t_all = n3d.T_LOW_LATENCY

  report = {"params": n_params, "thresholds": {"max_abs": MAX_ABS, "min_corr": MIN_CORR}, "fixtures": {}}
  all_pass = True
  for name in args.fixtures:
    d = np.load(os.path.join(results, f"chunk_io_{name}.npz"))
    n = d["length"].shape[0]
    steps = []
    t0 = time.time()
    for i in range(n):
      L = int(d["length"][i])
      packed = d["packed_embeds"][i][None]
      ref = d["chunk_logits"][i, : L * 8]
      got = run_encoder(encoder, packed, n3d.attn_bias_for(L), cos, sin)[: L * 8]
      mabs = float(np.abs(got - ref).max())
      c = corr(got, ref)

      # graph A: the chunk rows (with look-ahead) are the last ceil(F/8) rows of the packed input
      f = int(d["mel_frames"][i])
      rows = -(-f // 8)
      with torch.no_grad():
        emb = frontend(torch.from_numpy(d["input_features"][i][None]))[0].numpy()
      fa = float(np.abs(emb[:rows] - d["packed_embeds"][i, L - rows : L]).max())
      steps.append(dict(step=i, L=L, max_abs=mabs, corr=c, frontend_max_abs=fa,
                        compressed=bool(d["cache_pre"][i, 2]),
                        passed=bool(mabs <= MAX_ABS and c >= MIN_CORR)))
    dt = time.time() - t0

    lengths = d["length"]
    full = [s["step"] for s in steps if s["L"] == t_all]
    after_compress = [s["step"] for s in steps if s["compressed"]]
    worst = max(steps, key=lambda s: s["max_abs"])
    fx_pass = all(s["passed"] for s in steps)

    # negative controls on a few representative steps
    probe = sorted(set([0, int(np.argmax(lengths))]
                       + ([after_compress[0]] if after_compress else [])
                       + [i for i in range(n) if lengths[i] < t_all][:1]))
    neg = []
    for i in probe:
      L = int(lengths[i])
      packed = d["packed_embeds"][i][None]
      ref = d["chunk_logits"][i, : L * 8]
      got = run_encoder(encoder, packed, n3d.attn_bias_for(L), ones, zeros)[: L * 8]
      neg.append(dict(control="rope_off", step=i, L=L, max_abs=float(np.abs(got - ref).max()),
                      corr=corr(got, ref)))
      if L < t_all:
        got = run_encoder(encoder, packed, np.zeros((1, 1, 1, t_all), np.float32), cos, sin)[: L * 8]
        neg.append(dict(control="attn_bias_zero", step=i, L=L,
                        max_abs=float(np.abs(got - ref).max()), corr=corr(got, ref)))
    for c_ in neg:
      c_["red"] = bool(c_["max_abs"] > MAX_ABS or c_["corr"] < MIN_CORR)
    neg_ok = all(c_["red"] for c_ in neg) and {c_["control"] for c_ in neg} == {"rope_off", "attn_bias_zero"}

    # row-mask ablation: which logit rows depend on the mask
    ablation = []
    for i in probe:
      L = int(lengths[i])
      packed = d["packed_embeds"][i][None]
      ref = d["chunk_logits"][i, : L * 8]
      got = run_encoder_without_row_mask(encoder, packed, n3d.attn_bias_for(L), cos, sin)[: L * 8]
      last = slice((L - 1) * 8, L * 8)
      ablation.append(dict(step=i, L=L,
                           rows_before_last_max_abs=float(np.abs(got[: (L - 1) * 8] - ref[: (L - 1) * 8]).max())
                           if L > 1 else 0.0,
                           last_row_max_abs=float(np.abs(got[last] - ref[last]).max())))
    checks = capture_self_checks(
        d, np.load(os.path.join(results, f"ref_{name}_low_latency.npz"))["logits"])

    summary = dict(
        steps=n, seconds=round(dt, 1), passed_steps=sum(s["passed"] for s in steps),
        max_abs=worst["max_abs"], max_abs_step=worst["step"],
        min_corr=min(s["corr"] for s in steps),
        frontend_max_abs=max(s["frontend_max_abs"] for s in steps),
        steps_L541=full, steps_after_compress=len(after_compress),
        first_step_after_compress=after_compress[0] if after_compress else None,
        L541_max_abs=max((s["max_abs"] for s in steps if s["L"] == t_all), default=None),
        after_compress_max_abs=max((s["max_abs"] for s in steps if s["compressed"]), default=None),
        negative_controls=neg, negative_controls_all_red=neg_ok,
        row_mask_ablation=ablation, capture_self_checks=checks,
        PASS=bool(fx_pass and neg_ok and checks["out_logits_vs_chunk_logits_slice_max_abs"] == 0.0
                  and checks["concat_equals_streaming_logits"] and checks["length_equals_cache_fifo_rows"]
                  and checks["zero_rows_beyond_L"]),
    )
    all_pass &= summary["PASS"]
    report["fixtures"][name] = dict(summary=summary, steps=steps)
    print(json.dumps({name: summary}, indent=1), flush=True)

  report["PASS"] = bool(all_pass)
  with open(os.path.join(results, "gate_reauthor.json"), "w") as f:
    json.dump(report, f, indent=1)
  print("GATE_REAUTHOR", "PASS" if all_pass else "FAIL", flush=True)


if __name__ == "__main__":
  main()
