"""Tap: LayerNorm input magnitudes and the residual stream of the re-authored encoder (fp32, torch).

On the L=541 low_latency step: max|x| of the input of every LayerNorm (64 = input LN + 2 x 31 layer
LNs + final LN) with the largest per-row variance, and max|x| of the residual stream after each of
the 31 layers. A second probe splits valid rows from zero-padded rows on the shortest step (the pad
rows run through the graph too). Reports only; no verdict.

Usage: .venv/bin/python tap_ln.py --run-dir <run> --model-dir <dir> --fixture diarization_example_16k
"""

import argparse
import json
import os

import numpy as np
import torch

import nemotron3diar_model as n3d


def tap(encoder, packed, length, cos, sin):
  ln_rows, res_rows = [], []
  names = {m: n for n, m in encoder.named_modules()}
  hooks = []

  def ln_hook(module, args):
    x = args[0][0]  # [T,512]
    ln_rows.append((names[module], x.detach().clone()))

  def layer_hook(module, args, output):
    res_rows.append((names[module], output[0].detach().clone()))

  for m in encoder.modules():
    if isinstance(m, torch.nn.LayerNorm):
      hooks.append(m.register_forward_pre_hook(ln_hook))
    if isinstance(m, n3d.N3DLayer):
      hooks.append(m.register_forward_hook(layer_hook))
  with torch.no_grad():
    encoder(torch.from_numpy(packed), torch.from_numpy(n3d.attn_bias_for(length)), cos, sin)
  for h in hooks:
    h.remove()
  return ln_rows, res_rows


def stats(x, rows):
  x = x[rows]
  return dict(max_abs=float(x.abs().max()),
              max_row_var=float(x.var(dim=-1, unbiased=False).max()),
              max_row_abs_mean=float(x.mean(dim=-1).abs().max()))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--model-dir", required=True)
  ap.add_argument("--fixture", default="diarization_example_16k")
  args = ap.parse_args()
  results = os.path.join(args.run_dir, "results")
  _, encoder, _, _ = n3d.load_models(os.path.join(args.model_dir, "model.safetensors"))
  cos, sin = n3d.rope_tables()
  d = np.load(os.path.join(results, f"chunk_io_{args.fixture}.npz"))
  lengths = d["length"]
  full = [int(i) for i in np.nonzero(lengths == n3d.T_LOW_LATENCY)[0]]
  assert full, "no L=541 step in this fixture"
  i = full[0]
  L = int(lengths[i])
  ln_rows, res_rows = tap(encoder, d["packed_embeds"][i][None], L, cos, sin)
  assert len(ln_rows) == 2 * n3d.LAYERS + 2 and len(res_rows) == n3d.LAYERS
  all_rows = slice(0, L)
  report = dict(
      fixture=args.fixture, step=i, L=L, packed_input_max_abs=float(np.abs(d["packed_embeds"][i, :L]).max()),
      layer_norm_count=len(ln_rows),
      layer_norm_inputs=[dict(idx=k, name=n, **stats(x, all_rows)) for k, (n, x) in enumerate(ln_rows)],
      residual_after_layer=[dict(layer=k, name=n, **stats(x, all_rows)) for k, (n, x) in enumerate(res_rows)],
  )

  # valid vs zero-padded rows on the shortest step
  j = int(np.argmin(lengths))
  Lj = int(lengths[j])
  ln_rows, res_rows = tap(encoder, d["packed_embeds"][j][None], Lj, cos, sin)
  valid, pad = slice(0, Lj), slice(Lj, n3d.T_LOW_LATENCY)
  report["pad_probe"] = dict(
      step=j, L=Lj, pad_rows=n3d.T_LOW_LATENCY - Lj,
      ln_input_max_abs_valid=max(float(x[valid].abs().max()) for _, x in ln_rows),
      ln_input_max_abs_pad=max(float(x[pad].abs().max()) for _, x in ln_rows),
      residual_max_abs_valid=max(float(x[valid].abs().max()) for _, x in res_rows),
      residual_max_abs_pad=max(float(x[pad].abs().max()) for _, x in res_rows),
      all_finite=bool(all(torch.isfinite(x).all() for _, x in ln_rows + res_rows)),
  )
  with open(os.path.join(results, "ln_tap.json"), "w") as f:
    json.dump(report, f, indent=1)

  print(f"step {i} L={L} packed max|x| {report['packed_input_max_abs']:.3f}  LN count {len(report['layer_norm_inputs'])}")
  print(f"{'idx':>3} {'layer norm input':<22} {'max|x|':>10} {'max row var':>12} {'max |row mean|':>14}")
  for r in report["layer_norm_inputs"]:
    print(f"{r['idx']:>3} {r['name']:<22} {r['max_abs']:>10.3f} {r['max_row_var']:>12.3f} {r['max_row_abs_mean']:>14.3f}")
  print(f"{'layer':>5} {'residual after layer':<22} {'max|x|':>10}")
  for r in report["residual_after_layer"]:
    print(f"{r['layer']:>5} {r['name']:<22} {r['max_abs']:>10.3f}")
  print("pad probe:", json.dumps(report["pad_probe"]))


if __name__ == "__main__":
  main()
