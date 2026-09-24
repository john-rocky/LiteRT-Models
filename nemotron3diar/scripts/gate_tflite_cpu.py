"""Gate: exported .tflite graphs on the Mac CPU (LiteRT CompiledModel) vs transformers + op census.

graph A  n3d_frontend.tflite(mel [1,104,128]) vs the chunk rows of the transformers packed input
         PASS: max|d| <= 1e-5
graph B  n3d_encoder_ll.tflite logits[:L*8] vs transformers chunk_logits
         fp32 PASS: max|dlogit| <= 1e-3;  fp16 (weights): report max|dp| after sigmoid and
         speaker-activity agreement @0.5 (target >= 99.9 %)
census   per graph: op histogram, the BANNED set below (ops kept out of LiteRT GPU graphs; GELU reported
         apart, the design keeps native GELU), GELU count + approximate flag, tensors with rank > 4, op total
         (flatbuffer parse; an Interpreter would add one XNNPACK DELEGATE node)

--ln safe|safe_guide gates n3d_encoder_ll_<ln>{,_fp16}.tflite (SafeLayerNorm v2, see
nemotron3diar_model.py) instead and also compares that fp32 graph with the plain fp32 graph on the same
inputs (the LN swap alone); results go to gate_tflite_cpu_<ln>.json / opcheck_<ln>.json.

--mode offline (round 3) gates n3d_encoder_off_<ln>{,_fp16}.tflite (T=684, two-level row mask) on every chunk of the
transformers offline forward (results/chunk_io_<fixture>_offline.npz: the step mask marks the masked key row, fed
as attn_bias -16384; pad rows -32768), checks the low_latency graph A run over 104-frame blocks of the offline
features against the whole-recording embeddings, and runs a negative control (the masked row fed as padding).
Results: gate_tflite_cpu_offline_<ln>.json, opcheck_offline_<ln>.json.

Usage: .venv/bin/python gate_tflite_cpu.py --run-dir <run> --fixtures name1 name2 [--steps all|i,j,k]
           [--ln plain|safe|safe_guide] [--mode low_latency|offline]
"""

import argparse
import collections
import json
import math
import os
import time

import numpy as np

BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP",
          "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM"}
A_MAX = 1e-5
B_FP32_MAX = 1e-3
T_ALL = 541


def census(path):
  from ai_edge_litert import schema_py_generated as schema
  with open(path, "rb") as f:
    buf = f.read()
  model = schema.Model.GetRootAs(buf, 0)
  names = {v: k for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("_")}
  tnames = {v: k for k, v in vars(schema.TensorType).items() if not k.startswith("_")}
  codes = [model.OperatorCodes(i) for i in range(model.OperatorCodesLength())]
  ops = collections.Counter()
  gelu_approx = collections.Counter()
  over4, dtypes = [], collections.Counter()
  for s in range(model.SubgraphsLength()):
    sg = model.Subgraphs(s)
    for i in range(sg.OperatorsLength()):
      op = sg.Operators(i)
      c = codes[op.OpcodeIndex()]
      b = max(c.BuiltinCode(), c.DeprecatedBuiltinCode())
      name = names.get(b, str(b))
      if name == "CUSTOM":
        name = "CUSTOM:" + c.CustomCode().decode()
      ops[name] += 1
      if name == "GELU":
        o = schema.GeluOptions()
        t = op.BuiltinOptions()
        if t is not None:
          o.Init(t.Bytes, t.Pos)
          gelu_approx[bool(o.Approximate())] += 1
        else:
          gelu_approx["no-options(default approximate=False)"] += 1
    for i in range(sg.TensorsLength()):
      t = sg.Tensors(i)
      dtypes[tnames.get(t.Type(), str(t.Type()))] += 1
      if t.ShapeLength() > 4:
        over4.append((t.Name().decode(), [t.Shape(j) for j in range(t.ShapeLength())]))
  banned = {k: v for k, v in ops.items() if k in BANNED and k != "GELU"}
  return dict(
      path=path, size_bytes=os.path.getsize(path), subgraphs=model.SubgraphsLength(),
      op_total=sum(ops.values()), ops=dict(sorted(ops.items(), key=lambda kv: -kv[1])),
      banned_excluding_gelu=banned, banned_raw_parakeet_set={k: v for k, v in ops.items() if k in BANNED},
      gelu_count=ops.get("GELU", 0), gelu_approximate=dict((str(k), v) for k, v in gelu_approx.items()),
      tensors_rank_gt4=len(over4), tensors_rank_gt4_examples=over4[:5], tensor_dtypes=dict(dtypes),
      gpu_clean=not banned and not over4,
  )


class Runner:
  """LiteRT CompiledModel on CPU, inputs/outputs by signature name."""

  def __init__(self, path, threads):
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.options import CpuOptions, Options
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator
    self.cm = CompiledModel.from_file(
        path, options=Options(hardware_accelerators=HardwareAccelerator.CPU,
                              cpu_options=CpuOptions(num_threads=threads)))
    sigs = self.cm.get_signature_list()
    self.key = next(iter(sigs))
    self.inputs = list(sigs[self.key]["inputs"])
    self.outputs = list(sigs[self.key]["outputs"])
    self.in_bufs = {n: self.cm.create_input_buffer_by_name(self.key, n) for n in self.inputs}
    self.out_bufs = {n: self.cm.create_output_buffer_by_name(self.key, n) for n in self.outputs}
    self.out_details = self.cm.get_output_tensor_details(self.key)

  def __call__(self, **feeds):
    assert set(feeds) == set(self.inputs), (sorted(feeds), self.inputs)
    for n, x in feeds.items():
      self.in_bufs[n].write(np.ascontiguousarray(x, np.float32))
    self.cm.run_by_name(self.key, self.in_bufs, self.out_bufs)
    res = {}
    for n in self.outputs:
      shape = list(self.out_details[n]["shape"])
      res[n] = self.out_bufs[n].read(int(np.prod(shape)), np.float32).reshape(shape)
    return res


def corr(a, b):
  a = a.astype(np.float64).ravel()
  b = b.astype(np.float64).ravel()
  a -= a.mean()
  b -= b.mean()
  return float((a @ b) / math.sqrt((a @ a) * (b @ b)))


def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def gate_offline(args):
  import nemotron3diar_model as n3d
  exports = os.path.join(args.run_dir, "exports")
  results = os.path.join(args.run_dir, "results")
  t = n3d.T_OFFLINE
  path_b = os.path.join(exports, f"n3d_encoder_off_{args.ln}.tflite")
  path_b16 = os.path.join(exports, f"n3d_encoder_off_{args.ln}_fp16.tflite")
  opcheck = {"graph_B_fp32": census(path_b), "graph_B_fp16": census(path_b16)}
  with open(os.path.join(results, f"opcheck_offline_{args.ln}.json"), "w") as f:
    json.dump(opcheck, f, indent=1)
  cos, sin = (x.numpy() for x in n3d.rope_tables(t))
  run_a = Runner(os.path.join(exports, "n3d_frontend.tflite"), args.threads)
  run_b = Runner(path_b, args.threads)
  run_b16 = Runner(path_b16, args.threads)
  report = {"mode": "offline", "T": t, "ln": args.ln, "graph_B_fp32": os.path.basename(path_b),
            "graph_B_fp16": os.path.basename(path_b16),
            "thresholds": {"graph_B_fp32_max_abs": B_FP32_MAX, "graph_A_blocks_max_abs": A_MAX},
            "op_total": {k: v["op_total"] for k, v in opcheck.items()},
            "banned_excluding_gelu": {k: v["banned_excluding_gelu"] for k, v in opcheck.items()},
            "tensors_rank_gt4": {k: v["tensors_rank_gt4"] for k, v in opcheck.items()}, "fixtures": {}}
  ok = True
  for name in args.fixtures:
    with np.load(os.path.join(results, f"chunk_io_{name}_offline.npz")) as z:
      d = {k: z[k] for k in z.files}
    # graph A over 104-frame blocks of the offline features == the whole-recording embeddings
    feats, embeds = d["input_features"], d["embeds"]
    ne = embeds.shape[0]
    a_max = 0.0
    for b0 in range(0, ne, 13):
      m = np.zeros((104, 128), np.float32)
      blk = feats[b0 * 8 : b0 * 8 + 104]
      m[: blk.shape[0]] = blk
      e = run_a(mel=m[None])["chunk_embeds"][0]
      k = min(13, ne - b0)
      a_max = max(a_max, float(np.abs(e[:k] - embeds[b0 : b0 + k]).max()))
    rows = []
    for i in range(d["length"].shape[0]):
      L = int(d["length"][i])
      valid = d["row_valid"][i, :L]
      feeds = dict(packed_embeds=d["packed_embeds"][i][None],
                   attn_bias=n3d.attn_bias_for(L, t, valid=valid, two_level=True), rope_cos=cos, rope_sin=sin)
      ref = d["chunk_logits"][i, : L * 8]
      g32 = run_b(**feeds)["logits"][0, : L * 8]
      g16 = run_b16(**feeds)["logits"][0, : L * 8]
      nc, nf = int(d["cache_pre"][i, 0]), int(d["cache_pre"][i, 1])
      c0, c1 = (nc + nf) * 8, (nc + nf + int(d["num_chunk_frames"][i])) * 8
      p_ref, p16 = sigmoid(ref), sigmoid(g16)
      rec = dict(step=i, L=L, masked_rows=int((~valid).sum()),
                 B32_max_abs=float(np.abs(g32 - ref).max()), B32_corr=corr(g32, ref),
                 B32_out_max_abs=float(np.abs(g32[c0:c1] - ref[c0:c1]).max()),
                 B16_max_abs=float(np.abs(g16 - ref).max()), B16_max_dp=float(np.abs(p16 - p_ref).max()),
                 B16_flips=int(((p16 > 0.5) != (p_ref > 0.5)).sum()), cells=int(ref.size),
                 B16_out_flips=int(((p16[c0:c1] > 0.5) != (p_ref[c0:c1] > 0.5)).sum()))
      if (~valid).any():
        # negative control: the masked real row fed as padding (row zeroed before the k=3 head conv)
        bad = dict(feeds, attn_bias=n3d.attn_bias_for(int(valid.sum()), t, two_level=True))
        gb = run_b(**bad)["logits"][0, : L * 8]
        rec["neg_masked_row_as_pad_max_abs"] = float(np.abs(gb - ref).max())
        last = int(np.nonzero(valid)[0][-1])
        rec["neg_rows_affected"] = [int(r) for r in np.nonzero(np.abs(gb - ref).max(-1) > 1e-3)[0][[0, -1]] // 8]
        rec["neg_last_valid_row"] = last
      rows.append(rec)
    cells = sum(r["cells"] for r in rows)
    summary = dict(chunks=len(rows), lengths=[r["L"] for r in rows], A_blocks_max_abs=a_max,
                   B32_max_abs=max(r["B32_max_abs"] for r in rows), B32_min_corr=min(r["B32_corr"] for r in rows),
                   B16_max_abs=max(r["B16_max_abs"] for r in rows), B16_max_dp=max(r["B16_max_dp"] for r in rows),
                   B16_flips=sum(r["B16_flips"] for r in rows), B16_agree=1.0 - sum(r["B16_flips"] for r in rows) / cells,
                   B16_out_flips=sum(r["B16_out_flips"] for r in rows))
    summary["A_PASS"] = bool(a_max <= A_MAX)
    summary["B32_PASS"] = bool(summary["B32_max_abs"] <= B_FP32_MAX)
    ok &= summary["A_PASS"] and summary["B32_PASS"]
    report["fixtures"][name] = dict(summary=summary, steps=rows)
    print(json.dumps({name: dict(summary=summary, steps=rows)}, indent=1), flush=True)
  clean = opcheck["graph_B_fp32"]["gpu_clean"] and opcheck["graph_B_fp16"]["gpu_clean"]
  report["census_clean"] = bool(clean)
  report["PASS"] = bool(ok and clean)
  with open(os.path.join(results, f"gate_tflite_cpu_offline_{args.ln}.json"), "w") as f:
    json.dump(report, f, indent=1)
  print("GATE_TFLITE_CPU_OFFLINE", "PASS" if report["PASS"] else "FAIL", json.dumps(report["op_total"]), flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--fixtures", nargs="+", required=True)
  ap.add_argument("--steps", default="all")
  ap.add_argument("--threads", type=int, default=8)
  ap.add_argument("--ln", choices=["plain", "safe", "safe_guide"], default="plain")
  ap.add_argument("--mode", choices=["low_latency", "offline"], default="low_latency")
  args = ap.parse_args()
  if args.mode == "offline":
    return gate_offline(args)
  exports = os.path.join(args.run_dir, "exports")
  results = os.path.join(args.run_dir, "results")
  suffix = "" if args.ln == "plain" else f"_{args.ln}"
  path_a = os.path.join(exports, "n3d_frontend.tflite")
  path_b = os.path.join(exports, f"n3d_encoder_ll{suffix}.tflite")
  path_b16 = os.path.join(exports, f"n3d_encoder_ll{suffix}_fp16.tflite")
  path_plain = os.path.join(exports, "n3d_encoder_ll.tflite")

  opcheck = {"graph_A": census(path_a), "graph_B_fp32": census(path_b), "graph_B_fp16": census(path_b16)}
  with open(os.path.join(results, f"opcheck{suffix}.json"), "w") as f:
    json.dump(opcheck, f, indent=1)
  for k, v in opcheck.items():
    print(k, json.dumps({x: v[x] for x in ("op_total", "banned_excluding_gelu", "gelu_count",
                                           "gelu_approximate", "tensors_rank_gt4", "gpu_clean")}), flush=True)

  import torch  # rope tables only (same code path as the re-authored model)
  import nemotron3diar_model as n3d
  cos, sin = (x.numpy() for x in n3d.rope_tables())

  run_a = Runner(path_a, args.threads)
  run_b = Runner(path_b, args.threads)
  run_b16 = Runner(path_b16, args.threads)
  run_plain = Runner(path_plain, args.threads) if args.ln != "plain" else None
  print("signatures:", run_a.inputs, run_a.outputs, "|", run_b.inputs, run_b.outputs, flush=True)

  report = {"thresholds": {"graph_A_max_abs": A_MAX, "graph_B_fp32_max_abs": B_FP32_MAX,
                           "graph_B_fp16_agreement_target": 0.999},
            "ln": args.ln, "graph_B_fp32": os.path.basename(path_b),
            "graph_B_fp16": os.path.basename(path_b16),
            "threads": args.threads, "opcheck": f"opcheck{suffix}.json",
            "op_total": {k: v["op_total"] for k, v in opcheck.items()},
            "banned_excluding_gelu": {k: v["banned_excluding_gelu"] for k, v in opcheck.items()},
            "tensors_rank_gt4": {k: v["tensors_rank_gt4"] for k, v in opcheck.items()},
            "fixtures": {}}
  ok = True
  for name in args.fixtures:
    d = np.load(os.path.join(results, f"chunk_io_{name}.npz"))
    n = d["length"].shape[0]
    steps = range(n) if args.steps == "all" else [int(s) for s in args.steps.split(",") if int(s) < n]
    rows = []
    t0 = time.time()
    for i in steps:
      L = int(d["length"][i])
      f = int(d["mel_frames"][i])
      k = -(-f // 8)
      emb = run_a(mel=d["input_features"][i][None])["chunk_embeds"][0]
      a_max = float(np.abs(emb[:k] - d["packed_embeds"][i, L - k : L]).max())

      feeds = dict(packed_embeds=d["packed_embeds"][i][None], attn_bias=n3d.attn_bias_for(L),
                   rope_cos=cos, rope_sin=sin)
      ref = d["chunk_logits"][i, : L * 8]
      g32 = run_b(**feeds)["logits"][0, : L * 8]
      g16 = run_b16(**feeds)["logits"][0, : L * 8]
      vs_plain = {}
      if run_plain is not None:
        gp = run_plain(**feeds)["logits"][0, : L * 8]
        vs_plain = dict(B32_vs_plain_max_abs=float(np.abs(g32 - gp).max()),
                        B32_vs_plain_max_dp=float(np.abs(sigmoid(g32) - sigmoid(gp)).max()))
      p_ref, p32, p16 = sigmoid(ref), sigmoid(g32), sigmoid(g16)
      # the chunk's own output rows (what the step returns) inside the L*8 logit rows
      nc, nf = int(d["cache_pre"][i, 0]), int(d["cache_pre"][i, 1])
      c0 = (nc + nf) * 8
      c1 = c0 + int(d["out_frames"][i])
      flips = []
      for row, spk in zip(*np.nonzero((p16 > 0.5) != (p_ref > 0.5))):
        # after_output = look-ahead rows (or the stacking pad row of the last chunk)
        region = ("cache" if row < nc * 8 else "fifo" if row < c0 else "chunk_output" if row < c1
                  else "after_output")
        flips.append(dict(logit_row=int(row), encoder_row=int(row // 8), region=region, speaker=int(spk),
                          p_ref=float(p_ref[row, spk]), p_fp16=float(p16[row, spk])))
      rows.append(dict(
          step=i, L=L, compressed=bool(d["cache_pre"][i, 2]),
          A_max_abs=a_max,
          B32_max_abs=float(np.abs(g32 - ref).max()), B32_corr=corr(g32, ref),
          B32_max_dp=float(np.abs(p32 - p_ref).max()),
          B16_max_abs=float(np.abs(g16 - ref).max()), B16_corr=corr(g16, ref),
          B16_max_dp=float(np.abs(p16 - p_ref).max()),
          B16_agree=float(((p16 > 0.5) == (p_ref > 0.5)).mean()),
          B16_flips=int(((p16 > 0.5) != (p_ref > 0.5)).sum()),
          B16_agree_out=float(((p16[c0:c1] > 0.5) == (p_ref[c0:c1] > 0.5)).mean()),
          B16_flip_detail=flips,
          cells=int(ref.size),
          **vs_plain,
      ))
    dt = time.time() - t0
    cells = sum(r["cells"] for r in rows)
    flips = sum(r["B16_flips"] for r in rows)
    summary = dict(
        steps=len(rows), seconds=round(dt, 1),
        steps_L541=[r["step"] for r in rows if r["L"] == T_ALL],
        steps_after_compress=sum(r["compressed"] for r in rows),
        A_max_abs=max(r["A_max_abs"] for r in rows),
        B32_max_abs=max(r["B32_max_abs"] for r in rows),
        B32_min_corr=min(r["B32_corr"] for r in rows),
        B32_max_dp=max(r["B32_max_dp"] for r in rows),
        B16_max_abs=max(r["B16_max_abs"] for r in rows),
        B16_min_corr=min(r["B16_corr"] for r in rows),
        B16_max_dp=max(r["B16_max_dp"] for r in rows),
        B16_agree_all_cells=1.0 - flips / cells, B16_flips=flips, B16_cells=cells,
        B16_min_step_agree=min(r["B16_agree"] for r in rows),
        B16_min_step_agree_out_rows=min(r["B16_agree_out"] for r in rows),
        B16_flip_detail=[dict(step=r["step"], L=r["L"], **f_) for r in rows for f_ in r["B16_flip_detail"]],
    )
    if run_plain is not None:
      summary["B32_vs_plain_max_abs"] = max(r["B32_vs_plain_max_abs"] for r in rows)
      summary["B32_vs_plain_max_dp"] = max(r["B32_vs_plain_max_dp"] for r in rows)
    summary["A_PASS"] = bool(summary["A_max_abs"] <= A_MAX)
    summary["B32_PASS"] = bool(summary["B32_max_abs"] <= B_FP32_MAX)
    ok &= summary["A_PASS"] and summary["B32_PASS"]
    report["fixtures"][name] = dict(summary=summary, steps=rows)
    print(json.dumps({name: summary}, indent=1), flush=True)

  clean = opcheck["graph_B_fp32"]["gpu_clean"] and opcheck["graph_A"]["gpu_clean"]
  report["census_clean"] = bool(clean)
  report["PASS"] = bool(ok and clean)
  with open(os.path.join(results, f"gate_tflite_cpu{suffix}.json"), "w") as f:
    json.dump(report, f, indent=1)
  print("GATE_TFLITE_CPU", "PASS" if report["PASS"] else "FAIL", flush=True)


if __name__ == "__main__":
  main()
