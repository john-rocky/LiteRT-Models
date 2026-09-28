#!/usr/bin/env python3
"""Parity of the SmolVLA .tflite graphs (LiteRT CompiledModel API) against the torch
modules and against lerobot's sample_actions.

Needs the fixtures written by verify_reauthored.py (out/fixtures/shapes.json) and
the graphs/assets written by build_smolvla.py. For every configuration:

  1. per graph: tflite(fixture inputs) vs the torch module output on the same inputs
     (always the exact-norm torch modules; the prefix is compared on non-pad tokens too)
  2. the expert graph on each of the 10 Euler-step inputs of the torch chain
  3. full chain: numpy host + tflite V -> P -> 10 x E -> un-normalize, same
     image/task/state/noise, vs lerobot float32 sample_actions (and vs the torch chain)
  4. latency on this host (context only; nothing here is a device number)

Configurations: CPU with fp32 and fp16-weight graphs by default. --gpu adds the
LiteRT GPU accelerator of this host (Metal on macOS; not the Android ML Drift
OpenCL/Vulkan backends) at its default precision and with enforce_f32. --norms
exact uses the opt-in *_exactnorm vision/prefix reference graphs.

Run (repo root):
  KMP_DUPLICATE_LIB_OK=TRUE python smolvla/scripts/verify_tflite.py --out smolvla/out
  KMP_DUPLICATE_LIB_OK=TRUE python smolvla/scripts/verify_tflite.py --out smolvla/out --gpu
"""

import argparse
import json
import os
import time

import numpy as np

import smolvla_host as host
from litert_helpers import TFLiteRunner, fmt_parity, load_fixture_group, parity

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
FIXTURE_GROUP = {"vision": "vision", "prefix": "prefix", "expert": "expert_step0"}
OUT_NAMES = {"vision": ["img_emb"], "prefix": ["k_all", "v_all"], "expert": ["v_t"]}

RESULTS = {}


def report(key, a, b, rows=None):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if rows is not None:
        a, b = a[..., rows, :], b[..., rows, :]
    p = parity(a, b)
    RESULTS[key] = p
    print(f"  {key:<66s} {fmt_parity(p)}")
    return p


def timed(fn, reps):
    fn()                       # warm-up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def run_config(tag, runners, args, fx_root, meta, chain, valid):
    print(f"\n===== {tag} =====")
    for g, r in runners.items():
        print(f"  {os.path.basename(r.path)}: fully accelerated {r.fully_accelerated}; "
              f"inputs {r.input_names}")
        RESULTS[f"{tag} {g}: fully_accelerated"] = bool(r.fully_accelerated)

    print(f"[{tag}] per graph: tflite vs torch module on the same fixture inputs")
    inputs_of = {}
    for g, r in runners.items():
        arrays, inputs = load_fixture_group(fx_root, FIXTURE_GROUP[g])
        inputs_of[g] = inputs
        shapes = [list(arrays[n].shape) for n in OUT_NAMES[g]]
        outs = r.run_shaped(inputs, shapes)
        for n, o in zip(OUT_NAMES[g], outs):
            if g == "prefix":
                report(f"{tag} prefix: {n} (non-pad tokens)", o, arrays[n], valid)
            else:
                report(f"{tag} {g}: {n}", o, arrays[n])
        if g == "vision":
            report(f"{tag} vision: img_emb vs lerobot embed_image", outs[0],
                   arrays["lerobot_img_emb"])

    worst = None
    for step in range(10):
        ins = list(inputs_of["expert"])
        ins[0], ins[1] = chain[f"x_t_step{step}"], chain[f"time_emb_step{step}"]
        out = runners["expert"].run_shaped(ins, [[1, 50, 32]])[0]
        p = parity(out, chain[f"torch_v_t_step{step}"])
        if worst is None or p["max_abs_diff"] > worst[1]["max_abs_diff"]:
            worst = (step, p)
    RESULTS[f"{tag} expert: v_t, worst of the 10 step inputs"] = worst[1]
    print(f"  {tag + ' expert: v_t, worst of the 10 step inputs (step ' + str(worst[0]) + ')':<66s} "
          f"{fmt_parity(worst[1])}")

    print(f"[{tag}] full chain: host + tflite V -> P -> 10 x E vs lerobot sample_actions")
    n_cam = 1
    s_len = 64 * n_cam + 48 + 1
    kv = [1, 80, s_len, 64]

    def f_vision(image, pe):
        return runners["vision"].run_shaped([image, pe], [[1, 64, 960]])[0]

    def f_prefix(*a):
        return tuple(runners["prefix"].run_shaped(list(a), [kv, kv]))   # k_all, v_all

    def f_expert(*a):
        return runners["expert"].run_shaped(list(a), [[1, 50, 32]])[0]

    pipe = host.SmolVLAHost(
        f_vision, f_prefix, f_expert,
        host.load_embed_table(os.path.join(args.out, "embed_tokens_f16.bin")),
        np.fromfile(os.path.join(args.out, "vision_pos_embed.bin"), dtype="<f4").reshape(1, 1024, 768),
        host.load_tokenizer(), stats=host.NormStats.from_json(os.path.join(args.out, "norm_stats.json")),
        num_cameras=n_cam, action_dim=6)
    image01 = host.load_image_rgb01(os.path.join(REPO_ROOT, meta["image"]))
    t0 = time.time()
    res = pipe.run([image01], meta["task"], meta["state"], chain["noise"])
    wall = time.time() - t0
    ids_ok = bool(np.array_equal(res["prepared"]["ids"], chain["token_ids"]))
    print(f"  chain wall time {wall:.2f}s; token ids == fixture: {ids_ok}")
    for step in (0, 9):
        report(f"{tag} chain v_t step {step} vs lerobot", res["trace"][step][1],
               chain[f"lerobot_v_t_step{step}"])
    report(f"{tag} chain final x [1,50,32] vs lerobot", res["x"], chain["lerobot_x_final"])
    report(f"{tag} chain actions [50,6] vs lerobot (postprocessed)", res["actions"],
           chain["lerobot_actions"])
    report(f"{tag} chain final x vs torch chain", res["x"], chain["torch_x_final"])
    with open(os.path.join(args.out, "norm_stats.json")) as f:
        so100 = json.load(f).get("unmatched_stat_keys", {}).get("so100.buffer.action")
    if so100:
        std = np.asarray(so100["std"], np.float64)
        d = np.abs(res["x"][..., :6].astype(np.float64) - chain["lerobot_x_final"][..., :6]) * std
        RESULTS[f"{tag} chain actions, illustrative so100-std units, max_abs_diff"] = float(d.max())
        print(f"  (illustrative) max abs diff x so100 action std: {float(d.max()):.3e} "
              f"(these stats are NOT applied by lerobot for smolvla_base)")

    timing = {f"{g}_ms": timed(lambda r=r, i=inputs_of[g]: r.run(i), args.timing_reps)
              for g, r in runners.items()}
    timing["chain_wall_s"] = wall
    print(f"  median latency on this host: { {k: round(v, 1) for k, v in timing.items()} }")
    RESULTS[f"{tag} timings"] = timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "smolvla", "out"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--norms", choices=("exact", "fp16safe"), default=host.DEFAULT_NORMS)
    ap.add_argument("--gpu", action="store_true",
                    help="also run the fp32 graphs on this host's LiteRT GPU accelerator")
    ap.add_argument("--no-cpu", action="store_true", help="skip the CPU configurations")
    ap.add_argument("--timing-reps", type=int, default=5)
    args = ap.parse_args()
    fx_root = os.path.join(args.out, "fixtures")
    with open(os.path.join(fx_root, "shapes.json")) as f:
        meta = json.load(f)
    chain, _ = load_fixture_group(fx_root, "chain")
    valid = load_fixture_group(fx_root, "prefix")[0]["valid_token_index"]

    configs = []
    if not args.no_cpu:
        configs += [("cpu", False, False, "fp32 weights"), ("cpu", True, False, "fp16 weights")]
    if args.gpu:
        configs += [("gpu", False, False, "fp32 weights, default precision"),
                    ("gpu", False, True, "fp32 weights, enforce_f32")]
    for accel, fp16_weights, gpu_fp32, label in configs:
        files = {g: host.graph_file(g, 1, fp16_weights, args.norms) for g in FIXTURE_GROUP}
        tag = f"{accel.upper()} {label}" + (" [exact norms]" if args.norms == "exact" else "")
        runners = {g: TFLiteRunner(os.path.join(args.out, f), args.threads, accel, gpu_fp32)
                   for g, f in files.items()}
        run_config(tag, runners, args, fx_root, meta, chain, valid)
        del runners

    name = "parity_tflite" + ("_gpu" if args.gpu and args.no_cpu else "") + \
        ("_exactnorm" if args.norms == "exact" else "") + ".json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(RESULTS, f, indent=1)
    print(f"\nwrote {os.path.join(args.out, name)}")


if __name__ == "__main__":
    main()
