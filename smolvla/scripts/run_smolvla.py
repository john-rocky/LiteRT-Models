#!/usr/bin/env python3
"""SmolVLA on LiteRT: image(s) + task + state -> 50 x action_dim action chunk.

Host loop in smolvla_host.py (numpy), graphs through the LiteRT CompiledModel
Python API (CPU here). Needs the files written by build_smolvla.py.

Run (repo root):
  KMP_DUPLICATE_LIB_OK=TRUE python smolvla/scripts/run_smolvla.py \\
      --image tipsv2/scripts/test.jpg --task "pick up the red cube and place it in the box" \\
      --state 0.25 -0.6 0.9 -0.3 0.45 -1.1 --seed 0
"""

import argparse
import json
import os
import time

import numpy as np

import smolvla_host as host
from litert_helpers import TFLiteRunner

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))


def make_graph_fns(out_dir, contract, fp16, norms, gpu):
    graphs = contract["graphs"]
    n_cam = contract["num_cameras"]
    runners = {g: TFLiteRunner(os.path.join(out_dir, host.graph_file(g, n_cam, fp16, norms)),
                               accelerator="gpu" if gpu else "cpu")
               for g in graphs}
    shape = {g: [o["shape"] for o in graphs[g]["outputs"]] for g in graphs}

    def vision(image, pos_embed):
        return runners["vision"].run_shaped([image, pos_embed], shape["vision"])[0]

    def prefix(*inputs):
        return tuple(runners["prefix"].run_shaped(list(inputs), shape["prefix"]))

    def expert(*inputs):
        return runners["expert"].run_shaped(list(inputs), shape["expert"])[0]

    return vision, prefix, expert


def noise_from_seed(seed, shape):
    """Same draw as lerobot's sample_noise after torch.manual_seed(seed)."""
    import torch
    torch.manual_seed(seed)
    return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", nargs="+", required=True, help="one image per camera, config order")
    ap.add_argument("--task", required=True)
    ap.add_argument("--state", type=float, nargs="+", required=True, help="raw robot state")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "smolvla", "out"),
                    help="directory with the graphs and host assets")
    ap.add_argument("--fp16", action="store_true", help="use the *_f16 graphs")
    ap.add_argument("--norms", choices=("exact", "fp16safe"), default=host.DEFAULT_NORMS,
                    help="exact: the opt-in *_exactnorm vision/prefix reference graphs")
    ap.add_argument("--gpu", action="store_true",
                    help="run on this host's LiteRT GPU accelerator (default precision)")
    ap.add_argument("--seed", type=int, default=None,
                    help="noise = torch.manual_seed(seed) + normal (lerobot's draw); "
                         "default: numpy standard normal")
    ap.add_argument("--noise", default=None, help="raw float32 [1,50,32] noise file (overrides --seed)")
    ap.add_argument("--save", default=None, help="write the actions here (.npy or .json)")
    args = ap.parse_args()

    with open(os.path.join(args.out, "graph_contract.json")) as f:
        contract = json.load(f)
    n_cam = contract["num_cameras"]
    assert len(args.image) == n_cam, f"graphs were built for {n_cam} camera(s)"
    vision, prefix, expert = make_graph_fns(args.out, contract, args.fp16, args.norms, args.gpu)
    pipe = host.SmolVLAHost(
        vision, prefix, expert,
        embed_table=host.load_embed_table(os.path.join(args.out, "embed_tokens_f16.bin")),
        pos_embed=np.fromfile(os.path.join(args.out, "vision_pos_embed.bin"),
                              dtype="<f4").reshape(1, 1024, 768),
        tokenizer=host.load_tokenizer(contract["tokenizer"]["repo"]),
        stats=host.NormStats.from_json(os.path.join(args.out, "norm_stats.json")),
        num_cameras=n_cam, action_dim=contract["action_dim"])

    shape = (1, contract["chunk_size"], contract["max_action_dim"])
    if args.noise:
        noise = np.fromfile(args.noise, dtype="<f4").reshape(shape)
    elif args.seed is not None:
        noise = noise_from_seed(args.seed, shape)
    else:
        noise = np.random.default_rng().standard_normal(shape).astype(np.float32)

    images = [host.load_image_rgb01(p) for p in args.image]
    t0 = time.time()
    res = pipe.run(images, args.task, args.state, noise)
    dt = time.time() - t0
    actions = res["actions"][0]
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print(f"{actions.shape[0]} x {actions.shape[1]} actions in {dt:.2f}s "
          f"({'fp16' if args.fp16 else 'fp32'}-weight graphs, {args.norms} norms, "
          f"CompiledModel {'GPU' if args.gpu else 'CPU'})")
    print(actions)
    if args.save:
        if args.save.endswith(".json"):
            with open(args.save, "w") as f:
                json.dump(actions.tolist(), f)
        else:
            np.save(args.save, actions)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
