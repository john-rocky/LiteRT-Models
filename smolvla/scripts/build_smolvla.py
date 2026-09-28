#!/usr/bin/env python3
"""Build the SmolVLA (lerobot/smolvla_base) LiteRT graphs and host assets.

Outputs (in --out, default smolvla/out):
  smolvla_vision.tflite / _f16      image [1,3,512,512] + pos_embed -> img_emb [1,64,960]
  smolvla_prefix.tflite / _f16      VLM prefill (16 layers) -> k_all / v_all [1,80,S,64]
  smolvla_expert_step.tflite / _f16 one Euler velocity evaluation -> v_t [1,50,32]
  embed_tokens_f16.bin              [49280, 960] float16 token-embedding table (host lookup)
  vision_pos_embed.bin              [1024, 768] float32 SigLIP position table (graph input)
  norm_stats.json                   the state/action MEAN_STD stats lerobot actually applies
  graph_contract.json               every graph input/output (signature order) + host recipe
  build_log.json                    op distribution, max rank, banned ops, sizes, timings

The fp32 graphs are litert_torch.convert(...).export(...) of the modules in
smolvla_graphs.py; *_f16 stores the weights as float16 (ai_edge_quantizer
FLOAT_CASTING). The vision/prefix graphs use the exact fp16-safe RMSNorm/LayerNorm
forms by default; --norms exact adds reference *_exactnorm* graphs with the plain
formulas (wrong at fp16 GPU precision). --num-cameras N changes the prefix length
S = 64N + 49 (files get a _camN suffix when N != 1); the vision graph is per camera.

Run (repo root):
  KMP_DUPLICATE_LIB_OK=TRUE python smolvla/scripts/build_smolvla.py --stage all --out smolvla/out
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from litert_helpers import fmt_op_scan, op_scan, to_fp16
from smolvla_host import DEFAULT_NORMS, FP16_SAFE_STAGES, graph_file
from smolvla_graphs import (CHUNK, EXPERT_HIDDEN, HEAD_DIM, HIDDEN, IMG_TOKENS, LANG_LEN,
                            MASK_NEG, MAX_DIM, N_KV, N_LAYERS, SMOLVLA_REPO, VIS_HIDDEN,
                            VIS_PATCHES, ExpertStepGraph, PrefixGraph, VisionGraph,
                            load_policy, load_processors, norm_stats_from_processors,
                            prefix_len, set_fp16_safe_norms)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
STAGES = ("vision", "prefix", "expert")


def graph_files(stage, num_cameras, norms=DEFAULT_NORMS):
    return (graph_file(stage, num_cameras, False, norms),
            graph_file(stage, num_cameras, True, norms))


def io_spec(stage, num_cameras):
    """(name, shape, semantics) for inputs and outputs, in signature order."""
    s = prefix_len(num_cameras)
    kv = [1, N_LAYERS * N_KV, s, HEAD_DIM]
    if stage == "vision":
        return ([("image", [1, 3, 512, 512],
                  "letterboxed RGB camera image in [-1, 1]: lerobot resize_with_pad to "
                  "512x512 (bilinear, align_corners=False, zero pad on the LEFT and TOP, "
                  "computed in [0,1]) then x*2-1"),
                 ("pos_embed", [1, VIS_PATCHES, VIS_HIDDEN],
                  "SigLIP position-embedding table, constant: vision_pos_embed.bin "
                  "(a runtime input on purpose, so no large baked constant enters compute)")],
                [("img_emb", [1, IMG_TOKENS, HIDDEN],
                  "connector output (pixel shuffle x4 + Linear 12288->960), NOT scaled "
                  "by sqrt(960); concatenate cameras along axis 1 for the prefix")])
    if stage == "prefix":
        return ([("img_emb", [1, IMG_TOKENS * num_cameras, HIDDEN],
                  "vision graph output(s), cameras concatenated in config order"),
                 ("lang_emb", [1, LANG_LEN, HIDDEN],
                  "raw rows of embed_tokens_f16.bin for the 48 token ids (pads included), "
                  "as float32, NOT scaled"),
                 ("state", [1, MAX_DIM],
                  "normalized robot state (norm_stats.json) zero-padded to 32"),
                 ("attn_bias", [1, 1, s, s],
                  "additive prefix mask: 0 where lerobot make_att_2d_masks(pad, att) is "
                  "True, -30000 elsewhere; att = 0 for image+language, 1 for the state"),
                 ("rope_cos", [1, 1, s, 32], "cos(pos / 10000**(2i/64)), pos = cumsum(pad)-1"),
                 ("rope_sin", [1, 1, s, 32], "sin of the same radians")],
                [("k_all", kv, "post-RoPE keys of all 16 layers; layer l = channels 5l..5l+4"),
                 ("v_all", kv, "values of all 16 layers; layer l = channels 5l..5l+4")])
    return ([("x_t", [1, CHUNK, MAX_DIM], "current Euler state (noise at step 0)"),
             ("time_emb", [1, 1, EXPERT_HIDDEN],
              "create_sinusoidal_pos_embedding(t, 720, 0.004, 4.0) in float64 -> float32, "
              "t = 1.0 + step*(-0.1)"),
             ("k_all", kv, "prefix graph output, unchanged for all 10 steps"),
             ("v_all", kv, "prefix graph output, unchanged for all 10 steps"),
             ("bias_self", [1, 1, CHUNK, s + CHUNK],
              "additive mask for self-attn layers: prefix pad columns then causal 50x50"),
             ("bias_cross", [1, 1, CHUNK, s], "additive mask for cross-attn layers: prefix pad"),
             ("rope_cos_s", [1, 1, CHUNK, 32], "RoPE cos, positions prefix_offset + 0..49 "
              "(prefix_offset = number of non-pad prefix tokens)"),
             ("rope_sin_s", [1, 1, CHUNK, 32], "RoPE sin, same positions"),
             ("rope_cos_c", [1, 1, CHUNK, 32], "RoPE cos, positions 0..49 (cross-attn queries)"),
             ("rope_sin_c", [1, 1, CHUNK, 32], "RoPE sin, positions 0..49")],
            [("v_t", [1, CHUNK, MAX_DIM], "velocity; host does x_t <- x_t + (-0.1) * v_t")])


def check_config(policy, num_cameras):
    """Fail early on checkpoint settings this export does not implement."""
    cfg = policy.config
    expect = {"attention_mode": "cross_attn", "add_image_special_tokens": False,
              "empty_cameras": 0, "chunk_size": CHUNK, "max_state_dim": MAX_DIM,
              "max_action_dim": MAX_DIM, "tokenizer_max_length": LANG_LEN,
              "resize_imgs_with_padding": (512, 512), "pad_language_to": "max_length"}
    for key, want in expect.items():
        got = getattr(cfg, key)
        got = tuple(got) if isinstance(got, list) else got
        assert got == want, f"config.{key} = {got!r}; this export assumes {want!r}"
    assert cfg.prefix_length <= prefix_len(num_cameras), "prefix_length padding not implemented"
    n_img = len(cfg.image_features)
    if n_img != num_cameras:
        print(f"note: the checkpoint lists {n_img} camera features; graphs are built for "
              f"{num_cameras} (lerobot uses only the cameras present in the batch)")


def make_module(stage, policy, num_cameras, norms=DEFAULT_NORMS):
    model = policy.model
    vlm = model.vlm_with_expert.get_vlm_model()
    if stage == "vision":
        module = VisionGraph(vlm.vision_model, vlm.connector)
    elif stage == "prefix":
        module = PrefixGraph(model, num_cameras)
    else:
        module = ExpertStepGraph(model, policy.config.self_attn_every_n_layers)
    if norms == "fp16safe" and stage in FP16_SAFE_STAGES:
        set_fp16_safe_norms(module)
    return module.eval()


def sample_kwargs(stage, num_cameras):
    g = torch.Generator().manual_seed(0)
    ins, _ = io_spec(stage, num_cameras)
    kw = {}
    for name, shape, _ in ins:
        if name.startswith("attn_bias") or name.startswith("bias_"):
            kw[name] = torch.zeros(shape)
        else:
            kw[name] = torch.randn(shape, generator=g)
    return kw


def convert_stage(stage, policy, out_dir, num_cameras, norms=DEFAULT_NORMS):
    import litert_torch  # imported only after the torch model is loaded

    f32, f16 = graph_files(stage, num_cameras, norms)
    p32, p16 = os.path.join(out_dir, f32), os.path.join(out_dir, f16)
    module = make_module(stage, policy, num_cameras, norms)
    kw = sample_kwargs(stage, num_cameras)
    with torch.no_grad():
        ref = module(**kw)
    t0 = time.time()
    litert_torch.convert(module, sample_args=(), sample_kwargs=kw).export(p32)
    t_conv = time.time() - t0
    t0 = time.time()
    to_fp16(p32, p16)
    t_f16 = time.time() - t0
    entry = {"stage": stage, "norms": norms, "fp32": f32, "fp16": f16,
             "convert_seconds": round(t_conv, 1), "fp16_cast_seconds": round(t_f16, 1),
             "scan_fp32": op_scan(p32), "scan_fp16": op_scan(p16)}
    print(f"\n[{stage}] {f32}: converted in {t_conv:.1f}s")
    print(fmt_op_scan(entry["scan_fp32"]))
    print(f"[{stage}] {f16}: cast in {t_f16:.1f}s")
    print(fmt_op_scan(entry["scan_fp16"]))
    del module, ref
    return entry


def snapshot_revision(repo, filename):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo, filename)
    parts = path.split(os.sep)
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts else None


def write_assets(policy, pre, post, out_dir):
    vlm = policy.model.vlm_with_expert.get_vlm_model()
    table = vlm.text_model.embed_tokens.weight.detach().float().numpy()
    t16 = table.astype(np.float16)
    t16.astype("<f2").tofile(os.path.join(out_dir, "embed_tokens_f16.bin"))
    inexact = int((t16.astype(np.float32) != table).sum())
    pos = vlm.vision_model.embeddings.position_embedding.weight.detach().float().numpy()
    pos.astype("<f4").tofile(os.path.join(out_dir, "vision_pos_embed.bin"))
    stats = norm_stats_from_processors(pre, post)
    stats["note"] = (
        "state/action = the MEAN_STD stats lerobot 0.6.1 applies to this checkpoint "
        "(normalize: (x-mean)/(std+eps); unnormalize: x*std+mean); null = identity. lerobot "
        "looks the stats up by the exact keys 'observation.state' and 'action'; entries under "
        "unmatched_stat_keys exist in the processor files but are NOT applied by lerobot.")
    with open(os.path.join(out_dir, "norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(f"host assets: embed_tokens_f16.bin {t16.shape} ({inexact} entries not exactly "
          f"representable in fp16, max abs err {float(np.abs(t16.astype(np.float32) - table).max()):.2e}), "
          f"vision_pos_embed.bin {pos.shape}, norm_stats.json (state stats applied: "
          f"{stats['state'] is not None}, action stats applied: {stats['action'] is not None})")
    return {"embed_table_shape": list(t16.shape), "embed_fp16_inexact_entries": inexact,
            "embed_fp16_max_abs_err": float(np.abs(t16.astype(np.float32) - table).max()),
            "pos_embed_shape": list(pos.shape)}


def write_contract(policy, repo, out_dir, num_cameras, tokenizer_info):
    from ai_edge_litert.compiled_model import CompiledModel

    s = prefix_len(num_cameras)
    graphs = {}
    for stage in STAGES:
        f32, f16 = graph_files(stage, num_cameras)
        path = os.path.join(out_dir, f32)
        if not os.path.exists(path):
            continue
        sig = CompiledModel.from_file(path).get_signature_by_index(0)
        ins, outs = io_spec(stage, num_cameras)
        assert list(sig["inputs"]) == [n for n, _, _ in ins], (stage, sig["inputs"])
        assert len(sig["outputs"]) == len(outs)
        graphs[stage] = {
            "file": f32, "file_f16": f16, "signature": sig["key"],
            "inputs": [{"index": i, "name": n, "shape": sh, "dtype": "float32", "semantics": d}
                       for i, (n, sh, d) in enumerate(ins)],
            "outputs": [{"index": i, "signature_name": sig["outputs"][i], "name": n, "shape": sh,
                         "dtype": "float32", "semantics": d} for i, (n, sh, d) in enumerate(outs)],
        }
    freq_exp = np.float32(2.0 / HEAD_DIM) * np.arange(HEAD_DIM // 2, dtype=np.float32)
    timescale = np.power(np.float64(10000), freq_exp.astype(np.float64)).astype(np.float32)
    action_dim = int(policy.config.output_features["action"].shape[0])
    contract = {
        "model": repo,
        "model_revision": snapshot_revision(repo, "config.json"),
        "reference": "lerobot 0.6.1 SmolVLAPolicy.sample_actions, all modules float32",
        "num_cameras": num_cameras,
        "prefix_len": s,
        "prefix_layout": f"[{IMG_TOKENS}*num_cameras image tokens | {LANG_LEN} language tokens | 1 state token]",
        "chunk_size": CHUNK, "action_dim": action_dim, "max_action_dim": MAX_DIM,
        "num_steps": int(policy.config.num_steps),
        "graphs": graphs,
        "host_assets": {
            "embed_tokens_f16.bin": "float16 little-endian [49280, 960]; row = token id",
            "vision_pos_embed.bin": "float32 little-endian [1024, 768]; feed as pos_embed [1,1024,768]",
            "norm_stats.json": "state/action MEAN_STD stats (null = identity)",
        },
        "constants": {
            "mask_value": MASK_NEG, "rope_base": 10000,
            "rope_timescale_f32": [float(x) for x in timescale],
            "time_embedding": {"dim": EXPERT_HIDDEN, "min_period": 0.004, "max_period": 4.0,
                               "formula": "fraction=linspace(0,1,360); period=0.004*(4.0/0.004)**fraction; "
                                          "a=2*pi/period*t (float64); cat([sin(a), cos(a)]) -> float32"},
            "euler": {"dt": -0.1, "times": [float(np.float32(1.0 + k * -0.1)) for k in range(10)]},
        },
        "host_procedure": [
            "image(s): RGB uint8 -> float32/255 -> resize_with_pad(512, 512) (bilinear, "
            "align_corners=False, no antialias; pad value 0 on the left/top) -> x*2-1",
            "vision graph per camera -> img_emb; concatenate cameras along axis 1",
            "task: append '\\n' if missing; tokenize (see tokenizer); lang_emb = "
            "embed_tokens_f16[ids] as float32 (pads included)",
            "state: (s - mean)/(std + eps) if norm_stats.state else s; zero-pad to 32",
            "pad = [1]*64N + lang_mask + [1]; att = [0]*(64N+48) + [1]; "
            "mask2d[i,j] = cumsum(att)[j] <= cumsum(att)[i] and pad[i] and pad[j]; "
            "attn_bias = 0 / -30000; positions = cumsum(pad) - 1; prefix_offset = sum(pad)",
            "prefix graph once -> k_all, v_all",
            "expert tables: bias_self = [pad (broadcast over 50 rows) | causal 50x50], "
            "bias_cross = pad columns; rope_s positions prefix_offset + i; rope_c positions i",
            "x = noise [1,50,32] ~ N(0,1); for step in 0..9: t = float32(1.0 + step*(-0.1)); "
            "v = expert(x, time_emb(t), ...); x = x + float32(-0.1) * v",
            f"actions = x[..., :{action_dim}]; x*std + mean if norm_stats.action else identity",
        ],
        "tokenizer": tokenizer_info,
    }
    with open(os.path.join(out_dir, "graph_contract.json"), "w") as f:
        json.dump(contract, f, indent=1)
    print(f"wrote {os.path.join(out_dir, 'graph_contract.json')}")


def tokenizer_notes(policy):
    from transformers import AutoTokenizer
    repo = policy.config.vlm_model_name
    tok = AutoTokenizer.from_pretrained(repo)
    enc = tok("pick up the red cube and place it in the box\n", max_length=LANG_LEN,
              truncation=True, padding="max_length", padding_side="right")
    return {
        "repo": repo, "revision": snapshot_revision(repo, "tokenizer.json"),
        "files": ["tokenizer.json (or vocab.json + merges.txt)"],
        "class": type(tok).__name__, "type": "byte-level BPE (GPT-2 style)",
        "call": "tokenizer(task, max_length=48, truncation=True, padding='max_length', "
                "padding_side='right') with default add_special_tokens (adds none here)",
        "newline_rule": "lerobot NewLineTaskProcessorStep appends '\\n' when the task does not end with one",
        "pad_token_id": tok.pad_token_id, "pad_token": tok.pad_token,
        "bos_token_id": tok.bos_token_id, "bos_added": False,
        "max_length": LANG_LEN, "padding_side": "right", "truncation": True,
        "attention_mask": "1 for real tokens, 0 for pads",
        "example": {"task": "pick up the red cube and place it in the box",
                    "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
    }


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES + ("all", "assets", "scan"), default="all",
                    help="scan: re-run the flatbuffer checks on existing graphs, no conversion")
    ap.add_argument("--num-cameras", type=int, default=1)
    ap.add_argument("--norms", choices=("exact", "fp16safe"), default=DEFAULT_NORMS,
                    help="fp16safe (default): vision/prefix RMSNorm/LayerNorm in the exact "
                         "down-scaled forms, correct at fp16 GPU precision. exact (opt-in "
                         "reference): the unmodified norm formulas, *_exactnorm*.tflite, "
                         "vision/prefix only; they overflow at fp16 GPU precision")
    ap.add_argument("--repo", default=SMOLVLA_REPO,
                    help="lerobot SmolVLA checkpoint (hub id or local dir); only "
                         "lerobot/smolvla_base has been verified")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "smolvla", "out"))
    ap.add_argument("--subprocess", action="store_true",
                    help="with --stage all: convert every graph in its own python process")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    if args.stage == "all" and args.subprocess:
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")
        for stage in ("assets",) + STAGES:
            if args.norms == "exact" and stage not in FP16_SAFE_STAGES:
                continue
            cmd = [sys.executable, os.path.abspath(__file__), "--stage", stage,
                   "--num-cameras", str(args.num_cameras), "--repo", args.repo,
                   "--norms", args.norms, "--out", args.out]
            print("$", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, env=env)
        return

    log_path = os.path.join(args.out, "build_log.json")
    if args.stage == "scan":
        with open(log_path) as f:
            log = json.load(f)
        for key in list(log):
            if not isinstance(log[key], dict) or "scan_fp32" not in log[key]:
                continue
            stage = log[key]["stage"]
            for var in ("fp32", "fp16"):
                path = os.path.join(args.out, log[key][var])
                log[key][f"scan_{var}"] = op_scan(path)
                print(f"[{stage}] {log[key][var]}")
                print(fmt_op_scan(log[key][f"scan_{var}"]))
        write_json(log_path, log)
        return

    t0 = time.time()
    policy = load_policy(args.repo, float32=True)
    print(f"loaded {args.repo} (float32, cpu) in {time.time() - t0:.1f}s")
    check_config(policy, args.num_cameras)

    log = {}
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                log = json.load(f)
        except json.JSONDecodeError:
            print(f"warning: {log_path} is not valid JSON; starting a new log")
    if args.stage in ("all", "assets") and args.norms == DEFAULT_NORMS:
        pre, post = load_processors(args.repo, policy_config=policy.config)
        log["assets"] = write_assets(policy, pre, post, args.out)
        log["tokenizer"] = tokenizer_notes(policy)
    stages = STAGES if args.stage == "all" else ((args.stage,) if args.stage in STAGES else ())
    if args.norms == "exact":
        stages = tuple(st for st in stages if st in FP16_SAFE_STAGES)
    for stage in stages:
        key = graph_file(stage, args.num_cameras, False, args.norms)[:-len(".tflite")]
        log[key] = convert_stage(stage, policy, args.out, args.num_cameras, args.norms)
        write_json(log_path, log)
    if "tokenizer" not in log:
        log["tokenizer"] = tokenizer_notes(policy)
    write_json(log_path, log)
    write_contract(policy, args.repo, args.out, args.num_cameras, log["tokenizer"])


if __name__ == "__main__":
    main()
