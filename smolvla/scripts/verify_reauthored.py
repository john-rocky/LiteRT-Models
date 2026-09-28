#!/usr/bin/env python3
"""Parity of the re-authored SmolVLA torch modules against lerobot 0.6.1 (float32, CPU).

Reference = lerobot's own SmolVLAPolicy (lerobot/smolvla_base) with every module
upcast to float32 (`policy.model.float()`), driven through lerobot's own
pre/post-processors. Compared stage by stage:

  host glue   letterbox, token ids/mask, fp16 embedding lookup, masks, RoPE/time tables
  isolated    VisionGraph / PrefixGraph / ExpertStepGraph fed with lerobot's exact
              internal tensors (measures the re-authoring alone)
  chained     numpy host + the three torch modules end to end (10 Euler steps)
              vs lerobot sample_actions / predict_action_chunk with the same noise
  context     lerobot default dtype (bf16 VLM) vs lerobot float32

Writes <out>/parity_reauthored.json and the fixtures in <out>/fixtures/ (raw
little-endian float32 per graph input, in graph input order, + shapes.json).

Run (repo root):
  KMP_DUPLICATE_LIB_OK=TRUE python smolvla/scripts/verify_reauthored.py --out smolvla/out
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import smolvla_host as host
from litert_helpers import FixtureWriter, fmt_parity, parity
from smolvla_graphs import (N_KV, N_LAYERS, ExpertStepGraph, PrefixGraph, VisionGraph,
                            gqa_attention, load_policy, load_processors,
                            norm_stats_from_processors, set_fp16_safe_norms, tile_bias)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_IMAGE = os.path.join(REPO_ROOT, "tipsv2", "scripts", "test.jpg")
DEFAULT_TASK = "pick up the red cube and place it in the box"
# A fixed 6-dim state on the normalized scale (smolvla_base applies no state stats).
DEFAULT_STATE = [0.25, -0.60, 0.90, -0.30, 0.45, -1.10]

RESULTS = {}


def report(key, a, b, mask_rows=None):
    """Parity of a vs reference b (optionally only rows `mask_rows` of axis -2)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if mask_rows is not None:
        a = a[..., mask_rows, :]
        b = b[..., mask_rows, :]
    p = parity(a, b)
    RESULTS[key] = p
    print(f"  {key:<44s} {fmt_parity(p)}")
    return p


def t2n(x):
    return x.detach().cpu().float().numpy()


FP16_MAX = 65504.0
FP16_SQ_LIMIT = 255.9      # |x| above this overflows fp16 when squared


class RangeRecorder:
    """Forward hooks that record what an fp16 GPU would have to hold.

    For every norm module: the input's max|x| and, for RMSNorm, mean(x*x) per
    token / for LayerNorm, the centred (x - mean) and its mean square; for every
    nn.Linear / nn.Conv2d: max|input| and max|output|. `rows` restricts the prefix
    graph to non-pad tokens.
    """

    def __init__(self, module, rows=None):
        import torch.nn as nn
        from smolvla_graphs import LayerNorm, RMSNorm
        self.rows = rows
        self.norm, self.linear = {}, {}
        self.handles = []
        for name, m in module.named_modules():
            if isinstance(m, (RMSNorm, LayerNorm, nn.LayerNorm)):
                self.handles.append(m.register_forward_hook(self._norm_hook(name)))
            elif isinstance(m, (nn.Linear, nn.Conv2d)):
                self.handles.append(m.register_forward_hook(self._lin_hook(name)))

    def _sel(self, x):
        x = x.detach().double()
        if self.rows is not None and x.dim() == 3 and x.shape[1] > max(self.rows):
            x = x[:, torch.as_tensor(self.rows)]
        return x

    def _norm_hook(self, name):
        def hook(m, inp, out):
            x = self._sel(inp[0])
            if not hasattr(m, "variance_epsilon") and hasattr(m, "bias"):   # LayerNorm
                d = x - x.mean(-1, keepdim=True)
            else:
                d = x
            rec = self.norm.setdefault(name, {"max_abs": 0.0, "max_centered": 0.0, "max_mean_sq": 0.0,
                                              "n_sq_overflow": 0})
            rec["max_abs"] = max(rec["max_abs"], float(x.abs().max()))
            rec["max_centered"] = max(rec["max_centered"], float(d.abs().max()))
            rec["max_mean_sq"] = max(rec["max_mean_sq"], float((d * d).mean(-1).max()))
            rec["n_sq_overflow"] = max(rec["n_sq_overflow"], int((d.abs() > FP16_SQ_LIMIT).sum()))
        return hook

    def _lin_hook(self, name):
        def hook(m, inp, out):
            rec = self.linear.setdefault(name, {"max_in": 0.0, "max_out": 0.0})
            rec["max_in"] = max(rec["max_in"], float(self._sel(inp[0]).abs().max()))
            rec["max_out"] = max(rec["max_out"], float(self._sel(out).abs().max()))
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()

    def summary(self):
        worst_norm = max(self.norm.items(), key=lambda kv: kv[1]["max_mean_sq"])
        worst_lin = max(self.linear.items(), key=lambda kv: kv[1]["max_out"])
        return {
            "norm_inputs": self.norm, "linear_io": self.linear,
            "max_norm_input_abs": max(r["max_abs"] for r in self.norm.values()),
            "max_norm_mean_square": worst_norm[1]["max_mean_sq"], "worst_norm": worst_norm[0],
            "norms_with_fp16_square_overflow": sum(1 for r in self.norm.values() if r["n_sq_overflow"]),
            "num_norms": len(self.norm),
            "max_linear_in": max(r["max_in"] for r in self.linear.values()),
            "max_linear_out": worst_lin[1]["max_out"], "worst_linear_out": worst_lin[0],
        }


def torch_rope_tables(positions):
    """cos/sin exactly as lerobot apply_rope computes them (float32 torch ops)."""
    pos = torch.as_tensor(np.asarray(positions), dtype=torch.long)[None]
    d_half = 32
    freq_exponents = (2.0 / 64) * torch.arange(d_half, dtype=torch.float32)
    timescale = 10_000 ** freq_exponents
    radians = pos[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)
    return torch.cos(radians)[:, None], torch.sin(radians)[:, None]        # [1,1,N,32]


def bias_from_mask(mask_bool):
    return torch.where(mask_bool, torch.tensor(0.0), torch.tensor(-30000.0)).to(torch.float32)


# ---------------------------------------------------------------- lerobot side
@torch.no_grad()
def lerobot_run(policy, pre, post, image01, task, state6, noise):
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    model = policy.model
    vwe = model.vlm_with_expert
    batch = {"observation.images.camera1": torch.from_numpy(image01.copy()),
             "observation.state": torch.tensor(state6, dtype=torch.float32),
             "task": task}
    b = pre(batch)
    images, img_masks = policy.prepare_images(b)
    state = policy.prepare_state(b)
    tokens = b["observation.language.tokens"]
    lmask = b["observation.language.attention_mask"]

    ref = {"batch": b, "images": images, "img_masks": img_masks, "state": state,
           "tokens": tokens, "lang_mask": lmask}
    ref["img_emb"] = vwe.embed_image(images[0])
    ref["lang_raw"] = vwe.embed_language_tokens(tokens)
    prefix_embs, pad, att = model.embed_prefix(images, img_masks, tokens, lmask, state=state)
    att2d = make_att_2d_masks(pad, att)
    pos = torch.cumsum(pad, dim=1) - 1
    ref.update(prefix_embs=prefix_embs, pad=pad, att=att, att2d=att2d, pos=pos)
    _, pkv = vwe.forward(attention_mask=att2d, position_ids=pos, past_key_values=None,
                         inputs_embeds=[prefix_embs, None], use_cache=True)
    ref["K"] = [pkv.layers[l].keys.clone() for l in range(N_LAYERS)]
    ref["V"] = [pkv.layers[l].values.clone() for l in range(N_LAYERS)]

    # euler_integrate unrolled (identical arithmetic) to capture every step.
    num_steps = policy.config.num_steps
    dt = -1.0 / num_steps
    x = noise.clone()
    xs, vs = [], []
    for step in range(num_steps):
        tt = torch.tensor(1.0 + step * dt, dtype=torch.float32).expand(1)
        v = model.denoise_step(prefix_pad_masks=pad, past_key_values=pkv, x_t=x, timestep=tt)
        x = x + dt * v
        xs.append(x.clone())
        vs.append(v.clone())
    ref.update(xs=xs, vs=vs)
    ref["x_sample_actions"] = model.sample_actions(images, img_masks, tokens, lmask, state,
                                                   noise=noise.clone())
    policy.reset()
    chunk = policy.predict_action_chunk(b, noise=noise.clone())
    ref["actions"] = post(chunk)

    # the denoise_step masks, rebuilt exactly as lerobot does
    suffix_len = policy.config.chunk_size
    prefix_len = pad.shape[1]
    prefix_pad_2d = pad[:, None, :].expand(1, suffix_len, prefix_len)
    suffix_2d = make_att_2d_masks(torch.ones(1, suffix_len, dtype=torch.bool),
                                  torch.ones(1, suffix_len, dtype=torch.float32))
    ref["full_att_2d"] = torch.cat([prefix_pad_2d, suffix_2d], dim=2)
    ref["prefix_offset"] = int(torch.sum(pad, dim=-1).item())
    return ref


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--state", type=float, nargs="+", default=DEFAULT_STATE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "smolvla", "out"))
    ap.add_argument("--skip-bf16", action="store_true", help="skip the default-dtype context run")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    t0 = time.time()
    policy = load_policy(float32=True)
    pre, post = load_processors(policy_config=policy.config)
    print(f"loaded lerobot policy (float32, cpu) in {time.time() - t0:.1f}s")
    model = policy.model
    vwe = model.vlm_with_expert
    vlm = vwe.get_vlm_model()

    from lerobot.policies.common.flow_matching import sample_noise
    torch.manual_seed(args.seed)
    noise = sample_noise((1, policy.config.chunk_size, policy.config.max_action_dim), "cpu")

    image01 = host.load_image_rgb01(args.image)
    t0 = time.time()
    ref = lerobot_run(policy, pre, post, image01, args.task, args.state, noise)
    print(f"lerobot float32 run: {time.time() - t0:.1f}s")
    assert torch.equal(ref["xs"][-1], ref["x_sample_actions"]), "unrolled Euler != sample_actions"
    print("  unrolled euler_integrate == sample_actions: bit-exact")

    # ------------------------------------------------------------- host glue
    print("\n[host glue vs lerobot]")
    tokenizer = host.load_tokenizer(policy.config.vlm_model_name)
    ids, lmask = host.tokenize(tokenizer, args.task)
    ids_ok = bool(np.array_equal(ids, t2n(ref["tokens"][0]).astype(np.int64)))
    mask_ok = bool(np.array_equal(lmask, ref["lang_mask"][0].numpy()))
    RESULTS["token_ids_equal"] = ids_ok
    RESULTS["lang_mask_equal"] = mask_ok
    print(f"  token ids equal: {ids_ok}  mask equal: {mask_ok}  ({int(lmask.sum())} valid tokens)")
    assert ids_ok and mask_ok
    lb = host.letterbox_512(image01)
    report("letterbox host vs lerobot prepare_images", lb, t2n(ref["images"][0]))
    table16 = t2n(vlm.text_model.embed_tokens.weight).astype(np.float16)
    lang_emb = host.embed_lookup(table16, ids)
    report("lang_emb host fp16 table vs lerobot", lang_emb, t2n(ref["lang_raw"]))
    stats = norm_stats_from_processors(pre, post)
    RESULTS["norm_stats_applied"] = {"state": stats["state"] is not None,
                                     "action": stats["action"] is not None}
    ns = host.NormStats(stats["state"], stats["action"], stats["eps"] or 1e-8)
    print(f"  lerobot applies state stats: {stats['state'] is not None}, action stats: "
          f"{stats['action'] is not None}; unmatched keys: {sorted(stats['unmatched_stat_keys'])}")
    state_host = host.pad_state(ns.normalize_state(args.state))
    report("state host vs lerobot prepare_state", state_host, t2n(ref["state"]))
    layout = host.PrefixLayout(lmask, num_cameras=1)
    mask_eq = bool(np.array_equal(layout.mask_2d, ref["att2d"][0].numpy()))
    pos_eq = bool(np.array_equal(layout.positions, ref["pos"][0].numpy()))
    off_eq = layout.offset == ref["prefix_offset"]
    bias_self, bias_cross = layout.expert_biases()
    full_eq = bool(np.array_equal(bias_self[0, 0] == 0, ref["full_att_2d"][0].numpy()))
    RESULTS.update(prefix_mask_equal=mask_eq, prefix_positions_equal=pos_eq,
                   prefix_offset_equal=off_eq, expert_mask_equal=full_eq,
                   prefix_offset=layout.offset)
    print(f"  prefix 2D mask equal: {mask_eq}, positions equal: {pos_eq}, prefix_offset "
          f"{layout.offset} equal: {off_eq}, expert mask equal: {full_eq}")
    assert mask_eq and pos_eq and off_eq and full_eq
    cos_h, sin_h = host.rope_tables(layout.positions)
    cos_t, sin_t = torch_rope_tables(layout.positions)
    report("rope cos host vs torch (lerobot formula)", cos_h, t2n(cos_t))
    report("rope sin host vs torch (lerobot formula)", sin_h, t2n(sin_t))
    from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding
    times, dt = host.euler_times()
    temb_diff = 0.0
    for t in times:
        tt = torch.tensor(float(t), dtype=torch.float32).expand(1)
        te = create_sinusoidal_pos_embedding(tt, 720, policy.config.min_period,
                                             policy.config.max_period, device=torch.device("cpu"))
        temb_diff = max(temb_diff, float(np.abs(host.time_embedding(t)[0] - t2n(te.float())).max()))
    RESULTS["time_emb_host_vs_lerobot_max_abs_diff"] = temb_diff
    print(f"  time embedding host vs lerobot (10 steps): max|diff| = {temb_diff:.3e}")

    # ---------------------------------------------------- isolated modules
    print("\n[isolated: re-authored torch module vs lerobot, lerobot's own inputs]")
    vis = VisionGraph(vlm.vision_model, vlm.connector).eval()
    prefix_g = PrefixGraph(model, num_cameras=1).eval()
    expert_g = ExpertStepGraph(model, policy.config.self_attn_every_n_layers).eval()
    pos_embed = vlm.vision_model.embeddings.position_embedding.weight[None].float()

    img_emb_iso = vis(ref["images"][0], pos_embed)
    report("img_emb [1,64,960]", t2n(img_emb_iso), t2n(ref["img_emb"]))

    valid = layout.valid_index()
    bias_ref = bias_from_mask(ref["att2d"])[:, None]
    cos_ref, sin_ref = torch_rope_tables(ref["pos"][0].numpy())
    k_iso, v_iso = prefix_g(ref["img_emb"], ref["lang_raw"], ref["state"], bias_ref, cos_ref, sin_ref)
    kk = t2n(k_iso).reshape(1, N_LAYERS, N_KV, -1, 64)
    vv = t2n(v_iso).reshape(1, N_LAYERS, N_KV, -1, 64)
    worst_k = worst_v = None
    for l in range(N_LAYERS):
        pk = report(f"k layer {l:2d} (valid tokens)", kk[:, l], t2n(ref["K"][l]), valid)
        pv = report(f"v layer {l:2d} (valid tokens)", vv[:, l], t2n(ref["V"][l]), valid)
        worst_k = pk if worst_k is None or pk["max_abs_diff"] > worst_k["max_abs_diff"] else worst_k
        worst_v = pv if worst_v is None or pv["max_abs_diff"] > worst_v["max_abs_diff"] else worst_v
    k_ref_all = torch.cat(ref["K"], dim=1)
    v_ref_all = torch.cat(ref["V"], dim=1)
    report("k_all all layers (valid tokens)", t2n(k_iso), t2n(k_ref_all), valid)
    report("v_all all layers (valid tokens)", t2n(v_iso), t2n(v_ref_all), valid)

    bs_ref = bias_from_mask(ref["full_att_2d"])[:, None]
    bc_ref = bias_from_mask(ref["full_att_2d"][:, :, :layout.length])[:, None]
    cos_s, sin_s = torch_rope_tables(layout.offset + np.arange(50))
    cos_c, sin_c = torch_rope_tables(np.arange(50))
    te0 = create_sinusoidal_pos_embedding(torch.tensor([1.0]), 720, policy.config.min_period,
                                          policy.config.max_period,
                                          device=torch.device("cpu")).float()[:, None]
    v0_iso = expert_g(noise, te0, k_ref_all, v_ref_all, bs_ref, bc_ref, cos_s, sin_s, cos_c, sin_c)
    report("v_t step 0 [1,50,32]", t2n(v0_iso), t2n(ref["vs"][0]))

    # GQA rewrite vs lerobot's eager_attention_forward on real layer-0 tensors
    layer0 = vlm.text_model.layers[0]
    from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope
    h = layer0.input_layernorm(ref["prefix_embs"])
    q = apply_rope(layer0.self_attn.q_proj(h).view(1, -1, 15, 64), ref["pos"])
    k = apply_rope(layer0.self_attn.k_proj(h).view(1, -1, 5, 64), ref["pos"])
    v = layer0.self_attn.v_proj(h).view(1, -1, 5, 64)
    att_ref = vwe.eager_attention_forward(ref["att2d"], 1, 64, q, k, v)
    att_ours = gqa_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
                             v.permute(0, 2, 1, 3), tile_bias(bias_ref))
    report("GQA reshape+bias vs lerobot eager (valid rows)", t2n(att_ours), t2n(att_ref), valid)
    # naive expand (15-head) with the same additive bias: isolates the reshape itself
    ke = k.permute(0, 2, 1, 3).repeat_interleave(3, dim=1)
    ve = v.permute(0, 2, 1, 3).repeat_interleave(3, dim=1)
    w = torch.matmul(q.permute(0, 2, 1, 3), ke.transpose(2, 3)) * 0.125 + bias_ref
    naive = torch.matmul(torch.softmax(w, -1), ve).permute(0, 2, 1, 3).reshape(1, -1, 960)
    report("GQA reshape vs naive expand (all rows)", t2n(att_ours), t2n(naive))

    # --------------------------------------------------------- chained run
    print("\n[chained: numpy host + torch modules vs lerobot (same image/task/state/noise)]")
    vis.taps, prefix_g.taps, expert_g.taps = [], [], []

    def f_vision(image, pe):
        return t2n(vis(torch.from_numpy(image), torch.from_numpy(pe)))

    def f_prefix(*a):
        k_, v_ = prefix_g(*[torch.from_numpy(np.ascontiguousarray(x)) for x in a])
        return t2n(k_), t2n(v_)

    def f_expert(*a):
        return t2n(expert_g(*[torch.from_numpy(np.ascontiguousarray(x)) for x in a]))

    rec_vis = RangeRecorder(vis)
    rec_pre = RangeRecorder(prefix_g, rows=valid)
    rec_exp = RangeRecorder(expert_g)
    pe_np = t2n(pos_embed)
    runner = host.SmolVLAHost(f_vision, f_prefix, f_expert, table16, pe_np, tokenizer,
                              stats=ns, num_cameras=1, action_dim=6)
    noise_np = t2n(noise)
    t0 = time.time()
    res = runner.run([image01], args.task, args.state, noise_np)
    print(f"  chained torch run: {time.time() - t0:.1f}s")
    report("chain img_emb", res["img_emb"], t2n(ref["img_emb"]))
    report("chain k_all (valid tokens)", res["k_all"], t2n(k_ref_all), valid)
    report("chain v_all (valid tokens)", res["v_all"], t2n(v_ref_all), valid)
    for step in range(len(res["trace"])):
        report(f"chain v_t step {step}", res["trace"][step][1], t2n(ref["vs"][step]))
    report("chain final x [1,50,32]", res["x"], t2n(ref["x_sample_actions"]))
    report("chain final x[..., :6] (normalized units)", res["x"][..., :6],
           t2n(ref["x_sample_actions"])[..., :6])
    report("chain actions vs lerobot predict+postprocess", res["actions"], t2n(ref["actions"]))

    # residual-stream magnitudes on this real input
    mags = {
        "vision_layer_out_max_abs": [float(t.abs().max()) for t in vis.taps],
        "prefix_layer_in_max_abs_valid": [float(t[0, valid].abs().max()) for t in prefix_g.taps],
        "prefix_layer_in_max_abs_all": [float(t.abs().max()) for t in prefix_g.taps],
    }
    n_img = 64
    seg = {"image": np.arange(n_img), "language": valid[(valid >= n_img) & (valid < layout.length - 1)],
           "state": np.array([layout.length - 1])}
    mags["prefix_layer_in_max_abs_by_segment"] = {
        name: [float(t[0, idx].abs().max()) for t in prefix_g.taps] for name, idx in seg.items()}
    per = N_LAYERS + 1
    steps = [expert_g.taps[i * per:(i + 1) * per] for i in range(len(expert_g.taps) // per)]
    mags["expert_layer_in_max_abs_per_step"] = [[float(t.abs().max()) for t in s] for s in steps]
    RESULTS["residual_magnitudes"] = mags
    print(f"\n[residual stream max|x|] vision per layer: "
          f"{[round(m, 1) for m in mags['vision_layer_out_max_abs']]}")
    print(f"  prefix (valid tokens) input of layer 0..15: "
          f"{[round(m, 1) for m in mags['prefix_layer_in_max_abs_valid']]}")
    print(f"  prefix (all tokens):  {[round(m, 1) for m in mags['prefix_layer_in_max_abs_all']]}")
    for name, vals in mags["prefix_layer_in_max_abs_by_segment"].items():
        print(f"  prefix {name} tokens: {[round(m, 1) for m in vals]}")
    emax = [max(s[i] for s in mags["expert_layer_in_max_abs_per_step"]) for i in range(per)]
    print(f"  expert (max over 10 steps) input of layer 0..15, final: {[round(m, 1) for m in emax]}")

    print("\n[fp16 range check: inputs of every norm, inputs/outputs of every Linear/Conv "
          "(prefix: non-pad tokens), chained run]")
    ranges = {}
    for gname, rec in (("vision", rec_vis), ("prefix", rec_pre), ("expert", rec_exp)):
        sm = rec.summary()
        rec.remove()
        ranges[gname] = sm
        print(f"  {gname:7s} norm inputs: max|x| {sm['max_norm_input_abs']:.1f}, max mean-square "
              f"{sm['max_norm_mean_square']:.4g} ({sm['worst_norm']}), norms with some |x| > 255.9 "
              f"(x*x overflows fp16): {sm['norms_with_fp16_square_overflow']}/{sm['num_norms']}; "
              f"Linear/Conv max|in| {sm['max_linear_in']:.1f}, max|out| {sm['max_linear_out']:.1f} "
              f"({sm['worst_linear_out']})")
    RESULTS["fp16_range_check"] = ranges

    # -------------------- fp16-safe norms (the default graphs) vs the exact modules
    print("\n[fp16safe norms (default graphs): exact down-scaled RMSNorm/LayerNorm in vision + prefix]")
    vis_s = VisionGraph(vlm.vision_model, vlm.connector).eval()
    pre_s = PrefixGraph(model, num_cameras=1).eval()
    n_sw = set_fp16_safe_norms(vis_s) + set_fp16_safe_norms(pre_s)
    print(f"  switched {n_sw} norms")
    report("fp16safe vision vs exact module (fixture input)",
           t2n(vis_s(torch.from_numpy(res["prepared"]["images"][0]), pos_embed)), res["img_emb"])
    p_in = [res["img_emb"], res["prepared"]["lang_emb"], res["prepared"]["state"],
            res["prepared"]["attn_bias"], res["prepared"]["rope_cos"], res["prepared"]["rope_sin"]]
    k_s, v_s = pre_s(*[torch.from_numpy(np.ascontiguousarray(x)) for x in p_in])
    report("fp16safe prefix k_all vs exact module (non-pad tokens)", t2n(k_s), res["k_all"], valid)
    report("fp16safe prefix v_all vs exact module (non-pad tokens)", t2n(v_s), res["v_all"], valid)
    expert_g.taps = None
    runner_s = host.SmolVLAHost(
        lambda im, pe: t2n(vis_s(torch.from_numpy(im), torch.from_numpy(pe))),
        lambda *a: tuple(t2n(t) for t in pre_s(*[torch.from_numpy(np.ascontiguousarray(x)) for x in a])),
        f_expert, table16, pe_np, tokenizer, stats=ns, num_cameras=1, action_dim=6)
    res_s = runner_s.run([image01], args.task, args.state, noise_np)
    report("fp16safe chain final x vs lerobot", res_s["x"], t2n(ref["x_sample_actions"]))
    del vis_s, pre_s

    # ------------------------------------------------------------ fixtures
    fx = FixtureWriter(os.path.join(args.out, "fixtures"))
    p = res["prepared"]
    fx.add("vision", "image", p["images"][0], "input", 0)
    fx.add("vision", "pos_embed", pe_np, "input", 1)
    fx.add("vision", "img_emb", res["img_emb"], "output", 0, note="torch VisionGraph output")
    fx.add("vision", "lerobot_img_emb", t2n(ref["img_emb"]), note="lerobot embed_image (fp32)")
    prefix_inputs = [("img_emb", res["img_emb"]), ("lang_emb", p["lang_emb"]),
                     ("state", p["state"]), ("attn_bias", p["attn_bias"]),
                     ("rope_cos", p["rope_cos"]), ("rope_sin", p["rope_sin"])]
    for i, (n, a) in enumerate(prefix_inputs):
        fx.add("prefix", n, a, "input", i)
    fx.add("prefix", "k_all", res["k_all"], "output", 0, note="torch PrefixGraph output")
    fx.add("prefix", "v_all", res["v_all"], "output", 1, note="torch PrefixGraph output")
    fx.add("prefix", "valid_token_index", valid.astype(np.int32), dtype="int32",
           note="prefix positions that are not padding (compare K/V only there)")
    x0, temb0 = noise_np, host.time_embedding(times[0])
    expert_inputs = [("x_t", x0), ("time_emb", temb0), ("k_all", res["k_all"]),
                     ("v_all", res["v_all"]), ("bias_self", p["bias_self"]),
                     ("bias_cross", p["bias_cross"]), ("rope_cos_s", p["rope_cos_s"]),
                     ("rope_sin_s", p["rope_sin_s"]), ("rope_cos_c", p["rope_cos_c"]),
                     ("rope_sin_c", p["rope_sin_c"])]
    for i, (n, a) in enumerate(expert_inputs):
        fx.add("expert_step0", n, a, "input", i)
    fx.add("expert_step0", "v_t", res["trace"][0][1], "output", 0,
           note="torch ExpertStepGraph output at step 0")
    fx.add("chain", "noise", noise_np)
    for step, t in enumerate(times):
        x_in = noise_np if step == 0 else res["trace"][step - 1][0]
        fx.add("chain", f"x_t_step{step}", x_in, note="expert input x_t at this step (torch chain)")
        fx.add("chain", f"time_emb_step{step}", host.time_embedding(t))
        fx.add("chain", f"torch_v_t_step{step}", res["trace"][step][1],
               note="torch ExpertStepGraph(x_t_step, time_emb_step, expert_step0 statics)")
        fx.add("chain", f"lerobot_v_t_step{step}", t2n(ref["vs"][step]))
    fx.add("chain", "torch_x_final", res["x"])
    fx.add("chain", "lerobot_x_final", t2n(ref["x_sample_actions"]))
    fx.add("chain", "torch_actions", res["actions"])
    fx.add("chain", "lerobot_actions", t2n(ref["actions"]))
    fx.add("chain", "token_ids", ids.astype(np.int32), dtype="int32")
    fx.add("chain", "lang_mask", lmask.astype(np.int32), dtype="int32")
    fixture_meta = {
        "image": os.path.relpath(args.image, REPO_ROOT), "task": args.task,
        "state": list(args.state), "seed": args.seed, "prefix_offset": layout.offset,
        "num_valid_lang_tokens": int(lmask.sum()), "euler_dt_float32": float(dt),
        "euler_times_float32": [float(t) for t in times],
    }
    fx.write_index(fixture_meta)
    print(f"\nfixtures -> {fx.root}")

    # ------------------------------------------------- default dtype context
    if not args.skip_bf16:
        print("\n[context: lerobot default dtype (bf16 VLM/expert weights) vs lerobot float32]")
        del vis, prefix_g, expert_g
        pol16 = load_policy(float32=False)
        t0 = time.time()
        with torch.no_grad():
            b = ref["batch"]
            images, img_masks = pol16.prepare_images(b)
            state = pol16.prepare_state(b)
            x16 = pol16.model.sample_actions(images, img_masks, ref["tokens"], ref["lang_mask"],
                                             state, noise=noise.clone())
        print(f"  default-dtype run: {time.time() - t0:.1f}s")
        report("lerobot default dtype vs float32: final x", t2n(x16), t2n(ref["x_sample_actions"]))
        report("lerobot default dtype vs float32: x[..., :6]", t2n(x16)[..., :6],
               t2n(ref["x_sample_actions"])[..., :6])
        fx.add("chain", "lerobot_default_dtype_x_final", t2n(x16))
        fx.write_index(fixture_meta)

    with open(os.path.join(args.out, "parity_reauthored.json"), "w") as f:
        json.dump(RESULTS, f, indent=1)
    print(f"\nwrote {os.path.join(args.out, 'parity_reauthored.json')}")


if __name__ == "__main__":
    main()
