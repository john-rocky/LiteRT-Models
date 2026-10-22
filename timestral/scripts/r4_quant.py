"""r4 quantized forms of an exported Timestral graph (ai-edge-quantizer 0.9.0).

  python r4_quant.py drq --src out/tflite/X_fp32.tflite --sel mlp [--rest-fp16]
      dynamic-range int8 (int8 channelwise weights, activations quantized at run time by XNNPACK) on the
      FULLY_CONNECTED ops of one group only; every other op keeps its float weights (fp32, or fp16 with --rest-fp16).
      Writes out/tflite/<stem>_<sel>int8[w16].tflite.
  python r4_quant.py a16w8 --src out/tflite/X_fp32.tflite --n-cal 24
      static int16 activations x int8 weights (recipe static_wi8_ai16) calibrated on synthetic series made here
      (never the parity fixture). Writes out/tflite/<stem>_a16w8.tflite and <stem>_a16w8.calib.json.

The groups match the op scope (= output tensor names, ai_edge_quantizer.utils.tfl_flatbuffer_utils.get_op_scope) of
the litert-torch export, e.g. ".../timestral_graph.Block_3/timestral_graph.SwiGLU_mlp/torch.nn.modules.linear.Linear_up".
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

LIN = r"torch\.nn\.modules\.linear\.Linear_"
GROUPS = {
    "all": r".*",
    "blocks": r"Block_\d+/",
    "mlp": r"SwiGLU_mlp/" + LIN + r"(gate|up|w3)\b",
    "mlpgu": r"SwiGLU_mlp/" + LIN + r"(gate|up)\b",
    "mlpw3": r"SwiGLU_mlp/" + LIN + r"w3\b",
    "attn": r"(TimeAttention|VariateAttention)_\d+/" + LIN + r"(q|k|v|g|o)\b",
    "attnvgo": r"(TimeAttention|VariateAttention)_\d+/" + LIN + r"(v|g|o)\b",
    "attnqk": r"TimeAttention_\d+/" + LIN + r"(q|k)\b",
    "mlpvgo": r"(SwiGLU_mlp/" + LIN + r"(gate|up|w3)|(TimeAttention|VariateAttention)_\d+/" + LIN + r"(v|g|o))\b",
}


def layer_range(sel):
    """'mlp@2-23' -> (base group, layers 2..23) — the group restricted to those blocks."""
    m = re.fullmatch(r"([a-z0-9]+)@(\d+)-(\d+)", sel)
    if not m:
        return sel, None
    return m.group(1), (int(m.group(2)), int(m.group(3)))


def group_regex(sel):
    base, rng = layer_range(sel)
    rx = GROUPS[base]
    if rng is None:
        return rx
    layers = "|".join(str(i) for i in range(rng[0], rng[1] + 1))
    return rf"Block_({layers})/.*?" + rx if base != "blocks" else rf"Block_({layers})/"


def sel_tag(sel):
    return sel.replace("@", "L").replace("-", "t")


def fc_scopes(path):
    """[(op index, scope, weight elements)] of every FULLY_CONNECTED op, read as the quantizer reads them."""
    from ai_edge_quantizer import qtyping
    from ai_edge_quantizer.utils import tfl_flatbuffer_utils as fb
    model = fb.read_model(path)
    sg = model.subgraphs[0]
    out = []
    for i, op in enumerate(sg.operators):
        # the quantizer's table names only the ops it can quantize (REDUCE_MAX = 82 is not in it)
        if fb.TFL_OP_CODE_TO_NAME.get(model.operatorCodes[op.opcodeIndex].builtinCode) != qtyping.TFLOperationName.FULLY_CONNECTED:
            continue
        w = sg.tensors[op.inputs[1]]
        out.append((i, fb.get_op_scope(op, sg.tensors), int(np.prod(w.shape))))
    return out


def cmd_drq(a):
    from ai_edge_quantizer import algorithm_manager, qtyping, quantizer, recipe_manager
    rx = group_regex(a.sel)
    scopes = fc_scopes(a.src)
    hit = [s for s in scopes if re.search(rx, s[1])]
    stem = os.path.basename(a.src).replace("_fp32.tflite", "")
    dst = os.path.join(os.path.dirname(a.src), f"{stem}_{sel_tag(a.sel)}int8{'w16' if a.rest_fp16 else ''}.tflite")
    rm = recipe_manager.RecipeManager()
    if a.rest_fp16:
        rm.add_weight_only_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED, num_bits=16,
                                  algorithm_key=algorithm_manager.AlgorithmName.FLOAT_CASTING)
    rm.add_dynamic_config(regex=rx, operation_name=qtyping.TFLOperationName.FULLY_CONNECTED, num_bits=8)
    if os.path.exists(dst):
        os.remove(dst)
    t0 = time.time()
    qt = quantizer.Quantizer(float_model=a.src)
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(dst)
    info = {"src": os.path.basename(a.src), "dst": os.path.basename(dst), "sel": a.sel, "regex": rx, "rest_fp16": a.rest_fp16,
            "fc_total": len(scopes), "fc_int8": len(hit), "weight_elems_int8": sum(s[2] for s in hit),
            "weight_elems_fc": sum(s[2] for s in scopes), "bytes": os.path.getsize(dst), "seconds": time.time() - t0}
    info["int8_share_of_fc_weights"] = info["weight_elems_int8"] / max(1, info["weight_elems_fc"])
    common.write_json(dst.replace(".tflite", ".quant.json"), info)
    print(f"DRQ {a.sel}: {len(hit)}/{len(scopes)} FC int8 ({100 * info['int8_share_of_fc_weights']:.1f} % of FC weights) "
          f"-> {os.path.basename(dst)} {info['bytes']} B")


# ------------------------------------------------------------------ a16w8
def synthetic_calibration_series(n, seed=20261022 + 4):
    """Own series for the static calibration (the parity fixture is never used): random walks, seasonal sines with
    trend and noise, steps, spikes, intermittent counts, leading / inner gaps, scales from 1e-2 to 1e5."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        t_len = int(rng.choice([96, 200, 500, 1000, 1536, 2048]))
        t = np.arange(t_len, dtype=np.float64)
        kind = i % 6
        if kind == 0:
            x = rng.standard_normal(t_len).cumsum()
        elif kind == 1:
            p = int(rng.choice([7, 12, 24, 48, 168]))
            x = 10 + 5 * np.sin(2 * np.pi * t / p) + 0.01 * t + rng.standard_normal(t_len)
        elif kind == 2:
            x = np.repeat(rng.standard_normal(t_len // 50 + 1) * 3, 50)[:t_len] + 0.3 * rng.standard_normal(t_len)
        elif kind == 3:
            x = np.abs(rng.standard_normal(t_len)) + (rng.random(t_len) < 0.02) * 30.0
        elif kind == 4:
            x = rng.poisson(0.7, t_len).astype(np.float64)
        else:
            p = int(rng.choice([24, 96]))
            x = 50 * (1 + np.sin(2 * np.pi * t / p)) * (1 + 0.002 * t) + 3 * rng.standard_normal(t_len)
        x = x * float(10.0 ** rng.uniform(-2, 5)) + float(rng.uniform(-100, 100)) * (kind in (0, 2))
        if rng.random() < 0.4:
            a0 = int(rng.integers(0, t_len // 3))
            x[: a0] = np.nan
        if rng.random() < 0.4:
            g0 = int(rng.integers(t_len // 3, t_len - 10))
            x[g0: g0 + int(rng.integers(1, 40))] = np.nan
        out.append(x.astype(np.float32))
    return out


def calibration_inputs(path, series):
    import gate_mac
    import host_forecast as hf
    run_r, run_l, run_h = gate_mac.parse_stem(path)
    gi = gate_mac.graph_inputs()
    neg = hf.mask_value(path)
    samples = []
    for s in series:
        rows, obs = common.assemble_rows(s)
        lc = rows.shape[-1] // 32
        if lc > run_l:
            rows, obs = rows[:, -run_l * 32:], obs[:, -run_l * 32:]
        x = np.concatenate([rows, -rows], 0) if run_r == 2 else rows
        o = np.concatenate([obs, obs], 0) if run_r == 2 else obs
        loc, scale = hf.patch_loc_scale(x, o)
        scaled = hf.normalize(x, loc, scale, o)
        ins = gi.build(scaled, o, run_l, run_h, neg=neg)
        samples.append(dict(zip(("patches", "dec", "bias", "cos", "sin"), ins)))
    return samples


def cmd_a16w8(a):
    from ai_edge_quantizer import algorithm_manager, qtyping, quantizer, recipe, recipe_manager
    series = synthetic_calibration_series(a.n_cal)
    samples = calibration_inputs(a.src, series)
    stem = os.path.basename(a.src).replace("_fp32.tflite", "")
    dst = os.path.join(os.path.dirname(a.src), f"{stem}_a16w8{'f' if a.float_io else ''}.tflite")
    if os.path.exists(dst):
        os.remove(dst)
    info = {"src": os.path.basename(a.src), "dst": os.path.basename(dst), "recipe": "static_wi8_ai16", "n_cal": len(samples),
            "float_io": a.float_io,
            "calibration": "synthetic series of r4_quant.synthetic_calibration_series (no fixture window)"}
    t0 = time.time()
    qt = quantizer.Quantizer(float_model=a.src)
    if a.float_io:
        # the same static recipe, with the graph's inputs and outputs (the quantizer's virtual INPUT / OUTPUT ops) left
        # float32: the converter puts QUANTIZE after each input and DEQUANTIZE before the output, so a runner that writes
        # and reads float32 (the lane's C-API runner, the app) can feed the file
        rm = recipe_manager.RecipeManager()
        rm.add_static_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED, activation_num_bits=16,
                             weight_num_bits=8)
        for op in (qtyping.TFLOperationName.INPUT, qtyping.TFLOperationName.OUTPUT):
            rm.add_quantization_config(regex=".*", operation_name=op, op_config=qtyping.OpQuantizationConfig(),
                                       algorithm_key=algorithm_manager.AlgorithmName.NO_QUANTIZE)
        qt.load_quantization_recipe(rm.get_quantization_recipe())
    else:
        qt.load_quantization_recipe(recipe.static_wi8_ai16())
    try:
        cal = qt.calibrate({"serving_default": samples}, num_threads=a.threads)
        info["calibrate_seconds"] = time.time() - t0
        t1 = time.time()
        qt.quantize(cal).export_model(dst)
        info["quantize_seconds"] = time.time() - t1
        info["bytes"] = os.path.getsize(dst)
        info["status"] = "ok"
    except Exception as e:  # the attempt's outcome is the result: keep the error line
        info["status"] = "error"
        info["error"] = f"{type(e).__name__}: {str(e)[:2000]}"
    common.write_json(dst.replace(".tflite", ".calib.json"), info)
    print(f"A16W8 {info['status']}: {info.get('error', '')[:300]} -> {os.path.basename(dst)} {info.get('bytes')} B "
          f"(calibrate {info.get('calibrate_seconds', 0):.0f} s, quantize {info.get('quantize_seconds', 0):.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("drq")
    d.add_argument("--src", required=True)
    d.add_argument("--sel", required=True, help="group name or group@first-last (blocks)")
    d.add_argument("--rest-fp16", action="store_true")
    s = sub.add_parser("a16w8")
    s.add_argument("--src", required=True)
    s.add_argument("--n-cal", type=int, default=24)
    s.add_argument("--threads", type=int, default=12)
    s.add_argument("--float-io", action="store_true", help="keep the graph inputs / outputs float32")
    a = ap.parse_args()
    {"drq": cmd_drq, "a16w8": cmd_a16w8}[a.cmd](a)


if __name__ == "__main__":
    main()
