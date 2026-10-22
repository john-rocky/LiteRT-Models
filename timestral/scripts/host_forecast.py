"""Host side of the Timestral LiteRT port: the official forecast() around one graph call.

NumPy only (no torch), univariate series, the official defaults: symmetric averaging,
non-negative clipping, quantile cap k = 1e4, blocks of up to 32 patches with the advancing
refine frame, max_context 32768. Each stage keeps the official dtype: the causal statistics
and the refine frames are float64; normalization, denormalization, the cap and the averaging
are float32. asinh / sinh are evaluated in float64 and rounded to float32 (the official
float32 kernels agree to about 1 ulp); this is the reference for the Kotlin port.

A runner turns one decode block into normalized quantiles:
    runner(scaled [R, T] f32, observed [R, T] bool, horizon_patches h) -> [R, h*32, 21] f32
where scaled / observed are the normalized context of R rows that share `observed`.
GraphInputs builds the five graph inputs of timestral_graph.TimestralGraph for a bucket.
"""

import math
import os

import numpy as np

PATCH = 32
D_MODEL = 768
N_Q = 21
MEDIAN = 10
MIN_SCALE = 1e-6
MAX_CONTEXT = 32768
BLOCK = 32
CAP_K = 1e4
EXTREME = 1e30
# Additive mask for blocked pairs. Timestral-209M's attention logits reach ~1.6e5 and a blocked key can
# sit 4e4 above the best allowed one (measured on the parity windows), so -1e4 does not zero it; -1e30
# does in fp32 (and becomes -inf in fp16, where exp still gives 0).
NEG = -1.0e30
# Graphs exported with attn_k = k (file names with `_att`) add the mask to logits already divided by
# 2^k, so the finite fp16 value -3e4 (= -4.8e5 logit units at k = 4, far below the 4.2e4 gap) blocks in
# fp32 too, and it keeps -inf out of the row max: with -1e30 (-inf in fp16) Metal's fp16 kernels gave
# NaN rows (desktop GPU, LiteRT 2.2.0). TS_NEG overrides the value for every graph.
NEG_ATTN_K = -3.0e4
if os.environ.get("TS_NEG"):
    NEG = NEG_ATTN_K = float(os.environ["TS_NEG"])


def mask_value(graph_name):
    """The additive mask value the host uses for a graph file."""
    return NEG_ATTN_K if "_att" in os.path.basename(str(graph_name)) else NEG
F32 = np.float32
F64 = np.float64
F32_MAX = float(np.finfo(np.float32).max)
F32_EPS = float(np.finfo(np.float32).eps)
F64_TINY = float(np.finfo(np.float64).tiny)


# ------------------------------------------------------------------ normalization
def causal_statistics(x, w):
    """Official compute_causal_statistics: float64 running mean / Bessel std, floored."""
    x = x.astype(F64)
    w = w.astype(F64)
    weighted = w * x
    cum_w = np.cumsum(w, axis=-1)
    cum_x = np.cumsum(weighted, axis=-1)
    den = np.maximum(cum_w, 1.0)
    means = cum_x / den
    shifted = np.zeros_like(means)
    shifted[..., 1:] = means[..., :-1]
    inc = (x - shifted) * (x - means) * w
    m2 = np.cumsum(inc, axis=-1)
    var = m2 / np.maximum(den - 1.0, 1.0)
    scales = np.maximum(np.sqrt(var), MIN_SCALE)
    return means.astype(F32), scales.astype(F32)


def patch_loc_scale(x, observed):
    """Per-step (loc, scale) f32: each patch takes the statistics at its last step."""
    means, scales = causal_statistics(x, observed)
    shp = x.shape[:-1] + (x.shape[-1] // PATCH, PATCH)
    loc = np.repeat(means.reshape(shp)[..., -1], PATCH, axis=-1)
    scale = np.repeat(scales.reshape(shp)[..., -1], PATCH, axis=-1)
    return loc, scale


def normalize(x, loc, scale, observed):
    """asinh((x - loc) / max(scale, eps)) on observed steps (f32), 0 elsewhere."""
    y = (x.astype(F32) - loc) / np.maximum(scale, F32(F32_EPS))
    y = np.where(observed, y, F32(0.0))
    return np.where(observed, np.arcsinh(y.astype(F64)).astype(F32), y)


def clamp_nonfinite(v):
    """Official clamp_nonfinite along the last axis (NaN untouched)."""
    fin = np.isfinite(v)
    hi = np.where(fin, v, -np.inf).max(axis=-1, keepdims=True)
    lo = np.where(fin, v, np.inf).min(axis=-1, keepdims=True)
    out = np.where(v == np.inf, hi, np.where(v == -np.inf, lo, v))
    return np.where(np.isinf(out), F32(0.0), out).astype(F32)


def denormalize(preds, loc, scale):
    """clamp_nonfinite(sinh(preds) * scale + loc) in f32; loc / scale broadcast [R, 1, 1]."""
    with np.errstate(over="ignore", invalid="ignore"):
        s = np.sinh(preds.astype(F64)).astype(F32)
        return clamp_nonfinite(s * scale + loc)


# ------------------------------------------------------------------ refine frame
def welford_combine(w_a, mean_a, m2_a, w_b, mean_b, m2_b):
    w = w_a + w_b
    w_safe = np.maximum(w, F64_TINY)
    delta = mean_b - mean_a
    mean = mean_a + delta * w_b / w_safe
    m2 = m2_a + m2_b + np.square(delta) * w_a * w_b / w_safe
    zero = w == 0
    return w, np.where(zero, 0.0, mean), np.where(zero, 0.0, m2)


def cumulative_welford(w, mean, m2):
    n = w.shape[1]
    d = 1
    while d < n:
        def shift(a):
            out = np.zeros_like(a)
            out[:, d:] = a[:, : n - d]
            return out
        w, mean, m2 = welford_combine(shift(w), shift(mean), shift(m2), w, mean, m2)
        d *= 2
    return w, mean, m2


def advancing_block_frames(context, observed, sinh_median):
    """Official advancing_block_frames: per-patch float64 (loc, scale), each [R, H]."""
    r, n_pred, ps = sinh_median.shape
    blank = np.zeros((r, n_pred, ps), F64)
    values = np.concatenate([context.reshape(r, -1, ps).astype(F64), blank], axis=1)
    weights = np.concatenate([observed.reshape(r, -1, ps).astype(F64), blank], axis=1)
    cnt = weights.sum(-1)
    mean = (values * weights).sum(-1) / np.maximum(cnt, 1.0)
    m2 = (weights * np.square(values - mean[..., None])).sum(-1)
    zero = cnt == 0
    mean = np.where(zero, 0.0, mean)
    m2 = np.where(zero, 0.0, m2)
    cnt, mean, m2 = (s[:, -n_pred - 1] for s in cumulative_welford(cnt, mean, m2))
    offset = mean
    clip = F32_MAX ** 0.5 / 1e3
    locs, scales = [], []
    for h in range(n_pred):
        var = np.maximum(m2 / np.maximum(cnt - 1.0, 1.0), 0.0)
        scale = np.maximum(np.sqrt(var), MIN_SCALE)
        locs.append(mean)
        scales.append(scale)
        folded = np.clip(sinh_median[:, h] * scale[:, None] + mean[:, None] - offset[:, None], -clip, clip)
        folded = folded + offset[:, None]
        b_mean = folded.mean(-1)
        b_m2 = np.square(folded - b_mean[:, None]).sum(-1)
        cnt, mean, m2 = welford_combine(cnt, mean, m2, np.full_like(cnt, ps), b_mean, b_m2)
    return np.stack(locs, axis=1), np.stack(scales, axis=1)


def refine_block(quantiles, context, observed):
    """Official _refine_block for target rows: anchor frame -> advancing refine frame."""
    n_pred = quantiles.shape[1] // PATCH
    if n_pred <= 1:
        return quantiles
    loc, scale = patch_loc_scale(context, observed)
    loc0, scale0 = loc[:, -1], scale[:, -1]
    median = quantiles[..., MEDIAN].reshape(-1, n_pred, PATCH).astype(F64)
    sinh_median = (median - loc0[:, None, None].astype(F64)) / scale0[:, None, None].astype(F64)
    loc_j, scale_j = advancing_block_frames(context, observed, sinh_median)
    loc_j = loc_j.astype(F32)
    scale_j = scale_j.astype(F32)
    loc_j[:, 0] = loc0
    scale_j[:, 0] = scale0
    loc_j = np.repeat(loc_j, PATCH, axis=-1)
    scale_j = np.repeat(scale_j, PATCH, axis=-1)
    sinh_norm = (quantiles - loc0[:, None, None]) / scale0[:, None, None]
    return clamp_nonfinite(sinh_norm * scale_j[..., None] + loc_j[..., None])


# ------------------------------------------------------------------ graph inputs
class GraphInputs:
    """Builds patches / dec / bias / cos / sin for a bucket (L, H) from a normalized block."""

    def __init__(self, forecast_query, rope_cos, rope_sin, rope_p_min):
        self.fq = np.asarray(forecast_query, F32).reshape(-1)
        self.cos = np.asarray(rope_cos, F32)
        self.sin = np.asarray(rope_sin, F32)
        self.p_min = int(rope_p_min)

    def build(self, scaled, observed, bucket_l, bucket_h, neg=None):
        neg = NEG if neg is None else neg
        r, t = scaled.shape
        lc = t // PATCH
        assert t % PATCH == 0 and lc <= bucket_l, (t, bucket_l)
        pad = bucket_l - lc
        s = bucket_l + bucket_h
        patches = np.zeros((r, bucket_l, 2 * PATCH), F32)
        patches[:, :pad, PATCH:] = 1.0
        patches[:, pad:, :PATCH] = scaled.reshape(r, lc, PATCH)
        patches[:, pad:, PATCH:] = (~observed).reshape(r, lc, PATCH).astype(F32)
        obs = np.zeros(bucket_l, bool)
        obs[pad:] = observed[0].reshape(lc, PATCH).any(-1)
        assert all((observed[i] == observed[0]).all() for i in range(r)), "rows must share the mask"
        key_valid = obs | np.maximum.accumulate(obs)
        first_obs = int(np.argmax(obs)) if obs.any() else 0
        allow = np.zeros((s, s), bool)
        ll = bucket_l
        allow[:ll, :ll] = (np.tril(np.ones((ll, ll), bool)) & key_valid[None, :]) | np.eye(ll, dtype=bool)
        allow[ll:, :ll] = key_valid[None, :]
        allow[ll:, ll:] = np.tril(np.ones((bucket_h, bucket_h), bool))
        bias = np.where(allow, F32(0.0), F32(neg)).reshape(1, 1, s, s)
        rows = np.arange(s) - first_obs - self.p_min
        assert rows.min() >= 0 and rows.max() < self.cos.shape[0], "RoPE table too small"
        cos = self.cos[rows].reshape(1, 1, s, -1)
        sin = self.sin[rows].reshape(1, 1, s, -1)
        dec = np.broadcast_to(self.fq, (r, bucket_h, self.fq.size)).astype(F32)
        return [patches, dec, bias, cos, sin]


# ------------------------------------------------------------------ forecast
def nonnegative_context(inputs, observed):
    obs = np.isfinite(inputs) & observed
    return bool(obs.any() and ((inputs >= 0) | ~obs).all())


def ffill(inputs, observed):
    if observed.all():
        return inputs
    idx = np.where(observed, np.arange(inputs.shape[-1]), 0)
    idx = np.maximum.accumulate(idx, axis=-1)
    out = np.take_along_axis(inputs, idx, axis=-1)
    return np.nan_to_num(out, nan=0.0).astype(F32)


def pad_to_patch(x, observed):
    t = x.shape[-1]
    if t % PATCH:
        pad = PATCH - t % PATCH
        x = np.concatenate([np.zeros(x.shape[:-1] + (pad,), F32), x], axis=-1)
        observed = np.concatenate([np.zeros(observed.shape[:-1] + (pad,), bool), observed], axis=-1)
    return x, observed


def cap_quantiles(q, inputs, observed):
    with np.errstate(invalid="ignore"):
        ctx_max = np.where(observed, inputs, -np.inf).max(-1, keepdims=True)
        ctx_min = np.where(observed, inputs, np.inf).min(-1, keepdims=True)
    ctx_max = np.where(np.isneginf(ctx_max), 0.0, ctx_max).astype(F32)
    ctx_min = np.where(np.isposinf(ctx_min), 0.0, ctx_min).astype(F32)
    _, scale = patch_loc_scale(inputs, observed)
    anchor = scale[..., -1:] * F32(CAP_K)
    lo = (ctx_min - anchor)[..., None]
    hi = (ctx_max + anchor)[..., None]
    return np.minimum(np.maximum(q, lo), hi).astype(F32)


def block_ar_decode(context, observed, prediction_length, runner):
    """Rows of one pass group share `observed`; returns [R, prediction_length, 21] f32."""
    blocks = []
    remaining = math.ceil(prediction_length / PATCH)
    while remaining > 0:
        h = min(BLOCK, remaining)
        loc, scale = patch_loc_scale(context, observed)
        scaled = normalize(context, loc, scale, observed)
        preds = np.asarray(runner(scaled, observed, h), F32)
        preds = np.clip(preds, -F32_MAX, F32_MAX)
        q = denormalize(preds, loc[:, -1:, None], scale[:, -1:, None])
        q = refine_block(q, context, observed)
        blocks.append(q)
        remaining -= h
        if remaining <= 0:
            break
        med = q[..., MEDIAN]
        feedback = np.where(np.isfinite(med) & (np.abs(med) < EXTREME), med, F32(0.0)).astype(F32)
        context = np.concatenate([context, feedback], axis=-1)
        observed = np.concatenate([observed, np.ones_like(feedback, bool)], axis=-1)
        if context.shape[-1] > MAX_CONTEXT:
            keep = max(PATCH, (MAX_CONTEXT // PATCH) * PATCH)
            context, observed = context[:, -keep:], observed[:, -keep:]
    return np.concatenate(blocks, axis=1)[:, :prediction_length]


def forecast(series, prediction_length, runner, mirror_in_one_call=True, symmetric=True, clip=True):
    """One univariate series (NaN = missing) -> quantiles [prediction_length, 21] f32.

    mirror_in_one_call: run the context and its negation as R = 2 rows of one call per
    block (the runner must accept R = 2); False runs two R = 1 passes.
    """
    x = np.asarray(series, F32).reshape(1, -1)
    observed = np.isfinite(x)
    inputs = np.where(observed, x, F32(0.0)).astype(F32)
    rows = ffill(inputs, observed)
    obs = observed
    if rows.shape[-1] > MAX_CONTEXT:
        rows, obs = rows[:, -MAX_CONTEXT:], obs[:, -MAX_CONTEXT:]
    rows, obs = pad_to_patch(rows, obs)

    def finish(q, ctx):
        """_forecast_raw's tail: the cap uses the rows of that pass (the negated rows for the mirror)."""
        q = cap_quantiles(q, ctx, obs)
        if np.isnan(q).any():
            raise ValueError(f"forecast produced {int(np.isnan(q).sum())} NaN values")
        return q

    if symmetric and mirror_in_one_call:
        both = np.concatenate([rows, -rows], axis=0)
        q2 = block_ar_decode(both, np.concatenate([obs, obs], axis=0), prediction_length, runner)
        q = finish(q2[:1], rows)
        qm = finish(q2[1:], -rows)
    else:
        q = finish(block_ar_decode(rows, obs, prediction_length, runner), rows)
        qm = finish(block_ar_decode(-rows, obs, prediction_length, runner), -rows) if symmetric else None
    if symmetric:
        q = (F32(0.5) * (q + (-qm)[..., ::-1])).astype(F32)
    if clip and nonnegative_context(inputs, observed):
        q = np.maximum(q, F32(0.0))
    return q[0]
