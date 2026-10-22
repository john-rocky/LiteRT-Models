"""Shared helpers for the Timestral LiteRT scripts: checkpoint path, official block calls,
synthetic series, and a fixture reader. Torch is imported lazily (the host does not need it)."""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")
REVISION = "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"
WEIGHTS_SHA256 = "1456effb9927e3daa8c7234f10497e3424233869245a9c72b8ecebaf7ad2b7fa"
def _fixtures_dir():
    """The parity fixture's directory (not part of this repository): $TIMESTRAL_FIXTURES, else the
    "fixtures" entry of scripts/local_paths.json (a local, untracked file), else ROOT/fixtures."""
    if os.environ.get("TIMESTRAL_FIXTURES"):
        return os.environ["TIMESTRAL_FIXTURES"]
    local = os.path.join(HERE, "local_paths.json")
    if os.path.exists(local):
        with open(local) as f:
            path = json.load(f).get("fixtures")
        if path:
            return os.path.expanduser(path)
    return os.path.join(ROOT, "fixtures")


FIXTURES = _fixtures_dir()


def checkpoint_dir():
    """Local snapshot of mistralai/Timestral-209M at REVISION (downloaded beforehand)."""
    from huggingface_hub import snapshot_download
    return snapshot_download("mistralai/Timestral-209M", revision=REVISION, local_files_only=True,
                             allow_patterns=["config.json", "model.safetensors"])


def load_official(dtype=None):
    import torch
    from timestral import load_model
    torch.set_grad_enabled(False)
    return load_model(checkpoint_dir(), device="cpu", dtype=dtype or torch.float32)


def official_scale(model, context, observed):
    """The official model.scale on [R, T] f32 numpy rows: (scaled, loc, scale) numpy f32."""
    import torch
    s, loc, scale = model.scale(torch.from_numpy(context), torch.from_numpy(observed))
    return s.numpy(), loc.numpy(), scale.numpy()


def official_block_preds(model, scaled, observed, h, dtype=None):
    """Official _predict_context_only from the normalized rows on: preds [R, h*32, 21] (each row its own series)."""
    import torch
    from timestral.masks import (build_forecast_positions, build_forecast_time_mask, build_two_stream_variate_mask,
                                 compute_enc_key_validity, group_first_observed_patch)
    s = torch.from_numpy(np.ascontiguousarray(scaled))
    if dtype is not None:
        s = s.to(dtype)
    o = torch.from_numpy(np.ascontiguousarray(observed))
    r, t = s.shape
    ll = t // 32
    x_enc = model.representation(s, o)
    x_dec = model.representation.forecast_tokens(r, h)
    obs_per_patch = o.unflatten(-1, (ll, 32)).any(dim=-1)
    key_valid = compute_enc_key_validity(obs_per_patch, model.config.attend_missing_patches)
    time_mask = build_forecast_time_mask(ll, h, key_valid, s.device)
    group_ids = torch.arange(r)
    variate_mask = build_two_stream_variate_mask(group_ids, key_valid, h)
    unique_ids, inverse = torch.unique(group_ids, return_inverse=True)
    first_obs = group_first_observed_patch(obs_per_patch, inverse, unique_ids.numel())[inverse]
    positions = build_forecast_positions(ll, h, first_obs)
    preds = model._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions)
    return preds.detach().cpu().numpy()


def assemble_rows(series):
    """forecast()'s _as_context + _assemble_rows for one univariate series: (rows, observed) [1, T]."""
    import host_forecast as hf
    x = np.asarray(series, np.float32).reshape(1, -1)
    observed = np.isfinite(x)
    inputs = np.where(observed, x, np.float32(0.0)).astype(np.float32)
    rows = hf.ffill(inputs, observed)
    obs = observed
    if rows.shape[-1] > hf.MAX_CONTEXT:
        rows, obs = rows[:, -hf.MAX_CONTEXT:], obs[:, -hf.MAX_CONTEXT:]
    return hf.pad_to_patch(rows, obs)


def synthetic_series():
    """Deterministic test series covering the input classes the fixture must have."""
    rng = np.random.default_rng(20261022)
    out = {}
    out["randwalk_1000"] = rng.standard_normal(1000).cumsum()
    t = np.arange(2048)
    out["daily_sine_2048_nonneg"] = 10 + 5 * np.sin(2 * np.pi * t / 24) + rng.standard_normal(2048)
    s = rng.standard_normal(333).cumsum() * 3 - 50
    s[:50] = np.nan
    s[200:240] = np.nan
    s[300] = np.nan
    out["neg_gaps_333"] = s
    out["const_256"] = np.full(256, 5.0)
    out["trend_4000"] = 0.05 * np.arange(4000) - 80 + rng.standard_normal(4000) * 4
    out["short_40"] = rng.standard_normal(40)
    big = np.abs(rng.standard_normal(3000)) * 1e5
    big[:700] = np.nan
    out["spiky_3000_leading_nan"] = big
    return out


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
