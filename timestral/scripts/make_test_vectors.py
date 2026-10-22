"""Writes the app's JVM test vectors and example series from the Python host reference.

Every vector is a synthetic series made here (no outside data) run through
`host_forecast.forecast()` (this directory; NumPy only) with the sample's limit of 8,192 context
steps. The graph is replaced by a float32 formula that the Kotlin test recomputes bit for bit:

    out[r, t, i] = base[r] + slope[r] * t + z[i] * (width + growth * t)

(t = decoder step 0..1023, z = the standard normal quantiles of the 21 levels), so the vectors test
the host alone: the five graph inputs (sha256 of their float32 bytes), the anchor statistics and the
21 quantiles after denormalize / refine / cap / symmetric averaging / clip. The first vectors run on
the smallest two-row graph of B1 / B2 that holds them; the others name a graph form (L, H, R), among
them one-row graphs (forecast(..., mirror_in_one_call=False): one call per row) and graphs with 8
decoder patches.

Writes, under the module:
  app/src/test/resources/vectors/<name>/case.properties   parameters and expected values
  app/src/test/resources/vectors/<name>/series.f32        the series, float32 little-endian, NaN = missing
  app/src/test/resources/vectors/<name>/quantiles.f32     the expected [horizon, 21] float32 forecast
  app/src/test/resources/vectors/index.txt                the vector names
  app/src/main/assets/examples/<name>.csv                 the app's example series

Usage (the Python of the conversion environment; the host tables are the model files'
forecast_query.f32, rope_cos.f32 and rope_sin.f32):
  python scripts/make_test_vectors.py --host-tables <dir with the model files>
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import host_forecast as hf  # noqa: E402

MODULE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORS = os.path.join(MODULE, "app", "src", "test", "resources", "vectors")
EXAMPLES = os.path.join(MODULE, "app", "src", "main", "assets", "examples")
FORMS = os.path.join(MODULE, "app", "src", "main", "assets", "timestral_forms.json")
MAX_CONTEXT = 8192
BUCKETS = (("B1", 64, 32), ("B2", 256, 32))
ROPE_FIRST = -1023
LEVELS = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
          0.8, 0.85, 0.9, 0.95, 0.99)
NAMES = ("patches", "dec", "bias", "cos", "sin")
F32 = np.float32


def f32_hex(value):
    return "0x" + struct.pack(">f", float(value)).hex()


def sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array, "<f4").tobytes()).hexdigest()


def load_tables(directory):
    fq = np.fromfile(os.path.join(directory, "forecast_query.f32"), "<f4")
    cos = np.fromfile(os.path.join(directory, "rope_cos.f32"), "<f4").reshape(-1, 64)
    sin = np.fromfile(os.path.join(directory, "rope_sin.f32"), "<f4").reshape(-1, 64)
    return hf.GraphInputs(fq, cos, sin, ROPE_FIRST)


def z_values():
    dist = statistics.NormalDist()
    return np.array([dist.inv_cdf(level) for level in LEVELS], F32)


class FormulaRunner:
    """A host_forecast runner: records the graph inputs it would send and returns the formula.

    form = None: a two-row graph, the smallest of BUCKETS that holds the block. form = (name, L, H, R):
    that graph; with R = 1 the forecaster calls the runner twice, the context first, its negation second.
    """

    def __init__(self, tables, mask, params, form=None):
        self.tables = tables
        self.mask = mask
        self.params = params
        self.form = form
        self.calls = []

    def __call__(self, scaled, observed, h):
        r, t = scaled.shape
        lc = t // hf.PATCH
        if self.form is None:
            name, ll, hh = next(b for b in BUCKETS if lc <= b[1] and h <= b[2])
        else:
            name, ll, hh, rows = self.form
            assert r == rows and lc <= ll and h <= hh, (r, lc, h, self.form)
        inputs = self.tables.build(scaled, observed, ll, hh, neg=self.mask)
        p = self.params
        row_ids = list(range(r)) if r == 2 else [len(self.calls)]
        base = scaled[:, -1].astype(F32)
        slope = np.asarray(p["slope"], F32)[row_ids]
        steps = np.arange(hh * hf.PATCH, dtype=F32)
        z = z_values()
        out = (base[:, None, None] + slope[:, None, None] * steps[None, :, None]
               + z[None, None, :] * (F32(p["width"]) + F32(p["growth"]) * steps)[None, :, None])
        out = out.astype(F32)
        for k, row in enumerate(row_ids):
            if p.get("nan") and row == 0:
                out[k, 0, hf.MEDIAN] = np.nan
            if p.get("inf"):
                if row == 0:
                    out[k, 1, hf.N_Q - 1] = np.inf
                else:
                    out[k, 1, 0] = -np.inf
        self.calls.append({"bucket": name, "L": ll, "H": hh, "h": h, "inputs": inputs, "rows": row_ids,
                           "base": base, "first_obs": first_observed(observed, ll)})
        return out[:, : h * hf.PATCH]


def first_observed(observed, bucket_l):
    lc = observed.shape[-1] // hf.PATCH
    obs = np.zeros(bucket_l, bool)
    obs[bucket_l - lc:] = observed[0].reshape(lc, hf.PATCH).any(-1)
    return int(np.argmax(obs)) if obs.any() else 0


def anchors(series):
    """The anchor (last step) loc / scale of both rows, as forecast() computes them."""
    x = np.asarray(series, F32).reshape(1, -1)
    observed = np.isfinite(x)
    inputs = np.where(observed, x, F32(0.0)).astype(F32)
    rows = hf.ffill(inputs, observed)
    obs = observed
    if rows.shape[-1] > MAX_CONTEXT:
        rows, obs = rows[:, -MAX_CONTEXT:], obs[:, -MAX_CONTEXT:]
    rows, obs = hf.pad_to_patch(rows, obs)
    both = np.concatenate([rows, -rows], 0)
    loc, scale = hf.patch_loc_scale(both, np.concatenate([obs, obs], 0))
    return loc[:, -1], scale[:, -1], rows.shape[-1] // hf.PATCH


def vector_cases(rng):
    """(name, series, horizon, mask, formula parameters): synthetic series, made up here."""
    cases = []
    t = np.arange(1000)
    s = 20 + 6 * np.sin(2 * np.pi * t / 24) + 0.01 * t + rng.normal(0, 0.8, t.size)
    cases.append(("daily_sine_b1", s, 64, hf.NEG, {}))
    s = np.cumsum(rng.normal(-0.05, 1.0, 700))
    cases.append(("random_walk_h1", s, 20, hf.NEG, {}))
    day = np.arange(800)
    rate = 4 + 3 * (day % 7 >= 5) + 2 * np.sin(2 * np.pi * day / 30)
    s = rng.poisson(np.clip(rate, 0.2, None)).astype(float)
    s[rng.random(day.size) < 0.1] = 0.0
    cases.append(("weekly_counts_clip", s, 48, hf.NEG_ATTN_K, {}))
    s = 100 + np.cumsum(rng.normal(0, 0.5, 700))
    s[:100] = np.nan
    s[300:340] = np.nan
    s[500:503] = np.nan
    s[650] = np.inf
    cases.append(("gaps_leading_missing", s, 40, hf.NEG, {}))
    t = np.arange(2200)
    s = 0.02 * t + 5 * np.sin(2 * np.pi * t / 168) + 2 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1, t.size)
    cases.append(("trend_season_b2", s, 100, hf.NEG, {}))
    t = np.arange(9000)
    s = 50 + 10 * np.sin(2 * np.pi * t / 96) + np.cumsum(rng.normal(0, 0.05, t.size))
    s[:40] = np.nan
    cases.append(("long_over_8192", s, 40, hf.NEG, {}))
    s = 3 + np.sin(np.arange(300) / 5.0) + rng.normal(0, 0.2, 300)
    cases.append(("full_block_1024", s, 1024, hf.NEG, {}))
    cases.append(("constant_level", np.full(200, 5.0), 32, hf.NEG, {}))
    s = rng.normal(0, 1, 500).cumsum()
    cases.append(("saturated_output", s, 40, hf.NEG, {"width": 40.0, "inf": True}))
    s = 10 + rng.normal(0, 1, 400)
    cases.append(("nan_output", s, 80, hf.NEG, {"nan": True}))
    cases.append(("short_20", 2 + rng.normal(0, 1, 20), 32, hf.NEG, {}))
    s = 1.0e6 + np.cumsum(rng.normal(0, 3.0, 300))
    cases.append(("level_1e6", s, 40, hf.NEG, {}))
    s = np.concatenate([np.zeros(64), np.abs(rng.normal(2, 1, 336))])
    cases.append(("zeros_then_values", s, 40, hf.NEG_ATTN_K, {}))
    return cases


def form_cases(rng):
    """Vectors on a named graph form (name, L, H, R): one-row graphs (two calls per forecast) and
    graphs with 8 decoder patches (horizons up to 256 steps)."""
    cases = []
    t = np.arange(900)
    s = 30 + 8 * np.sin(2 * np.pi * t / 24) + 0.02 * t + rng.normal(0, 1.0, t.size)
    cases.append(("r1_daily_sine", s, 64, hf.NEG, {}, ("B1", 64, 32, 1)))
    s = np.cumsum(rng.normal(0.03, 1.0, 1300))
    cases.append(("h8_random_walk", s, 100, hf.NEG, {}, ("B1h8", 64, 8, 2)))
    day = np.arange(1000)
    s = rng.poisson(3 + 2 * (day % 7 >= 5)).astype(float)
    cases.append(("r1_h8_counts_clip", s, 200, hf.NEG_ATTN_K, {}, ("B1h8", 64, 8, 1)))
    t = np.arange(3000)
    s = 0.01 * t + 4 * np.sin(2 * np.pi * t / 96) + rng.normal(0, 0.5, t.size)
    cases.append(("h8_trend_b2", s, 256, hf.NEG_ATTN_K, {}, ("B2h8", 256, 8, 2)))
    s = 50 + np.cumsum(rng.normal(0, 0.3, 4000))
    s[1000:1100] = np.nan
    s[3900:3905] = np.nan
    cases.append(("r1_b2_gaps", s, 48, hf.NEG, {}, ("B2", 256, 32, 1)))
    s = 5 + rng.normal(0, 1, 350)
    cases.append(("r1_nan_output", s, 70, hf.NEG, {"nan": True}, ("B1h8", 64, 8, 1)))
    return cases


def write_vector(tables, name, series, horizon, mask, extra, form=None):
    series = np.asarray(series, F32)
    params = {"slope": [0.002, -0.0017], "width": 0.35, "growth": 0.0004}
    params.update(extra)
    runner = FormulaRunner(tables, mask, params, form)
    rows = 2 if form is None else form[3]
    error = ""
    q = None
    hf.MAX_CONTEXT = MAX_CONTEXT
    try:
        q = hf.forecast(series, horizon, runner, mirror_in_one_call=rows == 2)
    except ValueError as failure:
        error = str(failure)
    call = runner.calls[0]
    loc, scale, lc = anchors(series)
    d = os.path.join(VECTORS, name)
    os.makedirs(d, exist_ok=True)
    series.astype("<f4").tofile(os.path.join(d, "series.f32"))
    if rows == 2:
        base = list(call["base"])
    else:
        # A one-row graph: the negation's call is missing when the context's forecast failed first.
        base = [call["base"][0], runner.calls[1]["base"][0] if len(runner.calls) > 1 else F32(0.0)]
    props = {
        "name": name,
        "horizon": horizon,
        "mask": repr(float(mask)),
        "bucket": call["bucket"],
        "form_l": call["L"],
        "form_h": call["H"],
        "form_rows": rows,
        "context_patches": lc,
        "horizon_patches": call["h"],
        "first_observed_patch": call["first_obs"],
        "anchor_loc": ",".join(f32_hex(v) for v in loc),
        "anchor_scale": ",".join(f32_hex(v) for v in scale),
        "raw_base": ",".join(f32_hex(v) for v in base),
        "raw_slope": ",".join(f32_hex(v) for v in np.asarray(params["slope"], F32)),
        "raw_width": f32_hex(params["width"]),
        "raw_growth": f32_hex(params["growth"]),
        "raw_z": ",".join(f32_hex(v) for v in z_values()),
        "raw_nan": "true" if params.get("nan") else "false",
        "raw_inf": "true" if params.get("inf") else "false",
        "error": error,
    }
    for key, array in zip(NAMES, call["inputs"]):
        props[f"sha256_{key}"] = sha256(array)
    if rows == 1:
        # Call 0 holds the context's row, call 1 the negation's; the other four inputs are the same.
        props["sha256_patches_row1"] = ""
        if len(runner.calls) > 1:
            second = runner.calls[1]["inputs"]
            props["sha256_patches_row1"] = sha256(second[0])
            for key, array in list(zip(NAMES, second))[1:]:
                assert sha256(array) == props[f"sha256_{key}"], (name, key)
    if q is not None:
        q.astype("<f4").tofile(os.path.join(d, "quantiles.f32"))
    with open(os.path.join(d, "case.properties"), "w") as f:
        f.write(f"# {name}: written by scripts/make_test_vectors.py from host_forecast.py\n")
        for key in sorted(props):
            f.write(f"{key}={props[key]}\n")
    return {"name": name, "steps": int(series.size), "horizon": horizon, "bucket": call["bucket"],
            "rows": rows, "h": call["h"], "error": error or None}


def write_examples(rng):
    """The app's three example series, made up here; CSV with a header."""
    os.makedirs(EXAMPLES, exist_ok=True)
    hours = np.arange(2400)
    temp = 14 + 6 * np.sin(2 * np.pi * (hours - 9) / 24) + 0.004 * hours + rng.normal(0, 0.7, hours.size)
    with open(os.path.join(EXAMPLES, "hourly_temperature.csv"), "w") as f:
        f.write("hour,temperature_c\n")
        for h, v in zip(hours, temp):
            f.write(f"{h},{v:.2f}\n")
    days = np.arange(365)
    rate = 40 + 15 * (days % 7 >= 5) + 0.05 * days + 8 * np.sin(2 * np.pi * days / 91)
    orders = rng.poisson(rate)
    with open(os.path.join(EXAMPLES, "daily_orders.csv"), "w") as f:
        f.write("day,orders\n")
        for d, v in zip(days, orders):
            f.write(f"{d},{int(v)}\n")
    minutes = np.arange(600)
    reading = 2.5 + np.cumsum(rng.normal(0, 0.03, minutes.size)) + 0.4 * np.sin(2 * np.pi * minutes / 60)
    missing = np.zeros(minutes.size, bool)
    missing[200:230] = True
    missing[rng.random(minutes.size) < 0.03] = True
    with open(os.path.join(EXAMPLES, "sensor_with_gaps.csv"), "w") as f:
        f.write("minute,reading\n")
        for m, v, gap in zip(minutes, reading, missing):
            f.write(f"{m},\n" if gap else f"{m},{v:.3f}\n")


def check_forms():
    """Each graph of the app's configuration against its file name,
    `timestral-209m-c{context steps}-h{horizon steps}[-int8].tflite`: the context and horizon
    patches, int8 files on the CPU only, two rows, and the mask value of the float16-safe form
    (every published file is that form)."""
    with open(FORMS) as f:
        graphs = json.load(f)["graphs"]
    for graph in graphs:
        path = graph["file"]
        m = re.fullmatch(r"timestral-209m-c(\d+)-h(\d+)(-int8)?\.tflite", path)
        assert m and (int(m.group(1)), int(m.group(2))) == (
            32 * graph["context_patches"], 32 * graph["horizon_patches"]), (graph["name"], path)
        assert not m.group(3) or graph["backends"] == ["cpu"], (graph["name"], path)
        assert graph["rows"] == 2, (graph["name"], path)
        assert float(graph["mask"]) == hf.NEG_ATTN_K, (graph["name"], path, graph["mask"])
    return len(graphs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-tables", required=True)
    args = ap.parse_args()
    tables = load_tables(args.host_tables)
    rng = np.random.default_rng(20261022)
    if os.path.isdir(VECTORS):
        shutil.rmtree(VECTORS)
    rows = [write_vector(tables, *case) for case in vector_cases(rng)]
    # The graph-form vectors draw from their own stream, so the vectors above stay byte-identical.
    rows += [write_vector(tables, *case) for case in form_cases(np.random.default_rng(20261023))]
    with open(os.path.join(VECTORS, "index.txt"), "w") as f:
        f.write("".join(row["name"] + "\n" for row in rows))
    write_examples(np.random.default_rng(7))
    n_forms = check_forms() if os.path.exists(FORMS) else 0
    print(json.dumps({"vectors": rows, "forms_checked": n_forms}, indent=1))


if __name__ == "__main__":
    main()
