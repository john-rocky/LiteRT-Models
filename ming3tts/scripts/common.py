"""Paths and small helpers shared by the Ming-3.0-tts-0.5B conversion scripts (no upstream imports)."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

# K = this module (scripts/ is inside it). ROOT holds the upstream checkpoint and code and the
# outputs: MING3TTS_ROOT, or work/ in this module (a directory or a link to one). Each of the
# three can also be set on its own.
K = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("MING3TTS_ROOT", K / "work"))
HF_DIR = Path(os.environ.get("MING3TTS_HF_DIR", ROOT / "src" / "hf"))  # HF inclusionAI/Ming-3.0-tts-0.5B @8d66b70a
GH_DIR = Path(os.environ.get("MING3TTS_GH_DIR", ROOT / "src" / "gh"))  # GitHub inclusionAI/Ming-3.0-tts @028c1eace193
OUT = Path(os.environ.get("MING3TTS_OUT", ROOT / "out"))
REF = OUT / "ref"
EXPORT = OUT / "export"
GATES = OUT / "gates"
DEVFIX = OUT / "devfix"
VOICES = OUT / "voices"

SAMPLE_RATE = 44100
HOP = 882                 # waveform samples per decoder position
PATCH = 4                 # decoder positions per latent (x4 linear upsample)
SAMPLES_PER_LATENT = HOP * PATCH   # 3528 = 80 ms
MAX_CACHE_LEN = 512       # modeling_bailingmm.MAX_CACHE_LEN
SEED = 1895


def fixtures():
    with open(K / "fixtures" / "fixtures.json") as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=_default)
        f.write("\n")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def corr(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def max_abs(a, b):
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())


def tensor_rule(got, ref):
    """L-LM / L-agg tensor rule: max|d| <= max(1e-2, 2e-3*absmax) and corr >= 0.999."""
    d = max_abs(got, ref)
    amax = float(np.abs(np.asarray(ref, np.float64)).max())
    lim = max(1e-2, 2e-3 * amax)
    c = corr(got, ref)
    return {"max_abs_diff": d, "absmax": amax, "limit": lim, "corr": c,
            "pass": bool(d <= lim and c >= 0.999)}


def env_versions(pkgs):
    import importlib.metadata as md
    out = {}
    for p in pkgs:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def ensure_dirs():
    for d in (OUT, REF, EXPORT, GATES, DEVFIX, VOICES):
        d.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(ROOT / "hf_home"))
