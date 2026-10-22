#!/usr/bin/env python3
"""The recorded clip's reference answers: the same hub/ files on the Mac CPU (XNNPACK, 8 threads), the same request
(examples.json "take": the coffee example's state and two of its questions, the coffee photo at the 448 px tile),
through the app's own Model.decide, run twice.

    PYTHONDONTWRITEBYTECODE=1 venv-demo/bin/python -B scripts/cpu_reference.py [--length 1024]

The clip's request is not one of the public fixture records, so the provider's float32 reference does not hold it;
the line for the clip is this CPU run instead: check_run.py holds the clip's Metal answers against it (the most likely
option of every question, every probability within 0.02, the conversion's line). Writes out/cpu_ref_take.json: the
request (the photo as its file name and SHA-256), its SHA-256, the tokens, the row graph and the tile, the response,
the unrounded probabilities of both calls, the second call's request time (the first one compiles the graphs), the
environment. Refuses to overwrite.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import time
from pathlib import Path

D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(D / "app"))

from clef_demo import Model, canonical_sha256, environment, phys_footprint  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--hub", default=str(D / "hub"))
    a = ap.parse_args()
    hub = Path(a.hub)
    out = D / "out" / "cpu_ref_take.json"
    if out.exists():
        raise SystemExit(f"{out} exists: not overwritten")
    ex = json.loads((D / "app/examples.json").read_text())
    take = ex["take"]
    src = next(x for x in ex["examples"] if x["id"] == take["from"])
    photo = hub / src["picture"]
    data = photo.read_bytes()
    request = {"model": "clef-flash", "state": src["state"],
               "questions": {k: src["questions"][k] for k in take["questions"]}, "images": [data]}
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    model = Model(hub, "cpu", a.length)
    t = time.perf_counter()
    first = model.decide(request, 448)        # compiles the row graph and the 448 px tower
    first_s = time.perf_counter() - t
    second = model.decide(request, 448)       # the same request on the compiled graphs
    footprint = phys_footprint()
    model.close()
    same = first["probs"] == second["probs"] and first["response"] == second["response"]
    doc = {"what": "the clip's request on the hub/ files, Mac CPU (XNNPACK, 8 threads); the line check_run.py holds "
                   "the clip to", "started_at": started, "accelerator": "cpu (XNNPACK, 8 threads)",
           "request": {"model": request["model"], "state": request["state"], "questions": request["questions"],
                       "images": [{"file": src["picture"], "sha256": hashlib.sha256(data).hexdigest(),
                                   "bytes": len(data)}]},
           "request_sha256": canonical_sha256(request), "tile": second["tile"], "tokens": second["tokens"],
           "length": second["length"], "response": second["response"], "probs": second["probs"],
           "first_call_probs": first["probs"], "first_call_seconds_with_compile": round(first_s, 3),
           "second_call_ms": round(second["ms"], 3), "first_and_second_identical": same,
           "response_is_its_probs": second["response_is_its_probs"], "phys_footprint": footprint,
           "env": environment(hub)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    tops = {q: max(p, key=p.get) for q, p in second["probs"].items()}
    print(f"{out.name}: {second['tokens']} tokens on L{second['length']}, tile {second['tile']}, "
          + ", ".join(f"{q} {k} {second['probs'][q][k]:.6f}" for q, k in tops.items())
          + f"; second call {second['ms']:.1f} ms; identical to the first: {same}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
