#!/usr/bin/env python3
"""The reference answers of the bundled wishes: the same ../hub/ files on the Mac CPU (XNNPACK, 8 threads), each wish's
request (app/wishes.json: the wish's house and the five questions), through the app's own Model.decide, run twice.

    PYTHONDONTWRITEBYTECODE=1 venv-demo/bin/python -B scripts/cpu_reference.py [--length 512]

The wishes are not records of the model repository's fixtures, so the provider's float32 reference does not hold them;
the line for the app's answers (Metal, float32) is this CPU run instead: check_wishes.py and check_run.py hold them
against it (the most likely option of every question whose two most likely options here are more than 0.02 apart, and
every probability within 0.02). Writes out/cpu_ref_wishes.json: per wish the request, its SHA-256, the tokens, the row
graph, the response, the unrounded probabilities of both calls and the second call's request time (the first call of
the first wish compiles the graph), and the environment. Refuses to overwrite.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import time
from pathlib import Path

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "app"))

from home_demo import Model, canonical_sha256, environment, phys_footprint  # noqa: E402

NEAR_TIE = 0.02


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--hub", default=str(H.parent / "hub"))
    a = ap.parse_args()
    hub = Path(a.hub)
    out = H / "out" / "cpu_ref_wishes.json"
    if out.exists():
        raise SystemExit(f"{out} exists: not overwritten")
    doc = json.loads((H / "app/wishes.json").read_text())
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    model = Model(hub, "cpu", a.length)
    wishes = []
    for w in doc["wishes"]:
        request = {"model": "clef-flash", "state": w["state"], "questions": doc["questions"]}
        t = time.perf_counter()
        first = model.decide(request)
        first_s = time.perf_counter() - t
        second = model.decide(request)
        same = first["probs"] == second["probs"] and first["response"] == second["response"]
        near = {}
        for q, p in second["probs"].items():
            top2 = sorted(p.values(), reverse=True)[:2]
            near[q] = len(top2) == 2 and top2[0] - top2[1] <= NEAR_TIE
        wishes.append({"id": w["id"], "request": request, "request_sha256": canonical_sha256(request),
                       "tokens": second["tokens"], "length": second["length"], "response": second["response"],
                       "probs": second["probs"], "argmax": {q: max(p, key=p.get) for q, p in second["probs"].items()},
                       "near_tie": near, "first_call_probs": first["probs"],
                       "first_call_seconds": round(first_s, 3), "second_call_ms": round(second["ms"], 3),
                       "first_and_second_identical": same, "response_is_its_probs": second["response_is_its_probs"]})
        tops = ", ".join(f"{q} {k} {second['probs'][q][k]:.4f}" for q, k in wishes[-1]["argmax"].items())
        print(f"{w['id']}: {second['tokens']} tokens on L{second['length']}; {tops}; second call {second['ms']:.1f} ms; "
              f"identical to the first: {same}", flush=True)
    footprint = phys_footprint()
    model.close()
    res = {"what": "the bundled wishes on the ../hub/ files, Mac CPU (XNNPACK, 8 threads); the line check_wishes.py and "
                   "check_run.py hold the app's answers to", "started_at": started,
           "accelerator": "cpu (XNNPACK, 8 threads)", "near_tie": NEAR_TIE, "wishes": wishes,
           "phys_footprint": footprint, "env": environment(hub)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(f"{out.name}: {len(wishes)} wishes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
