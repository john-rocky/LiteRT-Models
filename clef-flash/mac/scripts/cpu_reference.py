#!/usr/bin/env python3
"""The reference answers of a recorded take: the request of each scene of an autoplay run, on the same ../hub/ files on
the Mac CPU (XNNPACK, 8 threads), through the app's own Model.decide, each request run twice.

    PYTHONDONTWRITEBYTECODE=1 venv-demo/bin/python -B scripts/cpu_reference.py --from-run out/t1.run.json

Writes out/cpu_ref_<tag>.json in the form of scripts/run_set.py (one entry per scene: the request, its SHA-256, the
tokens, the row graph, the unrounded probabilities, the most likely option and the near ties of the second call, and
whether the two calls were identical); check_run.py holds the take's answers to it. The requests are not records of
the model repository's fixtures, so the provider's float32 reference does not hold them; this CPU run is the line for
the app's answers (Metal, float32). A command set (eval/*.json) is run the same way with scripts/run_set.py. Refuses to
overwrite.
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

from mac_demo import DEFAULT_LENGTH, Model, canonical_sha256, environment, phys_footprint  # noqa: E402

NEAR_TIE = 0.02


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from-run", required=True, help="an autoplay run JSON: its scenes' requests")
    ap.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    ap.add_argument("--hub", default=str(H.parent / "hub"))
    a = ap.parse_args()
    hub = Path(a.hub)
    run = json.loads(Path(a.from_run).read_text())
    items = [{"id": f"{run['tag']}/{s}", "request": rec["request"]} for s, rec in run["scenes"].items()]
    out = H / "out" / f"cpu_ref_{run['tag']}.json"
    if out.exists():
        raise SystemExit(f"{out} exists: not overwritten")
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    model = Model(hub, "cpu", a.length)
    rows = []
    for it in items:
        request = it["request"]
        t = time.perf_counter()
        first = model.decide(request)
        first_s = time.perf_counter() - t
        second = model.decide(request)
        same = first["probs"] == second["probs"] and first["response"] == second["response"]
        near = {}
        for q, p in second["probs"].items():
            top2 = sorted(p.values(), reverse=True)[:2]
            near[q] = len(top2) == 2 and top2[0] - top2[1] <= NEAR_TIE
        rows.append({"id": it["id"], "request": request, "request_sha256": canonical_sha256(request),
                     "tokens": second["tokens"], "length": second["length"], "ms": round(second["ms"], 3),
                     "probs": second["probs"], "picked": {q: max(p, key=p.get) for q, p in second["probs"].items()},
                     "near_tie": near, "first_call_probs": first["probs"], "first_call_seconds": round(first_s, 3),
                     "first_and_second_identical": same, "response_is_its_probs": second["response_is_its_probs"]})
        print(f"{it['id']}: {second['tokens']} tokens on L{second['length']}; resolution "
              f"{rows[-1]['picked'].get('resolution')}; second call {second['ms']:.1f} ms; identical to the first: "
              f"{same}", flush=True)
    footprint = phys_footprint()
    model.close()
    res = {"what": "a take's requests on the ../hub/ files, Mac CPU (XNNPACK, 8 threads); the line check_run.py holds "
                   "the take's answers to", "started_at": started, "from_run": a.from_run,
           "accelerator": "cpu (XNNPACK, 8 threads)", "near_tie": NEAR_TIE, "commands": rows,
           "phys_footprint": footprint, "env": environment(hub)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(f"{out.name}: {len(rows)} requests", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
