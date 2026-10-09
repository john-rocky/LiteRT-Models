#!/usr/bin/env python3
"""Scene B's reference answer: the same Hub files on the Mac CPU (XNNPACK, 8 threads), the same request (the question of
fixtures/requests_public.json `card_cats_001` on the scene's Pexels photo), run once.

    PYTHONDONTWRITEBYTECODE=1 venv-demo/bin/python -B scripts/cpu_reference.py [--photo <jpg>]

The provider's float32 reference for a new photo would need the source checkpoint (not on this disk); the line for
scene B is this CPU run instead: check_run.py holds the app's Metal answers against it (|dp| <= 1e-4, the same most
likely option, one tile, route row L512). Writes out/cpu_ref_<photo sha256, first 12>.json: the
request (the photo as its file name and sha256), its sha256, the route, the tiles, the response (unrounded), the
request's wall time on the CPU (compile excluded), the environment. Refuses to overwrite.
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
sys.path[:0] = [str(D / "app"), str(D / "hub" / "host"), str(D / "hub" / "examples")]

from d1_demo import canonical_sha256, environment, phys_footprint  # noqa: E402
import run_example as RE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--photo", default=str(D / "fixtures/cats_10241192.jpg"))
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    hub, photo = D / "hub", Path(a.photo)
    data = photo.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    out = D / "out" / f"cpu_ref_{sha[:12]}.json"
    if out.exists():
        raise SystemExit(f"{out} exists: not overwritten")
    cats = RE.record(hub, "card_cats_001")["request"]
    request = {"state": cats["state"], "questions": cats["questions"], "images": [data]}
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with RE.load_host(hub, "cpu", a.threads, keep=1) as host:
        route = host.route(request)
        tiles = sum(len(p.tiles) for p in host.shared.host.vision.pictures([data]))
        t = time.perf_counter()
        first = host.decide(request)          # compiles the L512 graph, the tower and the projector
        first_s = time.perf_counter() - t
        t = time.perf_counter()
        response = host.decide(request)       # the same request on the compiled graphs
        ms = (time.perf_counter() - t) * 1000
        footprint = phys_footprint()
    same = json.dumps(first, sort_keys=True) == json.dumps(response, sort_keys=True)
    doc = {"what": "scene B's request on the Hub files, Mac CPU (XNNPACK); the line check_run.py holds scene B to",
           "started_at": started, "accelerator": f"cpu (XNNPACK, {a.threads} threads)",
           "request": {"state": request["state"], "questions": request["questions"],
                       "images": [{"file": f"fixtures/{photo.name}", "sha256": sha, "bytes": len(data)}]},
           "request_sha256": canonical_sha256(request), "route": route, "tiles": tiles, "response": response,
           "first_call_seconds_with_compile": round(first_s, 3), "second_call_ms": round(ms, 3),
           "first_and_second_identical": same, "phys_footprint": footprint, "env": environment(hub)}
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    ans = response["answers"]["cats"]
    print(f"{out.name}: route {route}, tiles {tiles}, tokens {response['usage']['input_tokens']}, "
          f"choice {ans['choice']} {ans['confidence']:.6f}, probabilities {ans['probabilities']}, "
          f"second call {ms:.1f} ms, identical to the first: {same}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
