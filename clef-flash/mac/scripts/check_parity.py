#!/usr/bin/env python3
"""Hold one run of a command set to another (scripts/run_set.py outputs), request by request. Exit 1 on any failure.

    venv-demo/bin/python -B scripts/check_parity.py eval/dev_metal.json eval/dev_cpu.json

The second file is the reference (the CPU run, XNNPACK 8 threads, of the same ../hub/ files): the provider's float32
reference does not hold these requests, so the Metal run is held to this one. The line, set before any answer was
seen: the same request (SHA-256 of the canonical JSON), tokens and row graph; on every question whose two most likely
options are more than 0.02 apart in the reference, the same most likely option (a near tie is recorded only); every
option's probability within 0.02 (the unrounded float32 probabilities). Prints one line per command and the totals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

LINE = 0.02


def main() -> int:
    a_path, r_path = Path(sys.argv[1]), Path(sys.argv[2])
    a, ref = json.loads(a_path.read_text()), json.loads(r_path.read_text())
    rows = {c["id"]: c for c in ref["commands"]}
    problems, worst, n_q, n_agree, n_near = [], 0.0, 0, 0, 0
    for c in a["commands"]:
        r = rows.get(c["id"])
        if r is None:
            problems.append(f"{c['id']}: not in {r_path.name}")
            continue
        if c["request_sha256"] != r["request_sha256"]:
            problems.append(f"{c['id']}: another request")
            continue
        if (c["tokens"], c["length"]) != (r["tokens"], r["length"]):
            problems.append(f"{c['id']}: {c['tokens']} tokens on L{c['length']} vs {r['tokens']} on L{r['length']}")
        cw, agree, near = 0.0, 0, 0
        for q, pref in r["probs"].items():
            got = c["probs"][q]
            dp = max(abs(got[k] - pref[k]) for k in pref)
            cw = max(cw, dp)
            n_q += 1
            top, rtop = max(got, key=got.get), max(pref, key=pref.get)
            if r["near_tie"][q]:
                near += 1
            elif top == rtop:
                agree += 1
            else:
                problems.append(f"{c['id']}/{q}: most likely {top}, the reference {rtop}")
            if dp > LINE:
                problems.append(f"{c['id']}/{q}: max |dp| {dp:.2e} > {LINE}")
        worst = max(worst, cw)
        n_agree += agree
        n_near += near
        print(f"{c['id']}: argmax {agree}/{len(r['probs']) - near} (near ties {near}), max |dp| {cw:.2e}, "
              f"{c['ms']:.1f} ms vs {r['ms']:.1f} ms")
    print(f"TOTAL: argmax {n_agree}/{n_q - n_near} questions that are not near ties (near ties {n_near}), "
          f"max |dp| {worst:.2e}")
    print(f"CHECK_PARITY {a_path.name} vs {r_path.name}: " + ("PASS" if not problems else
                                                              "FAIL: " + "; ".join(problems[:8])))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
