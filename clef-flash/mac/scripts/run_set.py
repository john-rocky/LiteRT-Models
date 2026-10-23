#!/usr/bin/env python3
"""Run a set of commands through the app's own Model.decide on the ../hub/ files and hold every answer to the set's
gold.

    venv-demo/bin/python -B scripts/run_set.py eval/dev.json --accel cpu --out eval/dev_cpu.json
    venv-demo/bin/python -B scripts/run_set.py eval/heldout.json --accel gpu --out eval/heldout_metal.json

A set is {"commands": [{"id", "state", "gold"}]}: the state is the full state (its "request" is the command), the gold
is {"resolution", "targets", "changes": {"<device>.<property>": value}} (app/form.py gold_options turns it into one
option per question). Each request is app/form.py request_of(state): the questions built from the state the way the
app builds them. --accel gpu runs on Metal at float32 precision, cpu on XNNPACK with 8 threads; every request runs
on the row graph of --length (default 2048 tokens).

Writes per command: the request and its SHA-256, the tokens, the request time, every question's unrounded
probabilities, the picked option, the gold option, near ties (the two most likely options within 0.02), whether
every question matches (case exact) and the decision the app would act on. Prints one line per command and the
totals: questions matched and cases exact. Refuses to overwrite --out.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import time
from pathlib import Path

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "app"))

import form as F  # noqa: E402
from mac_demo import Model, canonical_sha256, environment, phys_footprint  # noqa: E402

NEAR_TIE = 0.02


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("set")
    ap.add_argument("--accel", choices=["gpu", "cpu"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--hub", default=str(H.parent / "hub"))
    ap.add_argument("--repeat", type=int, default=1,
                    help="decide each request this many times (the first call is scored; the times of all are kept)")
    ap.add_argument("--sort-questions", action="store_true",
                    help="(a dev experiment) send the questions sorted by name, as the benchmark's rows are")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: not overwritten")
    src = Path(a.set)
    doc = json.loads(src.read_text())
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t = time.perf_counter()
    model = Model(Path(a.hub), a.accel, a.length)
    load_s = time.perf_counter() - t
    t = time.perf_counter()
    compiled = model.compile(model.row_file())
    rows, n_q, n_ok, n_case = [], 0, 0, 0
    for c in doc["commands"]:
        state = c["state"]
        request = F.request_of(state)
        if a.sort_questions:
            request = dict(request, questions=dict(sorted(request["questions"].items())))
        gold = F.gold_options(state, c["gold"]) if "gold" in c else None
        r = model.decide(request)
        again = [model.decide(request) for _ in range(a.repeat - 1)]
        picked = {q: max(p, key=p.get) for q, p in r["probs"].items()}
        near = {}
        for q, p in r["probs"].items():
            top2 = sorted(p.values(), reverse=True)[:2]
            near[q] = len(top2) == 2 and top2[0] - top2[1] <= NEAR_TIE
        row = {"id": c["id"], "scenario": c.get("scenario"), "request_text": state.get("request"),
               "request": request, "request_sha256": canonical_sha256(request), "tokens": r["tokens"],
               "length": r["length"], "ms": round(r["ms"], 3), "probs": r["probs"], "picked": picked,
               "near_tie": near, "response_is_its_probs": r["response_is_its_probs"],
               "decision": F.decision(state, picked)}
        line = f"{c['id']}: {r['tokens']} tokens on L{r['length']}, {r['ms']:.1f} ms; {picked['resolution']}"
        if again:
            ms_all = [r["ms"]] + [x["ms"] for x in again]
            row["ms_all"] = [round(m, 3) for m in ms_all]
            row["ms_median"] = round(sorted(ms_all)[len(ms_all) // 2], 3)
            row["repeats_identical"] = all(x["probs"] == r["probs"] for x in again)
            line += f" (median of {len(ms_all)}: {row['ms_median']:.1f} ms, identical {row['repeats_identical']})"
        if gold is not None:
            match = {q: picked.get(q) == g for q, g in gold.items()}
            row["gold"], row["match"] = gold, match
            row["case_exact"] = all(match.values())
            n_q += len(match)
            n_ok += sum(match.values())
            n_case += row["case_exact"]
            misses = [f"{q} {picked.get(q)} ({r['probs'][q][picked[q]]:.3f}) gold {g} ({r['probs'][q].get(g, 0):.3f})"
                      for q, g in gold.items() if not match[q]]
            line += (f" gold {gold['resolution']}; {sum(match.values())}/{len(match)}"
                     f"{' EXACT' if row['case_exact'] else ''}" + ("; " + "; ".join(misses) if misses else ""))
        print(line, flush=True)
        rows.append(row)
    total = {"commands": len(rows)}
    if n_q:
        total.update(questions=n_q, questions_matched=n_ok, per_question=round(n_ok / n_q, 4), cases_exact=n_case)
        print(f"TOTAL {a.set}: {n_ok}/{n_q} questions ({100 * n_ok / n_q:.1f} %), {n_case}/{len(rows)} cases exact",
              flush=True)
    footprint = phys_footprint()
    model.close()
    res = {"what": "the set's requests through the app's Model.decide, held to the set's gold", "set": str(src),
           "set_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
           "form_py_sha256": hashlib.sha256((H / "app/form.py").read_bytes()).hexdigest(), "started_at": started,
           "accelerator": "gpu fp32 (GpuOptions enforce_f32)" if a.accel == "gpu" else "cpu (XNNPACK, 8 threads)",
           "length": a.length, "repeat": a.repeat, "sort_questions": a.sort_questions,
           "load_host_seconds": round(load_s, 3),
           "compile": compiled, "near_tie": NEAR_TIE,
           "total": total, "commands": rows, "phys_footprint": footprint, "env": environment(Path(a.hub))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
