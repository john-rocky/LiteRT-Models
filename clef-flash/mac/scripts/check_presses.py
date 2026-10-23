#!/usr/bin/env python3
"""Hold the app's answers to a set's commands (scripts/press_commands.py) against the CPU run of the same requests
(scripts/run_set.py --accel cpu) and against the set's gold, and sum up the request times. Exit 1 on any failure.

    venv-demo/bin/python -B scripts/check_presses.py out/<tag>_manual.run.json eval/dev_cpu.json

A Decide is a command's when its request has that command's SHA-256 in the CPU run. The lines, set before any answer
of this app was seen:
  1. against the CPU run: the same tokens and row graph; on every question whose two most likely options are more than
     0.02 apart in the CPU run, the same most likely option (a near tie is recorded only); every option's probability
     within 0.02 (the unrounded float32 probabilities the app holds from the same call); the response is the one of
     those probabilities and no graph was compiled inside the call;
  2. on screen: the way of handling and every answer card show the most likely option of the app's own call, its
     probability rounded to 3 places;
  3. the request time (time.perf_counter() around clef.decide): the median, min and max of each command's presses.
The questions matched against the gold and the cases exact are counted from the app's first press of each command.
Every press of the same request gives the same probabilities (recorded: the largest difference between presses). The
too-long command is refused (RequestTooLong) with a red line on screen. "Apply to this Mac" was off for every press.
Writes <tag>.presses_check.json next to the record and prints the tables.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "app"))
sys.dont_write_bytecode = True
import mac_demo as M  # noqa: E402

LINE = 0.02


def main() -> int:
    run, ref_path = Path(sys.argv[1]), Path(sys.argv[2])
    doc = json.loads(run.read_text())
    ref = json.loads(ref_path.read_text())
    by_sha = {c["request_sha256"]: c for c in ref["commands"]}
    problems, groups, refused, other = [], {}, [], 0
    for rec in doc["decides"]:
        if "error" in rec:
            refused.append(rec)
            continue
        c = by_sha.get(rec["request_sha256"])
        if c is None:
            other += 1
            continue
        groups.setdefault(c["id"], []).append(rec)
    out = {"record": str(run), "reference": str(ref_path), "line": {"max_abs_dp": LINE, "near_tie": 0.02},
           "commands": []}
    n_q = n_ok = n_case = n_gold = 0
    for c in ref["commands"]:
        recs = groups.get(c["id"])
        if not recs:
            problems.append(f"{c['id']}: no Decide in the record")
            continue
        first = recs[0]
        item = {"id": c["id"], "presses": len(recs), "tokens": first["tokens"], "length": first["length"]}
        spread = 0.0
        for r in recs[1:]:
            for q, p in r["probs"].items():
                spread = max(spread, max(abs(p[k] - first["probs"][q][k]) for k in p))
        item["max_dp_between_presses"] = spread
        bad = [r["seq"] for r in recs if not r.get("response_is_its_probs") or r.get("compiled_during")]
        if bad:
            problems.append(f"{c['id']}: Decide {bad}: the response is not its probabilities, or a compile ran")
        applied = [r["seq"] for r in recs if (r.get("apply") or {}).get("on")]
        if applied:
            problems.append(f"{c['id']}: Decide {applied}: \"Apply to this Mac\" was on")
        ms = [r["ms"]["request_wall"] for r in recs]
        item["ms"] = {"median": statistics.median(ms), "min": min(ms), "max": max(ms), "all": ms}
        if (first["tokens"], first["length"]) != (c["tokens"], c["length"]):
            problems.append(f"{c['id']}: {first['tokens']} tokens on L{first['length']}, the CPU run "
                            f"{c['tokens']} on L{c['length']}")
        worst, agree, near = 0.0, 0, 0
        for q, pref in c["probs"].items():
            got = first["probs"][q]
            dp = max(abs(got[k] - pref[k]) for k in pref)
            worst = max(worst, dp)
            top = max(got, key=got.get)
            if c["near_tie"][q]:
                near += 1
            elif top == c["picked"][q]:
                agree += 1
            else:
                problems.append(f"{c['id']}/{q}: most likely {top}, the CPU run {c['picked'][q]}")
            if dp > LINE:
                problems.append(f"{c['id']}/{q}: max |dp| {dp:.2e} > {LINE}")
        item.update(argmax_agree=agree, near_ties=near, questions=len(c["probs"]), max_abs_dp=worst)
        # the screen: the payload of the app's own call, as the page shows it
        want = M.answer_payload(first["request"], first["probs"], first["ms"]["request_wall"], first["tokens"],
                                first["length"])
        shown = first["shown"]
        for q in want["questions"]:
            s = shown["questions"].get(q["name"])
            if s is None or (s["answer"], s["answer_p"]) != (q["answer"], q["answer_p"]):
                problems.append(f"{c['id']}/{q['name']}: the card shows {s and (s['answer'], s['answer_p'])}, "
                                f"the call {q['answer']} {q['answer_p']}")
        way = shown["mac"]["way"]
        if want["way"] and (way["word"], way["p"], way["detail"]) != (want["way"]["word"], want["way"]["p"],
                                                                     want["way"]["detail"]):
            problems.append(f"{c['id']}: the way shows {way}, the call {want['way']}")
        item["way"] = way
        if "gold" in c:
            picked = {q: max(p, key=p.get) for q, p in first["probs"].items()}
            match = {q: picked.get(q) == g for q, g in c["gold"].items()}
            item["gold_match"] = sum(match.values())
            item["case_exact"] = all(match.values())
            n_q += len(match)
            n_ok += sum(match.values())
            n_case += item["case_exact"]
            n_gold += 1
        out["commands"].append(item)
    out["refused"] = [{"error": r["error"], "shown": r.get("shown_error")} for r in refused]
    if not any("RequestTooLong" in r["error"] and r.get("shown_error") for r in refused):
        problems.append("no too-long command refused with a red line on screen")
    out["other_decides"] = other
    out["gold"] = {"questions": n_q, "matched": n_ok, "cases_exact": n_case, "commands": n_gold}
    out["problems"] = problems
    out["pass"] = not problems
    dst = run.parent / f"{doc['tag']}.presses_check.json"
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    print(f"# {doc['tag']}: {doc.get('accelerator')}, {doc.get('env', {}).get('chip')}, row graph L{doc.get('length')}"
          f"; held to {ref_path.name}")
    print("| command | tokens | way on screen | argmax = CPU run | max \\|Δp\\| | presses | max Δp between presses | "
          "request time median (min–max) | gold |")
    print("|---|---:|---|---|---:|---:|---:|---|---|")
    for r in out["commands"]:
        agree = f"{r['argmax_agree']}/{r['questions'] - r['near_ties']}" + (
            f" (near tie {r['near_ties']})" if r["near_ties"] else "")
        gold = "—" if "gold_match" not in r else f"{r['gold_match']}/{r['questions']}{' exact' if r['case_exact'] else ''}"
        print(f"| {r['id']} | {r['tokens']} | {r['way']['word']} {r['way']['p']} | {agree} | {r['max_abs_dp']:.2e} | "
              f"{r['presses']} | {r['max_dp_between_presses']:.1e} | {r['ms']['median']:.1f} "
              f"({r['ms']['min']:.1f}–{r['ms']['max']:.1f}) ms | {gold} |")
    if n_gold:
        print(f"\ngold (first press of each command): {n_ok}/{n_q} questions, {n_case}/{n_gold} cases exact")
    for r in out["refused"]:
        print(f"refused: {r['error'][:120]} | on screen: {r['shown']}")
    print(f"CHECK_PRESSES {doc['tag']}: " + ("PASS" if not problems else "FAIL: " + "; ".join(problems[:8])))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
