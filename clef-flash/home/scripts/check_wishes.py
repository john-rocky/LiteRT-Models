#!/usr/bin/env python3
"""Hold the app's answers to the bundled wishes against the CPU run of the same requests and against the settings each
device was expected to get, and sum up the request times. Exit 1 on any failure.

    venv-demo/bin/python -B scripts/check_wishes.py out/<tag>_manual.run.json

Reads the app's record (app/home_demo.py --tag, pressed by scripts/press_wishes.py) and out/cpu_ref_wishes.json
(scripts/cpu_reference.py). A Decide is a wish's when its request is that wish's request (app/wishes.json: the state
and the questions, key order included); "take" is the recorded scene's way to the same request (the wish typed).
The lines, set before any answer of this app was seen:
  1. against the CPU run: the same tokens and row graph; on every question whose two most likely options are more than
     0.02 apart in the CPU run, the same most likely option (a near tie is recorded only); every option's probability
     within 0.02 (the unrounded float32 probabilities the app holds from the same call); the response is the one of
     those probabilities and no graph was compiled inside the call;
  2. against the expected settings (wishes.json "expected"): the most likely option of the app's first press of each
     wish equals the expected one on at least 80 % of the wishes x devices; every miss is listed;
  3. the request time (time.perf_counter() around clef.decide): the median, min and max of each request's presses.
Every press of the same request gives the same probabilities (recorded: the largest difference between presses). The
too-long wish is refused (RequestTooLong) with a red line on screen. Writes <tag>.wishes_check.json next to the record
and prints the tables.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

H = Path(__file__).resolve().parents[1]
LINE, GOLD_SHARE = 0.02, 0.80


def ordered(x) -> str:
    return json.dumps(x, ensure_ascii=False)


def main() -> int:
    run = Path(sys.argv[1])
    doc = json.loads(run.read_text())
    wishes = json.loads((H / "app/wishes.json").read_text())
    ref_path = H / "out" / "cpu_ref_wishes.json"
    ref = {w["id"]: w for w in json.loads(ref_path.read_text())["wishes"]} if ref_path.is_file() else {}
    problems = []
    if not ref:
        problems.append(f"no CPU reference ({ref_path.name}: scripts/cpu_reference.py)")
    want = {w["id"]: {"model": "clef-flash", "state": w["state"], "questions": wishes["questions"]}
            for w in wishes["wishes"]}
    groups: dict = {}
    refused = []
    for rec in doc["decides"]:
        if "error" in rec:
            refused.append(rec)
            continue
        name = next((wid for wid, req in want.items() if ordered(rec["request"]) == ordered(req)), None)
        groups.setdefault(name, []).append(rec)
    # the take wish's request is pressed twice over (press_wishes.py: its button, then the wish typed, the same number
    # of presses each): the second half is the scene's way, "take"
    g = groups.get(wishes["take"], [])
    if len(g) >= 2:
        groups["take"], groups[wishes["take"]] = g[len(g) // 2:], g[:len(g) // 2]

    out = {"record": str(run), "line": {"max_abs_dp": LINE, "near_tie": 0.02, "expected_share": GOLD_SHARE},
           "requests": [], "expected": []}
    hits = total = 0
    for name in [w["id"] for w in wishes["wishes"]] + ["take", None]:
        recs = groups.get(name)
        if not recs:
            if name is not None:
                problems.append(f"{name}: no Decide in the record")
            continue
        if name is None:
            out["requests"].append({"name": None, "presses": len(recs), "note": "not a bundled wish"})
            continue
        wid = wishes["take"] if name == "take" else name
        first = recs[0]
        item = {"name": name, "wish": wid, "presses": len(recs), "tokens": first["tokens"], "length": first["length"]}
        spread = 0.0
        for r in recs[1:]:
            for q, p in r["probs"].items():
                spread = max(spread, max(abs(p[k] - first["probs"][q][k]) for k in p))
        item["max_dp_between_presses"] = spread
        bad = [r["seq"] for r in recs if not r.get("response_is_its_probs") or r.get("compiled_during")]
        if bad:
            problems.append(f"{name}: Decide {bad}: the response is not its probabilities, or a compile ran")
        ms = [r["ms"]["request_wall"] for r in recs]
        item["ms"] = {"median": statistics.median(ms), "min": min(ms), "max": max(ms), "all": ms}
        rr = ref.get(wid)
        if rr is not None:
            if (first["tokens"], first["length"]) != (rr["tokens"], rr["length"]):
                problems.append(f"{name}: {first['tokens']} tokens on L{first['length']}, the CPU run "
                                f"{rr['tokens']} on L{rr['length']}")
            worst, qs = 0.0, []
            for q, pref in rr["probs"].items():
                got = first["probs"][q]
                if sorted(got) != sorted(pref):
                    problems.append(f"{name}/{q}: options {sorted(got)} vs {sorted(pref)}")
                    continue
                dp = max(abs(got[k] - pref[k]) for k in pref)
                top = max(got, key=got.get)
                qs.append({"question": q, "argmax": top, "cpu_argmax": rr["argmax"][q], "near_tie": rr["near_tie"][q],
                           "max_abs_dp": dp, "p_top": got[top]})
                worst = max(worst, dp)
                if not rr["near_tie"][q] and top != rr["argmax"][q]:
                    problems.append(f"{name}/{q}: most likely {top}, the CPU run {rr['argmax'][q]}")
                if dp > LINE:
                    problems.append(f"{name}/{q}: max |dp| {dp:.2e} > {LINE}")
            item["questions"], item["max_abs_dp"] = qs, worst
        out["requests"].append(item)
        if name != "take":
            exp = next(w for w in wishes["wishes"] if w["id"] == wid)["expected"]
            for q, e in exp.items():
                got = max(first["probs"][q], key=first["probs"][q].get)
                total += 1
                hits += got == e
                out["expected"].append({"wish": wid, "device": q, "expected": e, "answer": got,
                                        "p": first["probs"][q][got], "match": got == e})
    out["expected_hits"], out["expected_total"] = hits, total
    share = hits / total if total else 0.0
    if share < GOLD_SHARE:
        problems.append(f"expected settings {hits}/{total} = {share:.1%} < {GOLD_SHARE:.0%}")
    out["refused"] = [{"error": r["error"], "shown": r.get("shown_error")} for r in refused]
    if not any("RequestTooLong" in r["error"] and r.get("shown_error") for r in refused):
        problems.append("no too-long wish refused with a red line on screen")
    out["problems"] = problems
    out["pass"] = not problems
    dst = run.parent / f"{doc['tag']}.wishes_check.json"
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    print(f"# {doc['tag']}: {doc.get('accelerator')}, {doc.get('env', {}).get('chip')}, row graph L{doc.get('length')}")
    print("| request | tokens | graph | argmax = CPU run | max \\|Δp\\| | presses | max Δp between presses | "
          "request time median (min–max) |")
    print("|---|---:|---|---|---:|---:|---:|---|")
    for r in out["requests"]:
        if not r.get("tokens"):
            continue
        qs = r.get("questions")
        agree = "—" if qs is None else f"{sum(q['argmax'] == q['cpu_argmax'] for q in qs)}/{len(qs)}" + (
            f" (near tie {sum(q['near_tie'] for q in qs)})" if any(q["near_tie"] for q in qs) else "")
        dp = "—" if qs is None else f"{r['max_abs_dp']:.2e}"
        print(f"| {r['name']} | {r['tokens']} | L{r['length']} | {agree} | {dp} | {r['presses']} | "
              f"{r['max_dp_between_presses']:.1e} | {r['ms']['median']:.1f} ({r['ms']['min']:.1f}–{r['ms']['max']:.1f}) ms |")
    print(f"\nexpected settings: {hits}/{total}")
    print("| wish | device | expected | answer | p |")
    print("|---|---|---|---|---:|")
    for e in out["expected"]:
        print(f"| {e['wish']} | {e['device']} | {e['expected']} | {e['answer']}{'' if e['match'] else ' (miss)'} | "
              f"{e['p']:.3f} |")
    for r in out["refused"]:
        print(f"refused: {r['error'][:120]} | on screen: {r['shown']}")
    print(f"CHECK_WISHES {doc['tag']}: " + ("PASS" if not problems else "FAIL: " + "; ".join(problems[:8])))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
