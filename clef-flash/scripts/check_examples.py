#!/usr/bin/env python3
"""Hold the app's answers to the bundled examples against the provider's float32 reference, and sum up the request
times. Exit 1 on any failure.

    venv-demo/bin/python -B scripts/check_examples.py out/<tag>_manual.run.json

Reads the editor's record (app/clef_demo.py --tag, pressed by scripts/press_examples.py). A Decide is an example's
when its request is that example's request (app/examples.json: the state and the questions, key order included; the
picture by its SHA-256). Its reference is hub/fixtures/reference_probs.json at the example's record and arm (text, or
g256 / g448 for the tile the photo was read at): the provider's own code (joint_schema_model.py systemone()) in
float32 on the CPU. The line, set before any number of this app was seen (the conversion's acceptance line):
  - the input tokens are the reference's;
  - on every question whose two most likely options are more than 0.02 apart (near_tie false), the most likely
    option is the reference's; a near tie is recorded only;
  - every option's probability is within 0.02 of the reference's (the unrounded float32 probabilities the app holds
    from the same call, not the response's 4-digit rounding);
  - the response is the one of those probabilities (response_is_its_probs) and no graph was compiled inside the call.
Every press of the same request gives the same probabilities (recorded: the largest difference between presses).
Times: request_wall per request (time.perf_counter() around clef.decide), the median, min and max over its presses.
Writes <tag>.examples_check.json next to the record and prints a table.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1]
HUB = D / "hub"
LINE = 0.02


def ordered(x) -> str:
    return json.dumps(x, ensure_ascii=False)


def main() -> int:
    run = Path(sys.argv[1])
    doc = json.loads(run.read_text())
    ex = json.loads((D / "app/examples.json").read_text())
    ref = {r["key"]: r for r in json.loads((HUB / "fixtures/reference_probs.json").read_text())["runs"]}
    pics = {x["id"]: hashlib.sha256((HUB / x["picture"]).read_bytes()).hexdigest()
            for x in ex["examples"] if x.get("picture")}
    take = ex["take"]
    src = next(x for x in ex["examples"] if x["id"] == take["from"])
    take_req = {"model": "clef-flash", "state": src["state"],
                "questions": {k: src["questions"][k] for k in take["questions"]}}
    groups: dict = {}
    problems, rows = [], []
    for rec in doc["decides"]:
        if "error" in rec:
            groups.setdefault(("refused", None), []).append(rec)
            continue
        req = {k: v for k, v in rec["request"].items() if k != "images"}
        sha = rec["request"]["images"][0]["sha256"] if rec["request"].get("images") else None
        name = None
        for x in ex["examples"]:
            want = {"model": "clef-flash", "state": x["state"], "questions": x["questions"]}
            if ordered(req) == ordered(want) and sha == pics.get(x["id"]):
                name = x["id"]
        if ordered(req) == ordered(take_req) and sha == pics.get(take["from"]):
            name = "take"
        groups.setdefault((name, rec.get("tile")), []).append(rec)

    out = {"record": str(run), "line": {"max_abs_dp": LINE, "near_tie": 0.02}, "requests": []}
    for (name, tile), recs in groups.items():
        if name == "refused":
            for rec in recs:
                out["requests"].append({"name": "refused", "error": rec["error"],
                                        "shown": (rec.get("shown") or {}).get("error")})
            continue
        if name is None:
            out["requests"].append({"name": None, "presses": len(recs), "note": "not a bundled example"})
            continue
        first = recs[0]
        item = {"name": name, "tile": tile, "presses": len(recs), "tokens": first["tokens"], "length": first["length"]}
        spread = 0.0
        for r in recs[1:]:
            for q, p in r["probs"].items():
                spread = max(spread, max(abs(p[k] - first["probs"][q][k]) for k in p))
        item["max_dp_between_presses"] = spread
        bad = [r["seq"] for r in recs if not r.get("response_is_its_probs") or r.get("compiled_during")]
        if bad:
            problems.append(f"{name}/{tile}: Decide {bad}: the response is not its probabilities, or a compile ran")
        ms = [r["ms"]["request_wall"] for r in recs]
        item["ms"] = {"median": statistics.median(ms), "min": min(ms), "max": max(ms), "all": ms}
        if name != "take":
            x = next(e for e in ex["examples"] if e["id"] == name)
            key = f"ref:{x['record']}/{'text' if tile is None else f'g{tile}'}"
            ref_run = ref[key]
            item["reference"] = key
            if first["tokens"] != ref_run["tokens"]:
                problems.append(f"{name}: {first['tokens']} tokens, the reference {ref_run['tokens']}")
            worst, qs = 0.0, []
            for rq in ref_run["questions"]:
                got = first["probs"][rq["question_id"]]
                if sorted(got) != sorted(rq["option_ids"]):
                    problems.append(f"{name}/{rq['question_id']}: options {sorted(got)} vs {sorted(rq['option_ids'])}")
                    continue
                dps = [abs(got[o] - p) for o, p in zip(rq["option_ids"], rq["probs"])]
                top = max(rq["option_ids"], key=lambda o: got[o])
                agree = top == rq["argmax"]
                qs.append({"question": rq["question_id"], "argmax": top, "reference_argmax": rq["argmax"],
                           "near_tie": rq["near_tie"], "max_abs_dp": max(dps),
                           "p_top": got[top], "reference_p_top": max(rq["probs"])})
                worst = max(worst, max(dps))
                if not rq["near_tie"] and not agree:
                    problems.append(f"{name}/{rq['question_id']}: most likely {top}, the reference {rq['argmax']}")
                if max(dps) > LINE:
                    problems.append(f"{name}/{rq['question_id']}: max |dp| {max(dps):.2e} > {LINE}")
            item["questions"], item["max_abs_dp"] = qs, worst
        out["requests"].append(item)
    names = {(r["name"], r.get("tile")) for r in out["requests"] if r.get("presses")}
    for x in ex["examples"]:
        for tile in ((448, 256) if x.get("picture") else (None,)):
            if (x["id"], tile) not in names:
                problems.append(f"{x['id']}{'' if tile is None else f'/{tile}'}: no Decide in the record")
    out["problems"] = problems
    out["pass"] = not problems
    dst = run.parent / f"{doc['tag']}.examples_check.json"
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    print(f"# {doc['tag']}: {doc.get('accelerator')}, {doc.get('env', {}).get('chip')}, row graph L{doc.get('length')}")
    print("| request | tile | tokens | graph | questions argmax = reference | max \\|Δp\\| | presses | "
          "max Δp between presses | request time median (min–max) |")
    print("|---|---|---:|---|---|---:|---:|---:|---|")
    for r in out["requests"]:
        if r["name"] == "refused":
            print(f"| refused | | | | {r['error'][:80]} | | | | |")
            continue
        if not r.get("tokens"):
            continue
        qs = r.get("questions")
        agree = "—" if qs is None else f"{sum(q['argmax'] == q['reference_argmax'] for q in qs)}/{len(qs)}" + (
            f" (near tie {sum(q['near_tie'] for q in qs)})" if any(q["near_tie"] for q in qs) else "")
        dp = "—" if qs is None else f"{r['max_abs_dp']:.2e}"
        print(f"| {r['name']} | {r['tile'] or ''} | {r['tokens']} | L{r['length']} | {agree} | {dp} | {r['presses']} | "
              f"{r['max_dp_between_presses']:.1e} | {r['ms']['median']:.1f} ({r['ms']['min']:.1f}–{r['ms']['max']:.1f}) ms |")
    print(f"CHECK_EXAMPLES {doc['tag']}: " + ("PASS" if not problems else "FAIL: " + "; ".join(problems[:8])))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
