#!/usr/bin/env python3
"""The numbers on screen in one demo run, as markdown tables (material for the post; nothing typed by hand).

    venv-demo/bin/python -B scripts/screen_numbers.py out/<tag>.run.json

Every cell is a string the page showed (`shown`, read back from the DOM by the app), next to the run's request time
and token count. Writes <tag>.screen_numbers.md next to the run JSON and prints it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NAMES = {"A": "A (support ticket)", "B": "B (photo)"}


def main() -> int:
    run = Path(sys.argv[1])
    doc = json.loads(run.read_text())
    if doc.get("ui_only"):
        raise SystemExit("a --ui-only run has no numbers")
    out = [f"# Screen numbers: {doc['tag']} ({doc['accelerator']}, {doc['hud']['device']})", "",
           "| Scene | Question | Answer | Probability | Footer |", "|---|---|---|---:|---|"]
    for s, rec in doc["scenes"].items():
        for q in rec["shown"]["questions"].values():
            out.append(f"| {NAMES.get(s, s)} | {q['text']} | {q['answer']} | {q['answer_p']} | {rec['shown']['footer']} |")
    out += ["", "| Scene | Question | Option | Probability |", "|---|---|---|---:|"]
    for s, rec in doc["scenes"].items():
        for q in rec["shown"]["questions"].values():
            for k, p in q["options"].items():
                out.append(f"| {s} | {q['text']} | {q['labels'][k]} | {p} |")
    out += ["", "| Scene | Route | Input tokens | Tiles | request_wall (ms, unrounded) |", "|---|---|---:|---:|---:|"]
    for s, rec in doc["scenes"].items():
        r = rec["route"]
        route = f"pair Ls{r['Ls']}+Lq{r['Lq']}" if r["route"] == "pair" else f"row L{r['L']}"
        out.append(f"| {s} | {route} | {rec['response']['usage']['input_tokens']} | {rec.get('tiles', '—')} | "
                   f"{rec['ms']['request_wall']:.3f} |")
    out += ["", f"request_wall: {next(iter(doc['scenes'].values()))['ms']['span']}."]
    text = "\n".join(out) + "\n"
    (run.parent / f"{doc['tag']}.screen_numbers.md").write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
