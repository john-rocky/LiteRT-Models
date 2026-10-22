#!/usr/bin/env python3
"""Press Decide on every bundled example in an editor started with --tag, through the page's own buttons.

    venv-demo/bin/python app/clef_demo.py --tag e1 &                  # the editor, scripted input accepted
    venv-demo/bin/python -B scripts/press_examples.py e1 [--presses 5] [--shots]
    venv-demo/bin/python -B scripts/check_examples.py out/e1_manual.run.json

For each example of app/examples.json (a picture example once at each tile, 448 and 256 px): tap its button, tap the
tile, then tap Decide --presses times (each Decide waits for the one before). Then the take's request: the coffee
example with the questions that examples.json "take" does not name removed with their − buttons (the editor of the
recorded clip, with the photo), --presses times. Then a state longer than the row graph, pasted into STATE: one
Decide, which the app refuses with a red line. --shots saves a screenshot of the window after the last Decide of each
request (out/<tag>_<name>.png). Every Decide is in the app's record (out/<tag>_manual.run.json); this script prints
one line per Decide and writes nothing else. It ends with QUIT (the app closes) unless --stay.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(D / "scripts"))
from drive_editor import Driver  # noqa: E402

LONG_STATE = " ".join(f"Reading {i}: the cold room was at {3 + (i % 9) / 10:.1f} degrees." for i in range(160))


def line(name: str, rec: dict) -> str:
    if "error" in rec:
        return f"{name}: refused: {rec['error']} | on screen: {rec.get('shown', {}).get('error') or ''}"
    s = rec["shown"]
    answers = ", ".join(f"{q} {v['answer']} {v['answer_p']}" for q, v in s["questions"].items())
    ms = rec["ms"]["request_wall"]
    return (f"{name}: {rec.get('tokens')} tokens on L{rec.get('length')}"
            + (f", tile {rec['tile']}" if rec.get("tile") else "")
            + f", {'—' if ms is None else f'{ms:.1f}'} ms; {answers}; footer {s['footer']!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tag")
    ap.add_argument("--presses", type=int, default=5)
    ap.add_argument("--shots", action="store_true")
    ap.add_argument("--stay", action="store_true", help="leave the app running at the end")
    a = ap.parse_args()
    drv = Driver(a.tag)
    ready = drv.ready()
    print(f"READY: window {ready['window']['id']} {ready['window']['px']}", flush=True)
    doc = json.loads((D / "app/examples.json").read_text())
    runs = []
    for x in doc["examples"]:
        for tile in ((448, 256) if x.get("picture") else (None,)):
            runs.append((x["id"] if tile is None else f"{x['id']}_{tile}", x["id"], tile, None))
    take = doc["take"]
    runs.append(("take", take["from"], 448, take["questions"]))
    for name, eid, tile, keep in runs:
        drv.example(eid)
        if tile is not None:
            drv.tile(tile)
        if keep is not None:   # remove the cards the take does not keep, from the last one up
            names = drv.js("[...document.querySelectorAll('#questions .q-card .q-name')].map((e) => e.value)")
            for i in reversed(range(len(names))):
                if names[i] not in keep:
                    drv.tap(f"#questions .q-card:nth-child({i + 1}) .q-remove")
        for k in range(a.presses):
            print(line(f"{name} #{k + 1}", drv.decide()), flush=True)
        if a.shots:
            print(f"{name}: screenshot {drv.shot(name)}", flush=True)
    drv.example("ticket")
    drv.paste("#state", LONG_STATE)
    rec = drv.decide()
    print(line("too long", rec), flush=True)
    if a.shots:
        print(f"too long: screenshot {drv.shot('too_long')}", flush=True)
    if not a.stay:
        drv.path("QUIT").touch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
