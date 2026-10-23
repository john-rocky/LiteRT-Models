#!/usr/bin/env python3
"""Press Decide on every bundled wish in the app started with --tag, through the page's own buttons and fields.

    venv-demo/bin/python app/home_demo.py --tag e1 &                  # the app, scripted input accepted
    venv-demo/bin/python -B scripts/press_wishes.py e1 [--presses 5] [--shots]
    venv-demo/bin/python -B scripts/check_wishes.py out/e1_manual.run.json

For each wish of app/wishes.json: tap its button, then tap Decide --presses times (each Decide waits for the one
before). Then the recorded scene's request the way the scene makes it: the take's wish button, the wish line cleared
and the wish typed into it, --presses times. Then a wish too long for the row graph, pasted into the wish line: one
Decide, which the app refuses with a red line. --shots saves a screenshot of the window after the last Decide of each
request (out/<tag>_<name>.png). Every Decide is in the app's record (out/<tag>_manual.run.json); this script prints
one line per Decide and writes nothing else. It ends with QUIT (the app closes) unless --stay.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "scripts"))
from drive_editor import Driver  # noqa: E402

LONG_WISH = ("We have friends coming over tonight for a long film marathon with snacks, so set up the living room "
             "for that, keep it comfortable for everyone through the evening, and make sure nobody has to get up to "
             "fiddle with anything until the last film ends well after midnight, thank you very much")


def line(name: str, rec: dict) -> str:
    if "error" in rec:
        return f"{name}: refused: {rec['error']} | on screen: {rec.get('shown_error') or ''}"
    s = rec["shown"]
    answers = ", ".join(f"{q} {v['answer']} {v['answer_p']}" for q, v in s["devices"].items())
    ms = rec["ms"]["request_wall"]
    return (f"{name}: {rec.get('tokens')} tokens on L{rec.get('length')}, {'—' if ms is None else f'{ms:.1f}'} ms; "
            f"{answers}; time {s['time']!r}")


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
    doc = json.loads((H / "app/wishes.json").read_text())
    take = next(w for w in doc["wishes"] if w["id"] == doc["take"])
    runs = [(w["id"], w["id"], None) for w in doc["wishes"]] + [("take", take["id"], take["state"]["request"])]
    for name, wid, typed in runs:
        drv.wish(wid)
        if typed is not None:   # the scene's way: the wish line cleared, the wish typed
            drv.type("#wish", typed, True, 27.0)
        for k in range(a.presses):
            print(line(f"{name} #{k + 1}", drv.decide()), flush=True)
        if a.shots:
            print(f"{name}: screenshot {drv.shot(name)}", flush=True)
    drv.wish(doc["wishes"][0]["id"])
    drv.paste("#wish", LONG_WISH)
    rec = drv.decide()
    print(line("too long", rec), flush=True)
    if a.shots:
        print(f"too long: screenshot {drv.shot('too_long')}", flush=True)
    if not a.stay:
        drv.path("QUIT").touch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
