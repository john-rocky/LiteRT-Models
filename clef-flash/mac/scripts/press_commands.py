#!/usr/bin/env python3
"""Press Decide on every command of a set in the app started with --tag, through the page's own fields and buttons,
with "Apply to this Mac" off (the answers are checked here; the Mac is not changed).

    venv-demo/bin/python app/mac_demo.py --tag e1 &                   # the app, scripted input accepted
    venv-demo/bin/python -B scripts/press_commands.py e1 eval/dev.json [--presses 5] [--shots]
    venv-demo/bin/python -B scripts/check_presses.py out/e1_manual.run.json eval/dev_cpu.json

First the "Apply to this Mac" switch is turned off. For each command of the set: its fixed state pasted into the State
tab (so the request is the set's own: the questions follow the pasted state, app/form.py), then Decide --presses times
(each Decide waits for the one before). Then a command too long for the row graph, pasted into the command line: one
Decide, which the app refuses with a red line. --shots saves a screenshot of the window after the last Decide of each
command (out/<tag>_<id>.png). Every Decide is in the app's record (out/<tag>_manual.run.json); this script prints one
line per Decide and writes nothing else. It ends with QUIT (the app closes) unless --stay.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "scripts"))
sys.path.insert(0, str(H / "app"))
from drive_editor import Driver  # noqa: E402
import editor as E  # noqa: E402

LONG_COMMAND = " ".join(["Quit every app except the TV app, keep the music playing, close both drafts, set the volume "
                         "to 30 percent, and then open Calculator again so I can add up the receipts."] * 6)


def line(name: str, rec: dict) -> str:
    if "error" in rec:
        return f"{name}: refused: {rec['error']} | on screen: {rec.get('shown_error') or ''}"
    s = rec["shown"]
    way = s["mac"]["way"]
    ms = rec["ms"]["request_wall"]
    return (f"{name}: {rec.get('tokens')} tokens on L{rec.get('length')}, {'—' if ms is None else f'{ms:.1f}'} ms; "
            f"{way['word']} {way['p']} | {way['detail']} | time {s['time']!r}; applied {rec['apply']['on']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tag")
    ap.add_argument("set")
    ap.add_argument("--presses", type=int, default=5)
    ap.add_argument("--shots", action="store_true")
    ap.add_argument("--stay", action="store_true", help="leave the app running at the end")
    a = ap.parse_args()
    drv = Driver(a.tag)
    ready = drv.ready()
    print(f"READY: window {ready['window']['id']} {ready['window']['px']}", flush=True)
    print(f"apply: {drv.apply(False)}", flush=True)
    doc = json.loads(Path(a.set).read_text())
    for c in doc["commands"]:
        drv.paste("#state", E.json_text(c["state"]))
        for k in range(a.presses):
            print(line(f"{c['id']} #{k + 1}", drv.decide()), flush=True)
        if a.shots:
            print(f"{c['id']}: screenshot {drv.shot(c['id'])}", flush=True)
    drv.paste("#command", LONG_COMMAND)
    rec = drv.decide()
    print(line("too long", rec), flush=True)
    if a.shots:
        print(f"too long: screenshot {drv.shot('too_long')}", flush=True)
    if not a.stay:
        drv.path("QUIT").touch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
