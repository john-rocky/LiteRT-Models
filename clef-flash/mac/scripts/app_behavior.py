#!/usr/bin/env python3
"""Count the commands of a run (scripts/run_set.py output) on which the app does what the expected answers would make
it do. This count was defined after the dev and held-out runs had been seen; it does not replace the lines set before
them (every question right, and the share of questions right).

    venv-demo/bin/python -B scripts/app_behavior.py eval/dev_cpu.json eval/heldout_metal.json

A command counts when the way of handling the app shows is the expected one ("execute now" and "execute available
parts now" are one way here: the app sets the same devices for both) and, when that way changes the Mac, the changes
the app would set (app/form.py decision: the targets whose answered setting differs from the setting now) are exactly
the expected changes. Prints one line per command that does not count, and the totals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "app"))
sys.dont_write_bytecode = True
import form as F  # noqa: E402


def way(resolution: str) -> str:
    return "execute" if resolution in F.CHANGING else resolution


def main() -> int:
    for path in sys.argv[1:]:
        doc = json.loads(Path(path).read_text())
        n_ok = 0
        for c in doc["commands"]:
            state = c["request"]["state"]
            dec = F.decision(state, c["picked"])
            expected = F.decision(state, c["gold"])   # the run keeps the expected option of every question
            got = {(x["device"], x["prop"], json.dumps(x["to"])) for x in dec["changes"]}
            want = {(x["device"], x["prop"], json.dumps(x["to"])) for x in expected["changes"]}
            same_way = way(dec["resolution"]) == way(expected["resolution"])
            same_changes = way(expected["resolution"]) != "execute" or got == want
            if same_way and same_changes:
                n_ok += 1
            else:
                print(f"{path} {c['id']}: shows {dec['resolution']}, expected {expected['resolution']}; changes "
                      f"{sorted(got)} vs expected {sorted(want)}")
        print(f"APP_BEHAVIOR {path}: {n_ok} / {len(doc['commands'])} commands")
    return 0


if __name__ == "__main__":
    sys.exit(main())
