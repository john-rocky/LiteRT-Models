#!/usr/bin/env python3
"""Check that the Mac now is as the app found it when it started (out/mac_before.json). Exit 1 if not.

    venv-demo/bin/python -B scripts/check_restore.py [out/mac_before.json]

Compares, read now with the file the app wrote at its start: whether the TV app and Calculator run, whether the drafts
are open (they are closed when the app ends) and the music plays (it stops), and the volume with its mute; prints one
line per device. The screen lock and the apps' memory are not compared (the app never sets them).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "app"))
sys.dont_write_bytecode = True
import actions as A  # noqa: E402


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else H / "out" / "mac_before.json"
    doc = json.loads(path.read_text())
    before = doc["settings"]
    now = A.read_all()
    raw = A.read_volume_raw()
    want = {"tv": {"running": (before.get("tv") or {}).get("running")},
            "calculator": {"running": (before.get("calculator") or {}).get("running")},
            "draft_1": {"open": False}, "draft_2": {"open": False}, "music": {"playing": False}}
    bad = []
    for d, w in want.items():
        got = {k: (now.get(d) or {}).get(k) for k in w}
        same = got == w
        print(f"{d}: wanted {w} (at the start {before.get(d)}), now {got}" + ("" if same else "  DIFFERENT"))
        bad += [] if same else [d]
    same_raw = raw == before.get("volume_raw")
    print(f"volume and mute: at the start {before.get('volume_raw')}, now {raw}" + ("" if same_raw else "  DIFFERENT"))
    bad += [] if same_raw else ["sound"]
    print(f"read at the start {before.get('read_at')}; put back {doc.get('restored_at')}")
    print("CHECK_RESTORE " + ("PASS" if not bad else "FAIL: " + ", ".join(bad)))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
