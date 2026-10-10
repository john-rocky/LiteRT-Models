#!/usr/bin/env python3
"""Scripted input for the editor started with --tag (app/d1_demo.py --tag <tag>): every command is JavaScript run on
the page, which reaches the same handlers a key press or a click does; screenshots are of the app's window only.

    venv-demo/bin/python scripts/drive_editor.py <tag> ready                     # wait for READY; print the window
    venv-demo/bin/python scripts/drive_editor.py <tag> type <selector> <text> [--replace] [--cps 27]
    venv-demo/bin/python scripts/drive_editor.py <tag> paste <selector> <text, or @file>
    venv-demo/bin/python scripts/drive_editor.py <tag> tap <selector>
    venv-demo/bin/python scripts/drive_editor.py <tag> example <id>              # ticket, incident, review, photo
    venv-demo/bin/python scripts/drive_editor.py <tag> drop <picture file>       # through the page's drop handler
    venv-demo/bin/python scripts/drive_editor.py <tag> decide                    # tap Decide, wait for the outcome
    venv-demo/bin/python scripts/drive_editor.py <tag> js <javascript>           # print its value
    venv-demo/bin/python scripts/drive_editor.py <tag> shot <name>               # out/<tag>_<name>.png
    venv-demo/bin/python scripts/drive_editor.py <tag> quit

Selectors: #state, #state-clear, #add-q, #decide, #photo-clear, #questions .q-card:nth-child(<n>) .q-name / .q-text /
.q-opts / .q-remove / .seg button[data-type=<noul|choice|score>]. `decide` prints the record the app wrote for that
Decide (out/<tag>_manual.run.json): the route, the request time, the answers on screen, or the error shown.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parents[1]
OUT = D / "out"


class Driver:
    def __init__(self, tag: str):
        self.tag = tag

    def path(self, suffix: str) -> Path:
        return OUT / f"{self.tag}.{suffix}"

    def js(self, script: str, timeout: float = 30.0):
        n = 1 + max([int(p.name.split(".")[-2]) for p in OUT.glob(f"{self.tag}.cmd.*.js")] or [0])
        cmd = self.path(f"cmd.{n}.js")
        tmp = cmd.with_name(cmd.name + ".tmp")
        tmp.write_text(script)
        tmp.rename(cmd)
        res = self.path(f"cmd.{n}.json")
        end = time.time() + timeout
        while not res.exists():
            if time.time() > end:
                raise TimeoutError(f"no answer to {cmd.name} within {timeout} s")
            time.sleep(0.05)
        doc = json.loads(res.read_text())
        if "error" in doc:
            raise RuntimeError(doc["error"])
        return doc.get("value")

    def mark(self, name: str, timeout: float = 60.0) -> dict:
        end = time.time() + timeout
        while time.time() < end:
            m = self.js(f"D1.mark({json.dumps(name)})")
            if m:
                return m
            time.sleep(0.1)
        raise TimeoutError(f"no mark {name}")

    def record(self) -> dict:
        p = OUT / f"{self.tag}_manual.run.json"
        return json.loads(p.read_text()) if p.exists() else {"decides": [], "events": []}

    def ready(self, timeout: float = 900.0) -> dict:
        p = self.path("READY")
        end = time.time() + timeout
        while not p.exists():
            if self.path("FAILED").exists():
                raise SystemExit(self.path("FAILED").read_text())
            if time.time() > end:
                raise SystemExit(f"no {p.name} within {timeout} s (did the app refuse the tag? see its output)")
            time.sleep(0.2)
        return json.loads(p.read_text())

    def type(self, selector: str, text: str, replace: bool, cps: float) -> dict:
        name = f"drv:{time.time():.6f}"
        self.js(f"D1.autoType({json.dumps(selector)}, {json.dumps(text)}, {cps}, {json.dumps(replace)}, "
                f"{json.dumps(name)})")
        return self.mark(name)

    def paste(self, selector: str, text: str) -> dict:
        name = f"drv:{time.time():.6f}"
        self.js(f"D1.autoPaste({json.dumps(selector)}, {json.dumps(text)}, {json.dumps(name)})")
        return self.mark(name)

    def tap(self, selector: str) -> dict:
        name = f"drv:{time.time():.6f}"
        self.js(f"D1.autoTap({json.dumps(selector)}, {json.dumps(name)})")
        return self.mark(name)

    def example(self, eid: str):
        before = len(self.record()["events"])
        self.tap(f'.ex[data-id="{eid}"]')
        self.wait_events(before)

    def wait_events(self, before: int, timeout: float = 30.0) -> list:
        end = time.time() + timeout
        while time.time() < end:
            ev = self.record()["events"]
            if len(ev) > before:
                return ev[before:]
            time.sleep(0.1)
        raise TimeoutError("no new event in the record")

    def drop(self, file: Path):
        before = len(self.record()["events"])
        b64 = base64.b64encode(file.read_bytes()).decode()
        kind = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        n = self.js(f"D1.dropFile({json.dumps(file.name)}, {json.dumps(b64)}, {json.dumps(kind)})")
        return {"files_dropped": n, "events": self.wait_events(before)}

    def decide(self, timeout: float = 300.0) -> dict:
        before = len(self.record()["decides"])
        self.js("D1.autoTap('#decide', 'drv-decide:' + Date.now(), 0)")
        end = time.time() + timeout
        while time.time() < end:
            decides = self.record()["decides"]
            if len(decides) > before:
                return decides[before]
            time.sleep(0.1)
        raise TimeoutError("no Decide recorded")

    def shot(self, name: str) -> dict:
        wid = self.ready()["window"]["id"]
        png = OUT / f"{self.tag}_{name}.png"
        subprocess.run(["screencapture", "-x", "-o", "-l", str(wid), str(png)], check=True)
        size = subprocess.run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(png)], capture_output=True,
                              text=True).stdout.split()
        return {"png": str(png), "px": [int(size[-3]), int(size[-1])]}


def brief(rec: dict) -> dict:
    """The parts of a Decide record worth printing."""
    keys = ("seq", "origin", "error", "route", "full_route", "missing_for_full_route", "row_tokens", "tiles", "tokens",
            "compile")
    out = {k: rec[k] for k in keys if k in rec}
    if "ms" in rec:
        out["request_wall_ms"] = rec["ms"]["request_wall"]
    if "shown" in rec:
        s = rec["shown"]
        out["shown"] = {"footer": s["footer"], "note": s["footer_note"], "hint": s["hint"], "error": s["error"],
                        "answers": {q: f"{v['answer']} {v['answer_p']}" for q, v in s["questions"].items()}}
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    drv, cmd, rest = Driver(argv[0]), argv[1], argv[2:]
    if cmd == "ready":
        out = drv.ready()
    elif cmd == "type":
        replace = "--replace" in rest
        cps = float(rest[rest.index("--cps") + 1]) if "--cps" in rest else 27.0
        args = [a for i, a in enumerate(rest) if a != "--replace" and a != "--cps"
                and not (i > 0 and rest[i - 1] == "--cps")]
        out = drv.type(args[0], args[1].replace("\\n", "\n"), replace, cps)
    elif cmd == "paste":
        text = Path(rest[1][1:]).read_text() if rest[1].startswith("@") else rest[1].replace("\\n", "\n")
        out = drv.paste(rest[0], text)
    elif cmd == "tap":
        out = drv.tap(rest[0])
    elif cmd == "example":
        out = drv.example(rest[0])
    elif cmd == "drop":
        out = drv.drop(Path(rest[0]))
    elif cmd == "decide":
        out = brief(drv.decide())
    elif cmd == "js":
        out = drv.js(rest[0])
    elif cmd == "shot":
        out = drv.shot(rest[0])
    elif cmd == "quit":
        drv.path("QUIT").touch()
        out = "QUIT written"
    else:
        print(__doc__)
        return 2
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
