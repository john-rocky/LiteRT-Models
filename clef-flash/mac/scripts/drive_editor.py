#!/usr/bin/env python3
"""Scripted input for the app started with --tag (app/mac_demo.py --tag <tag>): every command is JavaScript run on
the page, which reaches the same handlers a key press or a click does; screenshots are of the app's window only.

    venv-demo/bin/python scripts/drive_editor.py <tag> ready                     # wait for READY; print the window
    venv-demo/bin/python scripts/drive_editor.py <tag> type <selector> <text> [--replace] [--cps 27]
    venv-demo/bin/python scripts/drive_editor.py <tag> paste <selector> <text, or @file>
    venv-demo/bin/python scripts/drive_editor.py <tag> tap <selector>
    venv-demo/bin/python scripts/drive_editor.py <tag> command <id>              # a bundled command's button
    venv-demo/bin/python scripts/drive_editor.py <tag> tab <command|edit>        # the Command or the State & questions tab
    venv-demo/bin/python scripts/drive_editor.py <tag> apply <on|off>            # the "Apply to this Mac" switch
    venv-demo/bin/python scripts/drive_editor.py <tag> decide                    # tap Decide, wait for the outcome
    venv-demo/bin/python scripts/drive_editor.py <tag> js <javascript>           # print its value
    venv-demo/bin/python scripts/drive_editor.py <tag> shot <name>               # out/<tag>_<name>.png
    venv-demo/bin/python scripts/drive_editor.py <tag> quit

Selectors: #command, #state, #read-mac, #rebuild-q, #scene, #apply, #add-q, #decide, .tab[data-tab=edit],
#questions .q-card:nth-child(<n>) .q-name / .q-text / .q-opts / .q-remove / .seg button[data-type=<noul|choice|score>].
`decide` prints the record the app wrote for that Decide (out/<tag>_manual.run.json): the plan, the request time, the
way of handling and the answers on screen, the settings set, or the error shown.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

H = Path(__file__).resolve().parents[1]
OUT = H / "out"


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
            m = self.js(f"MAC.mark({json.dumps(name)})")
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
        self.js(f"MAC.autoType({json.dumps(selector)}, {json.dumps(text)}, {cps}, {json.dumps(replace)}, "
                f"{json.dumps(name)})")
        return self.mark(name)

    def paste(self, selector: str, text: str) -> dict:
        name = f"drv:{time.time():.6f}"
        self.js(f"MAC.autoPaste({json.dumps(selector)}, {json.dumps(text)}, {json.dumps(name)})")
        return self.mark(name)

    def tap(self, selector: str) -> dict:
        name = f"drv:{time.time():.6f}"
        self.js(f"MAC.autoTap({json.dumps(selector)}, {json.dumps(name)})")
        return self.mark(name)

    def command(self, cid: str):
        loads = self.js("MAC.loads()")
        self.tap(f'.ex[data-id="{cid}"]')
        end = time.time() + 30   # the page writes the command when the command's call returns
        while time.time() < end:
            if self.js("MAC.loads()") > loads:
                on = self.js("document.querySelector('.ex.on') && document.querySelector('.ex.on').dataset.id")
                if on != cid:
                    raise RuntimeError(f"the page shows command {on}, not {cid}")
                return {"command": cid, "line": self.js("document.querySelector('#command').value")}
            time.sleep(0.05)
        raise TimeoutError(f"the page did not show command {cid}")

    def apply(self, on: bool) -> dict:
        if bool(self.js("document.querySelector('#apply').checked")) != on:
            self.tap("#apply")
        got = bool(self.js("document.querySelector('#apply').checked"))
        if got != on:
            raise RuntimeError(f"the apply switch reads {got}, not {on}")
        return {"apply": got}

    def tab(self, name: str) -> dict:
        return self.tap(f'.tab[data-tab="{name}"]')

    def decide(self, timeout: float = 300.0) -> dict:
        before = len(self.record()["decides"])
        self.js("MAC.autoTap('#decide', 'drv-decide:' + Date.now(), 0)")
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
    keys = ("seq", "origin", "error", "shown_error", "plan", "tokens", "length", "compile", "compiled_during",
            "response_is_its_probs")
    out = {k: rec[k] for k in keys if k in rec}
    if "ms" in rec:
        out["request_wall_ms"] = rec["ms"]["request_wall"]
    if "shown" in rec:
        s = rec["shown"]
        out["shown"] = {"time": s["time"], "note": s["note"], "error": s["error"], "way": s["mac"]["way"],
                        "answers": {q: f"{v['answer']} {v['answer_p']}" for q, v in s["questions"].items()}}
    if "apply" in rec:
        ap = rec["apply"]
        out["apply"] = {"on": ap["on"], "seconds": ap.get("seconds"),
                        "results": {f"{r['device']}.{r['prop']}": f"{r.get('read_back')} "
                                    f"{'ok' if r['ok'] else r.get('error')}" for r in ap.get("results", [])}}
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
    elif cmd == "command":
        out = drv.command(rest[0])
    elif cmd == "tab":
        out = drv.tab(rest[0])
    elif cmd == "apply":
        out = drv.apply(rest[0] == "on")
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
