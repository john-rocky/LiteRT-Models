#!/usr/bin/env python3
"""Check, without the model, the requests of a command set, the editor's rules on them, what the rows read from them,
and the table from a decided change to a setting.

    venv-demo/bin/python -B scripts/check_editor.py [eval/dev.json]

For each command of the set: its request (app/form.py request_of: the state and the questions built from it) through
the editor and back (app/editor.py editor_of, then build_request) is the same request, key order included; every
device of the state reads as a setting on its row; the gold's changes (app/form.py decision on the gold options) each
have a fixed setting in app/actions.py TABLE (or are shown only: the screen); every option of every post question is a
value of its property's type or "not applicable or not listed". The bundled commands (app/commands.json) are the set's
commands. The host's tokenizer gives each request's length (no graph is loaded): every request fits the 2,048-token
row graph, and the too-long command of press_commands.py does not. Exit 1 on any failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "app"))
sys.path.insert(0, str(H / "scripts"))
sys.dont_write_bytecode = True
import actions as A  # noqa: E402
import editor as E  # noqa: E402
import form as F  # noqa: E402
from mac_demo import DEFAULT_LENGTH, device_now  # noqa: E402
from press_commands import LONG_COMMAND  # noqa: E402


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else H / "eval/dev.json"
    doc = json.loads(src.read_text())
    hub = H.parent / "hub"
    tok = None
    if (hub / "tokenizer/tokenizer.json").is_file():
        sys.path.insert(0, str(hub / "host"))
        from clef_litert.request import encode_record
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(hub / "tokenizer/tokenizer.json"))
    problems = []
    lengths = []
    for c in doc["commands"]:
        state = c["state"]
        req = F.request_of(state)
        back = E.build_request(E.editor_of(req))
        same = json.dumps(back, ensure_ascii=False) == json.dumps(req, ensure_ascii=False)
        if not same:
            problems.append(f"{c['id']}: the request through the editor and back is not the request")
        rows = {d["id"]: device_now(d) for d in F.devices_of(state)}
        if any("?" in v for v in rows.values()):
            problems.append(f"{c['id']}: a row cannot read its device: {rows}")
        gold = F.gold_options(state, c["gold"])
        if set(gold) != set(req["questions"]):
            problems.append(f"{c['id']}: the gold answers {sorted(set(gold) ^ set(req['questions']))} differently")
        for q, g in gold.items():
            if g not in req["questions"][q]["criteria"]:
                problems.append(f"{c['id']}/{q}: the gold option {g} is not an option")
        dec = F.decision(state, gold)
        for ch in dec["changes"]:
            key = (ch["device"], ch["prop"])
            if key not in A.TABLE and key not in A.SHOWN_ONLY:
                problems.append(f"{c['id']}: no fixed setting for {key}")
        for q, spec in req["questions"].items():
            if not q.startswith("post:"):
                continue
            dev, _, prop = q[5:].partition(".")
            want = type(next(d for d in F.devices_of(state) if d["id"] == dev)[prop])
            for k in spec["criteria"]:
                v = F.parse_value(k)
                if k != "na" and type(v) is not want:
                    problems.append(f"{c['id']}/{q}: option {k} is not a {want.__name__}")
        tokens = encode_record(tok, req).tokens if tok else None
        if tokens is not None:
            lengths.append(tokens)
            if tokens > DEFAULT_LENGTH:
                problems.append(f"{c['id']}: {tokens} tokens > {DEFAULT_LENGTH}")
        print(f"{c['id']}: through the editor and back {'same' if same else 'DIFFERENT'}; {len(req['questions'])} "
              f"questions; gold {c['gold']['resolution']} -> "
              + (", ".join(A.describe(x['device'], x['prop'], x['to']) for x in dec["changes"]) or "no change")
              + ("" if tokens is None else f"; {tokens} tokens"))
    bundled = json.loads((H / "app/commands.json").read_text())
    texts = [x["request"] for x in bundled["commands"]]
    if sorted(texts) != sorted(c["state"]["request"] for c in doc["commands"]) and src.name == "dev.json":
        problems.append("app/commands.json does not hold the set's commands")
    if tok:
        state = dict(doc["commands"][0]["state"], request=LONG_COMMAND)
        n = encode_record(tok, F.request_of(state)).tokens
        if n <= DEFAULT_LENGTH:
            problems.append(f"the too-long command makes {n} tokens, not more than {DEFAULT_LENGTH}")
        print(f"requests {min(lengths)}..{max(lengths)} tokens; the too-long command of press_commands.py: {n} tokens")
    print("CHECK_EDITOR " + ("PASS" if not problems else "FAIL: " + "; ".join(problems)))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
