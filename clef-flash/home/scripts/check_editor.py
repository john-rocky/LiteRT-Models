#!/usr/bin/env python3
"""Check, without the model, the editor's rules on the bundled wishes and what the house picture reads from them.

    venv-demo/bin/python -B scripts/check_editor.py

For each wish of app/wishes.json: its request (the wish's house and the five questions) through the editor and back
(app/editor.py editor_of, then build_request) is the same request, key order included; the house picture reads every
device of the state (app/house.py house_now: no "?"), and the expected settings read as the picture's settings
(house_after with the expected options). The recorded scene's opening (the take's house with an empty wish) is the
take's request but for the wish. The host's tokenizer gives each request's length and the row graph that holds it
(no graph is loaded). Exit 1 on any failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "app"))
sys.dont_write_bytecode = True
import editor as E  # noqa: E402
import house as HS  # noqa: E402


def main() -> int:
    doc = json.loads((H / "app/wishes.json").read_text())
    hub = H.parent / "hub"
    tok = None
    if (hub / "tokenizer/tokenizer.json").is_file():
        sys.path.insert(0, str(hub / "host"))
        from clef_litert.request import encode_record
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(hub / "tokenizer/tokenizer.json"))
    problems = []
    for w in doc["wishes"]:
        req = {"model": E.MODEL, "state": w["state"], "questions": doc["questions"]}
        back = E.build_request(E.editor_of(req))
        same = json.dumps(back, ensure_ascii=False) == json.dumps(req, ensure_ascii=False)
        now = HS.house_now(w["state"])
        unread = [d for d in HS.DEVICES if now[d] is None]
        after = HS.house_after(req, w["expected"])
        unread_after = [d for d in HS.DEVICES if after[d] is None]
        tokens = encode_record(tok, req).tokens if tok else None
        if not same:
            problems.append(f"{w['id']}: the request through the editor and back is not the request")
        if unread or unread_after:
            problems.append(f"{w['id']}: the picture cannot read {unread or unread_after}")
        print(f"{w['id']}: through the editor and back {'same' if same else 'DIFFERENT'}; "
              + " ".join(f"{d} {now[d]}->{after[d]}" for d in HS.DEVICES)
              + f"; sky {now['sky']}; changed {HS.changed(now, after)}"
              + ("" if tokens is None else f"; {tokens} tokens"))
    take = next(w for w in doc["wishes"] if w["id"] == doc["take"])
    opening = E.build_request(E.editor_of({"model": E.MODEL, "state": dict(take["state"], request=""),
                                           "questions": doc["questions"]}))
    want = {"model": E.MODEL, "state": dict(take["state"], request=""), "questions": doc["questions"]}
    if json.dumps(opening, ensure_ascii=False) != json.dumps(want, ensure_ascii=False):
        problems.append("the scene's opening is not the take's request with an empty wish")
    print(f"take {take['id']}: the opening = the take's request with request \"\": "
          f"{json.dumps(opening, ensure_ascii=False) == json.dumps(want, ensure_ascii=False)}")
    print("CHECK_EDITOR " + ("PASS" if not problems else "FAIL: " + "; ".join(problems)))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
