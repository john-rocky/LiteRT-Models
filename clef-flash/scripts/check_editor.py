#!/usr/bin/env python3
"""Check the editor's request rules (app/editor.py) and the bundled examples without a model or a window.

    venv-demo/bin/python -B scripts/check_editor.py

1. Every bundled example (app/examples.json) goes editor -> request and back unchanged, key order included.
2. Every example is its record of the model repository's public fixtures (hub/fixtures/requests_public.json): the
   request the editor builds equals the record's request, key order included; its picture is the record's picture,
   with the SHA-256 of hub/SHA256SUMS (and, for a photograph, of hub/fixtures/photos.json, CC0-1.0).
3. The take's editor (examples.json "take") is its example's state and the named questions, in the example's order.
4. The state rules: blank -> "", a JSON object or array -> the value, anything else -> the text as typed.
5. Each editor mistake is refused with a message that names the question: no question, no name, a name used twice,
   no question text, a choice with one option or a repeated option, a score with one level, a yes / no with only Yes:
   or a line that is neither Yes: nor No:.
Exit 1 on any failure.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1]
HUB = D / "hub"
sys.path.insert(0, str(D / "app"))
import editor as E  # noqa: E402

failed = 0


def check(ok: bool, text: str) -> None:
    global failed
    print(f"{'PASS' if ok else 'FAIL'}  {text}")
    failed += 0 if ok else 1


def ordered(x) -> str:
    return json.dumps(x, ensure_ascii=False)


def refused(draft: dict, words: str) -> None:
    try:
        E.build_request(draft)
    except E.DraftError as e:
        check(words in str(e), f"refused: {str(e)!r} (expected to mention {words!r})")
        return
    check(False, f"not refused: {ordered(draft)[:120]}")


def main() -> int:
    doc = json.loads((D / "app/examples.json").read_text())
    examples, take = doc["examples"], doc["take"]
    records = {r["id"]: r for r in json.loads((HUB / "fixtures/requests_public.json").read_text())["records"]}
    sums = {line.split()[1][2:]: line.split()[0] for line in (HUB / "SHA256SUMS").read_text().splitlines() if line}
    photos = {p["file"]: p for p in json.loads((HUB / "fixtures/photos.json").read_text())["photos"]}
    for x in examples:
        req = {"model": E.MODEL, "state": x["state"], "questions": x["questions"]}
        back = E.build_request(E.editor_of(req))
        check(ordered(back) == ordered(req), f"example {x['id']}: editor -> request is the example, in order")
        check(E.editor_of(back) == E.editor_of(req), f"example {x['id']}: request -> editor is stable")
        rec = records[x["record"]]
        check(ordered(back) == ordered(rec["request"]),
              f"example {x['id']} = fixture record {x['record']} ({rec['note']}; {rec['license']}), in order")
        if x.get("picture"):
            data = (HUB / x["picture"]).read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            name = Path(x["picture"]).name
            check(rec.get("image_files") == [name] and sums.get(x["picture"]) == sha,
                  f"example {x['id']}: picture {name} = the record's, SHA-256 {sha[:12]} = SHA256SUMS")
            p = photos.get(f"images/{name}")
            if p is not None:
                check(p["sha256"] == sha and p["license"] == "CC0-1.0",
                      f"example {x['id']}: {name} = photos.json's ({p['license']}, {p['author']}, {p['file_page']})")
        else:
            check(not rec.get("image_files"), f"example {x['id']}: no picture, as the record")

    src = next(x for x in examples if x["id"] == take["from"])
    want = [q for q in src["questions"] if q in take["questions"]]
    check(want == take["questions"] and len(want) >= 2,
          f"the take: {take['from']}'s state and questions {take['questions']}, in the example's order")

    check(E.state_value("") == "" and E.state_value("  \n ") == "", "a blank state is the empty text")
    check(E.state_value('{"a": [1, 2]}') == {"a": [1, 2]}, "a JSON object is the value")
    check(E.state_value("[1, 2]") == [1, 2], "a JSON array is the value")
    check(E.state_value("{not json") == "{not json", "text that is not JSON stays text")
    check(E.state_value('"quoted"') == '"quoted"', "a JSON string stays text (only objects and arrays are JSON)")
    check(E.state_value(" Two charges\non March 3 ") == " Two charges\non March 3 ", "text is kept as typed")

    q = {"name": "q1", "type": "choice", "text": "Which?", "options": "a: A\nb: B"}
    refused({"state": "x", "questions": []}, "Add a question")
    refused({"state": "x", "questions": [dict(q, name=" ")]}, "Question 1 needs a name")
    refused({"state": "x", "questions": [q, dict(q)]}, "Two questions are named")
    refused({"state": "x", "questions": [dict(q, text="  ")]}, "Question 1 needs the question")
    refused({"state": "x", "questions": [dict(q, options="a: A")]}, "at least 2 options")
    refused({"state": "x", "questions": [dict(q, options="a: A\na: B")]}, "there twice")
    refused({"state": "x", "questions": [dict(q, options=": A\nb: B")]}, "no name before the colon")
    refused({"state": "x", "questions": [dict(q, type="score", options="low")]}, "at least 2 levels")
    refused({"state": "x", "questions": [dict(q, type="noul", options="Yes: it is")]}, "both Yes: and No:")
    refused({"state": "x", "questions": [dict(q, type="noul", options="maybe: sometimes")]}, "Yes: <what yes means>")
    refused({"state": "x", "questions": [q, dict(q, name="q2", type="nope")]}, "Question 2 has no type")

    got = E.build_request({"state": "", "questions": [dict(q, type="noul", options="")]})
    check(got == {"model": "clef-flash", "state": "", "questions": {"q1": {"type": "noul", "instructions": "Which?"}}},
          "yes / no without options has no criteria; an empty state is the empty text")
    got = E.build_request({"state": "s", "questions": [dict(q, type="noul", options="no: not\nYES: so")]})
    check(ordered(got["questions"]["q1"]["criteria"]) == ordered({"true": "so", "false": "not"}),
          "yes / no options in any order and case become {true, false}")
    got = E.build_request({"state": "s", "questions": [dict(q, options=" a \nb_c:  \nd: D: e")]})
    check(ordered(got["questions"]["q1"]["criteria"]) == ordered({"a": None, "b_c": None, "d": "D: e"}),
          "choice lines: a name alone, an empty description, a colon inside the description")
    got = E.build_request({"state": "s", "questions": [dict(q, type="score", options="\n".join(map(str, range(12))))]})
    check(len(got["questions"]["q1"]["criteria"]) == 12, "a score of 12 levels is a request")
    got = E.build_request({"state": "s", "questions": [q]}, photo=b"\xff\xd8")
    check(got.get("images") == [b"\xff\xd8"], "the photo is the request's only picture")

    print(f"CHECK_EDITOR: {'PASS' if not failed else f'FAIL ({failed} failed)'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
