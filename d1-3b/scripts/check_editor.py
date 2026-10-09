#!/usr/bin/env python3
"""Check the editor's request rules (app/editor.py) without a model or a window.

    venv-demo/bin/python -B scripts/check_editor.py

1. Every bundled example (app/examples.json) goes editor -> request and back unchanged, key order included.
2. The ticket and photo examples are the model card's requests (hub/fixtures/requests_public.json card_text_001 and
   the questions of card_cats_001), key order included.
3. The state rules: blank -> None, a JSON object or array -> the value, anything else -> the text as typed.
4. Each editor mistake is refused with a message that names the question: no question, no name, a name used twice,
   no question text, a choice with one option or a repeated option, a score with 1 or 11 levels, a yes / no with only
   Yes: or a line that is neither Yes: nor No:.
Exit 1 on any failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1]
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
    examples = json.loads((D / "app/examples.json").read_text())["examples"]
    for x in examples:
        req = {"state": x["state"], "questions": x["questions"]}
        back = E.build_request(E.editor_of(req))
        check(ordered(back) == ordered(req), f"example {x['id']}: editor -> request is the example, in order")
        form = E.editor_of(back)
        check(form == E.editor_of(req), f"example {x['id']}: request -> editor is stable")

    records = {r["id"]: r for r in json.loads((D / "hub/fixtures/requests_public.json").read_text())["records"]}
    card_text = records["card_text_001"]["request"]
    ticket = next(x for x in examples if x["id"] == "ticket")
    check(ordered({"state": ticket["state"], "questions": ticket["questions"]}) == ordered(card_text),
          "the ticket example = card_text_001, in order")
    photo = next(x for x in examples if x["id"] == "photo")
    check(ordered(photo["questions"]) == ordered(records["card_cats_001"]["request"]["questions"]) and
          photo["state"] is None, "the photo example's questions = card_cats_001's, no state")

    check(E.state_value("") is None and E.state_value("  \n ") is None, "a blank state is no state")
    check(E.state_value('{"a": [1, 2]}') == {"a": [1, 2]}, "a JSON object is the value")
    check(E.state_value("[1, 2]") == [1, 2], "a JSON array is the value")
    check(E.state_value("{not json") == "{not json", "text that is not JSON stays text")
    check(E.state_value('"quoted"') == '"quoted"', "a JSON string stays text (only objects and arrays are JSON)")
    check(E.state_value(" I was charged\ntwice ") == " I was charged\ntwice ", "text is kept as typed")

    q = {"name": "q1", "type": "choice", "text": "Which?", "options": "a: A\nb: B"}
    refused({"state": "x", "questions": []}, "Add a question")
    refused({"state": "x", "questions": [dict(q, name=" ")]}, "Question 1 needs a name")
    refused({"state": "x", "questions": [q, dict(q)]}, "Two questions are named")
    refused({"state": "x", "questions": [dict(q, text="  ")]}, "Question 1 needs the question")
    refused({"state": "x", "questions": [dict(q, options="a: A")]}, "at least 2 options")
    refused({"state": "x", "questions": [dict(q, options="a: A\na: B")]}, "there twice")
    refused({"state": "x", "questions": [dict(q, options=": A\nb: B")]}, "no name before the colon")
    refused({"state": "x", "questions": [dict(q, type="score", options="low")]}, "2 to 10 levels")
    refused({"state": "x", "questions": [dict(q, type="score", options="\n".join(map(str, range(11))))]},
            "2 to 10 levels")
    refused({"state": "x", "questions": [dict(q, type="noul", options="Yes: it is")]}, "both Yes: and No:")
    refused({"state": "x", "questions": [dict(q, type="noul", options="maybe: sometimes")]}, "Yes: <what yes means>")
    refused({"state": "x", "questions": [q, dict(q, name="q2", type="nope")]}, "Question 2 has no type")

    got = E.build_request({"state": "", "questions": [dict(q, type="noul", options="")]})
    check(got == {"state": None, "questions": {"q1": {"type": "noul", "instructions": "Which?"}}},
          "yes / no without options has no criteria; an empty state is None")
    got = E.build_request({"state": "s", "questions": [dict(q, type="noul", options="no: not\nYES: so")]})
    check(ordered(got["questions"]["q1"]["criteria"]) == ordered({"true": "so", "false": "not"}),
          "yes / no options in any order and case become {true, false}")
    got = E.build_request({"state": "s", "questions": [dict(q, options=" a \nb_c:  \nd: D: e")]})
    check(ordered(got["questions"]["q1"]["criteria"]) == ordered({"a": None, "b_c": None, "d": "D: e"}),
          "choice lines: a name alone, an empty description, a colon inside the description")
    got = E.build_request({"state": "s", "questions": [q]}, photo=b"\xff\xd8")
    check(got.get("images") == [b"\xff\xd8"], "the photo is the request's only picture")

    print(f"CHECK_EDITOR: {'PASS' if not failed else f'FAIL ({failed} failed)'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
