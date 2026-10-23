"""The editor's form of a clef-flash request, and back (the clef-flash sample's own editor rules, unchanged).

The screen holds a state text (the house, as JSON; the wish line writes the state's "request" member) and question
cards. A card has a name, a type (yes / no, choice or score), the question, and an options text with one entry per
line. `build_request` turns that into the request the model repository's host takes (hub/host/clef_litert, docstring
item 1); `editor_of` turns a request into the editor's form (the bundled wishes). For every request this module
builds, `editor_of(build_request(draft))` is the draft up to the normalisation below, and `build_request(editor_of(
request))` is the request.

The rules:
- State: empty or blank -> ""; a text whose whole content parses as a JSON object or array -> that JSON value (the
  host renders it as compact JSON with sorted keys); any other text -> the text as typed.
- Name: trimmed, not empty, not used by another card. It is the question's id: the model reads it with the question
  (`ID: <name>`), and it is the answer's key.
- Question: trimmed, not empty.
- Options, one per line, blank lines skipped, each line trimmed (the model reads every option name and description):
  choice  "name: description" or "name" (the name ends at the first colon; an empty description is none); at least
          two, names unique;
  score   one level per line, from the lowest; at least two;
  yes/no  nothing (the model's own descriptions of yes and no), or both "Yes: <what yes means>" and
          "No: <what no means>" (true: / false: work too).
- At least one question. "model" is "clef-flash": the host returns it in the response; the model does not read it.
  This app sends no picture.
"""
from __future__ import annotations

import json
import re

MODEL = "clef-flash"
TYPES = ("noul", "choice", "score")
TYPE_WORDS = {"noul": "yes / no", "choice": "choice", "score": "score"}
MIN_CHOICE, MIN_LEVELS = 2, 2
YES_NO = re.compile(r"^(yes|no|true|false)\s*:\s*(.*)$", re.IGNORECASE)


class DraftError(ValueError):
    """The editor's content is not a request yet; the message says what to fix."""


def state_value(text: str):
    """The request's state for the STATE text."""
    if not text.strip():
        return ""
    if text.strip()[:1] in "{[":
        try:
            value = json.loads(text)
        except ValueError:
            return text
        if isinstance(value, (dict, list)):
            return value
    return text


def state_kind(text: str) -> str:
    """How the STATE text is read: "empty", "json", "text", or "text (not valid JSON)"."""
    if not text.strip():
        return "empty"
    value = state_value(text)
    if isinstance(value, (dict, list)):
        return "json"
    return "text (not valid JSON)" if text.strip()[:1] in "{[" else "text"


def option_lines(text: str) -> list[str]:
    return [line.strip() for line in (text or "").split("\n") if line.strip()]


def _criteria(kind: str, options: str, where: str):
    lines = option_lines(options)
    if kind == "score":
        if len(lines) < MIN_LEVELS:
            raise DraftError(f"{where} (score) needs at least {MIN_LEVELS} levels, one per line, from the lowest.")
        return lines
    if kind == "choice":
        out: dict = {}
        for line in lines:
            name, colon, desc = line.partition(":")
            name, desc = name.strip(), desc.strip()
            if not name:
                raise DraftError(f"{where}: the option line {line!r} has no name before the colon.")
            if name in out:
                raise DraftError(f"{where}: the option {name!r} is there twice.")
            out[name] = desc if colon and desc else None
        if len(out) < MIN_CHOICE:
            raise DraftError(f"{where} (choice) needs at least {MIN_CHOICE} options, one per line.")
        return out
    found: dict = {}
    for line in lines:
        m = YES_NO.match(line)
        if not m or not m.group(2).strip():
            raise DraftError(f"{where} (yes / no): write \"Yes: <what yes means>\" and \"No: <what no means>\", or "
                             "leave the options empty.")
        key = "true" if m.group(1).lower() in ("yes", "true") else "false"
        if key in found:
            raise DraftError(f"{where} (yes / no): {'Yes' if key == 'true' else 'No'} is there twice.")
        found[key] = m.group(2).strip()
    if not found:
        return None
    if set(found) != {"true", "false"}:
        raise DraftError(f"{where} (yes / no): write both Yes: and No:, or neither.")
    return {"true": found["true"], "false": found["false"]}


def build_request(draft: dict) -> dict:
    """The request for the editor's content: {"model", "state", "questions"}."""
    questions = draft.get("questions") or []
    if not questions:
        raise DraftError("Add a question.")
    out: dict = {}
    for i, card in enumerate(questions, 1):
        where = f"Question {i}"
        name = (card.get("name") or "").strip()
        if not name:
            raise DraftError(f"{where} needs a name.")
        if name in out:
            raise DraftError(f"Two questions are named {name!r}.")
        kind = card.get("type")
        if kind not in TYPES:
            raise DraftError(f"{where} has no type.")
        text = (card.get("text") or "").strip()
        if not text:
            raise DraftError(f"{where} needs the question.")
        question = {"type": kind, "instructions": text}
        criteria = _criteria(kind, card.get("options") or "", where)
        if criteria is not None:
            question["criteria"] = criteria
        out[name] = question
    return {"model": MODEL, "state": state_value(draft.get("state") or ""), "questions": out}


def json_text(value) -> str:
    """A JSON state as the STATE text: one line when it is short, else one member of the object (or item of the
    array) per line, each written compactly. Any JSON text is read back to the same value (state_value)."""
    one = json.dumps(value, ensure_ascii=False)
    if len(one) <= 60:
        return one
    if isinstance(value, dict):
        body = [f"  {json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}" for k, v in value.items()]
        return "{\n" + ",\n".join(body) + "\n}"
    return "[\n" + ",\n".join(f"  {json.dumps(v, ensure_ascii=False)}" for v in value) + "\n]"


def options_text(question: dict) -> str:
    kind = question.get("type", "choice")
    criteria = question.get("criteria")
    if kind == "score":
        return "\n".join(str(level) for level in criteria)
    if kind == "noul":
        return "" if not criteria else f"Yes: {criteria.get('true')}\nNo: {criteria.get('false')}"
    return "\n".join(name if desc is None else f"{name}: {desc}" for name, desc in criteria.items())


def editor_of(request: dict) -> dict:
    """The editor's form of a request: {"state": text, "questions": [{name, type, text, options}]}."""
    state = request.get("state")
    if state is None:
        text = ""
    elif isinstance(state, str):
        text = state
    else:
        text = json_text(state)
    return {"state": text, "questions": [
        {"name": name, "type": q.get("type", "choice"), "text": q["instructions"], "options": options_text(q)}
        for name, q in request["questions"].items()]}
