"""The editor's form of a d1-3B request, and back.

The screen holds a state text, at most one photo, and question cards. A card has a name (the answer's key), a type
(yes / no, choice or score), the question, and an options text with one entry per line. `build_request` turns that
into the request the host takes (host/d1_litert.py, docstring item 1); `editor_of` turns a request into the editor's
form (the bundled examples). For every request this module builds, `editor_of(build_request(draft))` is the draft
up to the normalisation below, and `build_request(editor_of(request))` is the request.

The rules:
- State: empty or blank -> no state (None); a text whose whole content parses as a JSON object or array -> that
  JSON value (the host renders it with json.dumps(indent=2)); any other text -> the text as typed.
- Name: trimmed, not empty, not used by another card. It never reaches the model; it names the answer.
- Question: trimmed, not empty.
- Options, one per line, blank lines skipped, each line trimmed:
  choice  "name: description" or "name" (the name ends at the first colon; an empty description is none); at least
          two, names unique;
  score   one level per line, from the lowest; 2 to 10 levels (the read-out takes one digit per level);
  yes/no  nothing, or both "Yes: <what yes means>" and "No: <what no means>" (true: / false: work too).
- At least one question. The photo, when there is one, is the request's only picture.
"""
from __future__ import annotations

import json
import re

TYPES = ("noul", "choice", "score")
TYPE_WORDS = {"noul": "yes / no", "choice": "choice", "score": "score"}
MIN_CHOICE, MIN_LEVELS, MAX_LEVELS = 2, 2, 10
YES_NO = re.compile(r"^(yes|no|true|false)\s*:\s*(.*)$", re.IGNORECASE)


class DraftError(ValueError):
    """The editor's content is not a request yet; the message says what to fix."""


def state_value(text: str):
    """The request's state for the STATE text."""
    if not text.strip():
        return None
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
        if not MIN_LEVELS <= len(lines) <= MAX_LEVELS:
            raise DraftError(f"{where} (score) needs {MIN_LEVELS} to {MAX_LEVELS} levels, one per line, from the lowest.")
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


def build_request(draft: dict, photo: bytes | None = None) -> dict:
    """The request for the editor's content: {"state", "questions"} (+ "images": [photo] with a photo)."""
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
    request = {"state": state_value(draft.get("state") or ""), "questions": out}
    if photo is not None:
        request["images"] = [photo]
    return request


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
        text = json.dumps(state, ensure_ascii=False, indent=2)
    return {"state": text, "questions": [
        {"name": name, "type": q.get("type", "choice"), "text": q["instructions"], "options": options_text(q)}
        for name, q in request["questions"].items()]}
