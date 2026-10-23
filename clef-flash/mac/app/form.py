"""The request of one command: this Mac's devices as a state in the shape of the Decision Index's home appliance benchmark,
the benchmark's questions on it, and the decision read from the answers.

The benchmark (the Decision Index kit's decision_index/suite/build/home_appliance.py, MIT) describes a house as
devices with an id, a name, a kind, whether they are available and their properties, with named groups, a household
policy, schedules, rules, the device the last request was about ("device_context") and the request itself, and asks
choice questions with its own wording: how the assistant should handle the request (nine ways), for every device
whether it is in the justified target set, and what a property is immediately after correct handling. This app asks
those questions in that wording about seven devices of this Mac:

    tv          app       running (bool), memory_mb       the TV app
    calculator  app       running (bool), memory_mb       Calculator
    draft_1     document  open (bool)                     a TextEdit document named Draft, written by this app
    draft_2     document  open (bool)                     a second one, also named Draft
    music       player    playing (bool)                  a chord loop this app plays
    sound       speaker   volume (0..100); muted (bool)   the output volume (muted is listed, not asked)
    screen      lock      locked (bool)                   shown only: this app never locks the screen

Where it differs from the benchmark's rows (which hold 14 devices and run to about 3,200 tokens): question and option
ids are words ("resolution", "target_member:tv", "post:tv.running"; "execute", "yes", "true") instead of hashes;
"post" is asked for every settable property of every device instead of the first two the request changes, so each
answer is the value the app sets; the benchmark's "outcome" and "state_read" questions are not asked; a Mac has no
rooms, so the state has no rooms, room aliases or outside temperature; the policy keeps the benchmark's
"multi_action", "allow_partial" and "security_mode", and its "away_unlock" becomes "presenting_lock" (this app says the
Mac is presenting, so a lock is blocked). A post question's options are the property's value now, the numbers in the request (and the value
now plus or minus each, inside the device's range), the opposite of a yes/no value, and "not applicable or not
listed".

decision() reads the answers: the most likely way to handle the request, the devices answered "yes", and for each of
those the properties whose answered value differs from the value now; only "execute now" and "execute available parts
now" change anything. The answers are never run: app/actions.py maps each (device, property, value) to a fixed setting.
"""
from __future__ import annotations

import json
import re
import time

MODEL = "clef-flash"

DEVICES = (
    {"id": "tv", "name": "TV", "kind": "app"},
    {"id": "calculator", "name": "Calculator", "kind": "app"},
    {"id": "draft_1", "name": "Draft", "kind": "document", "app": "TextEdit"},
    {"id": "draft_2", "name": "Draft", "kind": "document", "app": "TextEdit"},
    {"id": "music", "name": "Music", "kind": "player"},
    {"id": "sound", "name": "Sound", "kind": "speaker", "min_volume": 0, "max_volume": 100},
    {"id": "screen", "name": "Screen", "kind": "lock"},
)
DEVICE_IDS = tuple(d["id"] for d in DEVICES)
# the settable properties of each kind, in the order their post questions are asked
SETTABLE = {"app": ("running",), "document": ("open",), "player": ("playing",), "speaker": ("volume",),
            "lock": ("locked",)}
GROUPS = {"evening": ["tv", "music"], "work": ["draft_1", "draft_2", "calculator"]}
POLICY = {"multi_action": "best_effort", "allow_partial": True, "security_mode": "presenting",
          "presenting_lock": "blocked"}

# the benchmark's nine ways to handle a request (its status names as option ids, its wording as descriptions)
RESOLUTIONS = {
    "execute": "execute now",
    "execute_partial": "execute available parts now",
    "schedule": "schedule for later",
    "rule": "save an automation rule",
    "clarify": "ask a clarifying question",
    "no_change": "make no change because already satisfied",
    "unsupported": "reject as unsupported",
    "blocked": "block under household policy",
    "query": "answer a state query",
}
CHANGING = ("execute", "execute_partial")
TARGET = {"yes": "yes, this device is in the justified target set",
          "no": "no, this device is not in the justified target set"}
NA = "not applicable or not listed"
RESOLUTION_TEXT = "How should the assistant handle the request?"
MAX_NUMBER_OPTIONS = 6
TIME_RE = re.compile(r"\b\d{1,2}[:.]\d{2}\b")
NUMBER_RE = re.compile(r"(?<![\w.])\d+")


def now_text() -> str:
    """This Mac's clock as the benchmark writes "now": 2026-10-23T20:00:00+09:00."""
    t = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return t[:-2] + ":" + t[-2:]


def state_of(values: dict, request: str, context: str | None, now: str | None = None) -> dict:
    """The state for the device values read from this Mac ({device id: {property: value}}, app/actions.py
    read_all()) and the request."""
    devices = []
    for d in DEVICES:
        v = values.get(d["id"]) or {}
        out = {"id": d["id"], "name": d["name"], "kind": d["kind"], "available": bool(v.get("available", True))}
        if "app" in d:
            out["app"] = d["app"]
        for k, x in v.items():
            if k != "available":
                out[k] = x
        for k in ("min_volume", "max_volume"):
            if k in d:
                out[k] = d[k]
        devices.append(out)
    return {"now": now or now_text(), "devices": devices, "groups": GROUPS, "policy": POLICY, "schedules": [],
            "rules": [], "device_context": context, "request": request}


def devices_of(state) -> list[dict]:
    """The state's devices that have an id (any other entry is left to the model)."""
    if not isinstance(state, dict) or not isinstance(state.get("devices"), list):
        return []
    return [d for d in state["devices"] if isinstance(d, dict) and isinstance(d.get("id"), str) and d["id"]]


def settable(device: dict) -> list[str]:
    """The properties a post question asks about: the kind's settable ones the device lists (any other kind: its
    yes/no properties but "available")."""
    props = SETTABLE.get(device.get("kind"))
    if props is None:
        props = tuple(k for k, v in device.items() if isinstance(v, bool) and k != "available")
    return [p for p in props if p in device and isinstance(device[p], (bool, int))]


def numbers_in(request: str) -> list[int]:
    """The whole numbers a request names, in order, without clock times (10:15) and repeats."""
    text = TIME_RE.sub(" ", request or "")
    out = []
    for m in NUMBER_RE.finditer(text):
        n = int(m.group(0))
        if n not in out:
            out.append(n)
    return out


def value_text(v) -> str:
    """A property value as the benchmark writes its post options (json.dumps): true, false, 25."""
    return json.dumps(v)


def post_options(device: dict, prop: str, numbers: list[int]) -> dict:
    """{option id: description}: the value now, then for a yes/no property its opposite, for a number the request's
    numbers and the value now plus or minus each (inside the device's min_/max_ range when it has one), then "not
    applicable or not listed"."""
    v = device[prop]
    vals = [v]
    if isinstance(v, bool):
        vals.append(not v)
    else:
        lo, hi = device.get(f"min_{prop}"), device.get(f"max_{prop}")
        extra = list(numbers)
        for n in numbers:
            for w in (v + n, v - n):
                if (lo is None or w >= lo) and (hi is None or w <= hi):
                    extra.append(w)
        for w in extra:
            if w not in vals and len(vals) < MAX_NUMBER_OPTIONS:
                vals.append(w)
    out = {value_text(x): value_text(x) for x in vals}
    out["na"] = NA
    return out


def questions_of(state) -> dict:
    """The questions for a state, in this order: resolution, target_member:<id> per device, post:<id>.<property> per
    settable property."""
    devices = devices_of(state)
    request = state.get("request") if isinstance(state, dict) and isinstance(state.get("request"), str) else ""
    numbers = numbers_in(request)
    q = {"resolution": {"type": "choice", "instructions": RESOLUTION_TEXT, "criteria": dict(RESOLUTIONS)}}
    for d in devices:
        q[f"target_member:{d['id']}"] = {
            "type": "choice", "instructions": f"Is {d['id']} in the justified device target set for this request?",
            "criteria": dict(TARGET)}
    for d in devices:
        for p in settable(d):
            q[f"post:{d['id']}.{p}"] = {
                "type": "choice", "instructions": f"Immediately after correct handling, what is {p} for {d['id']}?",
                "criteria": post_options(d, p, numbers)}
    return q


def request_of(state) -> dict:
    return {"model": MODEL, "state": state, "questions": questions_of(state)}


def parse_value(option_id: str):
    """A post option id back to its value (None for "na" or anything that is not JSON)."""
    if option_id == "na":
        return None
    try:
        return json.loads(option_id)
    except ValueError:
        return None


def mentioned(state, request: str | None = None) -> list[str]:
    """The devices whose name the request names (whole words, any case): the candidates a clarifying question asks
    between."""
    text = (request if request is not None else (state.get("request") if isinstance(state, dict) else "")) or ""
    out = []
    for d in devices_of(state):
        name = d.get("name")
        if isinstance(name, str) and name and re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, re.IGNORECASE):
            out.append(d["id"])
    return out


def decision(state, picked: dict) -> dict:
    """What the answers decide, from the picked option of each question ({question name: option id}): the
    resolution, the devices in the target set, each post value, and the changes the app applies (only for
    execute / execute_partial: a target's property whose answered value differs from the value now)."""
    devices = {d["id"]: d for d in devices_of(state)}
    resolution = picked.get("resolution")
    targets = [i for i in devices if picked.get(f"target_member:{i}") == "yes"]
    posts = {}
    for name, opt in picked.items():
        if name.startswith("post:") and "." in name:
            dev, _, prop = name[5:].partition(".")
            posts[f"{dev}.{prop}"] = parse_value(opt)
    changes = []
    if resolution in CHANGING:
        for i in targets:
            for p in settable(devices[i]):
                key = f"{i}.{p}"
                if key in posts and posts[key] is not None and posts[key] != devices[i][p]:
                    changes.append({"device": i, "prop": p, "from": devices[i][p], "to": posts[key]})
    return {"resolution": resolution, "targets": targets, "posts": posts, "changes": changes,
            "candidates": mentioned(state) if resolution == "clarify" else []}


def gold_options(state, gold: dict) -> dict:
    """{question name: option id} for a gold decision {"resolution", "targets", "changes": {"<id>.<prop>": value}}:
    target_member yes for the targets (none for clarify, as in the benchmark), post = the changed value or the value
    now."""
    out = {"resolution": gold["resolution"]}
    targets = set() if gold["resolution"] == "clarify" else set(gold["targets"])
    changes = gold.get("changes", {})
    for d in devices_of(state):
        out[f"target_member:{d['id']}"] = "yes" if d["id"] in targets else "no"
        for p in settable(d):
            out[f"post:{d['id']}.{p}"] = value_text(changes.get(f"{d['id']}.{p}", d[p]))
    return out
