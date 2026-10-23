"""What the house picture shows: each device's setting, read from the state (the house now) or from the answers (the
house after Decide). app/ui/app.js reads the state the same way for the picture while you type (houseNow there).

The picture knows five devices by their question names, and their settings by the words of a text:
    light     off | dim | bright          (the living room light)
    curtains  closed | open
    tv        on | off
    aircon    off | cool | heat
    door      locked | unlocked          (the front door)
A setting is read from a text by its words, in lower case: the first setting in WORDS whose words include one of the
text's words. The house now: the state's member of the device's name ("light": "bright"), and "outside" (the sky:
night, dusk or day), "inside" and "time" as written. The house after Decide: per question of a device's name, the
option the model picked, read by its name and then its description (choice), its description (yes / no: "Closed" for
yes), or, for a yes / no question without descriptions, the device's yes and no (NOUL_DEFAULT). A device whose
question is not in the request keeps its setting; a setting that cannot be read is None (the picture marks it "?")."""
from __future__ import annotations

import re

DEVICES = ("light", "curtains", "tv", "aircon", "door")
NAMES = {"light": "Light", "curtains": "Curtains", "tv": "TV", "aircon": "Air conditioner", "door": "Front door"}
WORDS = {
    "light": (("off", ("off", "dark")), ("dim", ("dim", "dimmed", "low")),
              ("bright", ("bright", "brightness", "full", "on"))),
    "curtains": (("closed", ("closed", "close", "shut", "drawn")), ("open", ("open", "opened"))),
    "tv": (("on", ("on",)), ("off", ("off",))),
    "aircon": (("cool", ("cool", "cooling")), ("heat", ("heat", "heating", "warm")), ("off", ("off",))),
    "door": (("unlocked", ("unlocked", "unlock")), ("locked", ("locked", "lock"))),
}
NOUL_DEFAULT = {"curtains": ("closed", "open"), "tv": ("on", "off"), "door": ("locked", "unlocked")}   # (yes, no)
SKY = (("night", ("dark", "night", "midnight")),
       ("dusk", ("dusk", "dawn", "sunset", "sunrise", "evening", "twilight")))


def words(text) -> set[str]:
    return set(re.findall(r"[a-z]+", str(text).lower()))


def setting(device: str, text) -> str | None:
    """The device's setting named by the text's words, or None."""
    w = words(text)
    for value, names in WORDS[device]:
        if w & set(names):
            return value
    return None


def sky(outside) -> str:
    w = words(outside or "")
    for value, names in SKY:
        if w & set(names):
            return value
    return "day"


def house_now(state) -> dict:
    """The picture of the house the state describes (a JSON object; anything else: every device None)."""
    s = state if isinstance(state, dict) else {}
    out = {d: (setting(d, s[d]) if isinstance(s.get(d), (str, int, float, bool)) else None) for d in DEVICES}
    out["sky"] = sky(s.get("outside"))
    out["inside"] = str(s["inside"]) if isinstance(s.get("inside"), (str, int, float)) else ""
    out["time"] = str(s["time"]) if isinstance(s.get("time"), (str, int, float)) else ""
    return out


def answer_setting(device: str, question: dict, key: str) -> str | None:
    """The setting of the option `key` the model picked for the device's question."""
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "noul":
        desc = (criteria or {}).get(key)
        if desc:
            return setting(device, desc)
        pair = NOUL_DEFAULT.get(device)
        return None if pair is None else pair[0 if key == "true" else 1]
    if kind == "choice":
        return setting(device, key) or setting(device, (criteria or {}).get(key) or "")
    if kind == "score":
        return setting(device, criteria[int(key)])
    return None


def house_after(request: dict, picked: dict) -> dict:
    """The picture after Decide: the house now, with each device the request asks about set to the picked option."""
    out = house_now(request.get("state"))
    for device in DEVICES:
        question = request["questions"].get(device)
        if question is not None and device in picked:
            out[device] = answer_setting(device, question, picked[device])
    return out


def changed(before: dict, after: dict) -> list[str]:
    return [d for d in DEVICES if before.get(d) != after.get(d)]
