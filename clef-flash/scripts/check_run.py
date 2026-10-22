#!/usr/bin/env python3
"""Check one autoplay run JSON (app/clef_demo.py --scenes P) against the lines set before the take. Exit 1 on any
failure.

    venv-demo/bin/python -B scripts/check_run.py out/<tag>.run.json

(a) Scene P against out/cpu_ref_take.json (the same hub/ files and request on the Mac CPU, scripts/cpu_reference.py):
    the same request (SHA-256, the state, the questions in the same order), the same tokens, row graph and tile; on
    every question the same most likely option and every probability within 0.02 (the conversion's line; the largest
    difference is printed). The response is the one of the app's unrounded probabilities.
(b) Every string on screen, as read back from the page (`shown`): each option's probability = f"{p:.3f}" of its
    unrounded value; the answer = the label of the most likely option and its f"{p:.3f}"; the footer = "1 photo at
    <tile> px · <n> questions · round(request_wall) ms"; the note names the tokens and the row graph; the HUD and the
    question texts as sent; each option's bar drawn at its probability's share of the track (within 0.01,
    layout.bars); nothing drawn past its box and the editor not scrolled (layout.cut empty).
(c) request_wall at most 2 x the model card's median for a request with one photo at the 448 px tile on the
    1,024-token row graph (1,493.8 ms, Apple M4 Max, Metal float32).
(e) The editor = the request: the editor's content as the page sent it with Decide equals the content read back from
    the page after Decide was pressed, and the request equals that content under the editor's rules, written out
    here on their own (app/editor.py); key order included. The photo is the take's photo.
The accelerator is the GPU at float32 with every compiled file fully accelerated, and no graph compiled between READY
and the answers. When the run read a measurement lock (CLEF_LOCK_FILE), the lock held the same line from the compile
to the end of the scene and that line was not empty; without a lock this line is SKIP.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1]
LINE, SLOW = 0.02, 2.0
CARD_PHOTO_448_MS = 1493.8   # the model card's "One photo at the 448 px tile, 688 tokens | g448 tower, L1024"


def labels_of(question: dict) -> dict:
    kind = question.get("type", "choice")
    if kind == "noul":
        return {"true": "yes", "false": "no"}
    if kind == "score":
        return {str(i): str(level) for i, level in enumerate(question["criteria"])}
    return {str(name): str(name) for name in question["criteria"]}


def editor_request(editor: dict) -> dict:
    """The request for the editor's content, under the editor's rules, written out here on their own."""
    text = editor["state"]
    state = "" if not text.strip() else text
    if text.strip() and text.strip()[0] in "{[":
        try:
            v = json.loads(text)
            state = v if isinstance(v, (dict, list)) else text
        except ValueError:
            state = text
    questions = {}
    for card in editor["questions"]:
        lines = [x.strip() for x in card["options"].split("\n") if x.strip()]
        q = {"type": card["type"], "instructions": card["text"].strip()}
        if card["type"] == "score":
            q["criteria"] = lines
        elif card["type"] == "choice":
            q["criteria"] = {}
            for line in lines:
                name, colon, desc = line.partition(":")
                q["criteria"][name.strip()] = desc.strip() if colon and desc.strip() else None
        elif lines:
            pairs = [line.split(":", 1) for line in lines]
            got = {("true" if k.strip().lower() in ("yes", "true") else "false"): v.strip() for k, v in pairs}
            q["criteria"] = {"true": got["true"], "false": got["false"]}
        questions[card["name"].strip()] = q
    return {"model": "clef-flash", "state": state, "questions": questions}


class Checks:
    def __init__(self):
        self.lines, self.failed = [], 0

    def __call__(self, ok, text: str):
        word = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        self.lines.append(f"{word}  {text}")
        self.failed += 1 if ok is False else 0


def main() -> int:
    run = Path(sys.argv[1])
    doc = json.loads(run.read_text())
    tag = doc["tag"]
    c = Checks()
    c(not doc.get("ui_only"), "the run answered with the model (not --ui-only)")
    if doc.get("ui_only") or "failed" in doc or "scenes" not in doc:
        c("failed" not in doc, "the run finished (no traceback)")
        print("\n".join(c.lines))
        print(f"CHECK_RUN {tag}: FAIL ({c.failed} failed)")
        return 1
    scenes = doc["scenes"]
    ex = json.loads((D / "app/examples.json").read_text())
    take = ex["take"]
    src = next(x for x in ex["examples"] if x["id"] == take["from"])
    photo_sha = hashlib.sha256((D / "hub" / src["picture"]).read_bytes()).hexdigest()

    # (a) scene P against the CPU run of the same files and request
    if "P" in scenes:
        p = scenes["P"]
        ref_path = D / "out" / "cpu_ref_take.json"
        if not ref_path.is_file():
            c(False, f"(a) P: no CPU reference {ref_path.name} (scripts/cpu_reference.py)")
        else:
            ref = json.loads(ref_path.read_text())
            c(ref["request_sha256"] == p["request_sha256"],
              f"(a) P request sha256 {p['request_sha256'][:12]} = the CPU reference's")
            c(json.dumps(ref["request"]["questions"], ensure_ascii=False) == p["questions_ordered"]
              and ref["request"]["state"] == p["request"]["state"],
              "(a) P questions and options in the CPU reference's order, the same state")
            c(ref["accelerator"].startswith("cpu"), f"(a) P reference on {ref['accelerator']}")
            c((p["tokens"], p["length"], p["tile"]) == (ref["tokens"], ref["length"], ref["tile"]),
              f"(a) P {p['tokens']} tokens on L{p['length']} at the {p['tile']} px tile = the CPU reference's")
            for q, got in p["probs"].items():
                want = ref["probs"][q]
                dp = max(abs(got[k] - want[k]) for k in want)
                top, rtop = max(got, key=got.get), max(want, key=want.get)
                c(dp <= LINE and top == rtop,
                  f"(a) P {q}: max |dp| {dp:.2e} <= {LINE} vs the Mac CPU run, most likely {top} = {rtop}")
        c(p.get("response_is_its_probs") is True, "(a) P the response is the one of its unrounded probabilities")
        c(p["request"].get("images", [{}])[0].get("sha256") == photo_sha,
          f"(a) P the photo is {Path(src['picture']).name} (sha256 {photo_sha[:12]})")

    # (b) the strings on screen are the rounding of the unrounded values
    for s, rec in scenes.items():
        shown = rec["shown"]
        req_q = rec["request"]["questions"]
        bad = []
        for q, question in req_q.items():
            probs = rec["probs"][q]
            labels = labels_of(question)
            p_in_order = {k: probs[k] for k in labels}
            top = max(p_in_order, key=p_in_order.get)
            sq = shown["questions"].get(q)
            if sq is None:
                bad.append(f"{q}: no answer card on screen")
                continue
            if sq["text"] != question["instructions"]:
                bad.append(f"{q} text {sq['text']!r}")
            if sq["answer"] != labels[top] or sq["answer_p"] != f"{probs[top]:.3f}":
                bad.append(f"{q} answer {sq['answer']} {sq['answer_p']} vs {labels[top]} {probs[top]:.3f}")
            for k, pk in probs.items():
                if sq["options"].get(k) != f"{pk:.3f}":
                    bad.append(f"{q}/{k} {sq['options'].get(k)} vs {pk:.3f}")
                if sq["labels"].get(k) != labels[k]:
                    bad.append(f"{q}/{k} label {sq['labels'].get(k)!r} vs {labels[k]!r}")
        if set(shown["questions"]) != set(req_q):
            bad.append(f"answer cards {sorted(shown['questions'])} vs questions {sorted(req_q)}")
        n = len(req_q)
        word = f"{n} question" + ("s" if n != 1 else "")
        word = f"1 photo at {rec['tile']} px · {word}" if rec["request"].get("images") else word
        ms = rec["ms"]["request_wall"]
        foot = f"{word} · {round(ms):,} ms"
        if shown["footer"] != foot:
            bad.append(f"footer {shown['footer']!r} vs {foot!r}")
        note = f"{rec['tokens']:,} tokens · {rec['length']:,}-token row graph · "
        if not shown["footer_note"].startswith(note):
            bad.append(f"note {shown['footer_note']!r} vs {note!r}…")
        if shown["model"] != doc["hud"]["model"] or shown["device"] != doc["hud"]["device"] or \
                shown["pill"] != doc["hud"]["pill"] or shown["status"] != "DONE":
            bad.append(f"HUD {shown['model']!r} {shown['device']!r} {shown['pill']!r} {shown['status']!r}")
        if shown.get("error"):
            bad.append(f"an error line on screen: {shown.get('error')!r}")
        cut = rec["layout"].get("cut", [])
        if cut:
            bad.append(f"drawn past its box: {cut[:3]}")
        bars = rec["layout"].get("bars")
        if bars is None:
            bad.append("no bar widths in the layout")
        else:
            for q in req_q:
                for k, pk in rec["probs"][q].items():
                    w = bars.get(q, {}).get(k)
                    if w is None or abs(w - pk) > 0.01:
                        bad.append(f"{q}/{k} bar drawn at {w} of its track for p {pk:.3f}")
        strings = sum(1 + len(x["options"]) for x in shown["questions"].values()) + 1
        c(not bad, f"(b) {s}: {strings} numbers on screen = the rounding of the run's values (footer {shown['footer']!r})"
          + ("" if not bad else ": " + "; ".join(bad[:6])))

    # (c) the request time within 2 x the model card's median of its class
    for s, rec in scenes.items():
        ms = rec["ms"]["request_wall"]
        c(ms <= SLOW * CARD_PHOTO_448_MS,
          f"(c) {s} request_wall {ms:.3f} ms <= {SLOW:g} x {CARD_PHOTO_448_MS} = {SLOW * CARD_PHOTO_448_MS:.1f} ms "
          "(the model card)")

    # (e) the editor = the request
    for s, rec in scenes.items():
        ed, shown_ed = rec["editor"], rec["editor_shown"]
        same = ed == shown_ed
        want = editor_request(ed)
        got = {k: v for k, v in rec["request"].items() if k != "images"}
        ordered = json.dumps(got, ensure_ascii=False) == json.dumps(want, ensure_ascii=False)
        photo_ok = (ed.get("photo") or {}).get("id") == rec["request"]["images"][0]["sha256"][:12] \
            if rec["request"].get("images") else ed.get("photo") is None
        dropped = [st for st in rec.get("steps", []) if st["what"] == "photo"]
        c(same and ordered and photo_ok and bool(dropped),
          f"(e) {s}: the editor sent = the editor read back after Decide ({same}); the request = the editor under its "
          f"rules, in order ({ordered}); the photo ({photo_ok}); placed by autoplay ({bool(dropped)})")
        want_q = {k: src["questions"][k] for k in take["questions"]}
        c(json.dumps(got["questions"], ensure_ascii=False) == json.dumps(want_q, ensure_ascii=False)
          and got["state"] == src["state"], f"(e) {s}: the request is the take's (examples.json take)")

    # the accelerator, the compiles and the measurement lock
    c(doc.get("accelerator") == "gpu fp32 (GpuOptions enforce_f32)", f"accelerator {doc.get('accelerator')!r}")
    comp = doc.get("compile", [])
    c(bool(comp) and all(v.get("fully_accelerated") is True for v in comp),
      "every compiled file fully accelerated: " + ", ".join(f"{v['file']} {v.get('fully_accelerated')}" for v in comp))
    late = [f"{s}: {[x['file'] for x in rec.get('compile', [])] + rec.get('compiled_during', [])}"
            for s, rec in scenes.items() if rec.get("compile") or rec.get("compiled_during")]
    c(not late, "no graph compiled between READY and the answers" + ("" if not late else f": {late}"))
    lock = doc.get("lock", {})
    if lock.get("file"):
        start = doc.get("lock_at_compile", "")
        lines = {x["lock"] for x in lock.get("seen", [])} | {rec.get("lock_at_decide", "") for rec in scenes.values()}
        c(bool(start) and lines == {start},
          f"one measurement window from the compile to the end of the scenes (lock {start[:60]!r}; lines seen {len(lines)})")
    else:
        c(None, "measurement window: no lock file configured (CLEF_LOCK_FILE unset)")

    print(f"# check_run {tag}: {doc.get('accelerator')}, started {doc.get('started_at')}")
    print("\n".join(c.lines))
    print(f"CHECK_RUN {tag}: {'PASS' if not c.failed else f'FAIL ({c.failed} failed)'}")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
