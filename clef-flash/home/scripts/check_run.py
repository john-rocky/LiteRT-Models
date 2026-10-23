#!/usr/bin/env python3
"""Check one autoplay run JSON (app/home_demo.py --scenes M) against the lines set before the take. Exit 1 on any
failure.

    venv-demo/bin/python -B scripts/check_run.py out/<tag>.run.json

(a) Scene M against out/cpu_ref_wishes.json (the same ../hub/ files and request on the Mac CPU,
    scripts/cpu_reference.py): the same request (SHA-256, the state, the questions in the same order), the same tokens
    and row graph; on every question the same most likely option and every probability within 0.02 (the conversion's
    line; the largest difference is printed). The response is the one of the app's unrounded probabilities. And the
    take's wish got every device's expected setting (app/wishes.json "expected").
(b) Every string on screen, as read back from the page (`shown`): per question row and card, the answer = the label
    of the most likely option and f"{p:.3f}", each option's probability = f"{p:.3f}" of its unrounded value; the time =
    "round(request_wall) ms"; the note names the questions, the tokens and the row graph; the HUD and the question
    texts as sent; each option's bar drawn at its probability (within 0.01) when the cards are on screen (the House &
    questions tab), and no bar drawn when they are not (the Wish tab, the scene's); nothing drawn past its box and the
    wish pane not scrolled (layout.cut empty). The house: before Decide it showed the state (app/house.py house_now), after
    the answers it shows house_after of the answers, with the changed devices marked.
(c) request_wall at most 2 x the model card's median for a text request on the 512-token row graph (673.3 ms, Apple M4
    Max, Metal float32).
(e) The editor = the request: the editor's content as the page sent it with Decide equals the content read back from
    the page after Decide was pressed, and the request equals that content under the editor's rules, written out here
    on their own (app/editor.py); key order included. The wish was typed by autoplay into the wish line.
The accelerator is the GPU at float32 with the row graph fully accelerated, and no graph compiled between READY and the
answers. When the run read a measurement lock (HOME_LOCK_FILE), the lock held the same line from the compile to the end
of the scene and that line was not empty; without a lock this line is SKIP.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "app"))
import house as HS  # noqa: E402

LINE, SLOW = 0.02, 2.0
CARD_TEXT_512_MS = 673.3   # the model card's "one decision request, text, T 459 | L512" (Metal float32, M4 Max)


def labels_of(question: dict) -> dict:
    kind = question.get("type", "choice")
    c = question.get("criteria")
    if kind == "noul":
        c = c or {}
        return {"true": str(c.get("true") or "yes"), "false": str(c.get("false") or "no")}
    if kind == "score":
        return {str(i): str(level) for i, level in enumerate(c)}
    return {str(name): str(name if desc is None else desc) for name, desc in c.items()}


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
    wishes = json.loads((H / "app/wishes.json").read_text())
    take = next(w for w in wishes["wishes"] if w["id"] == wishes["take"])
    take_req = {"model": "clef-flash", "state": take["state"], "questions": wishes["questions"]}

    # (a) scene M against the CPU run of the same files and request
    if "M" in scenes:
        m = scenes["M"]
        ref_path = H / "out" / "cpu_ref_wishes.json"
        if not ref_path.is_file():
            c(False, f"(a) M: no CPU reference {ref_path.name} (scripts/cpu_reference.py)")
        else:
            refs = json.loads(ref_path.read_text())
            ref = next(w for w in refs["wishes"] if w["id"] == take["id"])
            c(ref["request_sha256"] == m["request_sha256"],
              f"(a) M request sha256 {m['request_sha256'][:12]} = the CPU reference's ({take['id']})")
            c(json.dumps(ref["request"]["questions"], ensure_ascii=False) == m["questions_ordered"]
              and ref["request"]["state"] == m["request"]["state"],
              "(a) M questions and options in the CPU reference's order, the same state")
            c(refs["accelerator"].startswith("cpu"), f"(a) M reference on {refs['accelerator']}")
            c((m["tokens"], m["length"]) == (ref["tokens"], ref["length"]),
              f"(a) M {m['tokens']} tokens on L{m['length']} = the CPU reference's")
            for q, got in m["probs"].items():
                want = ref["probs"][q]
                dp = max(abs(got[k] - want[k]) for k in want)
                top, rtop = max(got, key=got.get), max(want, key=want.get)
                c(dp <= LINE and top == rtop,
                  f"(a) M {q}: max |dp| {dp:.2e} <= {LINE} vs the Mac CPU run, most likely {top} = {rtop}")
            for q, e in take["expected"].items():
                top = max(m["probs"][q], key=m["probs"][q].get)
                c(top == e, f"(a) M {q}: {top} = the expected {e}")
        c(m.get("response_is_its_probs") is True, "(a) M the response is the one of its unrounded probabilities")

    # (b) the strings on screen are the rounding of the unrounded values; the house is the answers' house
    for s, rec in scenes.items():
        shown = rec["shown"]
        req_q = rec["request"]["questions"]
        bad = []
        picked = {}
        for q, question in req_q.items():
            probs = rec["probs"][q]
            labels = labels_of(question)
            p_in_order = {k: probs[k] for k in labels}
            top = max(p_in_order, key=p_in_order.get)
            picked[q] = top
            row, card = shown["devices"].get(q), shown["questions"].get(q)
            if row is None or card is None:
                bad.append(f"{q}: no answer row or card on screen")
                continue
            if row["answer"] != labels[top] or row["answer_p"] != f"{probs[top]:.3f}":
                bad.append(f"{q} row {row['answer']} {row['answer_p']} vs {labels[top]} {probs[top]:.3f}")
            if row["device"] != HS.NAMES.get(q, q):
                bad.append(f"{q} row name {row['device']!r}")
            if card["text"] != question["instructions"]:
                bad.append(f"{q} text {card['text']!r}")
            if card["answer"] != labels[top] or card["answer_p"] != f"{probs[top]:.3f}":
                bad.append(f"{q} card {card['answer']} {card['answer_p']} vs {labels[top]} {probs[top]:.3f}")
            for k, pk in probs.items():
                if card["options"].get(k) != f"{pk:.3f}":
                    bad.append(f"{q}/{k} {card['options'].get(k)} vs {pk:.3f}")
                if card["labels"].get(k) != labels[k]:
                    bad.append(f"{q}/{k} label {card['labels'].get(k)!r} vs {labels[k]!r}")
        if set(shown["devices"]) != set(req_q) or set(shown["questions"]) != set(req_q):
            bad.append(f"rows {sorted(shown['devices'])}, cards {sorted(shown['questions'])} vs {sorted(req_q)}")
        ms = rec["ms"]["request_wall"]
        if shown["time"] != f"{round(ms):,} ms":
            bad.append(f"time {shown['time']!r} vs {round(ms):,} ms")
        n = len(req_q)
        note = f"{n} question{'s' if n != 1 else ''} · {rec['tokens']:,} tokens · {rec['length']:,}-token row graph · "
        if not shown["note"].startswith(note):
            bad.append(f"note {shown['note']!r} vs {note!r}…")
        if shown["model"] != doc["hud"]["model"] or shown["device"] != doc["hud"]["device"] or \
                shown["pill"] != doc["hud"]["pill"] or shown["status"] != "DONE":
            bad.append(f"HUD {shown['model']!r} {shown['device']!r} {shown['pill']!r} {shown['status']!r}")
        if shown.get("error"):
            bad.append(f"an error line on screen: {shown.get('error')!r}")
        cut = rec["layout"].get("cut", [])
        if cut:
            bad.append(f"drawn past its box: {cut[:3]}")
        # the cards' bars are on screen only in the House & questions tab; in the Wish tab none may be drawn
        bars = rec["layout"].get("bars") or {}
        cards_shown = (rec["layout"].get("rects", {}).get("pane-edit") or {}).get("w", 0) > 0
        for q in req_q:
            for k, pk in rec["probs"][q].items():
                w = bars.get(q, {}).get(k)
                if cards_shown and (w is None or abs(w - pk) > 0.01):
                    bad.append(f"{q}/{k} bar drawn at {w} of its track for p {pk:.3f}")
                if not cards_shown and w is not None:
                    bad.append(f"{q}/{k} bar drawn ({w}) while the cards' tab is hidden")
        before, after = HS.house_now(rec["request"]["state"]), HS.house_after(rec["request"], picked)
        changed = HS.changed(before, after)
        hb, ha = rec["house_shown_before"], shown["house"]
        keys = list(HS.DEVICES) + ["sky", "inside", "time"]
        if any(hb.get(k) != before.get(k) for k in keys):
            bad.append(f"house before Decide {[(k, hb.get(k)) for k in keys]} vs the state's {before}")
        if any(ha.get(k) != after.get(k) for k in keys) or sorted(ha.get("changed", [])) != sorted(changed):
            bad.append(f"house after {[(k, ha.get(k)) for k in keys]} changed {ha.get('changed')} vs {after} {changed}")
        if rec["sent_to_page"]["house_after"] != after or rec["sent_to_page"]["changed"] != changed:
            bad.append("the house sent to the page is not the answers' house")
        strings = sum(2 + 1 + len(x["options"]) for x in shown["questions"].values()) + 1
        c(not bad, f"(b) {s}: {strings} strings on screen = the rounding of the run's values (time {shown['time']!r}); "
          f"house {before['light']}/{before['curtains']}/{before['tv']}/{before['aircon']}/{before['door']} -> "
          f"{after['light']}/{after['curtains']}/{after['tv']}/{after['aircon']}/{after['door']} (changed {changed})"
          + ("" if not bad else ": " + "; ".join(bad[:6])))

    # (c) the request time within 2 x the model card's median of its class
    for s, rec in scenes.items():
        ms = rec["ms"]["request_wall"]
        c(ms <= SLOW * CARD_TEXT_512_MS,
          f"(c) {s} request_wall {ms:.3f} ms <= {SLOW:g} x {CARD_TEXT_512_MS} = {SLOW * CARD_TEXT_512_MS:.1f} ms "
          "(the model card)")

    # (e) the editor = the request
    for s, rec in scenes.items():
        ed, shown_ed = rec["editor"], rec["editor_shown"]
        same = ed == shown_ed
        want = editor_request(ed)
        ordered = json.dumps(rec["request"], ensure_ascii=False) == json.dumps(want, ensure_ascii=False)
        typed = [st for st in rec.get("steps", []) if st["what"] == "type"]
        typed_ok = bool(typed) and typed[0].get("value") == take["state"]["request"] == ed.get("wish")
        c(same and ordered and typed_ok,
          f"(e) {s}: the editor sent = the editor read back after Decide ({same}); the request = the editor under its "
          f"rules, in order ({ordered}); the wish typed by autoplay ({typed_ok})")
        c(json.dumps(rec["request"], ensure_ascii=False) == json.dumps(take_req, ensure_ascii=False),
          f"(e) {s}: the request is the take's wish request (wishes.json take = {take['id']})")

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
        c(None, "measurement window: no lock file configured (HOME_LOCK_FILE unset)")

    print(f"# check_run {tag}: {doc.get('accelerator')}, started {doc.get('started_at')}")
    print("\n".join(c.lines))
    print(f"CHECK_RUN {tag}: {'PASS' if not c.failed else f'FAIL ({c.failed} failed)'}")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
