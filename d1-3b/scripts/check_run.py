#!/usr/bin/env python3
"""Check one autoplay run JSON (app/d1_demo.py --scenes A,B) against the lines set before the take. Exit 1 on any
failure.

    venv-demo/bin/python -B scripts/check_run.py out/<tag>.run.json

(a) Scene A: the unrounded probabilities against hub/examples/run_example.expected.json (`text`, the provider's code
    in float32 on the CPU): every option within 1e-5, the same most likely option, usage.input_tokens and the route
    equal to the file's (116; pair Ls64+Lq64). Scene B: against out/cpu_ref_<photo sha256, first 12>.json (the same
    files and request on the Mac CPU, scripts/cpu_reference.py): every option within 1e-4, the same most likely
    option, the same request (sha256, and the questions in the same order), 1 tile, route row L512. The card photo's
    check (the picture pipeline): out/run_example_check_gpu.log holds "matches run_example.expected.json (gpu)".
(b) Every string on screen, as read back from the page (`shown`): each option's probability = f"{p:.3f}" of its
    unrounded value; the answer = the label of the most likely option and its f"{p:.3f}"; the footer = "<count> ·
    round(request_wall) ms"; the HUD and the question texts as sent; each option's bar drawn at its probability's
    share of the track (within 0.01, layout.bars); nothing drawn past its box (layout.cut empty).
(c) request_wall at most 2 x the card's time of the same class: A 153.9249 ms (the card's three questions on the
    Ls64+Lq64 pair, Mac Metal float32), B 389.275 ms (the card's one picture and one question). Read from the
    conversion's result files when they are next to this demo, else these values.
(e) The editor = the request: the editor's content as the page sent it with Decide equals the content read back from
    the page after Decide was pressed, and the request equals that content under the editor's rules, written out
    here on their own (app/editor.py: the state blank -> None, a JSON object or array -> the value, else the text;
    each card -> {type, instructions: the question trimmed, criteria from the option lines}); key order included.
    The typed text is what autoplay typed (A: the card's ticket; B: the question and its options).
The accelerator is the GPU at float32 with every compiled file fully accelerated. When the run read a measurement
lock (D1_LOCK_FILE), each scene's Decide ran inside its own window (the lock named d1-demo-<tag>-<scene> at that
moment) and no other window was open during the compile and the warm-up; without a lock these two lines are SKIP.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1]
K = D.parent
TOL_A, TOL_B, SLOW = 1e-5, 1e-4, 2.0
CARD_MS = {"A": 153.9249, "B": 389.275}
WINDOW_LABEL = "d1-demo"


def labels_of(question: dict) -> dict:
    kind = question.get("type", "choice")
    if kind == "noul":
        return {"true": "yes", "false": "no"}
    if kind == "score":
        return {str(i): str(level) for i, level in enumerate(question["criteria"])}
    return {name: name for name in question["criteria"]}


def probs_of(answer: dict) -> dict:
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1.0 - answer["noul"]}
    return dict(answer["probabilities"])


def card_ms(scene: str) -> tuple[float, str]:
    """The card's time of the scene's class, from the result files when present."""
    try:
        if scene == "A":
            rows = [r for r in json.loads((K / "results/timing_mac_r10.json").read_text())["rows"]
                    if r["set"] == "three" and r["form"] == "pair Ls64+Lq64 (no sharing) embeds"
                    and r["accel"] == "gpu_f32" and r["handover"] == "direct"]
            return float(rows[0]["median_ms"]), "results/timing_mac_r10.json"
        doc = json.loads((K / "results/timing_mac_image.json").read_text())
        return float(doc["card_columns"]["one_picture_one_question_gpu_f32"]["total_median_ms"]), \
            "results/timing_mac_image.json"
    except (OSError, KeyError, IndexError):
        return CARD_MS[scene], "the model card"


def editor_request(editor: dict) -> dict:
    """The request for the editor's content, under the editor's rules, written out here on their own."""
    text = editor["state"]
    state = None if not text.strip() else text
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
    return {"state": state, "questions": questions}


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

    # (a) scene A against the provider's float32 CPU answers
    expected = json.loads((D / "hub/examples/run_example.expected.json").read_text())
    if "A" in scenes:
        a, exp = scenes["A"], expected["requests"]["text"]
        c(a["response"]["usage"]["input_tokens"] == exp["input_tokens"],
          f"(a) A input_tokens {a['response']['usage']['input_tokens']} = {exp['input_tokens']} (expected file)")
        c(a["route"].get("route") == exp["route"]["route"] and a["route"].get("Ls") == exp["route"].get("Ls")
          and a["route"].get("Lq") == exp["route"].get("Lq"),
          f"(a) A route {a['route'].get('route')} Ls{a['route'].get('Ls')}+Lq{a['route'].get('Lq')} = {exp['route']}")
        for q, e in exp["questions"].items():
            got = probs_of(a["response"]["answers"][q])
            dp = max(abs(got[k] - p) for k, p in zip(e["keys"], e["probabilities"]))
            top = max(got, key=got.get)
            c(dp <= TOL_A and top == e["argmax_key"],
              f"(a) A {q}: max |dp| {dp:.2e} <= {TOL_A:.0e} vs the provider's float32 CPU, most likely {top} = "
              f"{e['argmax_key']}")

    # (a) scene B against the CPU run of the same files and request
    if "B" in scenes:
        b = scenes["B"]
        img = b["request"]["images"][0]
        ref_path = D / "out" / f"cpu_ref_{img['sha256'][:12]}.json"
        if not ref_path.is_file():
            c(False, f"(a) B: no CPU reference {ref_path.name} (scripts/cpu_reference.py)")
        else:
            ref = json.loads(ref_path.read_text())
            c(ref["request_sha256"] == b["request_sha256"],
              f"(a) B request sha256 {b['request_sha256'][:12]} = the CPU reference's")
            c(json.dumps(ref["request"]["questions"], ensure_ascii=False) == b["questions_ordered"]
              and ref["request"]["state"] == b["request"]["state"],
              "(a) B questions and options in the CPU reference's order, the same state")
            c(ref["accelerator"].startswith("cpu"), f"(a) B reference on {ref['accelerator']}")
            for q, ans in b["response"]["answers"].items():
                got, want = probs_of(ans), probs_of(ref["response"]["answers"][q])
                dp = max(abs(got[k] - want[k]) for k in want)
                top, rtop = max(got, key=got.get), max(want, key=want.get)
                c(dp <= TOL_B and top == rtop,
                  f"(a) B {q}: max |dp| {dp:.2e} <= {TOL_B:.0e} vs the Mac CPU run, most likely {top} = {rtop}")
        c(b.get("tiles") == 1, f"(a) B tiles {b.get('tiles')} = 1")
        c(b["route"] == {"route": "row", "L": 512}, f"(a) B route {b['route']} = row L512")
        c(b["response"]["usage"]["input_tokens"] == b.get("tokens"), f"(a) B tokens {b.get('tokens')} recorded")
    log = D / "out/run_example_check_gpu.log"
    c(log.is_file() and "matches run_example.expected.json (gpu)" in log.read_text(),
      "(a) card photo pipeline check: out/run_example_check_gpu.log says matches run_example.expected.json (gpu)")

    # (b) the strings on screen are the rounding of the unrounded values
    for s, rec in scenes.items():
        shown = rec["shown"]
        req_q = rec["request"]["questions"]
        bad = []
        for q, question in req_q.items():
            probs = probs_of(rec["response"]["answers"][q])
            labels = labels_of(question)
            top = max(probs, key=probs.get)
            sq = shown["questions"].get(q)
            if sq is None:
                bad.append(f"{q}: no answer card on screen")
                continue
            if sq["text"] != question["instructions"]:
                bad.append(f"{q} text {sq['text']!r}")
            if sq["answer"] != labels[top] or sq["answer_p"] != f"{probs[top]:.3f}":
                bad.append(f"{q} answer {sq['answer']} {sq['answer_p']} vs {labels[top]} {probs[top]:.3f}")
            for k, p in probs.items():
                if sq["options"].get(k) != f"{p:.3f}":
                    bad.append(f"{q}/{k} {sq['options'].get(k)} vs {p:.3f}")
                if sq["labels"].get(k) != labels[k]:
                    bad.append(f"{q}/{k} label {sq['labels'].get(k)!r} vs {labels[k]!r}")
        if set(shown["questions"]) != set(req_q):
            bad.append(f"answer cards {sorted(shown['questions'])} vs questions {sorted(req_q)}")
        n = len(req_q)
        word = f"{n} question" + ("s" if n != 1 else "")
        word = (f"1 picture · {word}" if rec["request"].get("images") else word)
        ms = rec["ms"]["request_wall"]
        foot = f"{word} · {round(ms)} ms"
        if shown["footer"] != foot:
            bad.append(f"footer {shown['footer']!r} vs {foot!r}")
        if shown["model"] != doc["hud"]["model"] or shown["device"] != doc["hud"]["device"] or \
                shown["pill"] != doc["hud"]["pill"] or shown["status"] != "DONE":
            bad.append(f"HUD {shown['model']!r} {shown['device']!r} {shown['pill']!r} {shown['status']!r}")
        if shown.get("error") or shown.get("hint"):
            bad.append(f"an error or hint line on screen: {shown.get('error')!r} {shown.get('hint')!r}")
        cut = rec["layout"].get("cut", [])
        if cut:
            bad.append(f"drawn past its box: {cut[:3]}")
        bars = rec["layout"].get("bars")
        if bars is None:
            bad.append("no bar widths in the layout")
        else:
            for q in req_q:
                for k, p in probs_of(rec["response"]["answers"][q]).items():
                    w = bars.get(q, {}).get(k)
                    if w is None or abs(w - p) > 0.01:
                        bad.append(f"{q}/{k} bar drawn at {w} of its track for p {p:.3f}")
        strings = sum(1 + len(x["options"]) for x in shown["questions"].values()) + 1
        c(not bad, f"(b) {s}: {strings} numbers on screen = the rounding of the run's values (footer {shown['footer']!r})"
          + ("" if not bad else ": " + "; ".join(bad[:6])))

    # (c) the request time within 2 x the card's class
    for s in ("A", "B"):
        if s not in scenes:
            continue
        ms = scenes[s]["ms"]["request_wall"]
        ref, src = card_ms(s)
        c(ms <= SLOW * ref, f"(c) {s} request_wall {ms:.3f} ms <= {SLOW:g} x {ref} = {SLOW * ref:.3f} ms ({src})")

    # (e) the editor = the request
    for s, rec in scenes.items():
        ed, shown_ed = rec["editor"], rec["editor_shown"]
        same = ed == shown_ed
        want = editor_request(ed)
        got = {k: v for k, v in rec["request"].items() if k != "images"}
        ordered = json.dumps(got, ensure_ascii=False) == json.dumps(want, ensure_ascii=False)
        photo_ok = ((ed.get("photo") or {}).get("id") == rec["request"]["images"][0]["sha256"][:12]
                    if rec["request"].get("images") else ed.get("photo") is None)
        typed = [st for st in rec.get("steps", []) if st["what"] == "type"]
        typed_ok = all(st.get("value", "").endswith(st["text"]) for st in typed)
        c(same and ordered and photo_ok and typed_ok and bool(typed),
          f"(e) {s}: the editor sent = the editor read back after Decide ({same}); the request = the editor under its "
          f"rules, in order ({ordered}); the photo ({photo_ok}); {len(typed)} typed field(s) hold the typed text "
          f"({typed_ok})")
    if "A" in scenes:
        ticket = json.loads((D / "app/examples.json").read_text())["examples"][0]
        c(scenes["A"]["editor"]["state"] == ticket["state"] and
          any(st["what"] == "type" and st.get("field") == "state" and st["text"] == ticket["state"]
              for st in scenes["A"].get("steps", [])),
          "(e) A: the state is the card's ticket, typed in by autoplay")

    # the accelerator and the measurement windows
    c(doc.get("accelerator") == "gpu fp32 (GpuOptions enforce_f32)", f"accelerator {doc.get('accelerator')!r}")
    comp = doc.get("compile", [])
    c(bool(comp) and all(v.get("fully_accelerated") is True for v in comp),
      "every compiled file fully accelerated: " + ", ".join(f"{v['file']} {v.get('fully_accelerated')}" for v in comp))
    late = [f"{s}: {[x['file'] for x in rec.get('compile', [])]}" for s, rec in scenes.items() if rec.get("compile")]
    c(not late, "no graph compiled between READY and the answers" + ("" if not late else f": {late}"))
    lock = doc.get("lock", {})
    if lock.get("file"):
        for s, rec in scenes.items():
            label = f"{WINDOW_LABEL}-{tag}-{s}"
            c(rec.get("lock_at_decide", "").startswith(label),
              f"{s} decided inside its own measurement window (lock {rec.get('lock_at_decide', '')[:60]!r})")
        foreign = lock.get("foreign", [])
        c(not foreign, f"no other window during compile and warm-up ({len(foreign)} seen)"
          + ("" if not foreign else f": {foreign[:2]}"))
    else:
        c(None, "measurement windows: no lock file configured (D1_LOCK_FILE unset)")

    print(f"# check_run {tag}: {doc.get('accelerator')}, started {doc.get('started_at')}")
    print("\n".join(c.lines))
    print(f"CHECK_RUN {tag}: {'PASS' if not c.failed else f'FAIL ({c.failed} failed)'}")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
