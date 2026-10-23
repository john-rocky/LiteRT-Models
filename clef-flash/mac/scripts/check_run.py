#!/usr/bin/env python3
"""Check one autoplay run JSON (app/mac_demo.py --scenes M,D) against the lines set before the take. Exit 1 on any
failure.

    venv-demo/bin/python -B scripts/check_run.py out/<tag>.run.json

(a) Each scene against out/cpu_ref_<tag>.json (the same ../hub/ files and the scene's own request on the Mac CPU,
    scripts/cpu_reference.py --from-run): the same request (SHA-256), the same tokens and row graph; on every question
    the same most likely option and every probability within 0.02 (the largest difference is printed). The response is
    the one of the app's unrounded probabilities. And each scene's command was handled as the answers expected for it
    in eval/dev.json (its resolution, targets and changes) would handle it, on the scene's own state (the take's line,
    set before the first take): the same way of handling, with "execute now" and "execute available parts now" as one
    (the app sets the same devices for both), the same changes the app sets, and for a clarifying question the same
    devices it asks between. How many of the questions got the expected answer is printed (INFO), not a line: "Close
    Draft." gets 12 of 15 on dev (both drafts answered as targets, draft_1 as closing), and the app still only asks.
(b) Every string on screen, as read back from the page (`shown`): per question card, the answer = the label of the most
    likely option and f"{p:.3f}", each option's probability = f"{p:.3f}" of its unrounded value; the way of handling,
    its probability and its line (app/mac_demo.py answer_payload of the run's own values); per device row, the setting
    before = the request's state, after = the decided change, the probability = the target card's "yes"; the time =
    "round(request_wall) ms"; the note names the questions, the tokens and the row graph; the HUD; no error line;
    nothing drawn past its box (layout.cut empty).
(c) request_wall at most 2 x the 2,048-token row graph's own median on this Mac (2,707.0 ms, Metal float32, one call
    of the graph alone).
(d) The Mac: "Apply to this Mac" was on; scene M's changes (Calculator quit, both drafts closed) were each set and read
    back, the TV app and the music untouched, the rows marked; the Mac read at the end of the scene holds them. Scene
    D (a clarifying question) changed nothing, and its two drafts were open from before the command was typed to the
    end of the scene.
(e) The editor = the request: the state is the Mac read right before Decide with the typed command as its "request",
    the questions built from that state (app/form.py); the editor's content as Decide sent it equals the content read
    back from the page after Decide was pressed; key order included. The command was typed by autoplay, all of it,
    before Decide was pressed.
The accelerator is the GPU at float32 with the row graph fully accelerated, and no graph compiled between READY and the
answers. When the run read a measurement lock (MAC_LOCK_FILE), the lock held the same line from the compile to the end
of the scenes and that line was not empty; without a lock this line is SKIP.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(H / "app"))
sys.dont_write_bytecode = True
import editor as E  # noqa: E402
import form as F  # noqa: E402
import mac_demo as M  # noqa: E402

LINE, SLOW = 0.02, 2.0
GRAPH_L2048_MS = 2707.0   # the 2,048-token row graph alone, median of 20 calls (Metal float32, Apple M4 Max)
DEV_OF = {"quit_apps": "d02", "close_draft": "d04"}


class Checks:
    def __init__(self):
        self.lines, self.failed = [], 0

    def __call__(self, ok, text: str):
        word = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        self.lines.append(f"{word}  {text}")
        self.failed += 1 if ok is False else 0

    def info(self, text: str):   # printed, not a line
        self.lines.append(f"INFO  {text}")


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
    dev = {x["id"]: x for x in json.loads((H / "eval/dev.json").read_text())["commands"]}
    ref_path = H / "out" / f"cpu_ref_{tag}.json"
    refs = {x["id"]: x for x in json.loads(ref_path.read_text())["commands"]} if ref_path.is_file() else None

    for s, rec in scenes.items():
        state = rec["request"]["state"]
        # (a) the CPU run of the same request, and the command's expected answers
        if refs is None:
            c(False, f"(a) {s}: no CPU reference {ref_path.name} (scripts/cpu_reference.py --from-run)")
        else:
            ref = refs[f"{tag}/{s}"]
            c(ref["request_sha256"] == rec["request_sha256"],
              f"(a) {s} request sha256 {rec['request_sha256'][:12]} = the CPU reference's")
            c((rec["tokens"], rec["length"]) == (ref["tokens"], ref["length"]),
              f"(a) {s} {rec['tokens']} tokens on L{rec['length']} = the CPU reference's")
            worst, bad = 0.0, []
            for q, got in rec["probs"].items():
                want = ref["probs"][q]
                dp = max(abs(got[k] - want[k]) for k in want)
                worst = max(worst, dp)
                if dp > LINE or max(got, key=got.get) != max(want, key=want.get):
                    bad.append(f"{q} dp {dp:.2e} {max(got, key=got.get)} vs {max(want, key=want.get)}")
            c(not bad, f"(a) {s}: {len(rec['probs'])} questions, most likely = the CPU run's, max |dp| {worst:.2e} "
              f"<= {LINE}" + ("" if not bad else ": " + "; ".join(bad[:4])))
        gold = F.gold_options(state, dev[DEV_OF[rec["command_id"]]]["gold"])
        picked = {q: max(p, key=p.get) for q, p in rec["probs"].items()}
        got, want = F.decision(state, picked), F.decision(state, gold)
        same_way = got["resolution"] == want["resolution"] or {got["resolution"], want["resolution"]} <= set(F.CHANGING)
        changes = {k: sorted(f"{x['device']}.{x['prop']}={json.dumps(x['to'])}" for x in d["changes"])
                   for k, d in (("got", got), ("want", want))}
        way = got["resolution"] if got["resolution"] == want["resolution"] else f"{got['resolution']} for " \
            f"{want['resolution']}" + (" (one way here: the app sets the same devices for both)" if same_way else "")
        c(same_way and changes["got"] == changes["want"] and got["candidates"] == want["candidates"],
          f"(a) {s} ({rec['command_id']}): handled as expected: the way {way}, the changes the app sets "
          f"{changes['got'] or 'none'}" + (
              f", the devices it asks between {got['candidates']}" if got["resolution"] == "clarify" else ""))
        miss = [f"{q} {picked.get(q)} vs {g}" for q, g in gold.items() if picked.get(q) != g]
        c.info(f"(a) {s}: {len(gold) - len(miss)}/{len(gold)} questions with the expected answer"
               + ("" if not miss else ": " + "; ".join(miss)))
        c(rec.get("response_is_its_probs") is True, f"(a) {s}: the response is the one of its unrounded probabilities")

        # (b) the strings on screen are the run's values, rounded
        want = M.answer_payload(rec["request"], rec["probs"], rec["ms"]["request_wall"], rec["tokens"], rec["length"])
        shown = rec["shown"]
        bad = []
        for q in want["questions"]:
            card = shown["questions"].get(q["name"])
            if card is None:
                bad.append(f"{q['name']}: no card")
                continue
            if (card["answer"], card["answer_p"], card["text"]) != (q["answer"], q["answer_p"], q["text"]):
                bad.append(f"{q['name']} card {card['answer']} {card['answer_p']}")
            for o in q["options"]:
                if card["options"].get(o["key"]) != o["text"] or card["labels"].get(o["key"]) != o["label"]:
                    bad.append(f"{q['name']}/{o['key']} {card['options'].get(o['key'])}")
        way = shown["mac"]["way"]
        if (way["word"], way["p"], way["detail"]) != (want["way"]["word"], want["way"]["p"], want["way"]["detail"]):
            bad.append(f"way {way} vs {want['way']}")
        for r in want["rows"]:
            row = shown["mac"]["rows"].get(r["id"], {})
            if (row.get("before"), row.get("after"), row.get("p"), row.get("target")) != (
                    r["before"], r["after"], r["target_p"] or "", r["target"]):
                bad.append(f"row {r['id']} {row}")
        ms = rec["ms"]["request_wall"]
        if shown["time"] != f"{round(ms):,} ms":
            bad.append(f"time {shown['time']!r}")
        n = len(rec["request"]["questions"])
        note = f"{n} questions · {rec['tokens']:,} tokens · {rec['length']:,}-token row graph · "
        if not shown["note"].startswith(note):
            bad.append(f"note {shown['note']!r}")
        if (shown["model"], shown["device"], shown["pill"], shown["status"]) != (
                doc["hud"]["model"], doc["hud"]["device"], doc["hud"]["pill"], "DONE"):
            bad.append(f"HUD {shown['model']!r} {shown['device']!r} {shown['pill']!r} {shown['status']!r}")
        if shown.get("error"):
            bad.append(f"an error line: {shown['error']!r}")
        if rec["layout"].get("cut"):
            bad.append(f"drawn past its box: {rec['layout']['cut'][:3]}")
        c(not bad, f"(b) {s}: the screen = the run's values ({way['word']} {way['p']} | {way['detail']}; time "
          f"{shown['time']})" + ("" if not bad else ": " + "; ".join(bad[:6])))

        # (c) the request time
        c(ms <= SLOW * GRAPH_L2048_MS, f"(c) {s} request_wall {ms:.1f} ms <= {SLOW:g} x {GRAPH_L2048_MS} ms")

        # (d) the Mac
        ap = rec["apply"]
        at_done = rec.get("timeline", {}).get("mac_at_done", {})
        if rec["command_id"] == "quit_apps":
            res = {(r["device"], r["prop"]): r for r in ap.get("results", [])}
            want_ch = {("calculator", "running"): False, ("draft_1", "open"): False, ("draft_2", "open"): False}
            ok = ap.get("on") is True and set(res) == set(want_ch) and all(
                res[k]["ok"] and res[k].get("read_back") == v for k, v in want_ch.items())
            held = (at_done.get("calculator", {}).get("running") is False
                    and at_done.get("draft_1", {}).get("open") is False and at_done.get("draft_2", {}).get("open") is False
                    and at_done.get("tv", {}).get("running") is True and at_done.get("music", {}).get("playing") is True)
            marks = {d: rec["shown_after_apply"]["mac"]["rows"][d]["mark"] for d in ("calculator", "draft_1", "draft_2")}
            c(ok and held and set(marks.values()) == {"✓"},
              f"(d) {s}: Calculator quit and both drafts closed, each read back ("
              + ", ".join(f"{k[0]} {res[k]['seconds']} s" for k in res) + f"; {ap.get('seconds')} s in all, not in the "
              f"request time); rows {marks}; at the end the TV app runs and the music plays ({held})")
        else:
            opened = rec.get("timeline", {}).get("drafts_opened") or []
            ok = not ap.get("results") and all(x["ok"] for x in opened) and len(opened) == 2 and \
                at_done.get("draft_1", {}).get("open") is True and at_done.get("draft_2", {}).get("open") is True
            c(ok, f"(d) {s}: nothing set by the answer; both drafts opened before the command "
              f"({[x['read_back'] for x in opened]}) and open at the end")

        # (e) the editor = the request
        ed, shown_ed = rec["editor"], rec["editor_shown"]
        same = {k: ed[k] for k in ("state", "questions", "command")} == {k: shown_ed[k] for k in
                                                                         ("state", "questions", "command")}
        built = E.build_request(ed)
        ordered = json.dumps(rec["request"], ensure_ascii=False) == json.dumps(built, ensure_ascii=False)
        auto_q = json.dumps(rec["request"]["questions"], ensure_ascii=False) == json.dumps(
            F.questions_of(state), ensure_ascii=False)
        typed = [st for st in rec.get("steps", []) if st["what"] == "type"]   # the last one: Decide was pressed on it
        cmd = rec["request"]["state"].get("request")
        typed_ok = bool(typed) and typed[-1].get("value") == cmd == ed.get("command")
        mr = rec.get("mac_read", {})
        live_ok = rec.get("live") is True and mr.get("text") == ed["state"]
        c(same and ordered and auto_q and typed_ok and live_ok,
          f"(e) {s}: sent = read back ({same}); request = the editor under its rules ({ordered}); questions built from "
          f"the state ({auto_q}); the command typed ({typed_ok}); the state = the Mac read right before Decide "
          f"({live_ok}): {json.dumps({d['id']: M.device_now(d) for d in F.devices_of(state)})}")

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
        c(None, "measurement window: no lock file configured (MAC_LOCK_FILE unset)")

    print(f"# check_run {tag}: {doc.get('accelerator')}, started {doc.get('started_at')}")
    print("\n".join(c.lines))
    print(f"CHECK_RUN {tag}: {'PASS' if not c.failed else f'FAIL ({c.failed} failed)'}")
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())
