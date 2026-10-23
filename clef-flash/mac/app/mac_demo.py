#!/usr/bin/env python3
"""clef-flash mac: one command, seven devices of this Mac. The Mac's devices (the TV app, Calculator, two TextEdit
documents both named Draft, a music loop, the sound and the screen lock, read from the Mac as JSON) and a command you
type ("Quit every app except the TV app, and keep the music playing.") go to the model in one call, with the questions
of the Decision Index's home appliance benchmark: how to handle the command (execute now, execute what is available, schedule
it, save a rule, ask a clarifying question, change nothing because it is already so, reject it as unsupported, block it
under the policy, answer a question about the state), for every device whether it is a justified target, and what each
device's setting is right after correct handling. Every question comes back as one of its options with a probability
for each option. With "execute", the app sets the devices whose answered setting differs from the setting now; with
"ask" it shows the devices the command names; anything else changes nothing. The model is
litert-community/clef-flash-LiteRT (Cloudflare's clef-flash on LiteRT), run on this Mac's GPU through Metal at float32
precision, or on the CPU. It never writes text, and its answers are never run as commands: app/actions.py maps each
decided change to a fixed setting, and the screen is never locked (shown only).

    venv-demo/bin/python app/mac_demo.py                          # the app (puts the Mac back when its window closes)
    venv-demo/bin/python app/mac_demo.py --restore                # put the Mac back as the last run found it
    venv-demo/bin/python app/mac_demo.py --tag m1                 # the app; every Decide recorded in
                                                                  #   out/m1_manual.run.json; scripted input accepted
                                                                  #   from scripts/drive_editor.py
    venv-demo/bin/python app/mac_demo.py --scenes M,D --tag t1    # autoplay for a recording (scripts/take.sh)

The screen (app/ui): THIS MAC, one row per device (its setting now; after Decide the setting it gets, the probability
that it is a target, and whether the Mac took the setting), the answer's way of handling in large letters, and two
tabs. COMMAND: the command line (the state's "request"), the bundled commands, "Open the demo's devices", "Apply to
this Mac", Decide, and the time the request took. STATE & QUESTIONS: the state as JSON and one card per question (name,
type, the question, the options; + Add question, − removes). Decide sends the state and the cards as one request
(app/editor.py: the rules); the answers come back all at once.

The state is read from the Mac when the app starts, when you press "Read this Mac", and again right before each
Decide, so the request holds the Mac as it is; "device_context" is the first target of the last command that had one.
Once you edit the state text yourself, your text is used as you wrote it until "Read this Mac". The questions follow
the state (app/form.py builds them from its devices and its request) until you edit a card; then your cards are sent
as you wrote them until "Rebuild questions". A bundled command's button writes only the command.

The app reads every device when it starts and writes it to out/mac_before.json; it puts the Mac back when its window
closes, on Ctrl-C or a TERM signal, and with --restore (after a crash): the apps running as they were, the drafts
closed, the music stopped, the volume and its mute as they were. It refuses to start while that file holds a state not
put back yet.

The host is the model repository's own Python host (../hub/host/clef_litert), imported as downloaded. The app holds
one row graph, the one of --length (default 2,048 tokens: a command with these questions is about 1,850 tokens): it
compiles it at start and warms it up with one request of other content, so a Decide never waits for a compile. A
request longer than the row graph is refused (the host's RequestTooLong), never cut. The time on screen is
time.perf_counter() right before and after clef.decide(request): never a compile, never the Mac's settings. All model
calls run on one thread.

Autoplay (--scenes, for a recording) and scripted runs (--tag): the window stays hidden while the graph compiles and
appears at READY, floating, mouse-transparent, top left of the screen. Autoplay makes the app the active one and, with
--backdrop, lays a plain backdrop over the desktop, behind every window. It drives the page itself (typing into the
command line, tapping the buttons: the same handlers and the same Decide as a click), on triggers from the take script
in --trigger-dir: <tag>.READY (written), then per scene <tag>.ARM_<s> -> <tag>.ARMED_<s>, <tag>.GO_<s> -> the scene ->
<tag>.DONE_<s>, then <tag>.run.json; --stay keeps the window until <tag>.QUIT. Scene M: the first command of
commands.json "take" typed, Decide, applied. Scene D: both drafts opened again, the second command typed, Decide.

Measuring next to other GPU work is optional: MAC_WAIT_CMD (a command prefix run as `<prefix> -- <command>`; the app
runs it before compiling and waits for it) and MAC_LOCK_FILE (a lock file whose content is recorded during compile and
warm-up and at each Decide). Unset, the app waits for nothing and reads no lock.

--ui-only loads no model: the answers read "0.000" and "000 ms" and pick the first option of every question (window,
settings and recorder tests). --accel cpu runs on XNNPACK with 8 threads. Exit 4 (one line on stderr) when a file the
app needs is not in ../hub/; exit 5 when out/mac_before.json was not put back.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # the model repository's host/ stays byte for byte as downloaded

import argparse
import ctypes
import hashlib
import json
import os
import platform
import queue
import shlex
import signal
import subprocess
import threading
import time
import traceback
from importlib import metadata
from pathlib import Path

APP = Path(__file__).resolve().parent
H = APP.parent
UI = APP / "ui"
sys.path.insert(0, str(APP))

import actions as A  # noqa: E402
import editor as E  # noqa: E402
import form as F  # noqa: E402

WIDTH, HEIGHT = 1060, 660
MARGIN = 36
TITLE = "clef-flash mac"
LENGTHS = (256, 512, 1024, 2048, 4096)
DEFAULT_LENGTH = 2048
MARK_TIMEOUT_S = 5.0
HOLD_S = 1.6                  # autoplay: a scene ends this long after its answers are on screen
TYPE_CPS = 16.0
PAUSE_BEFORE_TYPING_S = 0.2
PAUSE_BEFORE_DECIDE_S = 0.2
BACKDROP = (0x5B / 255, 0x6B / 255, 0x7C / 255)   # a plain blue-grey, the same in light and dark
WAIT_CMD = shlex.split(os.environ.get("MAC_WAIT_CMD", ""))
LOCK_FILE = Path(os.environ["MAC_LOCK_FILE"]) if os.environ.get("MAC_LOCK_FILE") else None
NOTE = "timed by the app"
WARM_STATE = "The kettle stopped heating after a week and the light on its base no longer turns on."
WARM_QUESTIONS = {"faulty": {"type": "noul", "instructions": "The product is faulty."}}
SPAN = ("time.perf_counter() right before and after clef.decide(request): the tokenizer, the row graph's inputs "
        "(embedding rows, positions), the row graph call with its input writes and output read, the joint head and the "
        "response; not a compile, not the screen update, not the Mac's settings")
HOST_FILES = ("__init__.py", "_erf.py", "errors.py", "head.py", "host.py", "pixels.py", "positions.py", "request.py",
              "response.py", "runtime.py", "tables.py")
# the way of handling, as the screen shows it
WAYS = {"execute": "EXECUTE", "execute_partial": "EXECUTE WHAT IS AVAILABLE", "schedule": "SCHEDULED",
        "rule": "RULE SAVED", "clarify": "ASK", "no_change": "NO CHANGE", "unsupported": "REJECT",
        "blocked": "BLOCKED", "query": "ANSWER"}


def needed_files(length: int) -> list[str]:
    """The ../hub/ files the app loads for a row graph of `length` tokens (the host's FOLDER layout; no tower)."""
    return ([f"host/clef_litert/{f}" for f in HOST_FILES]
            + [f"clef-flash_rowprefill_embeds_L{length}_fp16fc.tflite"]
            + ["tables/embed_tokens.safetensors", "tables/lm_head.safetensors", "head/joint_head.safetensors",
               "head/joint_head_config.json", "tokenizer/tokenizer.json"])


# ----------------------------------------------------------------------------------------------- small helpers


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_lock() -> str:
    if LOCK_FILE is None:
        return ""
    try:
        return LOCK_FILE.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def sh(*cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:   # informational only
        return f"unavailable: {e}"


def version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "not installed"


def phys_footprint() -> int | None:
    """This process's phys_footprint (proc_pid_rusage RUSAGE_INFO_V2), the number Activity Monitor calls Memory."""
    class Info(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
            "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time", "child_system_time",
            "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime",
            "diskio_bytesread", "diskio_byteswritten")]
    try:
        info = Info()
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        if lib.proc_pid_rusage(os.getpid(), 2, ctypes.byref(info)) != 0:
            return None
        return int(info.phys_footprint)
    except Exception:
        return None


def as_int(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:
        return None


def relative(p) -> str:
    """A path as the run records keep it: relative to this app's folder (venv-demo/bin/python, ../hub)."""
    try:
        return os.path.relpath(p, H)
    except ValueError:
        return str(p)


def environment(hub: Path) -> dict:
    sw = dict(line.split(":", 1) for line in sh("/usr/bin/sw_vers").splitlines() if ":" in line)
    return {
        "python": sys.version.split()[0],
        "executable": relative(sys.executable),
        "packages": {d: version(d) for d in ("ai-edge-litert", "pywebview", "pyobjc-core", "pyobjc-framework-WebKit",
                                             "numpy", "tokenizers")},
        "macos": {k.strip(): v.strip() for k, v in sw.items()},
        "platform": platform.platform(),
        "chip": sh("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": as_int(sh("/usr/sbin/sysctl", "-n", "hw.memsize")),
        "hub": relative(hub),
    }


def canonical_sha256(request: dict) -> str:
    """sha256 of the request as canonical JSON (sorted keys, no spaces)."""
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
                          ).hexdigest()


def write_json(path: Path, doc) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    tmp.rename(path)


# ----------------------------------------------------------------------------------------------- the Mac's state


BEFORE = H / "out" / "mac_before.json"


def restore_before(path: Path = BEFORE) -> tuple[int, list]:
    """Put the Mac back to the snapshot in `path` (mac_before.json): (exit code, results). Marks the file restored."""
    if not path.is_file():
        return 0, [{"note": f"{path.name} not found: nothing to put back"}]
    doc = json.loads(path.read_text())
    if doc.get("restored_at"):
        return 0, [{"note": f"already put back at {doc['restored_at']}"}]
    results = A.restore(doc["settings"])
    doc["restored_at"], doc["restore"] = now_iso(), results
    write_json(path, doc)
    return (0 if all(r.get("ok") for r in results) else 1), results


def value_word(prop: str, v) -> str:
    """A setting as a row shows it: Running / Not running, Open / Closed, Playing / Off, 25, Locked / Unlocked."""
    if v is None:
        return "?"
    words = {"running": ("Running", "Not running"), "open": ("Open", "Closed"), "playing": ("Playing", "Off"),
             "locked": ("Locked", "Unlocked"), "muted": ("Muted", "Not muted"), "available": ("Available", "Not available")}
    if isinstance(v, bool) and prop in words:
        return words[prop][0 if v else 1]
    if isinstance(v, bool):
        return "Yes" if v else "No"
    return str(v)


def device_now(d: dict) -> str:
    """A device's settings now, as its row shows them: the settable ones, the volume with its mute."""
    props = F.settable(d)
    if d.get("kind") == "speaker" and "volume" in d:
        return f"{d['volume']}" + (" (muted)" if d.get("muted") is True else "")
    if not props:
        return "—"
    return " · ".join(value_word(p, d.get(p)) for p in props)


# ----------------------------------------------------------------------------------------------- what is shown


def option_rows(question: dict) -> list[dict]:
    """[{key, label}] in the request's order: an option's label is its description when it has one (yes / no: the
    descriptions of yes and no, else "yes" and "no"), else its name; a score level is its text."""
    kind = question.get("type", "choice")
    criteria = question.get("criteria")
    if kind == "noul":
        c = criteria or {}
        return [{"key": "true", "label": str(c.get("true") or "yes")}, {"key": "false", "label": str(c.get("false") or "no")}]
    if kind == "score":
        return [{"key": str(i), "label": str(level)} for i, level in enumerate(criteria)]
    return [{"key": str(name), "label": str(name if desc is None else desc)} for name, desc in criteria.items()]


def detail_of(state, dec: dict) -> str:
    """The line under the way of handling: what changes, which devices to ask between, what was asked."""
    devices = {d["id"]: d for d in F.devices_of(state)}
    way = dec.get("resolution")
    if way in F.CHANGING:
        if dec["changes"]:
            return " · ".join(A.describe(c["device"], c["prop"], c["to"]) if c["device"] in A.NAMES
                              else f"{c['device']}.{c['prop']} → {json.dumps(c['to'])}" for c in dec["changes"])
        return "Nothing to change: no target's setting differs from now."
    if way == "clarify":
        cands = dec.get("candidates") or []
        if len(cands) > 1:
            name = devices[cands[0]].get("name", cands[0])
            return f"Which {name}? {' or '.join(cands)}"
        if len(cands) == 1:
            return f"Did you mean {cands[0]}?"
        return "Which device? None of this Mac's devices is named in the command."
    targets = ", ".join(dec["targets"]) or "no device"
    if way == "schedule":
        return f"For later: {targets}. This demo keeps no schedule."
    if way == "rule":
        return f"A rule for {targets}. This demo keeps no rules."
    if way == "no_change":
        return f"Already so: {targets}."
    if way == "unsupported":
        return f"Not supported: {targets}."
    if way == "blocked":
        policy = state.get("policy") if isinstance(state, dict) else None
        mode = policy.get("security_mode") if isinstance(policy, dict) else None
        return f"Blocked by this Mac's policy{f' ({mode})' if mode else ''}: {targets}."
    if way == "query":
        parts = [f"{i}: {device_now(devices[i])}" for i in dec["targets"] if i in devices]
        return "; ".join(parts) or "No device asked about."
    return ""


def answer_payload(request: dict, probs: dict | None, ms: float | None, tokens: int | None,
                   length: int | None) -> dict:
    """Every string the answers show, rounded here: probabilities f"{p:.3f}", the time round(ms); the way of handling;
    per device of the state its setting now, the setting it gets, and the probability that it is a target."""
    qs, picked = [], {}
    state = request.get("state")
    for name, q in request["questions"].items():
        rows = option_rows(q)
        if probs is None:   # --ui-only placeholders: the first option
            p = {o["key"]: 0.0 for o in rows}
            top = rows[0]["key"]
        else:
            p = {o["key"]: probs[name][o["key"]] for o in rows}
            top = max(p, key=p.get)
        picked[name] = top
        labels = {o["key"]: o["label"] for o in rows}
        qs.append({"name": name, "text": q["instructions"], "answer": labels[top], "answer_p": f"{p[top]:.3f}",
                   "top": top, "options": [{"key": o["key"], "label": o["label"], "text": f"{p[o['key']]:.3f}",
                                            "width": max(0.0, min(1.0, p[o["key"]]))} for o in rows]})
    dec = F.decision(state, picked)
    by_name = {q["name"]: q for q in qs}
    way = None
    if "resolution" in by_name and dec["resolution"] in F.RESOLUTIONS:
        way = {"key": dec["resolution"], "word": WAYS[dec["resolution"]], "p": by_name["resolution"]["answer_p"],
               "text": F.RESOLUTIONS[dec["resolution"]], "detail": detail_of(state, dec)}
    rows = []
    for d in F.devices_of(state):
        tq = by_name.get(f"target_member:{d['id']}")
        tp = None
        if tq is not None:
            tp = next((o["text"] for o in tq["options"] if o["key"] == "yes"), None)
        ch = [c for c in dec["changes"] if c["device"] == d["id"]]
        rows.append({"id": d["id"], "name": A.NAMES.get(d["id"], str(d.get("name", d["id"]))),
                     "kind": d.get("kind"), "before": device_now(d),
                     "after": " · ".join(value_word(c["prop"], c["to"]) for c in ch) if ch else "",
                     "target": d["id"] in dec["targets"] and dec["resolution"] != "clarify", "target_p": tp,
                     "candidate": d["id"] in dec.get("candidates", []),
                     "shown_only": any((c["device"], c["prop"]) in A.SHOWN_ONLY for c in ch)})
    n = len(request["questions"])
    time_text = f"{round(ms):,} ms" if ms is not None else "000 ms"
    where = f"{n} question{'s' if n != 1 else ''}" + ("" if tokens is None else
                                                      f" · {tokens:,} tokens · {length:,}-token row graph")
    return {"questions": qs, "time": time_text, "note": f"{where} · {NOTE}", "picked": picked, "decision": dec,
            "way": way, "rows": rows}


def html_page() -> str:
    page = (UI / "index.html").read_text()
    page = page.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{(UI / 'style.css').read_text()}</style>")
    return page.replace('<script src="app.js"></script>', f"<script>\n{(UI / 'app.js').read_text()}</script>")


# ----------------------------------------------------------------------------------------------- the page


class Page:
    """The pywebview window and the page's MAC object."""

    def __init__(self, window):
        self.w = window

    def js(self, script: str):
        return self.w.evaluate_js(script)

    def call(self, fn: str, *args):
        return self.js(f"MAC.{fn}({', '.join(json.dumps(a, ensure_ascii=False) for a in args)})")

    def wait_mark(self, name: str, timeout: float = MARK_TIMEOUT_S, required: bool = True) -> dict:
        """The step's paint times in epoch seconds: raf1 (the frame that paints it), raf2 (the frame after), and what
        the step recorded with them. A covered or hidden window paints nothing: required (autoplay) raises, else
        {"painted": False} comes back and the app goes on."""
        end = time.time() + timeout
        while time.time() < end:
            m = self.js(f"MAC.mark({json.dumps(name)})")
            if m:
                return {k: (float(v) / 1000.0 if k in ("raf1", "raf2", "start", "end") else v) for k, v in m.items()}
            time.sleep(0.004)
        if required:
            raise TimeoutError(f"the page did not paint step {name!r} within {timeout} s (window hidden?)")
        return {"painted": False}


def on_main(fn, timeout=10.0):
    """Run fn on the AppKit main thread and return its result."""
    from PyObjCTools import AppHelper

    box, done = {}, threading.Event()

    def run():
        try:
            box["value"] = fn()
        except Exception as e:   # re-raised in the caller
            box["error"] = e
        finally:
            done.set()

    AppHelper.callAfter(run)
    if not done.wait(timeout):
        raise TimeoutError("the main thread did not answer")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def window_facts(w, screen) -> dict:
    """The window and its content in screen points, top-left origin (a display recording's pixels at scale 1)."""
    f, sf = w.frame(), screen.frame()
    c = w.contentRectForFrameRect_(f)
    scale = float(screen.backingScaleFactor())
    top = sf.origin.y + sf.size.height
    return {"id": int(w.windowNumber()), "title": str(w.title()), "pid": os.getpid(),
            "frame_topleft_pt": [f.origin.x - sf.origin.x, top - (f.origin.y + f.size.height), f.size.width,
                                 f.size.height],
            "content_topleft_pt": [c.origin.x - sf.origin.x, top - (c.origin.y + c.size.height), c.size.width,
                                   c.size.height],
            "scale": scale, "px": [round(c.size.width * scale), round(c.size.height * scale)],
            "screen_pt": [sf.origin.x, sf.origin.y, sf.size.width, sf.size.height],
            "style_mask": int(w.styleMask()), "level": int(w.level()), "ignores_mouse": bool(w.ignoresMouseEvents())}


def place_for_recording(window, activate: bool) -> dict:
    """A titled window, floating, mouse-transparent, top left of its screen's visible area; with `activate` the app
    becomes the active one (its menus in the menu bar)."""
    import AppKit

    def fn():
        w = window.native
        w.setLevel_(AppKit.NSFloatingWindowLevel)
        w.setIgnoresMouseEvents_(True)
        screen = w.screen() or AppKit.NSScreen.mainScreen()
        vf = screen.visibleFrame()
        frame = w.frameRectForContentRect_(AppKit.NSMakeRect(0, 0, WIDTH, HEIGHT))
        x = vf.origin.x + MARGIN
        top = vf.origin.y + vf.size.height
        w.setFrame_display_(AppKit.NSMakeRect(x, top - frame.size.height - MARGIN, frame.size.width,
                                              frame.size.height), True)
        if activate:
            AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
            w.makeKeyAndOrderFront_(None)
        else:
            w.orderFrontRegardless()
        return window_facts(w, screen)

    return on_main(fn)


def place_for_editing(window) -> dict:
    """A normal window, WIDTH x HEIGHT pt (less on a smaller screen), centred, in front with the keyboard; light or
    dark with the Mac."""
    import AppKit

    def fn():
        w = window.native
        screen = w.screen() or AppKit.NSScreen.mainScreen()
        vf = screen.visibleFrame()
        frame = w.frameRectForContentRect_(AppKit.NSMakeRect(0, 0, WIDTH, HEIGHT))
        width = min(frame.size.width, vf.size.width - 2 * MARGIN)
        h = min(frame.size.height, vf.size.height - 2 * MARGIN)
        x = vf.origin.x + (vf.size.width - width) / 2
        y = vf.origin.y + (vf.size.height - h) / 2
        w.setFrame_display_(AppKit.NSMakeRect(x, y, width, h), True)
        AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        w.makeKeyAndOrderFront_(None)
        return window_facts(w, screen)

    return on_main(fn)


def show_backdrop() -> dict:
    """A borderless window of one plain colour over the whole screen, just above the desktop's icons and below every
    window (for a recording: the desktop picture and icons stay out of it, and nothing is changed to hide them)."""
    import AppKit
    import Quartz

    def fn():
        screen = AppKit.NSScreen.mainScreen()
        b = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            screen.frame(), AppKit.NSWindowStyleMaskBorderless, AppKit.NSBackingStoreBuffered, False)
        b.setBackgroundColor_(AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(*BACKDROP, 1.0))
        b.setLevel_(Quartz.CGWindowLevelForKey(Quartz.kCGDesktopIconWindowLevelKey) + 1)
        b.setIgnoresMouseEvents_(True)
        b.setHasShadow_(False)
        b.setReleasedWhenClosed_(False)
        b.setCollectionBehavior_(AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                                 | AppKit.NSWindowCollectionBehaviorStationary)
        b.orderFrontRegardless()
        show_backdrop.window = b   # kept alive
        return {"id": int(b.windowNumber()), "level": int(b.level()), "rgb": [round(c * 255) for c in BACKDROP]}

    return on_main(fn)


# ----------------------------------------------------------------------------------------------- the model thread


class ModelThread(threading.Thread):
    """Every call into the host (load, compile, decide) runs here, one at a time, on one thread."""

    def __init__(self):
        super().__init__(daemon=True, name="model")
        self.jobs: queue.Queue = queue.Queue()

    def run(self):
        while True:
            fn, box, done = self.jobs.get()
            try:
                box["value"] = fn()
            except BaseException as e:   # handed back to the caller
                box["error"] = e
                box["traceback"] = traceback.format_exc()
            finally:
                done.set()

    def call(self, fn):
        box, done = {}, threading.Event()
        self.jobs.put((fn, box, done))
        done.wait()
        if "error" in box:
            raise box["error"]
        return box.get("value")


class LockWatch(threading.Thread):
    """Reads the lock file every 0.2 s (when MAC_LOCK_FILE is set); records each change with the app's phase."""

    def __init__(self):
        super().__init__(daemon=True, name="lockwatch")
        self.phase, self.seen, self.halt, self._last = None, [], threading.Event(), None

    def run(self):
        while not self.halt.wait(0.2):
            content = read_lock()
            if content != self._last:
                self._last = content
                self.seen.append({"epoch": time.time(), "phase": self.phase, "lock": content[:240]})

    def report(self) -> dict:
        return {"file": str(LOCK_FILE) if LOCK_FILE else None, "seen": self.seen}


class Model:
    """The model repository's host on the files in ../hub/ (one row graph), and what this app needs around it: the plan
    of a request (its tokens, the graph it runs on), compiling the graph (timed), and Decide with the unrounded
    probabilities of the same call."""

    def __init__(self, hub: Path, accel: str, length: int):
        self.hub, self.accel, self.length = hub, accel, length
        sys.path.insert(0, str(hub / "host"))
        import clef_litert as C   # noqa: E402  (the repository's host, as downloaded)

        self.C = C
        t = time.perf_counter()
        self.clef = C.ClefLiteRT.from_dir(hub, accelerator=accel, threads=8, lengths=[length])
        self.load_seconds = round(time.perf_counter() - t, 3)
        host = self.clef.host
        self._scores = host.scores
        host.scores = self._keep_scores   # ClefHost.decide calls self.scores(enc, hidden): keep what it returns
        self.kept = None

    def _keep_scores(self, enc, hidden, lexical=None):
        logits = self._scores(enc, hidden, lexical)
        self.kept = (enc, logits)
        return logits

    def row_file(self) -> dict:
        return {"kind": "row", "key": self.length, "name": f"row L{self.length}",
                "file": self.clef.graphs.row_files[self.length].name}

    def plan(self, request: dict) -> dict:
        """The host's checks and encoding of the request (refusals raise the host's RequestError); runs no graph."""
        enc = self.clef.host.encode(request)
        return {"tokens": enc.record.tokens, "length": enc.length, "files": [self.row_file()]}

    def compiled(self, f: dict) -> bool:
        g = self.clef.graphs
        return g.row is not None and g.row[0] == f["key"]

    def compile(self, f: dict) -> dict:
        """Compile the row graph now (what the host does on a request's first use), timed."""
        g = self.clef.graphs
        t = time.perf_counter()
        graph = g._row_graph(f["key"])
        return {"file": f["file"], "kind": f["name"], "seconds": round(time.perf_counter() - t, 3),
                "fully_accelerated": graph.fully_accelerated(), "options": graph.options,
                "phys_footprint_after": phys_footprint()}

    def decide(self, request: dict) -> dict:
        """clef.decide(request) timed, with the float32 probabilities behind the response's 4-digit values."""
        C, g = self.C, self.clef.graphs
        n_compiled = len(g.compiled)
        self.kept = None
        e0 = time.time()
        t0 = time.perf_counter()
        response = self.clef.decide(request)
        t1 = time.perf_counter()
        e1 = time.time()
        enc, logits = self.kept
        probs = {q.question_id: dict(zip(q.option_ids, C.question_probs(z).tolist()))
                 for q, z in zip(enc.record.questions, logits)}
        return {"response": response, "probs": probs, "ms": (t1 - t0) * 1000.0, "epoch": [e0, e1],
                "tokens": enc.record.tokens, "length": enc.length, "compiled_during": g.compiled[n_compiled:],
                "response_is_its_probs": self.clef.host.respond(request, enc, logits) == response}

    def close(self):
        self.clef.close()


# ----------------------------------------------------------------------------------------------- the app


class App:
    """The page's state on the Python side (the commands, the Mac, the device the last command was about), Decide and
    the settings, and the run's record."""

    def __init__(self, a, before: dict):
        self.a = a
        self.hub = Path(a.hub)
        self.out = Path(a.trigger_dir)
        self.tag = a.tag
        self.autoplay = bool(a.scenes)
        self.hidden_until_ready = bool(a.scenes or a.tag)   # autoplay and scripted runs: shown at READY
        self.page: Page | None = None
        self.model: Model | None = None
        self.worker = ModelThread()
        self.worker.start()
        self.watch = LockWatch()
        doc = json.loads((APP / "commands.json").read_text())
        self.commands, self.take = doc["commands"], doc["take"]
        self.context: str | None = None   # the state's device_context
        self.ready = threading.Event()
        self.decide_lock = threading.Lock()
        self.shutting = threading.Lock()
        self.seq = 0
        self.decided = threading.Condition()
        self.before = before
        self.doc = {"tag": a.tag, "mode": "autoplay" if self.autoplay else "app", "started_at": now_iso(),
                    "ui_only": bool(a.ui_only), "argv": sys.argv, "length": a.length, "mac_before": before,
                    "decides": [], "events": []}

    # ---- record
    def path(self, suffix: str) -> Path:
        return self.out / f"{self.tag}.{suffix}"

    def write_new(self, suffix: str, doc) -> None:
        p = self.path(suffix)
        if p.exists():
            raise FileExistsError(f"{p} exists: not overwritten")
        text = doc if isinstance(doc, str) else json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(text)
        tmp.rename(p)

    def save_manual(self) -> None:
        """The app's record (--tag without --scenes): rewritten after every Decide and command."""
        if self.tag and not self.autoplay:
            self.doc["lock"] = self.watch.report()
            self.doc["saved_at"] = now_iso()
            write_json(self.out / f"{self.tag}_manual.run.json", self.doc)

    def event(self, what: str, **kw) -> None:
        self.doc["events"].append({"epoch": time.time(), "what": what, **kw})

    # ---- the end: the Mac goes back as the app found it
    def shutdown(self, why: str, code: int) -> None:
        """Put the Mac back (out/mac_before.json), record it, and end the process. Runs once."""
        if not self.shutting.acquire(blocking=False):
            return
        try:
            rc, results = restore_before()
            self.doc["restored"] = {"why": why, "at": now_iso(), "rc": rc, "results": results}
            print(f"RESTORED ({why}): " + ", ".join(
                f"{r.get('device', '')} {json.dumps(r.get('read_back', r.get('note', '')))}"
                f"{'' if r.get('ok', True) else ' FAILED'}" for r in results), flush=True)
            if self.tag and not self.autoplay:
                self.doc.setdefault("finished_at", now_iso())
                self.save_manual()
        finally:
            os._exit(code)

    # ---- status
    def paint(self, name: str, timeout: float = MARK_TIMEOUT_S) -> dict:
        return self.page.wait_mark(name, timeout, required=self.autoplay)

    def status(self, phase: str, word: str) -> None:
        if self.page is not None:
            self.page.call("status", phase, word)

    def compile_files(self, files: list[dict], phase: str) -> list[dict]:
        """Compile what is not compiled yet, one file at a time, with "COMPILING <name> <s> s…" on screen."""
        out = []
        for f in files:
            if self.model.compiled(f):
                continue
            t0, stop = time.time(), threading.Event()

            def tick(name=f["name"]):
                while not stop.wait(0.5):
                    self.status(phase, f"COMPILING {name} {int(time.time() - t0)} s…")

            self.status(phase, f"COMPILING {f['name']} 0 s…")
            ticker = threading.Thread(target=tick, daemon=True)
            ticker.start()
            try:
                out.append(self.worker.call(lambda f=f: self.model.compile(f)))
            finally:
                stop.set()
                ticker.join(1)
        return out

    # ---- boot
    def boot(self, window, previous_app):
        a, doc = self.a, self.doc
        self.page = page = Page(window)
        window.events.loaded.wait(30)
        if self.hidden_until_ready and not self.autoplay:   # scripted: the keyboard goes back to the user's app
            if previous_app is not None:
                on_main(lambda: previous_app.activateWithOptions_(0))
        elif not self.hidden_until_ready:
            doc["window"] = place_for_editing(window)
        doc["env"] = environment(self.hub)
        chip = doc["env"]["chip"]
        chip = chip if chip and not chip.startswith("unavailable") else "Mac"
        accel_word = "Metal float32" if a.accel == "gpu" else "CPU 8 threads"
        doc["hud"] = {"model": TITLE, "device": f"{chip} · LiteRT {version('ai-edge-litert')} · {accel_word}",
                      "pill": "● on-device"}
        page.call("hud", doc["hud"])
        page.call("commands", [{"id": x["id"], "title": x["title"]} for x in self.commands], not a.no_apply)
        if self.autoplay:
            page.call("autoplay", True)
        page.call("load", self.editor_live(""), None, True)
        page.call("status", "loading", "LOADING")
        if not self.hidden_until_ready:
            self.paint("loading")
        if not a.ui_only:
            if LOCK_FILE is not None:
                self.watch.start()
            doc["accelerator"] = ("gpu fp32 (GpuOptions enforce_f32)" if a.accel == "gpu"
                                  else "cpu (XNNPACK, 8 threads)")
            if WAIT_CMD:
                self.status("loading", "WAITING FOR A QUIET GPU")
                t = time.time()
                waited = subprocess.run(WAIT_CMD + ["--", "/usr/bin/true"], capture_output=True, text=True)
                doc["wait_before_compile"] = {"seconds": round(time.time() - t, 1),
                                              "stderr": waited.stderr.strip()[-400:]}
            self.watch.phase = "compile"
            doc["lock_at_compile"] = read_lock()[:240]
            self.status("loading", "LOADING")
            self.model = self.worker.call(lambda: Model(self.hub, a.accel, a.length))
            doc["load_host_seconds"] = self.model.load_seconds
            doc["compile"] = self.compile_files([self.model.row_file()], "loading")
            doc["phys_footprint_after_compile"] = phys_footprint()
            self.watch.phase = "warmup"
            self.status("loading", "WARMING UP")
            doc["warmup"] = self.worker.call(self.warm_up)
            self.watch.phase = None
            doc["phys_footprint_after_warmup"] = phys_footprint()
        self.ready.set()
        if self.autoplay and a.backdrop:
            doc["backdrop"] = show_backdrop()
        if self.hidden_until_ready:
            doc["window"] = place_for_recording(window, activate=self.autoplay)
        page.call("load", self.editor_live(""), None, True)   # the Mac as it is at READY
        page.call("ready", "ready")
        doc["timeline"] = {"READY": self.paint("ready")}
        if self.tag:
            self.write_new("READY", {"pid": os.getpid(), "window": doc["window"], "epoch": time.time(),
                                     "ui_only": bool(a.ui_only), "mode": doc["mode"],
                                     "backdrop": doc.get("backdrop")})
        print(f"READY {self.tag or ''} window {doc['window']['id']}", flush=True)

    def warm_up(self) -> list:
        """One request of other content on the row graph; timed and recorded, not shown."""
        req = {"model": E.MODEL, "state": WARM_STATE, "questions": WARM_QUESTIONS}
        r = self.model.decide(req)
        return [{"for": "text", "request_sha256": canonical_sha256(req), "request": req, "tokens": r["tokens"],
                 "length": r["length"], "ms": round(r["ms"], 3), "epoch": r["epoch"],
                 "compiled_during": r["compiled_during"], "response": r["response"]}]

    # ---- the editor's content
    def command(self, cid: str) -> dict:
        return next(x for x in self.commands if x["id"] == cid)

    def state_now(self, command: str, values: dict | None = None) -> tuple[dict, dict]:
        """(the state of the Mac as it is now, or as `values` read it, with the command as its "request", what was
        read)."""
        values = A.read_all() if values is None else values
        return F.state_of(values, command, self.context), values

    def editor_live(self, command: str, values: dict | None = None) -> dict:
        """The editor's form for the Mac as it is now (or as `values` read it), the command as its "request", and the
        questions it gets."""
        state, values = self.state_now(command, values)
        form = E.editor_of(F.request_of(state))
        form["read_errors"] = values.get("errors", {})
        return form

    # ---- Decide
    def decide(self, draft: dict, origin: str) -> dict:
        """Decide on the page's content: the same steps for a click and for autoplay."""
        if not self.ready.is_set():
            return {"error": "The model is still loading."}
        if not self.decide_lock.acquire(blocking=False):
            return {"error": "A request is running."}
        try:
            return self._decide(draft, origin)
        finally:
            self.decide_lock.release()

    def _decide(self, draft: dict, origin: str) -> dict:
        page, model = self.page, self.model
        self.seq += 1
        seq = self.seq
        rec = {"seq": seq, "origin": origin, "clicked_at": time.time(), "editor": draft,
               "live": bool(draft.get("live")), "auto_questions": bool(draft.get("autoQuestions")),
               "apply_on": bool(draft.get("apply"))}
        if draft.get("live"):   # the Mac read again: the request holds it as it is now
            t = time.perf_counter()
            state, values = self.state_now(draft.get("command") or "")
            text = E.json_text(state)
            rec["mac_read"] = {"values": values, "seconds": round(time.perf_counter() - t, 3), "text": text}
            page.call("setState", text)
            draft = dict(draft, state=text)
        if draft.get("autoQuestions"):   # the questions follow the state
            value = E.state_value(draft.get("state") or "")
            if isinstance(value, dict):
                cards = E.editor_of({"state": value, "questions": F.questions_of(value)})["questions"]
                page.call("setQuestions", cards)
                draft = dict(draft, questions=cards)
        rec["editor"] = draft
        try:
            request = E.build_request(draft)
            plan = self.worker.call(lambda: model.plan(request)) if model else None
        except Exception as e:   # the host's RequestError (RequestTooLong, ...) and the editor's DraftError
            rec["error"] = f"{type(e).__name__}: {e}"
            message = str(e)
            if type(e).__name__ == "RequestTooLong":
                message = f"Refused: {e}. Shorten the command, the state or the questions."
            elif type(e).__name__ != "DraftError":
                message = f"Refused ({type(e).__name__}): {e}"
            page.call("error", message)
            rec["shown_error"] = page.call("shown")["error"]
            self.finish(rec)
            return {"error": message}
        rec["request"] = request
        rec["request_sha256"] = canonical_sha256(request)
        page.call("running", f"press:{seq}")
        rec["marks"] = {"press": self.paint(f"press:{seq}")}
        rec["editor_shown"] = page.call("readEditor")
        rec["mac_shown_before"] = page.call("macShown")
        if plan is not None:
            rec["plan"] = plan
            rec["compile"] = self.compile_files(plan["files"], "running")
            if rec["compile"]:
                self.status("running", "RUNNING")
        rec["lock_at_decide"] = read_lock()[:240]
        if model is not None:
            r = self.worker.call(lambda: model.decide(request))
            rec["response"], rec["probs"] = r["response"], r["probs"]
            rec["ms"] = {"request_wall": r["ms"], "span": SPAN, "epoch": r["epoch"]}
            rec["tokens"], rec["length"] = r["tokens"], r["length"]
            rec["compiled_during"] = r["compiled_during"]
            rec["response_is_its_probs"] = r["response_is_its_probs"]
            payload = answer_payload(request, r["probs"], r["ms"], r["tokens"], r["length"])
        else:
            rec["ms"] = {"request_wall": None, "span": "--ui-only: nothing ran"}
            payload = answer_payload(request, None, None, None, None)
        page.call("answers", payload, f"answers:{seq}")
        rec["marks"]["answers"] = self.paint(f"answers:{seq}")
        rec["sent_to_page"] = payload
        rec["shown"] = page.call("shown")
        rec["layout"] = page.call("layout")
        rec["apply"] = self.apply(payload["decision"]["changes"], bool(draft.get("apply")), seq)
        rec["shown_after_apply"] = page.call("shown")
        dec = payload["decision"]
        if dec["resolution"] != "clarify" and dec["targets"]:
            self.context = dec["targets"][0]
        rec["context_after"] = self.context
        self.finish(rec)
        return {"ok": True, "seq": seq}

    def apply(self, changes: list[dict], on: bool, seq: int) -> dict:
        """Set the Mac to the decided changes, one at a time, each read back; the rows show each result. Timed on its
        own (it is not part of the request time)."""
        if not on or not changes:
            self.page.call("applyOff", "apply-off:" + str(seq), bool(changes))
            return {"on": on, "changes": changes, "results": [], "shown": self.paint(f"apply-off:{seq}")}
        self.status("done", "APPLYING")
        self.page.call("applying", sorted({c["device"] for c in changes}))
        results = []
        e0, t0 = time.time(), time.perf_counter()
        for c in changes:
            r = A.apply_change(c["device"], c["prop"], c["to"])
            r["epoch_done"] = time.time()
            results.append(r)
            self.page.call("applied", c["device"], {"ok": r["ok"], "shown_only": bool(r.get("shown_only")),
                                                    "error": r.get("error", "")}, f"applied:{c['device']}:{seq}")
        seconds = time.perf_counter() - t0   # every change set and read back, and its row updated
        e1 = time.time()
        self.status("done", "DONE")
        for r in results:
            r["shown"] = self.paint(f"applied:{r['device']}:{seq}")
        try:
            windows = A.device_windows()
        except Exception as e:   # informational (a recording's checks read it)
            windows = f"{type(e).__name__}: {e}"
        return {"on": True, "changes": changes, "results": results, "seconds": round(seconds, 3), "epoch": [e0, e1],
                "mac_after": A.read_all(), "windows": windows}

    def finish(self, rec: dict) -> None:
        with self.decided:
            self.doc["decides"].append(rec)
            self.decided.notify_all()
        self.save_manual()

    def wait_decide(self, seq_after: int, timeout: float = 120.0) -> dict:
        end = time.time() + timeout
        with self.decided:
            while True:
                done = [r for r in self.doc["decides"] if r["seq"] > seq_after]
                if done:
                    return done[0]
                left = end - time.time()
                if left <= 0:
                    raise TimeoutError("no Decide finished")
                self.decided.wait(left)


# ----------------------------------------------------------------------------------------------- what JS can call


class Api:
    """The methods the page calls (pywebview js_api). Each runs on its own pywebview thread."""

    def __init__(self, app: App):
        self._app = app

    def decide(self, draft):
        return self._app.decide(draft, "click")

    def command(self, cid):
        app = self._app
        app.event("command", id=cid)
        app.save_manual()
        return {"command": app.command(cid)["request"], "id": cid}

    def mac(self, command):
        """The Mac read now, as the STATE text with the command as its "request"."""
        app = self._app
        state, values = app.state_now(command or "")
        app.event("read mac", values=values)
        return {"state": E.json_text(state), "errors": values.get("errors", {})}

    def questions(self, state_text):
        """The question cards for a state text (app/form.py), or an error when it is not a JSON object."""
        value = E.state_value(state_text or "")
        if not isinstance(value, dict):
            return {"error": "The state is not a JSON object: the questions are built from its devices."}
        return {"questions": E.editor_of({"state": value, "questions": F.questions_of(value)})["questions"]}

    def scene(self):
        """Open the demo's devices (the TV app, Calculator, both drafts), play the music, volume 25."""
        app = self._app
        results = A.scene()
        app.event("scene", results=results)
        app.save_manual()
        return {"ok": all(r["ok"] for r in results),
                "errors": [f"{r['what']}: {r['error']}" for r in results if not r["ok"]]}


# ----------------------------------------------------------------------------------------------- autoplay


class Autoplay:
    """The recorded scenes; each step goes through the page as a user's would. M: the take's first command typed,
    Decide, applied. D: both drafts opened again (the app's own fixed setting), the second command typed, Decide."""

    def __init__(self, app: App):
        self.app, self.a = app, app.a

    def wait_trigger(self, suffix: str) -> float:
        p = self.app.path(suffix)
        end = time.time() + self.a.trigger_timeout
        while not p.exists():
            if time.time() > end:
                raise TimeoutError(f"no {p.name} within {self.a.trigger_timeout} s")
            time.sleep(0.01)
        return time.time()

    def step(self, scene: str, steps: list, what: str, fn: str, *args, **extra) -> dict:
        # the page keeps every mark it painted: a name an earlier scene used would come back at once, before this step
        mark = f"{what}:{scene}:{len(steps)}"
        self.app.page.call(fn, *args, mark)
        m = self.app.page.wait_mark(mark, timeout=30)
        steps.append({"what": what, **extra, **m})
        return m

    def type_command(self, scene: str, steps: list, command: str) -> None:
        """Type the command into the command line; return once the line holds all of it (typed again once if it does
        not), so that Decide is pressed on the command, never on part of it."""
        for _ in range(2):
            m = self.step(scene, steps, "type", "autoType", "#command", command, TYPE_CPS, True, command=command)
            if m.get("value") == command == self.app.page.js("document.getElementById('command').value"):
                return
        raise RuntimeError(f"scene {scene}: the command line does not hold the command after typing it twice")

    def run(self):
        app, a = self.app, self.a
        doc = app.doc
        doc["scenes"] = {}
        scenes = [s.strip().upper() for s in a.scenes.split(",") if s.strip()]
        for s in scenes:
            if s not in ("M", "D"):
                raise ValueError(f"scene {s}: this app has scenes M and D")
            cmd = app.command(app.take[0 if s == "M" else 1])
            tl, steps = {}, []
            tl["ARM"] = self.wait_trigger(f"ARM_{s}")
            tl["mac_at_arm"] = A.read_all()   # the Mac as the scene begins: the editor's state and the rows
            app.page.call("load", app.editor_live("", tl["mac_at_arm"]), None, True)   # an empty command line
            app.page.call("ready", f"armed:{s}")
            tl["armed_shown"] = app.page.wait_mark(f"armed:{s}")
            tl["windows_at_arm"] = A.device_windows()
            app.write_new(f"ARMED_{s}", {"epoch": time.time(), "mac": tl["mac_at_arm"]})
            tl["GO"] = self.wait_trigger(f"GO_{s}")
            seq_before = app.seq
            if s == "D":   # both drafts open again before the command is typed (scripts/take.sh opens and places them
                # between the scenes: then nothing is left to open, and the rows show them open since the scene began)
                closed = [d for d in ("draft_1", "draft_2") if not tl["mac_at_arm"].get(d, {}).get("open")]
                tl["drafts_opened"] = [A.apply_change(d, "open", True) for d in ("draft_1", "draft_2")]
                tl["drafts_opened_epoch"] = time.time()
                if closed:
                    app.page.call("macRead", E.json_text(app.state_now("")[0]), f"drafts:{s}")
                    tl["drafts_shown"] = app.page.wait_mark(f"drafts:{s}")
            time.sleep(PAUSE_BEFORE_TYPING_S)
            self.type_command(s, steps, cmd["request"])
            time.sleep(PAUSE_BEFORE_DECIDE_S)
            app.page.call("autoTap", "#decide", f"decide-tap:{s}")
            rec = app.wait_decide(seq_before)
            if "error" in rec:
                raise RuntimeError(f"scene {s}: Decide refused the request: {rec['error']}")
            tl["decide_pressed"] = rec["marks"]["press"]
            tl["answers_shown"] = rec["marks"]["answers"]
            tl["windows_at_done"] = A.device_windows()
            tl["mac_at_done"] = A.read_all()
            time.sleep(max(0.0, tl["answers_shown"]["raf1"] + HOLD_S - time.time()))
            tl["DONE"] = time.time()
            rec = dict(rec, timeline=tl, steps=steps, command_id=cmd["id"])
            doc["scenes"][s] = rec
            app.write_new(f"DONE_{s}", {"epoch": tl["DONE"], "request_wall_ms": rec["ms"]["request_wall"]})
            ms = rec["ms"]["request_wall"]
            print(f"DONE_{s} {app.tag} {'' if ms is None else f'{ms:.1f} ms'}", flush=True)
        doc["lock"] = app.watch.report()
        doc["finished_at"] = now_iso()
        app.write_new("run.json", doc)
        print(f"RUN_JSON {app.path('run.json')}", flush=True)


# ----------------------------------------------------------------------------------------------- scripted input


def control_loop(app: App) -> None:
    """With --tag in the app: run each <tag>.cmd.<n>.js that scripts/drive_editor.py writes (JavaScript on the page,
    the same handlers a click or a key press reaches) and write <tag>.cmd.<n>.json with its value; stop at
    <tag>.QUIT."""
    n = 1
    while True:
        if app.path("QUIT").exists():
            return
        cmd = app.out / f"{app.tag}.cmd.{n}.js"
        if cmd.exists():
            try:
                result = {"value": app.page.js(cmd.read_text())}
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
            write_json(app.out / f"{app.tag}.cmd.{n}.json", {"n": n, "epoch": time.time(), **result})
            n += 1
            continue
        time.sleep(0.05)


# ----------------------------------------------------------------------------------------------- main


def run(app: App, window, previous_app) -> None:
    try:
        app.boot(window, previous_app)
        if app.autoplay:
            Autoplay(app).run()
            if app.a.stay:
                Autoplay(app).wait_trigger("QUIT")
        elif app.tag:
            app.save_manual()
            control_loop(app)
        else:
            return   # the app runs until the window closes
        app.doc["finished_at"] = now_iso()
        app.save_manual()
        app.watch.halt.set()
        if app.model is not None:
            app.worker.call(app.model.close)
        on_main(lambda: window.native.close())
        app.shutdown("scripted run finished", 0)
    except BaseException:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        try:
            app.doc["failed"] = tb
            if app.tag:
                app.path("FAILED").write_text(tb)
                if app.autoplay:
                    app.write_new("run.json", app.doc)
                else:
                    app.save_manual()
        finally:
            app.shutdown("failed", 1)


SIGNALS = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}


def block_signals() -> None:
    """Block Ctrl-C, TERM and HUP in this thread before any other thread starts: every thread inherits the block, and
    watch_signals' thread takes them (a Python handler would wait for the Cocoa run loop)."""
    signal.pthread_sigmask(signal.SIG_BLOCK, SIGNALS)


def watch_signals(app: App) -> None:
    """Ctrl-C, TERM and HUP put the Mac back before the process ends."""
    def waiter():
        sig = signal.sigwait(SIGNALS)
        app.shutdown(f"signal {signal.Signals(sig).name}", 130)

    threading.Thread(target=waiter, daemon=True, name="signals").start()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--restore", action="store_true", help="put the Mac back to out/mac_before.json and exit")
    ap.add_argument("--length", type=int, default=DEFAULT_LENGTH, choices=LENGTHS,
                    help=f"the row graph in ../hub/ that every request runs on (default {DEFAULT_LENGTH} tokens)")
    ap.add_argument("--tag", help="name of this run's files in --trigger-dir (the app: a record of every Decide, "
                                  "scripted input; autoplay: required)")
    ap.add_argument("--scenes", help="autoplay these scenes (M, D) for a recording instead of the app")
    ap.add_argument("--backdrop", action="store_true", help="autoplay: a plain backdrop over the desktop")
    ap.add_argument("--no-apply", action="store_true", help="start with 'Apply to this Mac' off")
    ap.add_argument("--accel", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--trigger-dir", default=str(H / "out"))
    ap.add_argument("--hub", default=str(H.parent / "hub"))
    ap.add_argument("--stay", action="store_true", help="autoplay: keep the window after the run JSON until <tag>.QUIT")
    ap.add_argument("--trigger-timeout", type=float, default=1800.0)
    ap.add_argument("--ui-only", action="store_true", help="placeholder answers, no model (window and recorder tests)")
    a = ap.parse_args()
    if a.scenes and not a.tag:
        ap.error("--scenes needs --tag")

    if a.restore:
        rc, results = restore_before()
        for r in results:
            if "note" in r:
                print(r["note"])
            else:
                print(f"{r['device']}: {json.dumps(r.get('read_back'))}"
                      + ("" if r.get("ok") else f"  FAILED: {r.get('error')}"))
        return rc

    hub = Path(a.hub)
    missing = [f for f in needed_files(a.length) if not (hub / f).is_file()]
    if missing and not a.ui_only:
        fetch = "zsh scripts/fetch_hub.sh" + ("" if a.length == 512 else f" --length {a.length}")
        print(f"missing in {hub}: {', '.join(missing)} (run: {fetch})", file=sys.stderr)
        return 4

    out = Path(a.trigger_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.tag:
        # take.sh's own files, written before it starts the app: its logs, the Mac and the TV app's preferences it saved
        mine = {f"{a.tag}.take.log", f"{a.tag}.app.log", f"{a.tag}.mac_original.json", f"{a.tag}.tv_prefs.plist"}
        taken = sorted(p.name for p in [*out.glob(f"{a.tag}.*"), *out.glob(f"{a.tag}_*")] if p.name not in mine)
        if taken:
            print(f"tag {a.tag} has files already ({', '.join(taken[:5])}): pick a new tag", file=sys.stderr)
            return 2

    BEFORE.parent.mkdir(parents=True, exist_ok=True)
    if BEFORE.is_file() and not json.loads(BEFORE.read_text()).get("restored_at"):
        print(f"the last run did not put the Mac back ({BEFORE}): run `venv-demo/bin/python app/mac_demo.py "
              "--restore` first", file=sys.stderr)
        return 5
    block_signals()
    before = A.snapshot()
    write_json(BEFORE, {"settings": before, "pid": os.getpid(), "argv": sys.argv})

    from Foundation import NSBundle
    info = NSBundle.mainBundle().infoDictionary()
    if info is not None:
        info["CFBundleName"] = TITLE   # the menu bar's app name (else "Python")
    import AppKit
    import webview
    import webview.platforms.cocoa  # noqa: F401  (sets the activation policy at import; changed below)

    scripted = bool(a.tag and not a.scenes)
    AppKit.NSApplication.sharedApplication().setActivationPolicy_(
        AppKit.NSApplicationActivationPolicyAccessory if scripted else AppKit.NSApplicationActivationPolicyRegular)
    activity = AppKit.NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
        AppKit.NSActivityUserInitiated | AppKit.NSActivityLatencyCritical, "clef-flash mac requests (timed)")
    previous = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
    app = App(a, before)
    watch_signals(app)
    hidden = bool(a.scenes or a.tag)
    window = webview.create_window(TITLE, html=html_page(), js_api=Api(app), width=WIDTH, height=HEIGHT,
                                   resizable=not hidden, min_size=(WIDTH, HEIGHT), easy_drag=False,
                                   focus=not scripted, hidden=hidden, text_select=True, zoomable=False)
    webview.start(run, (app, window, previous), gui="cocoa", debug=False, private_mode=True)
    # the window is closed: whatever the other threads were doing, the app ends here, with the Mac put back
    if app.tag and not app.autoplay:
        app.doc.setdefault("finished_at", now_iso())
        app.doc["window_closed"] = True
    del activity
    app.shutdown("window closed", 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
