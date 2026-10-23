#!/usr/bin/env python3
"""clef-flash home on a Mac: a house (its devices and the weather, as JSON) and a wish in a few words ("Movie night").
Press Decide and the model sets every device at once, in one call: each question of the request (one per device) comes
back as one of its options with a probability for each option, and the house on screen changes to match. The model is
litert-community/clef-flash-LiteRT (Cloudflare's clef-flash on LiteRT), run on this Mac's GPU through Metal at float32
precision, or on the CPU. It never writes text.

    venv-demo/bin/python app/home_demo.py                          # the app
    venv-demo/bin/python app/home_demo.py --length 1024            # every request on another row graph in ../hub/
    venv-demo/bin/python app/home_demo.py --tag m1                 # the app; every Decide recorded in
                                                                   #   out/m1_manual.run.json; scripted input accepted
                                                                   #   from scripts/drive_editor.py
    venv-demo/bin/python app/home_demo.py --scenes M --tag t1      # autoplay for a recording (scripts/take.sh)

The screen (app/ui): the house (a drawing of the state: the living room light, the curtains, the TV, the air
conditioner, the front door, the sky of "outside", "inside" and "time" as written) and two tabs. WISH: the wish line
(the state's "request"), the bundled wishes, Decide, one row per question (the device, the setting picked, its
probability) and the time the request took. HOUSE & QUESTIONS: the state as JSON and one card per question (name,
type, the question, the options; + Add question, − removes). Decide sends the state and the cards as one request
(app/editor.py: the rules); the answers come back all at once and the house changes (app/house.py: how a setting is
read). Any edit clears the answers and puts the house back as the state describes it.

The host is the model repository's own Python host (../hub/host/clef_litert), imported as downloaded. The app holds
one row graph, the one of --length (default 512 tokens): it compiles it at start and warms it up with one request of
other content, so a Decide never waits for a compile. A request longer than the row graph is refused (the host's
RequestTooLong), never cut. The time on screen is time.perf_counter() right before and after clef.decide(request):
never a compile. All model calls run on one thread.

Autoplay (--scenes, for a recording) and scripted runs (--tag): the window stays hidden while the graph compiles and
appears at READY: 960 x 540 pt, without a title bar, floating, mouse-transparent, top right of the screen; the
keyboard stays with the app you were using. It drives the page itself (typing into the wish line, tapping the
buttons: the same handlers and the same Decide as a click), on triggers from the take script in --trigger-dir:
<tag>.READY (written), then per scene <tag>.ARM_<s> -> <tag>.ARMED_<s>, <tag>.GO_<s> -> the scene -> <tag>.DONE_<s>,
then <tag>.run.json; --stay keeps the window until <tag>.QUIT. Scene M: the house of the take's wish (wishes.json
"take") with an empty wish line; the wish is typed, Decide, the answers and the house after them.

Measuring next to other GPU work is optional: HOME_WAIT_CMD (a command prefix run as `<prefix> -- <command>`; the app
runs it before compiling and waits for it) and HOME_LOCK_FILE (a lock file whose content is recorded during compile
and warm-up and at each Decide). Unset, the app waits for nothing and reads no lock.

--ui-only loads no model: the answers read "0.000" and "000 ms" (window and recorder tests). --accel cpu runs on
XNNPACK with 8 threads. Exit 4 (one line on stderr) when a file the app needs is not in ../hub/.
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

import editor as E  # noqa: E402
import house as HS  # noqa: E402

WIDTH, HEIGHT = 960, 540
MARGIN = 24
TITLE = "clef-flash home"
LENGTHS = (256, 512, 1024, 2048, 4096)
MARK_TIMEOUT_S = 5.0
HOLD_S = 3.0
TYPE_CPS = 14.0
PAUSE_BEFORE_DECIDE_S = 0.8
WAIT_CMD = shlex.split(os.environ.get("HOME_WAIT_CMD", ""))
LOCK_FILE = Path(os.environ["HOME_LOCK_FILE"]) if os.environ.get("HOME_LOCK_FILE") else None
NOTE = "timed by the app"
WARM_STATE = "The kettle stopped heating after a week and the light on its base no longer turns on."
WARM_QUESTIONS = {"faulty": {"type": "noul", "instructions": "The product is faulty."}}
SPAN = ("time.perf_counter() right before and after clef.decide(request): the tokenizer, the row graph's inputs "
        "(embedding rows, positions), the row graph call with its input writes and output read, the joint head and the "
        "response; not a compile, not the screen update")
HOST_FILES = ("__init__.py", "_erf.py", "errors.py", "head.py", "host.py", "pixels.py", "positions.py", "request.py",
              "response.py", "runtime.py", "tables.py")


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


def environment(hub: Path) -> dict:
    sw = dict(line.split(":", 1) for line in sh("/usr/bin/sw_vers").splitlines() if ":" in line)
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": {d: version(d) for d in ("ai-edge-litert", "pywebview", "pyobjc-core", "pyobjc-framework-WebKit",
                                             "numpy", "tokenizers")},
        "macos": {k.strip(): v.strip() for k, v in sw.items()},
        "platform": platform.platform(),
        "chip": sh("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": as_int(sh("/usr/sbin/sysctl", "-n", "hw.memsize")),
        "hub": str(hub),
    }


def canonical_sha256(request: dict) -> str:
    """sha256 of the request as canonical JSON (sorted keys, no spaces)."""
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
                          ).hexdigest()


def write_json(path: Path, doc) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    tmp.rename(path)


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


def answer_payload(request: dict, probs: dict | None, ms: float | None, tokens: int | None,
                   length: int | None) -> dict:
    """Every string the answers show, rounded here: probabilities f"{p:.3f}", the time round(ms); and the house after
    them (app/house.py)."""
    qs, picked = [], {}
    for name, q in request["questions"].items():
        rows = option_rows(q)
        if probs is None:   # --ui-only placeholders
            p, top = {o["key"]: 0.0 for o in rows}, rows[0]["key"]
        else:
            p = {o["key"]: probs[name][o["key"]] for o in rows}
            top = max(p, key=p.get)
        picked[name] = top
        labels = {o["key"]: o["label"] for o in rows}
        qs.append({"name": name, "device": HS.NAMES.get(name, name), "text": q["instructions"], "answer": labels[top],
                   "answer_p": f"{p[top]:.3f}", "top": top,
                   "options": [{"key": o["key"], "label": o["label"], "text": f"{p[o['key']]:.3f}",
                                "width": max(0.0, min(1.0, p[o["key"]]))} for o in rows]})
    before = HS.house_now(request.get("state"))
    after = HS.house_after(request, picked)
    n = len(request["questions"])
    time_text = f"{round(ms):,} ms" if ms is not None else "000 ms"
    where = f"{n} question{'s' if n != 1 else ''}" + ("" if tokens is None else
                                                      f" · {tokens:,} tokens · {length:,}-token row graph")
    return {"questions": qs, "time": time_text, "note": f"{where} · {NOTE}", "house_before": before,
            "house_after": after, "changed": HS.changed(before, after)}


def html_page() -> str:
    page = (UI / "index.html").read_text()
    page = page.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{(UI / 'style.css').read_text()}</style>")
    return page.replace('<script src="app.js"></script>', f"<script>\n{(UI / 'app.js').read_text()}</script>")


# ----------------------------------------------------------------------------------------------- the page


class Page:
    """The pywebview window and the page's HOME object."""

    def __init__(self, window):
        self.w = window

    def js(self, script: str):
        return self.w.evaluate_js(script)

    def call(self, fn: str, *args):
        return self.js(f"HOME.{fn}({', '.join(json.dumps(a, ensure_ascii=False) for a in args)})")

    def wait_mark(self, name: str, timeout: float = MARK_TIMEOUT_S, required: bool = True) -> dict:
        """The step's paint times in epoch seconds: raf1 (the frame that paints it), raf2 (the frame after), and what
        the step recorded with them. A covered or hidden window paints nothing: required (autoplay) raises, else
        {"painted": False} comes back and the app goes on."""
        end = time.time() + timeout
        while time.time() < end:
            m = self.js(f"HOME.mark({json.dumps(name)})")
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
    f, sf, cv = w.frame(), screen.frame(), w.contentView().frame()
    scale = float(screen.backingScaleFactor())
    return {"id": int(w.windowNumber()), "title": str(w.title()), "pid": os.getpid(),
            "bounds_pt": [f.origin.x, f.origin.y, f.size.width, f.size.height],
            "content_pt": [cv.size.width, cv.size.height], "scale": scale,
            "px": [round(cv.size.width * scale), round(cv.size.height * scale)],
            "screen_pt": [sf.origin.x, sf.origin.y, sf.size.width, sf.size.height],
            "style_mask": int(w.styleMask()), "level": int(w.level()), "ignores_mouse": bool(w.ignoresMouseEvents())}


def place_for_recording(window, previous_app) -> dict:
    """Borderless (no title bar, square corners), floating, mouse-transparent, top right of its screen; the app the
    user was in keeps the keyboard."""
    import AppKit

    def fn():
        w = window.native
        w.setStyleMask_(AppKit.NSWindowStyleMaskBorderless)
        w.setHasShadow_(False)
        w.setLevel_(AppKit.NSFloatingWindowLevel)
        w.setIgnoresMouseEvents_(True)
        screen = w.screen() or AppKit.NSScreen.mainScreen()
        vf = screen.visibleFrame()
        x = vf.origin.x + vf.size.width - WIDTH - MARGIN
        top = vf.origin.y + vf.size.height
        w.setFrame_display_(AppKit.NSMakeRect(x, top - HEIGHT - MARGIN, WIDTH, HEIGHT), True)
        w.orderFrontRegardless()
        if previous_app is not None:
            previous_app.activateWithOptions_(0)
        return window_facts(w, screen)

    return on_main(fn)


def place_for_editing(window) -> dict:
    """A normal window with a dark title bar, 960 x 540 pt (less on a smaller screen), centred, in front with the
    keyboard."""
    import AppKit

    def fn():
        w = window.native
        w.setAppearance_(AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameDarkAqua))
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
    """Reads the lock file every 0.2 s (when HOME_LOCK_FILE is set); records each change with the app's phase."""

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
    """The page's state on the Python side (the wishes), Decide, and the run's record."""

    def __init__(self, a):
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
        doc = json.loads((APP / "wishes.json").read_text())
        self.questions, self.wishes, self.take = doc["questions"], doc["wishes"], doc["take"]
        self.ready = threading.Event()
        self.decide_lock = threading.Lock()
        self.seq = 0
        self.decided = threading.Condition()
        self.doc = {"tag": a.tag, "mode": "autoplay" if self.autoplay else "app", "started_at": now_iso(),
                    "ui_only": bool(a.ui_only), "argv": sys.argv, "length": a.length, "decides": [], "events": []}

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
        """The app's record (--tag without --scenes): rewritten after every Decide and wish."""
        if self.tag and not self.autoplay:
            self.doc["lock"] = self.watch.report()
            self.doc["saved_at"] = now_iso()
            write_json(self.out / f"{self.tag}_manual.run.json", self.doc)

    def event(self, what: str, **kw) -> None:
        self.doc["events"].append({"epoch": time.time(), "what": what, **kw})

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
        if self.hidden_until_ready:   # pywebview activated the app at start: the keyboard goes back to the user's app
            if previous_app is not None:
                on_main(lambda: previous_app.activateWithOptions_(0))
        else:
            doc["window"] = place_for_editing(window)
        doc["env"] = environment(self.hub)
        chip = doc["env"]["chip"]
        chip = chip if chip and not chip.startswith("unavailable") else "Mac"
        accel_word = "Metal float32" if a.accel == "gpu" else "CPU 8 threads"
        doc["hud"] = {"model": "clef-flash home", "device": f"{chip} · LiteRT {version('ai-edge-litert')} · {accel_word}",
                      "pill": "● on-device"}
        page.call("hud", doc["hud"])
        page.call("wishes", [{"id": x["id"], "title": x["title"]} for x in self.wishes])
        if self.autoplay:
            page.call("load", self.take_editor(), None)
        else:
            page.call("load", *self.wish_payload(self.wishes[0]["id"]))
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
        if self.hidden_until_ready:
            doc["window"] = place_for_recording(window, previous_app)
        page.call("ready", "ready")
        doc["timeline"] = {"READY": self.paint("ready")}
        if self.tag:
            self.write_new("READY", {"pid": os.getpid(), "window": doc["window"], "epoch": time.time(),
                                     "ui_only": bool(a.ui_only), "mode": doc["mode"]})
        print(f"READY {self.tag or ''} window {doc['window']['id']}", flush=True)

    def warm_up(self) -> list:
        """One request of other content on the row graph; timed and recorded, not shown."""
        req = {"model": E.MODEL, "state": WARM_STATE, "questions": WARM_QUESTIONS}
        r = self.model.decide(req)
        return [{"for": "text", "request_sha256": canonical_sha256(req), "request": req, "tokens": r["tokens"],
                 "length": r["length"], "ms": round(r["ms"], 3), "epoch": r["epoch"],
                 "compiled_during": r["compiled_during"], "response": r["response"]}]

    # ---- the wishes
    def wish(self, wid: str) -> dict:
        return next(x for x in self.wishes if x["id"] == wid)

    def wish_request(self, wid: str) -> dict:
        return {"model": E.MODEL, "state": self.wish(wid)["state"], "questions": self.questions}

    def wish_payload(self, wid: str) -> tuple:
        """(editor form, wish id) for page.load."""
        return E.editor_of(self.wish_request(wid)), wid

    def take_editor(self) -> dict:
        """The recorded clip's opening: the house of the take's wish with an empty wish line (its "request" is "")."""
        state = dict(self.wish(self.take)["state"], request="")
        return E.editor_of({"state": state, "questions": self.questions})

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
        rec = {"seq": seq, "origin": origin, "clicked_at": time.time(), "editor": draft}
        try:
            request = E.build_request(draft)
            plan = self.worker.call(lambda: model.plan(request)) if model else None
        except Exception as e:   # the host's RequestError (RequestTooLong, ...) and the editor's DraftError
            rec["error"] = f"{type(e).__name__}: {e}"
            message = str(e)
            if type(e).__name__ == "RequestTooLong":
                message = f"Refused: {e}. Shorten the wish, the house or the questions."
            elif type(e).__name__ != "DraftError":
                message = f"Refused ({type(e).__name__}): {e}"
            page.call("error", message)
            rec["shown_error"] = page.call("shown")["error"]
            self.finish(rec)
            return {"error": message}
        rec["request"] = request
        rec["request_sha256"] = canonical_sha256(request)
        rec["questions_ordered"] = json.dumps(request["questions"], ensure_ascii=False)
        page.call("running", f"press:{seq}")
        rec["marks"] = {"press": self.paint(f"press:{seq}")}
        rec["editor_shown"] = page.call("readEditor")
        rec["house_shown_before"] = page.call("houseShown")
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
        self.finish(rec)
        return {"ok": True, "seq": seq}

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

    def wish(self, wid):
        app = self._app
        form, wid = app.wish_payload(wid)
        app.event("wish", id=wid)
        app.save_manual()
        return {"editor": form, "id": wid}


# ----------------------------------------------------------------------------------------------- autoplay


class Autoplay:
    """Scene M for a recording; each step goes through the page as a user's would."""

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

    def step(self, steps: list, what: str, fn: str, *args, **extra) -> dict:
        mark = f"{what}:{len(steps)}"
        self.app.page.call(fn, *args, mark)
        m = self.app.page.wait_mark(mark, timeout=30)
        steps.append({"what": what, **extra, **m})
        return m

    def run(self):
        app, a = self.app, self.a
        doc = app.doc
        doc["scenes"] = {}
        scenes = [s.strip().upper() for s in a.scenes.split(",") if s.strip()]
        for s in scenes:
            if s != "M":
                raise ValueError(f"scene {s}: this app has scene M")
            tl, steps = {}, []
            tl["ARM"] = self.wait_trigger(f"ARM_{s}")
            app.page.call("load", app.take_editor(), None)
            app.page.call("ready", f"armed:{s}")
            tl["armed_shown"] = app.page.wait_mark(f"armed:{s}")
            app.write_new(f"ARMED_{s}", {"epoch": time.time()})
            tl["GO"] = self.wait_trigger(f"GO_{s}")
            seq_before = app.seq
            time.sleep(0.6)
            text = app.wish(app.take)["state"]["request"]
            self.step(steps, "type", "autoType", "#wish", text, TYPE_CPS, True, wish=text)
            time.sleep(PAUSE_BEFORE_DECIDE_S)
            app.page.call("autoTap", "#decide", f"decide-tap:{s}")
            rec = app.wait_decide(seq_before)
            if "error" in rec:
                raise RuntimeError(f"scene {s}: Decide refused the request: {rec['error']}")
            tl["decide_pressed"] = rec["marks"]["press"]
            tl["answers_shown"] = rec["marks"]["answers"]
            time.sleep(HOLD_S)
            tl["DONE"] = time.time()
            rec = dict(rec, timeline=tl, steps=steps)
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
        os._exit(0)
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
            os._exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--length", type=int, default=512, choices=LENGTHS,
                    help="the row graph in ../hub/ that every request runs on (default 512 tokens)")
    ap.add_argument("--tag", help="name of this run's files in --trigger-dir (the app: a record of every Decide, "
                                  "scripted input; autoplay: required)")
    ap.add_argument("--scenes", help="autoplay these scenes (M) for a recording instead of the app")
    ap.add_argument("--accel", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--trigger-dir", default=str(H / "out"))
    ap.add_argument("--hub", default=str(H.parent / "hub"))
    ap.add_argument("--stay", action="store_true", help="autoplay: keep the window after the run JSON until <tag>.QUIT")
    ap.add_argument("--trigger-timeout", type=float, default=1800.0)
    ap.add_argument("--ui-only", action="store_true", help="placeholder answers, no model (window and recorder tests)")
    a = ap.parse_args()
    if a.scenes and not a.tag:
        ap.error("--scenes needs --tag")

    hub = Path(a.hub)
    missing = [f for f in needed_files(a.length) if not (hub / f).is_file()]
    if missing and not a.ui_only:
        fetch = "zsh scripts/fetch_hub.sh" + ("" if a.length == 512 else f" --length {a.length}")
        print(f"missing in {hub}: {', '.join(missing)} (run: {fetch})", file=sys.stderr)
        return 4

    out = Path(a.trigger_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.tag:
        mine = {f"{a.tag}.take.log", f"{a.tag}.app.log"}   # take.sh's own logs, written before it starts the app
        taken = sorted(p.name for p in [*out.glob(f"{a.tag}.*"), *out.glob(f"{a.tag}_*")] if p.name not in mine)
        if taken:
            print(f"tag {a.tag} has files already ({', '.join(taken[:5])}): pick a new tag", file=sys.stderr)
            return 2

    import AppKit
    import webview
    import webview.platforms.cocoa  # noqa: F401  (sets the activation policy at import; changed below)

    autoplay_or_scripted = bool(a.scenes or a.tag)
    AppKit.NSApplication.sharedApplication().setActivationPolicy_(
        AppKit.NSApplicationActivationPolicyAccessory if autoplay_or_scripted
        else AppKit.NSApplicationActivationPolicyRegular)
    activity = AppKit.NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
        AppKit.NSActivityUserInitiated | AppKit.NSActivityLatencyCritical, "clef-flash home requests (timed)")
    previous = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
    app = App(a)
    window = webview.create_window(TITLE, html=html_page(), js_api=Api(app), width=WIDTH, height=HEIGHT,
                                   resizable=not autoplay_or_scripted, min_size=(WIDTH, HEIGHT),
                                   frameless=autoplay_or_scripted, easy_drag=False, shadow=not autoplay_or_scripted,
                                   focus=not a.scenes, hidden=autoplay_or_scripted, background_color="#0E1116",
                                   text_select=True, zoomable=False)
    webview.start(run, (app, window, previous), gui="cocoa", debug=False, private_mode=True)
    # the window is closed: whatever the other threads were doing, the app ends here
    if app.tag and not app.autoplay:
        app.doc.setdefault("finished_at", now_iso())
        app.doc["window_closed"] = True
        app.save_manual()
    del activity
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
