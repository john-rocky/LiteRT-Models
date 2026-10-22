#!/usr/bin/env python3
"""clef-flash Decide on a Mac: write a state (a text or JSON), add a photo if you like, write questions with their
options, press Decide. Every answer comes back at once, as one of your options with a probability for each option. The
model is litert-community/clef-flash-LiteRT (Cloudflare's clef-flash on LiteRT), run on this Mac's GPU through Metal
at float32 precision, or on the CPU. It never writes text.

    venv-demo/bin/python app/clef_demo.py                          # the editor
    venv-demo/bin/python app/clef_demo.py --length 512             # the editor on another row graph in hub/
    venv-demo/bin/python app/clef_demo.py --tag m1                 # the editor; every Decide recorded in
                                                                   #   out/m1_manual.run.json; scripted input accepted
                                                                   #   from scripts/drive_editor.py
    venv-demo/bin/python app/clef_demo.py --scenes P --tag t1      # autoplay for a recording (scripts/take.sh)

The screen (app/ui): the status (READY, COMPILING, RUNNING, DONE), the EXAMPLES (a support ticket, a JSON list of
rooms, a photo, a drawn clock), STATE, PHOTO (drop a JPEG or PNG, or Choose photo…; the tile it is read at, 448 or
256 px), QUESTIONS (one card per question: name, type, the question, the options; + Add question, − removes),
Decide, and the footer. Decide sends the editor's content as one request (app/editor.py: the rules); the question
cards turn into the answers, all at once, with the time the request took. Any edit clears the answers; Decide again
runs the edited request.

The host is the model repository's own Python host (hub/host/clef_litert), imported as downloaded. The app holds one
row graph, the one of --length (default 1,024 tokens), and both picture towers: it compiles them at start and warms
each up with one request of other content, so a Decide never waits for a compile. A request longer than the row
graph is refused (the host's RequestTooLong), never cut. The time in the footer is time.perf_counter() right before
and after clef.decide(request): never a compile. All model calls run on one thread.

Autoplay (--scenes, for a recording) and scripted runs (--tag): the window stays hidden while the graphs compile and
appears at READY: 540 x 960 pt, without a title bar, floating, mouse-transparent, top right of the screen; the
keyboard stays with the app you were using. It drives the editor through the page itself (typing into the fields,
tapping the buttons: the same handlers and the same Decide as a click), on triggers from the take script in
--trigger-dir: <tag>.READY (written), then per scene <tag>.ARM_<s> -> <tag>.ARMED_<s>, <tag>.GO_<s> -> the scene ->
<tag>.DONE_<s>, then <tag>.run.json; --stay keeps the window until <tag>.QUIT. Scene P: the editor opens with the
coffee example's state and three of its questions (examples.json "take") and no photo; the coffee photo is placed in
the drop zone, Decide, the three answers.

Measuring next to other GPU work is optional: CLEF_WAIT_CMD (a command prefix run as `<prefix> -- <command>`; the app
runs it before compiling and waits for it) and CLEF_LOCK_FILE (a lock file whose content is recorded during compile
and warm-up and at each Decide). Unset, the app waits for nothing and reads no lock.

--ui-only loads no model: the answers read "0.000" and "000 ms" (window and recorder tests). --accel cpu runs on
XNNPACK with 8 threads. Exit 4 (one line on stderr) when a file the app needs is not in hub/.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True   # the model repository's host/ stays byte for byte as downloaded

import argparse
import base64
import ctypes
import hashlib
import io
import json
import os
import platform
import queue
import shlex
import subprocess
import threading
import time
import traceback
import types
from importlib import metadata
from pathlib import Path

APP = Path(__file__).resolve().parent
D = APP.parent
UI = APP / "ui"
sys.path.insert(0, str(APP))

import editor as E  # noqa: E402

WIDTH, HEIGHT = 540, 960
MARGIN = 24
TITLE = "clef-flash"
LENGTHS = (256, 512, 1024, 2048, 4096)
TILES = (448, 256)            # the first is the editor's default
MARK_TIMEOUT_S = 5.0
HOLD_S = 3.0
TYPE_CPS = 27.0
PAUSE_BEFORE_DECIDE_S = 0.8
WAIT_CMD = shlex.split(os.environ.get("CLEF_WAIT_CMD", ""))
LOCK_FILE = Path(os.environ["CLEF_LOCK_FILE"]) if os.environ.get("CLEF_LOCK_FILE") else None
NOTE = "timed by the app"
WARM_TEXT_STATE = "The kettle stopped heating after a week and the light on its base no longer turns on."
WARM_TEXT_QUESTIONS = {"faulty": {"type": "noul", "instructions": "The product is faulty."}}
WARM_PICTURE_QUESTIONS = {"flat": {"type": "noul", "instructions": "The picture is one flat colour."}}
SPAN = ("time.perf_counter() right before and after clef.decide(request): the tokenizer, the picture's tile and "
        "patches, the tower call, the row graph's inputs (embedding rows, positions), the row graph call with its input "
        "writes and output read, the joint head and the response; not a compile, not the screen update")
PHOTO_TYPES = ("JPEG", "PNG")
THUMB_PX = 960
HOST_FILES = ("__init__.py", "_erf.py", "errors.py", "head.py", "host.py", "pixels.py", "positions.py", "request.py",
              "response.py", "runtime.py", "tables.py")


def needed_files(length: int) -> list[str]:
    """The hub/ files the app loads for a row graph of `length` tokens (the host's FOLDER layout)."""
    return ([f"host/clef_litert/{f}" for f in HOST_FILES]
            + [f"clef-flash_rowprefill_embeds_L{length}_fp16fc.tflite"]
            + [f"clef-flash_vision_tower_g{t}_fp16fc.tflite" for t in sorted(TILES)]
            + [f"tables/vision_pos_g{t}.safetensors" for t in sorted(TILES)]
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
                                             "numpy", "tokenizers", "pillow")},
        "macos": {k.strip(): v.strip() for k, v in sw.items()},
        "platform": platform.platform(),
        "chip": sh("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_bytes": as_int(sh("/usr/sbin/sysctl", "-n", "hw.memsize")),
        "hub": str(hub),
    }


def canonical_sha256(request: dict) -> str:
    """sha256 of the request as canonical JSON, each picture replaced by the sha256 of its bytes (or its description)."""
    def pic(x):
        if isinstance(x, (bytes, bytearray)):
            return {"sha256": hashlib.sha256(x).hexdigest(), "bytes": len(x)}
        if hasattr(x, "size") and hasattr(x, "mode"):
            return {"pil": f"{x.mode} {x.size[0]}x{x.size[1]}", "sha256_raw": hashlib.sha256(x.tobytes()).hexdigest()}
        return {"path": str(x)}
    doc = dict(request)
    if "images" in doc:
        doc["images"] = [pic(x) for x in doc["images"]]
    return hashlib.sha256(json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, doc) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    tmp.rename(path)


# ----------------------------------------------------------------------------------------------- what is shown


def option_rows(question: dict) -> list[dict]:
    """[{key, label}] in the request's order: noul true/false as yes/no, choice names, score levels."""
    kind = question.get("type", "choice")
    if kind == "noul":
        return [{"key": "true", "label": "yes"}, {"key": "false", "label": "no"}]
    if kind == "score":
        return [{"key": str(i), "label": str(level)} for i, level in enumerate(question["criteria"])]
    return [{"key": str(name), "label": str(name)} for name in question["criteria"]]


def count_words(request: dict, tile: int | None) -> str:
    n = len(request["questions"])
    words = f"{n} question" + ("s" if n != 1 else "")
    return f"1 photo at {tile} px · {words}" if request.get("images") else words


def answer_payload(request: dict, probs: dict | None, ms: float | None, tokens: int | None, length: int | None,
                   tile: int | None) -> dict:
    """Every string the answers show, rounded here: probabilities f"{p:.3f}", the footer's round(ms)."""
    qs = []
    for name, q in request["questions"].items():
        rows = option_rows(q)
        if probs is None:   # --ui-only placeholders
            p, top = {o["key"]: 0.0 for o in rows}, rows[0]["key"]
        else:
            p = {o["key"]: probs[name][o["key"]] for o in rows}
            top = max(p, key=p.get)
        labels = {o["key"]: o["label"] for o in rows}
        qs.append({"name": name, "text": q["instructions"], "answer": labels[top], "answer_p": f"{p[top]:.3f}",
                   "top": top, "options": [{"key": o["key"], "label": o["label"], "text": f"{p[o['key']]:.3f}",
                                            "width": max(0.0, min(1.0, p[o["key"]]))} for o in rows]})
    footer = count_words(request, tile) + (f" · {round(ms):,} ms" if ms is not None else " · 000 ms")
    where = "" if tokens is None else f"{tokens:,} tokens · {length:,}-token row graph · "
    return {"questions": qs, "footer": footer, "note": where + NOTE}


def html_page() -> str:
    page = (UI / "index.html").read_text()
    page = page.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{(UI / 'style.css').read_text()}</style>")
    return page.replace('<script src="app.js"></script>', f"<script>\n{(UI / 'app.js').read_text()}</script>")


# ----------------------------------------------------------------------------------------------- the photo


class Photo(types.SimpleNamespace):
    """id (sha256, first 12), name, data (the file's bytes), sha256, format, size (w, h)."""

    def meta(self) -> dict:
        """What the page shows: the name, the size, and a JPEG thumbnail of the pixels as the model reads them (no
        EXIF rotation: the host opens the bytes with PIL and converts them to RGB as they are)."""
        from PIL import Image

        img = Image.open(io.BytesIO(self.data)).convert("RGB")
        img.thumbnail((THUMB_PX, THUMB_PX), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        return {"id": self.id, "name": self.name, "w": self.size[0], "h": self.size[1], "bytes": len(self.data),
                "format": self.format, "thumb": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}


def open_photo(name: str, data: bytes) -> Photo:
    """A JPEG or PNG file's bytes as the request's picture; anything else is refused with the reason."""
    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise E.DraftError(f"{name} is not a picture this app can read ({type(e).__name__}).") from e
    if img.format not in PHOTO_TYPES:
        raise E.DraftError(f"{name} is a {img.format} file: use a JPEG or PNG.")
    sha = hashlib.sha256(data).hexdigest()
    return Photo(id=sha[:12], name=name, data=data, sha256=sha, format=img.format, size=tuple(img.size))


# ----------------------------------------------------------------------------------------------- the page


class Page:
    """The pywebview window and the page's CLEF object."""

    def __init__(self, window):
        self.w = window

    def js(self, script: str):
        return self.w.evaluate_js(script)

    def call(self, fn: str, *args):
        return self.js(f"CLEF.{fn}({', '.join(json.dumps(a, ensure_ascii=False) for a in args)})")

    def wait_mark(self, name: str, timeout: float = MARK_TIMEOUT_S, required: bool = True) -> dict:
        """The step's paint times in epoch seconds: raf1 (the frame that paints it), raf2 (the frame after), and what
        the step recorded with them. A covered or hidden window paints nothing: required (autoplay) raises, else
        {"painted": False} comes back and the app goes on."""
        end = time.time() + timeout
        while time.time() < end:
            m = self.js(f"CLEF.mark({json.dumps(name)})")
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
        w.setFrame_display_(AppKit.NSMakeRect(x, top - HEIGHT, WIDTH, HEIGHT), True)
        w.orderFrontRegardless()
        if previous_app is not None:
            previous_app.activateWithOptions_(0)
        return window_facts(w, screen)

    return on_main(fn)


def place_for_editing(window) -> dict:
    """A normal window with a dark title bar, 540 pt wide and up to 960 pt tall (less on a smaller screen), centred,
    in front with the keyboard."""
    import AppKit

    def fn():
        w = window.native
        w.setAppearance_(AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameDarkAqua))
        screen = w.screen() or AppKit.NSScreen.mainScreen()
        vf = screen.visibleFrame()
        frame = w.frameRectForContentRect_(AppKit.NSMakeRect(0, 0, WIDTH, HEIGHT))
        h = min(frame.size.height, vf.size.height - 2 * MARGIN)
        x = vf.origin.x + (vf.size.width - frame.size.width) / 2
        y = vf.origin.y + (vf.size.height - h) / 2
        w.setFrame_display_(AppKit.NSMakeRect(x, y, frame.size.width, h), True)
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
    """Reads the lock file every 0.2 s (when CLEF_LOCK_FILE is set); records each change with the app's phase."""

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
    """The model repository's host on the files in hub/ (one row graph, both towers), and what this app needs around
    it: the plan of a request (its tokens, the graph it runs on), compiling the graphs (timed), and Decide with the
    unrounded probabilities of the same call."""

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

    def tower_file(self, tile: int) -> dict:
        return {"kind": "tower", "key": tile, "name": f"tower {tile} px", "file": self.clef.graphs.tower_files[tile].name}

    def plan(self, request: dict, tile: int) -> dict:
        """The host's checks and encoding of the request (refusals raise the host's RequestError); runs no graph.
        files = the graphs the request runs on: the row graph, and the tower of its tile when it has a picture."""
        enc = self.clef.host.encode(request, tile=tile)
        files = [self.row_file()] + ([self.tower_file(enc.tile)] if enc.tile is not None else [])
        return {"tokens": enc.record.tokens, "length": enc.length, "tile": enc.tile, "files": files}

    def compiled(self, f: dict) -> bool:
        g = self.clef.graphs
        if f["kind"] == "row":
            return g.row is not None and g.row[0] == f["key"]
        return f["key"] in g.towers

    def compile(self, f: dict) -> dict:
        """Compile one file now (what the host does on a request's first use), timed."""
        g = self.clef.graphs
        t = time.perf_counter()
        graph = g._row_graph(f["key"]) if f["kind"] == "row" else g._tower(f["key"])
        return {"file": f["file"], "kind": f["name"], "seconds": round(time.perf_counter() - t, 3),
                "fully_accelerated": graph.fully_accelerated(), "options": graph.options,
                "phys_footprint_after": phys_footprint()}

    def startup_files(self) -> list[dict]:
        return [self.row_file()] + [self.tower_file(t) for t in TILES]

    def decide(self, request: dict, tile: int) -> dict:
        """clef.decide(request) timed, with the float32 probabilities behind the response's 4-digit values."""
        C, g = self.C, self.clef.graphs
        n_compiled = len(g.compiled)
        self.kept = None
        e0 = time.time()
        t0 = time.perf_counter()
        response = self.clef.decide(request, tile=tile)
        t1 = time.perf_counter()
        e1 = time.time()
        enc, logits = self.kept
        probs = {q.question_id: dict(zip(q.option_ids, C.question_probs(z).tolist()))
                 for q, z in zip(enc.record.questions, logits)}
        return {"response": response, "probs": probs, "ms": (t1 - t0) * 1000.0, "epoch": [e0, e1],
                "tokens": enc.record.tokens, "length": enc.length, "tile": enc.tile,
                "compiled_during": g.compiled[n_compiled:],
                "response_is_its_probs": self.clef.host.respond(request, enc, logits) == response}

    def close(self):
        self.clef.close()


# ----------------------------------------------------------------------------------------------- the app


class App:
    """The editor's state on the Python side (the photo, the examples), Decide, and the run's record."""

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
        self.photo: Photo | None = None
        doc = json.loads((APP / "examples.json").read_text())
        self.examples, self.take = doc["examples"], doc["take"]
        self.ready = threading.Event()
        self.decide_lock = threading.Lock()
        self.seq = 0
        self.decided = threading.Condition()
        self.doc = {"tag": a.tag, "mode": "autoplay" if self.autoplay else "editor", "started_at": now_iso(),
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
        """The editor's record (--tag without --scenes): rewritten after every Decide and photo."""
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
        doc["hud"] = {"model": "clef-flash", "device": f"{chip} · LiteRT {version('ai-edge-litert')} · {accel_word}",
                      "pill": "● on-device"}
        page.call("hud", doc["hud"])
        page.call("examples", [{"id": x["id"], "title": x["title"]} for x in self.examples], list(TILES))
        if self.autoplay:
            page.call("load", self.take_editor(), None, None, TILES[0])
        else:
            page.call("load", *self.example_payload(self.examples[0]["id"]))
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
            doc["compile"] = self.compile_files(self.model.startup_files(), "loading")
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
        """One request of other content on the row graph, and one per tower; timed and recorded, not shown."""
        from PIL import Image

        grey = Image.new("RGB", (512, 512), (128, 128, 128))
        warm = [("text", {"model": E.MODEL, "state": WARM_TEXT_STATE, "questions": WARM_TEXT_QUESTIONS}, TILES[0])]
        warm += [(f"tile {t}", {"model": E.MODEL, "state": "", "questions": WARM_PICTURE_QUESTIONS, "images": [grey]}, t)
                 for t in TILES]
        out = []
        for name, req, tile in warm:
            r = self.model.decide(req, tile)
            out.append({"for": name, "request_sha256": canonical_sha256(req),
                        "request": {k: v for k, v in req.items() if k != "images"}
                        | ({"images": ["PIL RGB 512x512 (128,128,128)"]} if "images" in req else {}),
                        "tile": r["tile"], "tokens": r["tokens"], "length": r["length"], "ms": round(r["ms"], 3),
                        "epoch": r["epoch"], "compiled_during": r["compiled_during"], "response": r["response"]})
        return out

    # ---- examples and the photo
    def example(self, eid: str) -> dict:
        return next(x for x in self.examples if x["id"] == eid)

    def example_payload(self, eid: str) -> tuple:
        """(editor form, photo meta or None, example id, tile) for page.load; sets the photo."""
        x = self.example(eid)
        self.photo = open_photo(Path(x["picture"]).name, (self.hub / x["picture"]).read_bytes()) \
            if x.get("picture") else None
        form = E.editor_of({"state": x["state"], "questions": x["questions"]})
        return form, (self.photo.meta() if self.photo else None), eid, TILES[0]

    def take_editor(self) -> dict:
        """The recorded clip's opening editor: the take's example state and questions (examples.json "take"), no photo."""
        x = self.example(self.take["from"])
        self.photo = None
        return E.editor_of({"state": x["state"], "questions": {k: x["questions"][k] for k in self.take["questions"]}})

    def set_photo(self, name: str, data: bytes, origin: str) -> dict:
        """Every way in (Choose photo…, a drop on the page, an example, autoplay) ends here."""
        try:
            photo = open_photo(name, data)
        except E.DraftError as e:
            self.event("photo refused", origin=origin, name=name, reason=str(e))
            self.save_manual()
            return {"error": str(e)}
        self.photo = photo
        meta = photo.meta()
        self.event("photo", origin=origin, name=name, sha256=photo.sha256, bytes=len(data), size=list(photo.size),
                   format=photo.format)
        self.save_manual()
        return meta

    def clear_photo(self, origin: str) -> dict:
        self.photo = None
        self.event("photo removed", origin=origin)
        return {"ok": True}

    # ---- Decide
    def decide(self, draft: dict, origin: str) -> dict:
        """Decide on the editor's content: the same steps for a click and for autoplay."""
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
        photo = self.photo
        try:
            want = (draft.get("photo") or {}).get("id")
            if want != (photo.id if photo else None):
                raise E.DraftError("The photo changed while Decide was pressed; press Decide again.")
            tile = int(draft.get("tile") or TILES[0])
            if tile not in TILES:
                raise E.DraftError(f"Tile {tile}: the app reads a photo at {' or '.join(map(str, TILES))} px.")
            request = E.build_request(draft, photo.data if photo else None)
            plan = self.worker.call(lambda: model.plan(request, tile)) if model else None
        except Exception as e:   # the host's RequestError (RequestTooLong, ...) and the editor's DraftError
            rec["error"] = f"{type(e).__name__}: {e}"
            message = str(e)
            if type(e).__name__ == "RequestTooLong":
                message = f"Refused: {e}. Shorten the state or the questions."
            elif type(e).__name__ != "DraftError":
                message = f"Refused ({type(e).__name__}): {e}"
            page.call("error", message)
            self.finish(rec)
            return {"error": message}
        rec["request"] = self.describe_request(request, photo)
        rec["tile"] = tile if photo else None
        rec["request_sha256"] = canonical_sha256(request)
        rec["questions_ordered"] = json.dumps(request["questions"], ensure_ascii=False)
        page.call("running", f"press:{seq}")
        rec["marks"] = {"press": self.paint(f"press:{seq}")}
        rec["editor_shown"] = page.call("readEditor")
        if plan is not None:
            rec["plan"] = plan
            rec["compile"] = self.compile_files(plan["files"], "running")
            if rec["compile"]:
                self.status("running", "RUNNING")
        rec["lock_at_decide"] = read_lock()[:240]
        if model is not None:
            r = self.worker.call(lambda: model.decide(request, tile))
            rec["response"], rec["probs"] = r["response"], r["probs"]
            rec["ms"] = {"request_wall": r["ms"], "span": SPAN, "epoch": r["epoch"]}
            rec["tokens"], rec["length"] = r["tokens"], r["length"]
            rec["compiled_during"] = r["compiled_during"]
            rec["response_is_its_probs"] = r["response_is_its_probs"]
            payload = answer_payload(request, r["probs"], r["ms"], r["tokens"], r["length"], rec["tile"])
        else:
            rec["ms"] = {"request_wall": None, "span": "--ui-only: nothing ran"}
            payload = answer_payload(request, None, None, None, None, rec["tile"])
        page.call("answers", payload, f"answers:{seq}")
        rec["marks"]["answers"] = self.paint(f"answers:{seq}")
        rec["sent_to_page"] = payload
        rec["shown"] = page.call("shown")
        rec["layout"] = page.call("layout")
        self.finish(rec)
        return {"ok": True, "seq": seq}

    def describe_request(self, request: dict, photo: Photo | None) -> dict:
        out = {k: v for k, v in request.items() if k != "images"}
        if photo is not None:
            img = {"name": photo.name, "sha256": photo.sha256, "bytes": len(photo.data), "size": list(photo.size),
                   "format": photo.format}
            try:   # a fixture picture: where it comes from (hub/fixtures/photos.json, SHA256SUMS)
                for p in json.loads((self.hub / "fixtures/photos.json").read_text())["photos"]:
                    if p["sha256"] == photo.sha256:
                        img.update(file=p["file"], file_page=p["file_page"], author=p["author"], license=p["license"])
            except (OSError, KeyError, ValueError):
                pass
            out["images"] = [img]
        return out

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

    def example(self, eid):
        app = self._app
        form, meta, eid, tile = app.example_payload(eid)
        app.event("example", id=eid)
        app.save_manual()
        return {"editor": form, "photo": meta, "id": eid, "tile": tile}

    def choose_photo(self):
        import webview

        app = self._app
        # a scripted run (--tag) opens the panel in hub/fixtures/images, so a test sees only the repository's files
        start = str(app.hub / "fixtures/images") if app.tag else ""
        paths = app.page.w.create_file_dialog(webview.FileDialog.OPEN, directory=start, allow_multiple=False,
                                              file_types=("Pictures (*.jpg;*.jpeg;*.png)",))
        if not paths:
            app.event("photo chooser closed")
            return {"cancelled": True}
        p = Path(paths[0] if isinstance(paths, (list, tuple)) else paths)
        return app.set_photo(p.name, p.read_bytes(), "chooser")

    def drop_photo(self, name, b64):
        return self._app.set_photo(str(name), base64.b64decode(b64), "drop")

    def clear_photo(self):
        return self._app.clear_photo("click")


# ----------------------------------------------------------------------------------------------- autoplay


class Autoplay:
    """Scene P for a recording; each step goes through the page as a user's would."""

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
            if s != "P":
                raise ValueError(f"scene {s}: this app has scene P")
            tl, steps = {}, []
            tl["ARM"] = self.wait_trigger(f"ARM_{s}")
            app.page.call("load", app.take_editor(), None, None, TILES[0])
            app.page.call("ready", f"armed:{s}")
            tl["armed_shown"] = app.page.wait_mark(f"armed:{s}")
            app.write_new(f"ARMED_{s}", {"epoch": time.time()})
            tl["GO"] = self.wait_trigger(f"GO_{s}")
            seq_before = app.seq
            time.sleep(0.6)
            photo = app.hub / app.example(app.take["from"])["picture"]
            self.step(steps, "drop", "autoDropStart", target="the photo drop zone")
            time.sleep(0.35)
            meta = app.set_photo(photo.name, photo.read_bytes(), "autoplay")
            self.step(steps, "photo", "photoSet", meta, name=photo.name)
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
    """With --tag in the editor: run each <tag>.cmd.<n>.js that scripts/drive_editor.py writes (JavaScript on the
    page, the same handlers a click or a key press reaches) and write <tag>.cmd.<n>.json with its value; stop at
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
            return   # the editor runs until the window closes
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
    ap.add_argument("--length", type=int, default=1024, choices=LENGTHS,
                    help="the row graph in hub/ that every request runs on (default 1024 tokens)")
    ap.add_argument("--tag", help="name of this run's files in --trigger-dir (the editor: a record of every Decide, "
                                  "scripted input; autoplay: required)")
    ap.add_argument("--scenes", help="autoplay these scenes (P) for a recording instead of the editor")
    ap.add_argument("--accel", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--trigger-dir", default=str(D / "out"))
    ap.add_argument("--hub", default=str(D / "hub"))
    ap.add_argument("--stay", action="store_true", help="autoplay: keep the window after the run JSON until <tag>.QUIT")
    ap.add_argument("--trigger-timeout", type=float, default=1800.0)
    ap.add_argument("--ui-only", action="store_true", help="placeholder answers, no model (window and recorder tests)")
    a = ap.parse_args()
    if a.scenes and not a.tag:
        ap.error("--scenes needs --tag")

    hub = Path(a.hub)
    pictures = [x["picture"] for x in json.loads((APP / "examples.json").read_text())["examples"] if x.get("picture")]
    missing = [f for f in needed_files(a.length) + pictures if not (hub / f).is_file()]
    if missing and not a.ui_only:
        fetch = "zsh scripts/fetch_hub.sh" + ("" if a.length == 1024 else f" --length {a.length}")
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
        AppKit.NSActivityUserInitiated | AppKit.NSActivityLatencyCritical, "clef-flash requests (timed)")
    previous = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
    app = App(a)
    window = webview.create_window(TITLE, html=html_page(), js_api=Api(app), width=WIDTH, height=HEIGHT,
                                   resizable=not autoplay_or_scripted, min_size=(WIDTH, 560),
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
