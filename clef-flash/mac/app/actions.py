"""The devices of this Mac that the app reads and sets, and the fixed table from a decided change to a setting.

Seven devices (app/form.py writes them into the request):
    tv          the TV app       running: its process (pgrep -x TV); memory_mb: its resident memory (ps); set with
                                 `open -g -a TV` and SIGTERM to its process (no Apple Event: macOS would ask first)
    calculator  Calculator       the same
    draft_1     a TextEdit document named Draft (out/drafts/1/Draft.txt), opened in a TextEdit process of its own
    draft_2     a second one (out/drafts/2/Draft.txt); "open -n" gives each its own process, so one Draft is closed by
                ending its process, and a TextEdit you opened yourself is never touched
    music       a chord loop this app writes to out/music.wav and plays with afplay, again and again, in a process of
                its own (out/music.pid) that ends with the process that started it (the Music app and its library
                are never touched)
    sound       the output volume (0..100) and whether it is muted (osascript `get volume settings` / `set volume`);
                setting a volume unmutes, as it does on macOS
    screen      locked: read from the login session; never set: this app does not lock the screen, it only shows what
                the answer would do

TABLE is the only way from a decision to the Mac: (device, property, value) gives one of the fixed setters below;
anything else gives nothing. No text of the request or of the answers is run: every command is fixed, and its only
variable parts are the value from TABLE (a yes/no or a whole number 0..100) or what the app read from the Mac at its
start (to put it back).

snapshot() reads every device; restore() puts each one back: the apps running as they were, the drafts closed, the
music stopped, the volume and its mute as they were. As a script: `python app/actions.py read` prints a snapshot.
"""
from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path

APP_OUT = Path(__file__).resolve().parents[1] / "out"
APPS = {"tv": os.environ.get("MAC_TV_APP", "TV"), "calculator": "Calculator"}
DRAFTS = {"draft_1": APP_OUT / "drafts" / "1" / "Draft.txt", "draft_2": APP_OUT / "drafts" / "2" / "Draft.txt"}
DRAFT_TEXT = {"draft_1": "Draft\n\nNotes for Monday: the agenda, three open questions, and who brings the slides.\n",
              "draft_2": "Draft\n\nA shopping list: coffee beans, oat milk, batteries for the remote.\n"}
DRAFT_PIDS = APP_OUT / "drafts" / "pids.json"
DEVICES = ("tv", "calculator", "draft_1", "draft_2", "music", "sound", "screen")
NAMES = {"tv": "TV app", "calculator": "Calculator", "draft_1": "Draft (1)", "draft_2": "Draft (2)", "music": "Music",
         "sound": "Sound", "screen": "Screen"}
OSA_TIMEOUT_S = 15
WAIT_S = 8.0
MUSIC_VOLUME = "0.5"          # afplay -v: the loop's own level under the Mac's volume


class SettingError(RuntimeError):
    """A setting could not be read or set; the message says what macOS answered."""


def osa(script: str) -> str:
    r = subprocess.run(["/usr/bin/osascript", "-e", script], capture_output=True, text=True, timeout=OSA_TIMEOUT_S)
    if r.returncode != 0:
        raise SettingError((r.stderr or r.stdout).strip() or f"osascript exit {r.returncode}")
    return r.stdout.strip()


# ----------------------------------------------------------------------------------------------- reading

def app_pids(device: str) -> list[int]:
    r = subprocess.run(["/usr/bin/pgrep", "-x", APPS[device]], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split()]


def rss_mb(pids: list[int]) -> int:
    """The resident memory of the processes, in MB (ps rss is in KB)."""
    if not pids:
        return 0
    r = subprocess.run(["/bin/ps", "-o", "rss=", "-p", ",".join(str(p) for p in pids)], capture_output=True,
                       text=True)
    return round(sum(int(x) for x in r.stdout.split()) / 1024)


def read_app(device: str) -> dict:
    pids = app_pids(device)
    return {"running": bool(pids), "memory_mb": rss_mb(pids)}


def draft_pids() -> dict:
    try:
        return {k: int(v) for k, v in json.loads(DRAFT_PIDS.read_text()).items()}
    except (OSError, ValueError):
        return {}


def draft_alive(pid: int | None) -> bool:
    """The process is a TextEdit this app opened (the pid could be reused by another process after it ended)."""
    if not pid:
        return False
    cmd = subprocess.run(["/bin/ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
    return "TextEdit.app/Contents/MacOS/TextEdit" in cmd


def read_draft(device: str) -> dict:
    return {"open": draft_alive(draft_pids().get(device))}


def read_volume_raw() -> dict:
    """{"level": 0..100 or None, "muted": bool} of the default output device."""
    out = osa("get volume settings")
    m_level = re.search(r"output volume:(\d+)", out)
    m_muted = re.search(r"output muted:(true|false)", out)
    return {"level": int(m_level.group(1)) if m_level else None,
            "muted": bool(m_muted and m_muted.group(1) == "true")}


def read_sound() -> dict:
    raw = read_volume_raw()
    if raw["level"] is None:
        raise SettingError("this output device has no volume")
    return {"muted": raw["muted"], "volume": raw["level"]}


def read_screen() -> dict:
    import Quartz

    d = Quartz.CGSessionCopyCurrentDictionary() or {}
    return {"locked": bool(d.get("CGSSessionScreenIsLocked"))}


READERS = {"tv": lambda: read_app("tv"), "calculator": lambda: read_app("calculator"),
           "draft_1": lambda: read_draft("draft_1"), "draft_2": lambda: read_draft("draft_2"),
           "music": lambda: {"playing": PLAYER.state() == "playing"}, "sound": read_sound, "screen": read_screen}


def read_all() -> dict:
    """{device: {property: value}} as the Mac has it now; a device that cannot be read is left out, with its error in
    "errors"."""
    out, errors = {}, {}
    for device in DEVICES:
        try:
            out[device] = READERS[device]()
        except Exception as e:   # shown on the device's row
            errors[device] = f"{type(e).__name__}: {e}"
    if errors:
        out["errors"] = errors
    return out


def snapshot() -> dict:
    """What restore() needs: every device, the volume under its mute, and when."""
    s = read_all()
    try:
        s["volume_raw"] = read_volume_raw()
    except Exception as e:
        s["volume_raw"] = None
        s.setdefault("errors", {})["volume_raw"] = f"{type(e).__name__}: {e}"
    s["read_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    s["apps"] = dict(APPS)
    return s


# ----------------------------------------------------------------------------------------------- setting

def wait_until(test, what: str) -> None:
    end = time.time() + WAIT_S
    while not test():
        if time.time() > end:
            raise SettingError(f"{what} within {WAIT_S:g} s")
        time.sleep(0.05)


def set_app(device: str, running: bool) -> None:
    name = APPS[device]
    if running:
        r = subprocess.run(["/usr/bin/open", "-g", "-a", name], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise SettingError((r.stderr or r.stdout).strip() or f"open exit {r.returncode}")
        wait_until(lambda: app_pids(device), f"{name} did not start")
        return
    for pid in app_pids(device):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    wait_until(lambda: not app_pids(device), f"{name} did not quit")


def write_draft(device: str) -> Path:
    path = DRAFTS[device]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(DRAFT_TEXT[device])
    return path


def set_draft(device: str, opened: bool) -> None:
    pids = draft_pids()
    pid = pids.get(device)
    if opened:
        if draft_alive(pid):
            return
        path = write_draft(device)
        before = set(int(x) for x in subprocess.run(["/usr/bin/pgrep", "-x", "TextEdit"], capture_output=True,
                                                    text=True).stdout.split())
        # a TextEdit of its own (-n) in the background (-g), not reopening any earlier window
        r = subprocess.run(["/usr/bin/open", "-g", "-n", "-a", "TextEdit", str(path), "--args",
                            "-ApplePersistenceIgnoreState", "YES"], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise SettingError((r.stderr or r.stdout).strip() or f"open exit {r.returncode}")
        new: list[int] = []

        def started():
            now = set(int(x) for x in subprocess.run(["/usr/bin/pgrep", "-x", "TextEdit"], capture_output=True,
                                                     text=True).stdout.split())
            new[:] = sorted(now - before)
            return bool(new)

        wait_until(started, "TextEdit did not start")
        pids[device] = new[-1]
        DRAFT_PIDS.write_text(json.dumps(pids))
        return
    if draft_alive(pid):
        os.kill(pid, signal.SIGTERM)
        wait_until(lambda: not draft_alive(pid), f"{NAMES[device]} did not close")
    pids.pop(device, None)
    DRAFT_PIDS.write_text(json.dumps(pids))


def set_volume(level: int) -> None:
    """The output volume (0..100); macOS unmutes when a volume is set."""
    if isinstance(level, bool) or not isinstance(level, int) or not 0 <= level <= 100:
        raise SettingError(f"volume {level!r} is not a whole number 0..100")
    osa(f"set volume output volume {level} without output muted")


# (device, property) -> a setter of the value; the screen is never set (SHOWN_ONLY)
TABLE = {
    ("tv", "running"): lambda v: set_app("tv", v),
    ("calculator", "running"): lambda v: set_app("calculator", v),
    ("draft_1", "open"): lambda v: set_draft("draft_1", v),
    ("draft_2", "open"): lambda v: set_draft("draft_2", v),
    ("music", "playing"): lambda v: PLAYER.start() if v else PLAYER.stop(),
    ("sound", "volume"): set_volume,
}
SHOWN_ONLY = {("screen", "locked")}
KINDS = {("tv", "running"): bool, ("calculator", "running"): bool, ("draft_1", "open"): bool,
         ("draft_2", "open"): bool, ("music", "playing"): bool, ("sound", "volume"): int, ("screen", "locked"): bool}


def describe(device: str, prop: str, value) -> str:
    """A change in words: "Quit Calculator", "Close Draft (1)", "Play music", "Volume 40", "Lock the screen"."""
    if prop == "running":
        return f"{'Open' if value else 'Quit'} {NAMES[device]}"
    if prop == "open":
        return f"{'Open' if value else 'Close'} {NAMES[device]}"
    if prop == "playing":
        return "Play music" if value else "Stop music"
    if prop == "volume":
        return f"Volume {value}"
    if prop == "locked":
        return "Lock the screen" if value else "Unlock the screen"
    return f"{device}.{prop} = {value}"


def apply_change(device: str, prop: str, value) -> dict:
    """Set one decided change and read the device back: {device, prop, value, what, ok, read_back, error, seconds};
    a change TABLE does not hold is not run (ok False, "no fixed setting"), the screen is shown only."""
    t0 = time.perf_counter()
    rec = {"device": device, "prop": prop, "value": value, "what": describe(device, prop, value)}
    kind = KINDS.get((device, prop))
    if (device, prop) in SHOWN_ONLY:
        rec.update(ok=True, shown_only=True, read_back=READERS[device]().get(prop),
                   note="shown only: this app never locks the screen")
    elif (device, prop) not in TABLE or kind is None or type(value) is not kind:
        rec.update(ok=False, error=f"no fixed setting for {device}.{prop} = {json.dumps(value)}")
    else:
        try:
            TABLE[(device, prop)](value)
            rec["read_back"] = READERS[device]().get(prop)
            rec["ok"] = rec["read_back"] == value
            if not rec["ok"]:
                rec["error"] = f"set to {value}, reads {rec['read_back']}"
        except Exception as e:
            rec["ok"], rec["error"] = False, f"{e}"
            try:
                rec["read_back"] = READERS[device]().get(prop)
            except Exception:
                rec["read_back"] = None
    rec["seconds"] = round(time.perf_counter() - t0, 3)
    return rec


def restore(before: dict) -> list[dict]:
    """Put the Mac back to the snapshot `before`: each app running or not as it was, the drafts closed, the music
    stopped, the volume and its mute as they were. Each device on its own: one that fails does not stop the others."""
    out = []
    for device in ("tv", "calculator"):
        want = (before.get(device) or {}).get("running")
        t0 = time.perf_counter()
        rec = {"device": device, "target": {"running": want}}
        try:
            now = read_app(device)["running"]
            if want is None:
                rec.update(ok=False, error="not in the snapshot")
            elif now == want:
                rec.update(ok=True, unchanged=True, read_back=now)
            else:
                set_app(device, want)
                rec["read_back"] = read_app(device)["running"]
                rec["ok"] = rec["read_back"] == want
        except Exception as e:
            rec.update(ok=False, error=str(e))
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        out.append(rec)
    for device in DRAFTS:
        t0 = time.perf_counter()
        rec = {"device": device, "target": {"open": False}}
        try:
            set_draft(device, False)
            rec["read_back"] = read_draft(device)["open"]
            rec["ok"] = rec["read_back"] is False
        except Exception as e:
            rec.update(ok=False, error=str(e))
        rec["seconds"] = round(time.perf_counter() - t0, 3)
        out.append(rec)
    t0 = time.perf_counter()
    PLAYER.stop()
    out.append({"device": "music", "target": {"playing": False}, "read_back": PLAYER.state() == "playing",
                "ok": PLAYER.state() == "off", "seconds": round(time.perf_counter() - t0, 3)})
    raw = before.get("volume_raw") or {}
    t0 = time.perf_counter()
    rec = {"device": "sound", "target": raw}
    try:
        if raw.get("level") is not None:
            osa(f"set volume output volume {int(raw['level'])} {'with' if raw.get('muted') else 'without'} "
                "output muted")
        rec["read_back"] = read_volume_raw()
        rec["ok"] = rec["read_back"] == raw
    except Exception as e:
        rec.update(ok=False, error=str(e))
    rec["seconds"] = round(time.perf_counter() - t0, 3)
    out.append(rec)
    return out


def scene(volume: int = 25) -> list[dict]:
    """The bundled opening: the TV app and Calculator running, both drafts open, the music playing, the volume at
    `volume` (unmuted)."""
    steps = [("tv", "running", True), ("calculator", "running", True), ("draft_1", "open", True),
             ("draft_2", "open", True), ("sound", "volume", volume), ("music", "playing", True)]
    return [apply_change(*s) for s in steps]


# ----------------------------------------------------------------------------------------------- music

def write_music(path: Path, seconds: float = 32.0, rate: int = 22050) -> None:
    """A soft chord loop (C, A minor, F, G; two seconds each, twice), synthesised here: sine tones with a slow attack
    and release and a quiet arpeggio. Mono, 16-bit PCM."""
    import numpy as np

    chords = [(261.63, 329.63, 392.00), (220.00, 261.63, 329.63), (174.61, 220.00, 261.63), (196.00, 246.94, 293.66)]
    beat = 2.0
    n = int(seconds * rate)
    t = np.arange(n) / rate
    y = np.zeros(n)
    for k in range(int(seconds / beat)):
        a, b = int(k * beat * rate), int((k + 1) * beat * rate)
        tt = t[a:b] - k * beat
        env = np.minimum(1.0, tt / 0.25) * np.minimum(1.0, (beat - tt) / 0.4)
        for f in chords[k % 4]:
            y[a:b] += 0.18 * env * np.sin(2 * math.pi * f * tt)
        for j, f in enumerate(chords[k % 4]):   # one arpeggio note per half beat
            s = a + int(j * beat / 4 * rate)
            e = min(b, s + int(0.45 * rate))
            te = t[s:e] - t[s]
            y[s:e] += 0.10 * np.exp(-te * 6) * np.sin(2 * math.pi * 2 * f * te)
    y = np.clip(y / max(1e-9, np.abs(y).max()) * 0.6, -1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((y * 32767).astype("<i2").tobytes())
    tmp.rename(path)


# the music's loop: afplay on the WAV again each time it ends, while its owner process runs ($1 owner pid, $2 afplay's
# volume, $3 the WAV); after the owner has ended, the WAV plays to its end and the loop stops
MUSIC_LOOP = 'while kill -0 "$1" 2>/dev/null && /usr/bin/afplay -v "$2" "$3"; do :; done'


class Player:
    """The music: a loop of afplay on out/music.wav in a process of its own (a session of its own, out/music.pid), so
    it plays on after the process that started it has ended when that process names an owner in MAC_MUSIC_OWNER
    (scripts/take.sh sets the opening with `actions.py scene` before the app starts: the music plays for the whole
    take); without one the owner is the process that started it. Any process of this app reads and stops it through
    out/music.pid: the app when it puts the Mac back, --restore after a crash, take.sh at its end."""

    def __init__(self, wav: Path):
        self.wav = wav
        self.pidfile = wav.with_name("music.pid")
        self.proc = None   # the loop, when this process started it (reaped when it stops)

    def loop_pid(self) -> int | None:
        """The loop of out/music.pid when it still plays this app's music."""
        try:
            pid = int(self.pidfile.read_text())
        except (OSError, ValueError):
            return None
        cmd = subprocess.run(["/bin/ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
        return pid if "afplay" in cmd and str(self.wav) in cmd else None

    def state(self) -> str:
        return "playing" if self.loop_pid() else "off"

    def start(self) -> None:
        if self.loop_pid():
            return
        if not self.wav.is_file():
            write_music(self.wav)
        owner = os.environ.get("MAC_MUSIC_OWNER") or str(os.getpid())
        self.proc = subprocess.Popen(["/bin/sh", "-c", MUSIC_LOOP, "music", owner, MUSIC_VOLUME, str(self.wav)],
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     start_new_session=True)
        self.pidfile.write_text(str(self.proc.pid))
        wait_until(lambda: self.loop_pid() or self.proc.poll() is not None, "the music did not start")

    def stop(self) -> None:
        pid = self.loop_pid()
        if pid:
            try:   # the loop leads its own process group: the loop and its afplay; KILL, because a loop the app
                # started inherits the app's blocked TERM (mac_demo.py watch_signals)
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self.proc is not None:
            self.proc.wait(5)
            self.proc = None
        wait_until(lambda: not self.loop_pid(), "the music did not stop")
        self.pidfile.unlink(missing_ok=True)


PLAYER = Player(APP_OUT / "music.wav")


def windows_of(pids: set[int]) -> list[dict]:
    """The on-screen windows of the processes: id, owner pid and bounds in points, top-left origin."""
    import Quartz

    out = []
    for w in Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID):
        if w.get("kCGWindowOwnerPID") in pids and w.get("kCGWindowLayer") == 0:
            b = w.get("kCGWindowBounds", {})
            out.append({"id": int(w["kCGWindowNumber"]), "pid": int(w["kCGWindowOwnerPID"]),
                        "bounds_pt": [b.get("X"), b.get("Y"), b.get("Width"), b.get("Height")]})
    return out


def device_windows() -> dict:
    """{device: [windows]} of the apps and drafts (for a recording's checks)."""
    out = {d: windows_of(set(app_pids(d))) for d in APPS}
    pids = draft_pids()
    for d in DRAFTS:
        out[d] = windows_of({pids[d]}) if draft_alive(pids.get(d)) else []
    return out


USAGE = """usage:
  python app/actions.py read                         a snapshot of every device, as JSON
  python app/actions.py set <device>.<prop>=<value>  e.g. calculator.running=false sound.volume=25 draft_1.open=true
  python app/actions.py scene [volume]               the bundled opening (apps, drafts, music, volume 25)
  python app/actions.py restore <snapshot.json>      put every device back to a snapshot written by `read`
Music started here plays while the process of MAC_MUSIC_OWNER (a pid) runs; unset, that is this command, and the loop
plays once to its end."""

if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["read"]:
        print(json.dumps(snapshot(), indent=1))
    elif args[:1] == ["set"] and len(args) > 1:
        results = []
        for item in args[1:]:
            key, _, text = item.partition("=")
            device, _, prop = key.partition(".")
            results.append(apply_change(device, prop, json.loads(text)))
        print(json.dumps(results))
        sys.exit(0 if all(r["ok"] for r in results) else 1)
    elif args[:1] == ["scene"] and len(args) <= 2:
        results = scene(int(args[1]) if len(args) == 2 else 25)
        print(json.dumps(results))
        sys.exit(0 if all(r["ok"] for r in results) else 1)
    elif args[:1] == ["restore"] and len(args) == 2:
        results = restore(json.loads(Path(args[1]).read_text()))
        print(json.dumps(results))
        sys.exit(0 if all(r.get("ok") for r in results) else 1)
    else:
        print(USAGE, file=sys.stderr)
        sys.exit(2)
