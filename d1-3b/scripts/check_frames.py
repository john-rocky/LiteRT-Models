#!/usr/bin/env python3
"""Hold one scene's video against the run JSON: what the frames show, and when.

    venv-demo/bin/python -B scripts/check_frames.py out/<tag>.run.json <scene>

Reads <tag>_<scene>.mp4 and .frames.jsonl next to the run JSON (written by record_window), decodes small boxes of
every frame with ffmpeg, and checks:
  - the video: H.264, 1080 x 1920, no audio stream, at most 30 s, one frames.jsonl line per decoded frame;
  - window only: the four corners and the top strip where a title bar would be are the app's background (#0E1116)
    in every frame (no title bar, no rounded corner, nothing of another window);
  - typing: for every field the scene typed into (the run's steps: the field's box, when the typing started and
    ended), the white text in that box grows while it is typed: the bright pixels (all three channels > 160) in the
    frames captured during the typing rise from the first such frame to the last (after the field was cleared, for a
    replaced name), rise in at least a quarter as many frames as there are characters (2 at least), and never fall by
    more than a tenth of the final count; the typing ends before the Decide frame;
  - the status pill's colour per frame (the pill's box from the run JSON layout x the device pixel ratio): idle
    #3A414C, running #1565C0, done #2E7D32. The first frame is idle, or, for a scene that opens on the previous
    scene's answers, done and then idle before Decide. The first running frame after the first idle one = Decide
    pressed, the first done frame after it = the answers on screen; the gap between them is at least request_wall
    less one frame interval (the screen cannot show the answers before the request returned) and at most
    request_wall + 0.25 s;
  - those frames' capture times (src_epoch) against the app's paint marks (timeline decide_pressed / answers_shown
    raf1): reported, with the difference.
Prints a table and writes <tag>_<scene>.frames_check.json. Exit 1 when a check fails.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

COLOURS = {"idle": (0x3A, 0x41, 0x4C), "running": (0x15, 0x65, 0xC0), "done": (0x2E, 0x7D, 0x32)}
BG = (0x0E, 0x11, 0x16)
BG_TOL = 14          # max channel distance of a background patch (H.264, yuv420p)
PILL_TOL = 40        # max channel distance to the nearest pill colour
BRIGHT = 160         # a text pixel: every channel above this (white text; the placeholder and the field are darker)
W, H = 1080, 1920


def probe(mp4: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                          "stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_frames:"
                          "format=duration", "-of", "json", str(mp4)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def boxes(mp4: Path, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Every frame's pixels in the box (x, y, w, h): [frames, h, w, 3] uint8."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", f"crop={w}:{h}:{x}:{y}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def dist(a, b) -> float:
    return max(abs(p - q) for p, q in zip(a, b))


def px_box(rect: dict, dpr: float, pad: int = 2) -> tuple[int, int, int, int]:
    x0 = max(0, int(rect["x"] * dpr) + pad)
    y0 = max(0, int(rect["y"] * dpr) + pad)
    x1 = min(W, int((rect["x"] + rect["w"]) * dpr) - pad)
    y1 = min(H, int((rect["y"] + rect["h"]) * dpr) - pad)
    return x0, y0, (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2


def typing_check(mp4: Path, lines: list, step: dict, dpr: float) -> dict:
    x, y, w, h = px_box(step["rect"], dpr)
    pix = boxes(mp4, x, y, w, h)
    counts = (pix > BRIGHT).all(axis=3).sum(axis=(1, 2))
    span = [k for k, f in enumerate(lines) if step["start"] - 0.05 <= f["src_epoch"] <= step["end"] + 0.15]
    out = {"field": step.get("field"), "chars": step.get("chars"), "box_px": [x, y, w, h], "frames": len(span),
           "problems": []}
    if len(span) < 2:
        out["problems"].append(f"{len(span)} frame(s) captured while {step.get('field')} was typed")
        return out
    c = counts[span[0]:span[-1] + 1]
    low = int(np.argmin(c))            # a replaced value is cleared first: the rise starts at its lowest point
    rise = c[low:]
    ups = int((np.diff(rise) > 0).sum())
    final = int(rise[-1])
    worst_fall = int(max(0, (np.maximum.accumulate(rise) - rise).max()))
    need = max(2, (step.get("chars") or 0) // 4)
    out.update(first_frame=span[0], last_frame=span[-1], bright_first=int(c[0]), bright_lowest=int(rise[0]),
               bright_last=final, rises=ups, rises_needed=need, worst_fall=worst_fall,
               counts=[int(v) for v in c])
    if final <= int(rise[0]):
        out["problems"].append(f"{step.get('field')}: the text did not grow ({int(rise[0])} -> {final} bright px)")
    if ups < need:
        out["problems"].append(f"{step.get('field')}: the text grew in {ups} frame(s), fewer than {need}")
    if worst_fall > 0.1 * final:
        out["problems"].append(f"{step.get('field')}: the text shrank by {worst_fall} bright px while typed")
    return out


def main() -> int:
    run_path, scene = Path(sys.argv[1]), sys.argv[2]
    doc = json.loads(run_path.read_text())
    tag = doc["tag"]
    rec = doc["scenes"][scene]
    stem = run_path.parent / f"{tag}_{scene}"
    mp4, frames_path = stem.with_suffix(".mp4"), stem.with_suffix(".frames.jsonl")
    problems, report = [], {"tag": tag, "scene": scene, "video": str(mp4)}

    info = probe(mp4)
    streams = info["streams"]
    video = [s for s in streams if s["codec_type"] == "video"]
    audio = [s for s in streams if s["codec_type"] == "audio"]
    v = video[0]
    duration = float(info["format"]["duration"])
    report["probe"] = {"codec": v["codec_name"], "size": [v["width"], v["height"]], "r_frame_rate": v["r_frame_rate"],
                       "avg_frame_rate": v["avg_frame_rate"], "frames": int(v["nb_read_frames"]),
                       "duration_s": duration, "audio_streams": len(audio)}
    if v["codec_name"] != "h264" or (v["width"], v["height"]) != (W, H):
        problems.append(f"video {v['codec_name']} {v['width']}x{v['height']}, expected h264 {W}x{H}")
    if audio:
        problems.append(f"{len(audio)} audio stream(s): the take is silent")
    if duration > 30.0:
        problems.append(f"video {duration:.2f} s > 30 s")

    lines = [json.loads(x) for x in frames_path.read_text().splitlines() if x.strip()]
    if len(lines) != int(v["nb_read_frames"]):
        problems.append(f"frames.jsonl has {len(lines)} lines, the video {v['nb_read_frames']} frames")

    # window only: corners and the top strip are the background in every frame
    corners = {"top_left": (0, 0, 24, 24), "top_right": (W - 24, 0, 24, 24), "bottom_left": (0, H - 24, 24, 24),
               "bottom_right": (W - 24, H - 24, 24, 24), "top_strip": (120, 0, W - 240, 40)}
    bg = {}
    for name, (x, y, w, h) in corners.items():
        means = boxes(mp4, x, y, w, h).reshape(-1, w * h, 3).mean(axis=1)
        worst = max(dist(m, BG) for m in means)
        bg[name] = {"box": [x, y, w, h], "max_distance_to_bg": round(float(worst), 2)}
        if worst > BG_TOL:
            problems.append(f"{name} is not the background in some frame (distance {worst:.1f} > {BG_TOL})")
    report["background"] = bg

    # the pill per frame
    lay = rec["layout"]
    dpr = float(lay["device_pixel_ratio"])
    r = lay["rects"]["pill"]
    px, py = round((r["x"] + 3) * dpr), round((r["y"] + r["h"] / 2 - 3) * dpr)
    means = boxes(mp4, px, py, 8, 8).reshape(-1, 64, 3).mean(axis=1)
    classes = []
    for m in means:
        name = min(COLOURS, key=lambda c: dist(m, COLOURS[c]))
        classes.append(name if dist(m, COLOURS[name]) <= PILL_TOL else "other")
    report["pill_box_px"] = [px, py, 8, 8]
    report["pill_classes"] = {c: classes.count(c) for c in ("idle", "running", "done", "other")}
    if "other" in classes:
        problems.append(f"{classes.count('other')} frame(s) with the pill in no known colour")
    first_idle = next((k for k, x in enumerate(classes) if x == "idle"), None)
    if classes and classes[0] not in ("idle", "done"):
        problems.append(f"the first frame's pill is {classes[0]}, expected idle (or done: the previous scene's answers)")
    report["opens_on"] = "the previous scene's answers" if classes and classes[0] == "done" else "the editor"
    k_run = next((k for k in range(first_idle or 0, len(classes)) if classes[k] == "running"), None) \
        if first_idle is not None else None
    k_done = next((k for k in range(k_run, len(classes)) if classes[k] == "done"), None) if k_run is not None else None
    first = {"running": k_run, "done": k_done}

    # typing: the text grows in each typed field, before Decide
    typed = []
    for st in rec.get("steps", []):
        if st["what"] != "type":
            continue
        t = typing_check(mp4, lines, st, dpr)
        typed.append(t)
        problems += t["problems"]
        if k_run is not None and t.get("last_frame") is not None and t["last_frame"] >= k_run:
            problems.append(f"{t['field']}: typed until frame {t['last_frame']}, Decide at {k_run}")
    report["typing"] = typed
    if not typed:
        problems.append("no typed field in the run's steps")

    tl = rec["timeline"]
    wall_ms = rec["ms"]["request_wall"]
    rows = []
    for key, cls in (("decide_pressed", "running"), ("answers_shown", "done")):
        k = first[cls]
        if k is None or k >= len(lines):
            problems.append(f"no {cls} frame (step {key})")
            continue
        f = lines[k]
        mark = tl[key]["raf1"]
        rows.append({"step": key, "frame": k, "pts_s": f["pts"], "src_epoch": f["src_epoch"], "app_raf1": mark,
                     "frame_minus_app_ms": round((f["src_epoch"] - mark) * 1000, 1)})
    report["steps"] = rows
    if first["running"] is not None and first["done"] is not None:
        a, b = lines[first["running"]], lines[first["done"]]
        gap_ms = (b["src_epoch"] - a["src_epoch"]) * 1000
        frame_ms = 1000.0 / 30
        report["pressed_to_answers_frames_ms"] = round(gap_ms, 1)
        report["request_wall_ms"] = wall_ms
        if wall_ms is not None:
            report["gap_minus_request_wall_ms"] = round(gap_ms - wall_ms, 1)
            if gap_ms < wall_ms - frame_ms:
                problems.append(f"answers on screen {gap_ms:.1f} ms after Decide, before the request's {wall_ms:.1f} ms")
            if gap_ms > wall_ms + 250:
                problems.append(f"answers on screen {gap_ms:.1f} ms after Decide: more than the request's "
                                f"{wall_ms:.1f} ms + 250 ms")
    report["problems"] = problems
    report["pass"] = not problems
    (stem.parent / f"{stem.name}.frames_check.json").write_text(json.dumps(report, indent=1) + "\n")

    p = report["probe"]
    print(f"# {tag} scene {scene}: {p['codec']} {p['size'][0]}x{p['size'][1]}, {p['frames']} frames, "
          f"{p['duration_s']:.3f} s, r {p['r_frame_rate']}, avg {p['avg_frame_rate']}, audio streams {p['audio_streams']}"
          f"; opens on {report['opens_on']}")
    for t in typed:
        if "bright_last" in t:
            print(f"typing {t['field']}: {t['chars']} characters, frames {t['first_frame']}..{t['last_frame']} "
                  f"({t['frames']}), bright px {t['bright_first']} -> lowest {t['bright_lowest']} -> {t['bright_last']}, "
                  f"rose in {t['rises']} frames (>= {t['rises_needed']}), worst fall {t['worst_fall']}")
    print("| step | first frame | pts s | frame capture (epoch) | app paint raf1 (epoch) | frame - app ms |")
    print("|---|---:|---:|---:|---:|---:|")
    for x in rows:
        print(f"| {x['step']} | {x['frame']} | {x['pts_s']:.3f} | {x['src_epoch']:.3f} | {x['app_raf1']:.3f} | "
              f"{x['frame_minus_app_ms']} |")
    if "pressed_to_answers_frames_ms" in report:
        print(f"pressed -> answers frames: {report['pressed_to_answers_frames_ms']} ms; request_wall: "
              f"{'—' if wall_ms is None else f'{wall_ms:.1f}'} ms")
    print("background patches (max distance to #0E1116): "
          + ", ".join(f"{k} {x['max_distance_to_bg']}" for k, x in bg.items()))
    print(f"pill frames: {report['pill_classes']}")
    print("FRAMES PASS" if not problems else "FRAMES FAIL:\n  " + "\n  ".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
