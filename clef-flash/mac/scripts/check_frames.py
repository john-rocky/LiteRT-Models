#!/usr/bin/env python3
"""Hold a take's display recording against its run JSON: what the frames show, and when.

    venv-demo/bin/python -B scripts/check_frames.py out/<tag>.run.json

Reads <tag>_take.mp4 and .frames.jsonl next to the run JSON (written by record_display), decodes small boxes of every
frame with ffmpeg, and checks:
  - the video: H.264, 1920 x 1080, no audio stream, at most 20 s, one frames.jsonl line per decoded frame;
  - nothing else on the display: a box 80 pt below the app's window (out of its shadow), left of the TV app's window,
    shows the app's plain backdrop in every frame;
  - the status pill's colour per frame (idle light #D1D1D6 or dark #3A414C, running #1565C0, done #2E7D32): idle in
    the first frame; per scene, in order, a first running frame (Decide pressed) and the first done frame after it
    (the answers on screen), at least request_wall less one frame interval and at most request_wall + 0.25 s apart;
  - the command: in the command line's box, typed after its scene began and settled before its Decide frame;
  - scene M: Calculator's and both drafts' windows (their boxes at the scene's start) show their app until the
    answers are on screen and the backdrop after them, by the end of the scene; the TV app's box shows the TV app in
    every frame; the way of handling turns green (EXECUTE) at the answers;
  - scene D: both drafts' boxes show the drafts again before the command is typed and until the end; the way of
    handling turns blue (ASK) at the answers;
  - those frames' capture times (src_epoch) against the app's paint marks (raf1): reported, with the difference.
Every colour is compared as the recording holds it: the recorder captures sRGB and writes BT.709 video, and macOS
re-encodes the picture for the BT.709 transfer, which it takes as a 1.961 gamma (on the Mac it was tested on, the
backdrop #5B6B7C, sRGB (91, 107, 124), decodes as (80, 94, 111): `recorded()` gives (80.6, 95.9, 112.7)).
Prints a table and writes <tag>_take.frames_check.json. Exit 1 when a check fails.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def recorded(rgb) -> tuple:
    """An sRGB colour (0..255) as the recording holds it (see above): sRGB to linear light, then a 1.961 gamma."""
    def one(v):
        c = v / 255
        lin = c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        return 255 * lin ** (1 / 1.961)
    return tuple(one(v) for v in rgb)


PILL = {"idle": [recorded((0xD1, 0xD1, 0xD6)), recorded((0x3A, 0x41, 0x4C))], "running": [recorded((0x15, 0x65, 0xC0))],
        "done": [recorded((0x2E, 0x7D, 0x32))]}
BACKDROP = recorded((0x5B, 0x6B, 0x7C))   # app/mac_demo.py BACKDROP
TOL = 14             # max channel distance of a plain patch (H.264, yuv420p)
PILL_TOL = 40        # max channel distance to the nearest pill colour
SAME = 6.0           # mean channel difference under which a box matches another frame's
GREEN, BLUE = 50, 80  # the way word: max over its box of g - max(r, b) (EXECUTE) / b - r (ASK)
MAX_S = 20.0
W, H = 1920, 1080


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


def means(mp4: Path, box) -> np.ndarray:
    x, y, w, h = box
    return boxes(mp4, x, y, w, h).reshape(-1, w * h, 3).mean(axis=1)


def dist(a, b) -> float:
    return max(abs(p - q) for p, q in zip(a, b))


def main() -> int:
    run_path = Path(sys.argv[1])
    doc = json.loads(run_path.read_text())
    tag = doc["tag"]
    scenes = doc["scenes"]
    stem = run_path.parent / f"{tag}_take"
    mp4, frames_path = stem.with_suffix(".mp4"), stem.with_suffix(".frames.jsonl")
    problems, report = [], {"tag": tag, "video": str(mp4)}

    info = probe(mp4)
    video = [s for s in info["streams"] if s["codec_type"] == "video"]
    audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
    v = video[0]
    duration = float(info["format"]["duration"])
    report["probe"] = {"codec": v["codec_name"], "size": [v["width"], v["height"]], "r_frame_rate": v["r_frame_rate"],
                       "avg_frame_rate": v["avg_frame_rate"], "frames": int(v["nb_read_frames"]),
                       "duration_s": duration, "audio_streams": len(audio)}
    if v["codec_name"] != "h264" or (v["width"], v["height"]) != (W, H):
        problems.append(f"video {v['codec_name']} {v['width']}x{v['height']}, expected h264 {W}x{H}")
    if audio:
        problems.append(f"{len(audio)} audio stream(s): the take is silent")
    if duration > MAX_S:
        problems.append(f"video {duration:.2f} s > {MAX_S:g} s")
    lines = [json.loads(x) for x in frames_path.read_text().splitlines() if x.strip()]
    if len(lines) != int(v["nb_read_frames"]):
        problems.append(f"frames.jsonl has {len(lines)} lines, the video {v['nb_read_frames']} frames")
    epoch = [f["src_epoch"] for f in lines]

    def frame_at(t: float) -> int:
        """The first frame captured at or after epoch t (len(lines) when none)."""
        return next((k for k, e in enumerate(epoch) if e >= t), len(lines))

    win = doc["window"]
    scale = W / win["screen_pt"][2]
    cx, cy = win["content_topleft_pt"][0] * scale, win["content_topleft_pt"][1] * scale
    fx, fy, fw, fh = [p * scale for p in win["frame_topleft_pt"]]

    def page_box(rect: dict, pad: int = 2):
        x0, y0 = int(cx + rect["x"] * scale) + pad, int(cy + rect["y"] * scale) + pad
        x1, y1 = int(cx + (rect["x"] + rect["w"]) * scale) - pad, int(cy + (rect["y"] + rect["h"]) * scale) - pad
        return x0, y0, max(2, (x1 - x0) // 2 * 2), max(2, (y1 - y0) // 2 * 2)

    def win_box(bounds_pt):
        x, y, w, h = [p * scale for p in bounds_pt]
        return int(x + w * 0.35), int(y + h * 0.55), 24, 24

    # the pill per frame (its place is the same in every scene)
    first_scene = next(iter(scenes.values()))
    r = first_scene["layout"]["rects"]["pill"]
    pill_box = (int(cx + (r["x"] + 3) * scale), int(cy + (r["y"] + r["h"] / 2 - 3) * scale), 8, 8)
    classes = []
    for m in means(mp4, pill_box):
        best = min(((c, dist(m, col)) for c, cols in PILL.items() for col in cols), key=lambda t: t[1])
        classes.append(best[0] if best[1] <= PILL_TOL else "other")
    report["pill"] = {"box_px": list(pill_box), "frames": {c: classes.count(c) for c in ("idle", "running", "done", "other")}}
    if "other" in classes:
        problems.append(f"{classes.count('other')} frame(s) with the pill in no known colour")
    if classes and classes[0] != "idle":
        problems.append(f"the first frame's pill is {classes[0]}, expected idle")

    # nothing else on the display: below the app's window, past its shadow (50 pt below it, the shadow still darkened
    # the box's top rows), left of the TV app's window
    plain = (40, int(fy + fh) + 80, 40, 40)
    worst = max(dist(x, BACKDROP) for x in means(mp4, plain))
    report["backdrop"] = {"box": list(plain), "max_distance": round(float(worst), 2)}
    if worst > TOL:
        problems.append(f"the box below the app's window is not the backdrop in some frame ({worst:.1f} > {TOL})")

    steps_out, start = [], 0
    for s, rec in scenes.items():
        tl, lay = rec["timeline"], rec["layout"]
        k_go = frame_at(tl["GO"])
        k_run = next((k for k in range(max(start, k_go), len(classes)) if classes[k] == "running"), None)
        k_done = next((k for k in range(k_run, len(classes)) if classes[k] == "done"), None) if k_run is not None else None
        sc = {"go_frame": k_go, "running_frame": k_run, "done_frame": k_done}
        if k_run is None or k_done is None:
            problems.append(f"{s}: no running / done frames after its GO")
            report[s] = sc
            continue
        wall = rec["ms"]["request_wall"]
        gap = (epoch[k_done] - epoch[k_run]) * 1000
        sc.update(pressed_to_answers_ms=round(gap, 1), request_wall_ms=round(wall, 1))
        if gap < wall - 1000 / 30:
            problems.append(f"{s}: answers on screen {gap:.1f} ms after Decide, before the request's {wall:.1f} ms")
        if gap > wall + 250:
            problems.append(f"{s}: answers on screen {gap:.1f} ms after Decide, more than {wall:.1f} ms + 250 ms")
        k_end = frame_at(tl["DONE"])
        # the command: typed after the scene began, settled before Decide (the last typing: Decide was pressed on it)
        st = ([x for x in rec.get("steps", []) if x["what"] == "type"] or [None])[-1]
        if st is None:
            problems.append(f"{s}: no typing step")
        else:
            pix = boxes(mp4, *page_box(st["rect"], pad=8)).astype(np.float32)
            ref = pix[k_run - 1]
            diff = np.abs(pix - ref).mean(axis=(1, 2, 3))
            settled = next((k for k in range(k_go, k_run) if (diff[k:k_run] <= SAME).all()), None)
            sc["command_settled_frame"] = settled
            if settled is None or epoch[settled] < st["start"] - 0.05:
                problems.append(f"{s}: the command box does not settle after its typing started")
        # the way word's colour at the answers
        wb = page_box(lay["rects"]["way-word"], pad=1)
        px = boxes(mp4, *wb).astype(np.int32)
        green = (px[..., 1] - np.maximum(px[..., 0], px[..., 2])).reshape(len(px), -1).max(axis=1)
        blue = (px[..., 2] - px[..., 0]).reshape(len(px), -1).max(axis=1)
        want = "green" if rec["command_id"] == "quit_apps" else "blue"
        col = green if want == "green" else blue
        thr = GREEN if want == "green" else BLUE
        k_way = next((k for k in range(k_run, len(col)) if col[k] > thr), None)
        sc["way"] = {"box_px": list(wb), "colour": want, "first_frame": k_way,
                     "before_answers_max": int(col[max(k_go, k_run - 3):k_done].max()) if k_done > k_run else None}
        if k_way is None or k_way < k_done - 1 or not (col[k_done:k_end] > thr).all():
            problems.append(f"{s}: the way of handling is not {want} from the answers to the end of the scene")
        # the windows
        if rec["command_id"] == "quit_apps":
            at_arm = tl.get("windows_at_arm") or {}
            for d in ("calculator", "draft_1", "draft_2"):
                w = at_arm.get(d) or []
                if not w:
                    problems.append(f"{s}: no {d} window at the scene's start")
                    continue
                bm = means(mp4, win_box(w[0]["bounds_pt"]))
                is_bd = [dist(m, BACKDROP) <= TOL for m in bm]
                k_gone = next((k for k in range(k_done, len(is_bd)) if is_bd[k]), None)
                sc[f"{d}_gone_frame"] = k_gone
                if any(is_bd[k_go:k_done]):
                    problems.append(f"{s}: {d}'s box shows the backdrop before the answers")
                if k_gone is None or k_gone > k_end or not all(is_bd[k_gone:k_end]):
                    problems.append(f"{s}: {d}'s window does not close after the answers")
            tvw = (tl.get("windows_at_done") or {}).get("tv") or []
            if not tvw:
                problems.append(f"{s}: no TV app window at the end of the scene")
            else:
                tm = means(mp4, win_box(tvw[0]["bounds_pt"]))
                if any(dist(m, BACKDROP) <= TOL for m in tm[k_go:k_end]):
                    problems.append(f"{s}: the TV app's box shows the backdrop during the scene")
        else:
            at_done = tl.get("windows_at_done") or {}
            k_typed = frame_at(st["start"]) if st else k_run
            for d in ("draft_1", "draft_2"):
                w = at_done.get(d) or []
                if not w:
                    problems.append(f"{s}: no {d} window at the end of the scene")
                    continue
                bm = means(mp4, win_box(w[0]["bounds_pt"]))
                shown = [dist(m, BACKDROP) > TOL for m in bm]
                k_back = next((k for k in range(k_go, len(shown)) if shown[k]), None)
                sc[f"{d}_back_frame"] = k_back
                if k_back is None or k_back > k_typed or not all(shown[k_back:k_end]):
                    problems.append(f"{s}: {d} is not open from before the command to the end of the scene")
        for key, k in (("decide_pressed", k_run), ("answers_shown", k_done)):
            steps_out.append({"scene": s, "step": key, "frame": k, "pts_s": lines[k]["pts"],
                              "src_epoch": epoch[k], "app_raf1": tl[key]["raf1"],
                              "frame_minus_app_ms": round((epoch[k] - tl[key]["raf1"]) * 1000, 1)})
        report[s] = sc
        start = k_end
    report["steps"] = steps_out
    report["problems"] = problems
    report["pass"] = not problems
    (stem.parent / f"{stem.name}.frames_check.json").write_text(json.dumps(report, indent=1) + "\n")

    p = report["probe"]
    print(f"# {tag}: {p['codec']} {p['size'][0]}x{p['size'][1]}, {p['frames']} frames, {p['duration_s']:.3f} s, "
          f"r {p['r_frame_rate']}, avg {p['avg_frame_rate']}, audio streams {p['audio_streams']}")
    for s in scenes:
        print(f"{s}: {json.dumps(report.get(s))}")
    print("| scene | step | frame | pts s | frame capture (epoch) | app paint raf1 (epoch) | frame - app ms |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for x in steps_out:
        print(f"| {x['scene']} | {x['step']} | {x['frame']} | {x['pts_s']:.3f} | {x['src_epoch']:.3f} | "
              f"{x['app_raf1']:.3f} | {x['frame_minus_app_ms']} |")
    print(f"backdrop box max distance {report['backdrop']['max_distance']}; pill frames {report['pill']['frames']}")
    print("FRAMES PASS" if not problems else "FRAMES FAIL:\n  " + "\n  ".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
