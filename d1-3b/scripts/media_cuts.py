#!/usr/bin/env python3
"""Where each scene's X cut starts and ends in its raw window recording, found from the frames themselves.

    venv-demo/bin/python -B scripts/media_cuts.py <tag> <scene>

Reads out/<tag>_<scene>.mp4 (record_window: constant 30 fps, one output frame per tick) and its frames.jsonl (which
screen capture each frame holds: src_seq). "Same picture" here = every 8 x 8 pixel block's mean grey level within
20 of the other frame's. Basis (two earlier takes of this app): one capture differs from itself by up to 60 levels
at single pixels and 5.55 in a block mean through the recorder's own H.264 keyframes, and x264 at the videos'
settings moves a block mean by up to 12.25 on its own (the end card encoded alone); one changed number in the footer
moves a block mean by 114 to 145 and a screen change by 213; the typing caret (2 pt wide) moves the block it stands
in by about 58 or more. Found:
  scene start   the first frame that is not the same picture as frame 0: the caret of the first field typed into
                (scene A) or the first tap (scene B); it must start a new capture;
  answers       the first frame whose status pill is the done colour after the first running one (the pill's box
                from the run JSON), as in check_frames.py; it must start a new capture;
  final screen  the first frame from which every later frame is the same picture as the recording's last frame. It
                must be the answers' frame: nothing on screen changes after the answers appear.
The cut keeps 1.0 s of READY before the scene start (all of it when shorter) and ends 3.0 s (90 frames) after the
final screen starts: start = max(0, scene start - 30), end = final screen + 90 (exclusive). Writes
out/<tag>_<scene>.cuts.json and prints "start end frames". Exit 1 when a check above fails or the recording has
fewer than `end` frames.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1]
OUT = D / "out"
FPS = 30
LEAD_FRAMES = 30        # 1.0 s of READY
HOLD_FRAMES = 90        # 3.0 s of the final screen
W, H = 270, 480         # 4 x 4 area means of the 1080 x 1920 frame; 2 x 2 of them = one 8 x 8 block
START_MEAN = 0.5
SAME_BLOCK = 20.0
DONE = (0x2E, 0x7D, 0x32)
RUNNING = (0x15, 0x65, 0xC0)
PILL_TOL = 40


def grey_frames(mp4: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", f"scale={W}:{H}:flags=area,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)


def blocks(f: np.ndarray) -> np.ndarray:
    """8 x 8 block means of the full frame, from the 4 x 4 area means."""
    return f.reshape(*f.shape[:-2], H // 2, 2, W // 2, 2).mean(axis=(-3, -1))


def pill_means(mp4: Path, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", f"crop={w}:{h}:{x}:{y}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h * w, 3).astype(np.float32).mean(axis=1)


def main() -> int:
    tag, scene = sys.argv[1], sys.argv[2]
    mp4 = OUT / f"{tag}_{scene}.mp4"
    run = json.loads((OUT / f"{tag}.run.json").read_text())
    rec = run["scenes"][scene]
    seq = [json.loads(x)["src_seq"] for x in (OUT / f"{tag}_{scene}.frames.jsonl").read_text().splitlines() if x.strip()]
    f = grey_frames(mp4)
    n = len(f)
    problems = []
    if len(seq) != n:
        problems.append(f"frames.jsonl has {len(seq)} lines, the video {n} frames")
    d0 = np.abs(f - f[0]).mean(axis=(1, 2))
    b = blocks(f)
    bl = np.abs(b - b[-1]).max(axis=(1, 2))           # block difference from the last frame, per frame
    b0 = np.abs(b - b[0]).max(axis=(1, 2))            # block difference from the first frame, per frame
    k_start = next((k for k in range(n) if b0[k] > SAME_BLOCK), None)
    differ = [k for k in range(n) if bl[k] > SAME_BLOCK]
    k_final = (differ[-1] + 1) if differ else 0
    lay = rec["layout"]
    dpr = float(lay["device_pixel_ratio"])
    r = lay["rects"]["pill"]
    box = (round((r["x"] + 3) * dpr), round((r["y"] + r["h"] / 2 - 3) * dpr), 8, 8)
    pill = pill_means(mp4, box)
    is_done = [np.abs(pill[k] - DONE).max() <= PILL_TOL for k in range(len(pill))]
    is_running = [np.abs(pill[k] - RUNNING).max() <= PILL_TOL for k in range(len(pill))]
    k_run = next((k for k in range(len(pill)) if is_running[k]), None)
    k_done = next((k for k in range(k_run, len(pill)) if is_done[k]), None) if k_run is not None else None
    new_capture = lambda k: k is not None and 0 < k < len(seq) and seq[k] != seq[k - 1]
    if k_start is None:
        problems.append("no frame differs from READY")
    elif not new_capture(k_start):
        problems.append(f"scene start frame {k_start} does not start a new capture")
    if k_done is None:
        problems.append("no frame with the done pill")
    elif not new_capture(k_done):
        problems.append(f"answers frame {k_done} does not start a new capture")
    if k_final != k_done:
        problems.append(f"final screen starts at frame {k_final}, the answers at {k_done}")
    start = max(0, (k_start or 0) - LEAD_FRAMES)
    end = k_final + HOLD_FRAMES
    if end > n:
        problems.append(f"the recording has {n} frames, the cut needs {end}")
    captures = [(k, s) for k, s in enumerate(seq) if k == 0 or s != seq[k - 1]]
    doc = {"tag": tag, "scene": scene, "video": mp4.name, "frames": n, "fps": FPS,
           "captures": [{"frame": k, "src_seq": s} for k, s in captures],
           "scene_start": k_start, "answers": k_done, "final_screen": k_final,
           "lead_frames": (k_start or 0) - start, "hold_frames": HOLD_FRAMES,
           "cut": {"start_frame": start, "end_frame": end, "frames": end - start, "seconds": (end - start) / FPS},
           "rule": {"scene_start": f"first frame not the same picture as frame 0 (an 8x8 block mean off by more "
                                   f"than {SAME_BLOCK})",
                    "same_picture": f"every 8x8 block mean within {SAME_BLOCK} grey levels",
                    "final_screen": "first frame from which every frame is the same picture as the last frame",
                    "answers": f"pill box {box} within {PILL_TOL} of #2E7D32, after the first #1565C0 frame"},
           "block_diff_from_last": {"before_final": round(float(bl[k_final - 1]), 2) if k_final else None,
                                    "max_from_final_on": round(float(bl[k_final:].max()), 2)},
           "d0_at_start": [round(float(x), 3) for x in d0[max(0, (k_start or 0) - 2):(k_start or 0) + 3]],
           "problems": problems}
    (OUT / f"{tag}_{scene}.cuts.json").write_text(json.dumps(doc, indent=1) + "\n")
    if problems:
        print(f"media_cuts {tag} {scene}: FAIL: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"{start} {end} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
