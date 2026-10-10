#!/usr/bin/env python3
"""Check the X videos of one take (make_media_mac.sh) against the recordings they were cut from and the end card.

    venv-demo/bin/python -B scripts/check_media.py <tag> [--other <tag>]

For out/<tag>_x.mp4 (scene A, scene B, end card; at most 30 s), <tag>_A_x.mp4 and <tag>_B_x.mp4 (one scene and the
end card; at most 15 s each):
  1. ffprobe: H.264 High, yuv420p, 1080 x 1920, r_frame_rate and avg_frame_rate 30/1, no audio stream, BT.709 tags;
     the duration within its limit; decoded frames = the cuts' frames + the end card's 60 = duration x 30 (within 1).
  2. Every frame of each scene segment against the recording frame it was cut from (cuts.json start + i): the same
     picture (below). At every frame where the recording's screen changes (consecutive recording frames not the same
     picture: the ticket or photo appearing, Decide pressed, the answers), the video's frame is the same picture as
     the recording's new frame and not the old one, and the video's frame before it is the old one: nothing shifted,
     dropped, repeated or edited.
  3. The hold: the first and the last frame of the 3.0 s hold of each scene against the recording's last frame (the
     answers), and the first and last end-card frames against the PNG, full size in RGB (BT.709 limited range decoded
     the same way for all): the same picture. Also printed: the largest single-pixel difference (the launch asked for
     <= 2 / 255) next to two floors that no re-encode can go under: the recording against itself (its answers frame
     against its last frame: one capture, two of its keyframes) and the encoder alone (the picture encoded as a 1 s
     clip with the same settings).
  "Same picture" = every 8 x 8 block's mean within 20 levels (per RGB channel at full size; grey 2 x 2 of the 270 x 480
  area means for the per-frame pass). Basis (media_cuts.py): one capture differs from itself by up to 5.55 in a
  block mean through the recorder's keyframes, x264 alone by up to 12.25 (the end card's first frame), and one changed
  number in the footer by 114 or more. With --other <tag>, the same test is run on the other take's last frame of
  each scene against this take's (a different ms in the footer) and must say "different": the test sees one changed
  number. The background's mean colour on the last scene frame and on the first end-card frame is printed too (no
  visible step at the cut).
  4. A contact sheet of the main video, one frame per second (t = 0, 1, 2, ... s), labelled: <tag>_x_contact.png.
Prints the ffprobe lines and the numbers; writes <tag>_media_check.json; exit 1 on any failure.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

D = Path(__file__).resolve().parents[1]
OUT = D / "out"
FPS = 30
CARD_FRAMES = 60
W, H = 1080, 1920
GW, GH = 270, 480
LIMITS = {"x": 30.0, "A_x": 15.0, "B_x": 15.0}
SEGMENTS = {"x": ["A", "B"], "A_x": ["A"], "B_x": ["B"]}
ENC = ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-profile:v", "high", "-pix_fmt", "yuv420p",
       "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
RGB = "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24"
SAME_BLOCK = 20.0


def probe(mp4: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                          "stream=index,codec_type,codec_name,profile,pix_fmt,width,height,r_frame_rate,avg_frame_rate,"
                          "nb_read_frames,color_space,color_primaries,color_transfer,color_range:format=duration",
                          "-of", "json", str(mp4)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def grey(mp4: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", f"scale={GW}:{GH}:flags=area,format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, GH, GW).astype(np.float32)


def rgb_frame(mp4: Path, k: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", f"select=eq(n\\,{k}),{RGB}", "-frames:v",
                          "1", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3).astype(np.int16)


def png_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def block_max(a: np.ndarray, b: np.ndarray) -> float:
    """The largest difference of 8 x 8 block means between two full frames (per RGB channel when 3-D)."""
    def bm(x):
        return x.reshape(H // 8, 8, W // 8, 8, *x.shape[2:]).mean(axis=(1, 3))
    return float(np.abs(bm(a.astype(np.float32)) - bm(b.astype(np.float32))).max())


def floor_of(picture: np.ndarray, colour_in: str) -> dict:
    """The encoder's own difference on one picture: it alone as a 1 s clip with the same settings, decoded the same way.
    colour_in: the conversion into yuv420p used for the real video (the recordings are already yuv; the end card goes
    through the same scale as in make_media_mac.sh)."""
    with tempfile.TemporaryDirectory() as tmp:
        src, mp4 = Path(tmp) / "p.png", Path(tmp) / "p.mp4"
        Image.fromarray(picture.astype(np.uint8)).save(src)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-i", str(src), "-vf",
                        f"{colour_in},fps={FPS}", "-frames:v", str(FPS), *ENC, str(mp4)], check=True)
        first, last = rgb_frame(mp4, 0), rgb_frame(mp4, FPS - 1)
    return {"max": int(max(np.abs(first - picture).max(), np.abs(last - picture).max())),
            "mean": round(float(max(np.abs(first - picture).mean(), np.abs(last - picture).mean())), 4),
            "block_max": round(max(block_max(first, picture), block_max(last, picture)), 2)}


def diff(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    return {"max": int(d.max()), "mean": round(float(d.mean()), 4), "share_over_2": round(float((d > 2).mean()), 6),
            "block_max": round(block_max(a, b), 2)}


def contact_sheet(mp4: Path, frames: int, out: Path) -> dict:
    ks = list(range(0, frames, FPS))
    tw, th, cols, pad, label_h = 216, 384, 8, 8, 30
    rows = (len(ks) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + pad) + pad, rows * (th + label_h + pad) + pad), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 20)
    for i, k in enumerate(ks):
        img = Image.fromarray(rgb_frame(mp4, k).astype(np.uint8)).resize((tw, th), Image.LANCZOS)
        x, y = pad + (i % cols) * (tw + pad), pad + (i // cols) * (th + label_h + pad)
        sheet.paste(img, (x, y + label_h))
        d.text((x + 2, y + 4), f"t={k // FPS} s  #{k}", font=font, fill=(235, 235, 235))
    sheet.save(out, optimize=True)
    return {"file": out.name, "frames": ks, "size": list(sheet.size)}


def main() -> int:
    tag = sys.argv[1]
    other = sys.argv[sys.argv.index("--other") + 1] if "--other" in sys.argv else None
    cuts = {s: json.loads((OUT / f"{tag}_{s}.cuts.json").read_text()) for s in ("A", "B")}
    raw = {s: OUT / f"{tag}_{s}.mp4" for s in ("A", "B")}
    card_png = OUT / f"{tag}_endcard.png"
    lines, problems, report = [], [], {"tag": tag, "videos": {}}

    raw_grey = {s: grey(raw[s]) for s in ("A", "B")}
    raw_last = {s: rgb_frame(raw[s], cuts[s]["frames"] - 1) for s in ("A", "B")}
    card = png_rgb(card_png)
    floors = {s: floor_of(raw_last[s], "format=yuv420p") for s in ("A", "B")}
    floors["card"] = floor_of(card, "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p")
    self_floor = {s: diff(rgb_frame(raw[s], cuts[s]["final_screen"]), raw_last[s]) for s in ("A", "B")}
    report["encoder_floor"] = floors
    report["recording_against_itself"] = self_floor
    lines.append("floors: recording against itself (answers frame vs last frame) " + ", ".join(
        f"{s} max {d['max']} mean {d['mean']} block {d['block_max']}" for s, d in self_floor.items())
        + "; encoder alone " + ", ".join(f"{k} max {d['max']} mean {d['mean']} block {d['block_max']}"
                                          for k, d in floors.items()))
    if other:
        sens = {}
        for s in ("A", "B"):
            o = OUT / f"{other}_{s}.mp4"
            n_o = probe(o)["streams"][0]["nb_read_frames"]
            sens[s] = round(block_max(rgb_frame(o, int(n_o) - 1), raw_last[s]), 2)
        report["sensitivity"] = {"other": other, "block_max": sens, "says_different": all(v > SAME_BLOCK for v in sens.values())}
        lines.append(f"sensitivity: {other}'s last frame vs {tag}'s (another ms in the footer): block max "
                     + ", ".join(f"{s} {v}" for s, v in sens.items())
                     + f" -> {'different (PASS)' if report['sensitivity']['says_different'] else 'SAME (FAIL)'}")
        if not report["sensitivity"]["says_different"]:
            problems.append(f"the same-picture test does not see the other take's footer: {sens}")

    def hold_ok(dd: dict) -> bool:
        return dd["block_max"] <= SAME_BLOCK

    for name, segs in SEGMENTS.items():
        mp4 = OUT / f"{tag}_{name}.mp4"
        info = probe(mp4)
        v = [s for s in info["streams"] if s["codec_type"] == "video"][0]
        audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
        dur = float(info["format"]["duration"])
        n = int(v["nb_read_frames"])
        want = sum(cuts[s]["cut"]["frames"] for s in segs) + CARD_FRAMES
        rep = {"probe": {k: v.get(k) for k in ("codec_name", "profile", "pix_fmt", "width", "height", "r_frame_rate",
                                                "avg_frame_rate", "nb_read_frames", "color_space", "color_primaries",
                                                "color_transfer", "color_range")},
               "duration": dur, "audio_streams": len(audio), "frames_expected": want, "segments": {}}
        lines.append(f"{mp4.name}: {v['codec_name']} {v.get('profile')} {v['pix_fmt']} {v['width']}x{v['height']} "
                     f"r {v['r_frame_rate']} avg {v['avg_frame_rate']} {n} frames {dur:.3f} s, audio streams "
                     f"{len(audio)}, colour {v.get('color_space')}/{v.get('color_primaries')}/"
                     f"{v.get('color_transfer')}/{v.get('color_range')}")
        bad = []
        if (v["codec_name"], v.get("profile"), v["pix_fmt"], v["width"], v["height"]) != ("h264", "High", "yuv420p", W, H):
            bad.append("not h264 High yuv420p 1080x1920")
        if v["r_frame_rate"] != "30/1" or v["avg_frame_rate"] != "30/1":
            bad.append(f"frame rate {v['r_frame_rate']} / {v['avg_frame_rate']}")
        if audio:
            bad.append(f"{len(audio)} audio stream(s)")
        if (v.get("color_space"), v.get("color_primaries"), v.get("color_transfer")) != ("bt709",) * 3:
            bad.append("colour tags not BT.709")
        if dur > LIMITS[name]:
            bad.append(f"{dur:.3f} s > {LIMITS[name]} s")
        if n != want or abs(n - dur * FPS) > 1:
            bad.append(f"{n} frames, expected {want} = {dur:.3f} s x 30 within 1")

        g = grey(mp4)
        off = 0
        for s in segs:
            c = cuts[s]["cut"]
            start, frames = c["start_frame"], c["frames"]
            seg = g[off:off + frames]
            src = raw_grey[s][start:start + frames]
            means = np.abs(seg - src).mean(axis=(1, 2))
            gb = lambda x: x.reshape(*x.shape[:-2], GH // 2, 2, GW // 2, 2).mean(axis=(-3, -1))
            blocks = np.abs(gb(seg) - gb(src)).max(axis=(1, 2))
            sb, vb = gb(src), gb(seg)
            changes = [i for i in range(1, frames) if np.abs(sb[i] - sb[i - 1]).max() > SAME_BLOCK]
            shifted = [i for i in changes
                       if np.abs(vb[i] - sb[i]).max() > SAME_BLOCK or np.abs(vb[i] - sb[i - 1]).max() <= SAME_BLOCK
                       or np.abs(vb[i - 1] - sb[i - 1]).max() > SAME_BLOCK]
            hold_first = off + (cuts[s]["final_screen"] - start)
            hold_last = off + frames - 1
            hf, hl = diff(rgb_frame(mp4, hold_first), raw_last[s]), diff(rgb_frame(mp4, hold_last), raw_last[s])
            rep["segments"][s] = {"frames": [off, off + frames - 1], "from_recording": [start, start + frames - 1],
                                  "grey_mean_diff_max": round(float(means.max()), 3),
                                  "grey_block_diff_max": round(float(blocks.max()), 2),
                                  "recording_changes_at": changes, "shifted": shifted,
                                  "hold": {"first_frame": hold_first, "last_frame": hold_last,
                                           "frames": hold_last - hold_first + 1, "first_vs_recording_last": hf,
                                           "last_vs_recording_last": hl, "floor": floors[s]}}
            lines.append(f"  scene {s}: video frames {off}..{off + frames - 1} = recording frames {start}.."
                         f"{start + frames - 1}; per frame: grey mean diff max {means.max():.3f}, block max "
                         f"{blocks.max():.2f}; recording changes at {changes}, shifted {shifted}; hold frames "
                         f"{hold_first}..{hold_last} ({hold_last - hold_first + 1}) vs the recording's last frame: "
                         f"first block {hf['block_max']} (pixel max {hf['max']}, mean {hf['mean']}), last block "
                         f"{hl['block_max']} (pixel max {hl['max']}, mean {hl['mean']})")
            if blocks.max() > SAME_BLOCK:
                bad.append(f"scene {s}: a frame is not the same picture as its recording frame (block {blocks.max():.2f})")
            if shifted:
                bad.append(f"scene {s}: frames {shifted} nearer the recording's previous frame")
            if hold_last - hold_first + 1 != 90:
                bad.append(f"scene {s}: hold of {hold_last - hold_first + 1} frames, expected 90")
            for which, dd in (("first", hf), ("last", hl)):
                if not hold_ok(dd):
                    bad.append(f"scene {s}: hold {which} frame block {dd['block_max']} > {SAME_BLOCK}")
            off += frames
        last_scene, first_card = rgb_frame(mp4, off - 1), rgb_frame(mp4, off)
        bg_patch = lambda x: [round(float(c), 2) for c in x[1880:1904, 40:1040].reshape(-1, 3).mean(axis=0)]
        rep["background_at_the_cut"] = {"last_scene_frame": bg_patch(last_scene), "first_card_frame": bg_patch(first_card),
                                        "png": bg_patch(card)}
        lines.append(f"  background at the cut (mean RGB of a strip at y 1880..1903): last scene frame "
                     f"{bg_patch(last_scene)}, first end-card frame {bg_patch(first_card)}, PNG {bg_patch(card)}")
        cf, cl = diff(first_card, card), diff(rgb_frame(mp4, off + CARD_FRAMES - 1), card)
        rep["end_card"] = {"frames": [off, off + CARD_FRAMES - 1], "first_vs_png": cf, "last_vs_png": cl,
                           "floor": floors["card"]}
        lines.append(f"  end card: video frames {off}..{off + CARD_FRAMES - 1} vs the PNG: first block {cf['block_max']} "
                     f"(pixel max {cf['max']}, mean {cf['mean']}), last block {cl['block_max']} (pixel max "
                     f"{cl['max']}, mean {cl['mean']})")
        for which, dd in (("first", cf), ("last", cl)):
            if not hold_ok(dd):
                bad.append(f"end card {which} frame block {dd['block_max']} > {SAME_BLOCK}")
        if off + CARD_FRAMES != n:
            bad.append(f"segments end at {off + CARD_FRAMES}, the video has {n} frames")
        rep["problems"] = bad
        report["videos"][name] = rep
        problems += [f"{mp4.name}: {b}" for b in bad]

    x = OUT / f"{tag}_x.mp4"
    report["contact_sheet"] = contact_sheet(x, int(report["videos"]["x"]["probe"]["nb_read_frames"]),
                                            OUT / f"{tag}_x_contact.png")
    report["problems"] = problems
    report["pass"] = not problems
    (OUT / f"{tag}_media_check.json").write_text(json.dumps(report, indent=1) + "\n")
    print("\n".join(lines))
    print(f"contact sheet: {report['contact_sheet']['file']} ({len(report['contact_sheet']['frames'])} frames, "
          f"{report['contact_sheet']['size'][0]}x{report['contact_sheet']['size'][1]})")
    print("MEDIA PASS" if not problems else "MEDIA FAIL:\n  " + "\n  ".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
