#!/usr/bin/env python3
"""Build the LiteRT showcase reel from curated demo media.

Port of CoreML-Models/scripts/hf_demo_video.py (the format Lu praised) to
the LiteRT-Models zoo: pairs wipe from input to output, stills hold, clips
play. Frames are generated with PIL and piped straight into ffmpeg — no
intermediate image files, and no drawtext dependency (the local ffmpeg
build has no libfreetype).

v2 lessons (user review): fill the frame (cover/contain with UPSCALING —
thumbnail() never enlarges), wipe only true image->image transforms,
full-res sources only.
v3 lessons (user review): sequences move (detection on video, LLM token
stream from a real device screen recording), TTS is audible (wav mixed
into the track at the segment offset), inpainting wipes from the REAL
original, and demo subjects are not all the same stock model.

Media: ~/Downloads/showcase-video/media_v2/<stem>.<ext>
Output: ~/Downloads/showcase-video/showcase_reel.mp4
        (1080x1080, H.264 + AAC, inside X's 140 s limit)

Run with a python that has PIL + numpy (e.g. ~/venvs/showcase/bin/python).

Stem syntax in REEL:
    "edsr"        whole image
    "edsr:L"      left half        "edsr:R"      right half
    "klein:0/3"   column 0 of a 3-panel composite (0-based)
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import wave as wavemod
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MEDIA = Path.home() / "Downloads" / "showcase-video" / "media_v2"
OUT = Path.home() / "Downloads" / "showcase-video" / "showcase_reel.mp4"

SIZE = 1080
BAR = 104                       # label strip at the bottom
BOX = SIZE - BAR                # square-ish content area
FPS = 24
BG = (16, 16, 18)
FG = (236, 236, 240)
DIM = (150, 150, 158)
ACCENT = (138, 180, 248)        # light blue

HOLD_A, WIPE, HOLD_B = 0.55, 0.55, 1.25   # before / wipe / after
STILL = 2.2
MOTION = 3.2
WIPE_TAG = "LiteRT GPU"

TTS_TEXT = {
    "kokoro": "Hello world. Welcome to on device text to speech.",
    "cmgan": "noisy input, then enhanced on device",
    "matcha": "Hello, this is Matcha running on the GPU.",
}
TTS_COLOR = {
    "matcha": (110, 230, 140),
}

# (label, task, kind, sources, display, duration_override)
#   pair  -> two stems, wipes from the first to the second
#   still -> one stem
#   anim  -> one stem (gif or mp4), spread across the slot
#   tts   -> one wav stem: text + progressing waveform, audio mixed in
#   display: "cover" fills the box (photos), "contain" letterboxes
REEL = [
    ("GFPGAN",            "face restoration",      "pair",  ["gfpgan_in", "gfpgan_out"],  "cover",   None),
    ("MI-GAN",            "object removal",        "pair",  ["inpaint_in", "inpaint_out"], "cover",  None),
    ("Depth Anything 3",  "monocular depth",       "pair",  ["da3:L", "da3:R"],           "cover",   None),
    ("MoGe-2",            "geometry + normals",    "pair",  ["moge_in", "moge_out"],      "cover",   None),
    ("Real-ESRGAN",       "4x super-resolution",   "pair",  ["esr_in", "esr_out"],        "cover",   None),
    ("MODNet",            "portrait matting",      "pair",  ["modnet_in", "modnet_out"],  "cover",   None),
    ("U2-Net Portrait",   "portrait sketch",       "pair",  ["portrait_in", "portrait_out"], "cover", None),
    ("Fast Style Transfer", "video style transfer", "anim", ["style"],                    "cover",   3.6),
    ("FLUX.2-klein 4B",   "image editing",         "pair",  ["klein:0/3", "klein:1/3"],   "cover",   None),
    ("DINOv2",            "dense features (PCA)",  "pair",  ["dinov2:L", "dinov2:R"],     "cover",   None),
    ("YOLOX-Tiny",        "object detection",      "anim",  ["detect"],                   "cover",   3.6),
    ("RTMPose",           "body pose",             "anim",  ["pose"],                     "cover",   3.4),
    ("RTMPose-Animal",    "animal pose",           "anim",  ["animal"],                   "cover",   3.4),
    ("SAM 2.1 Tiny",      "tap to segment",        "anim",  ["sam"],                      "cover",   3.6),
    ("YOLACT",            "instance segmentation", "still", ["yolact"],                   "cover",   None),
    ("BiSeNet",           "face parsing",          "pair",  ["faceparsing_in", "faceparsing_out"], "cover", None),
    ("Cloth Segmentation", "clothes parsing",      "pair",  ["clothseg_in", "clothseg_out"], "cover", None),
    # Live-count video needs an attribution-free crowd clip (Shibuya was
    # CC BY-SA -> removed per user rule). Static pair until one is provided.
    ("DM-Count",          "crowd counting",        "pair",  ["crowdcount:L", "crowdcount:R"], "cover", None),
    ("UniSal",            "video saliency",        "anim",  ["saliency"],                 "cover",   3.4),
    ("TwinLiteNet",       "lane + drivable area",  "still", ["twinlite:R"],               "cover",   None),
    ("M-LSD",             "line detection",        "pair",  ["mlsd:L", "mlsd:R"],         "cover",   None),
    ("PP-OCRv5",          "OCR",                   "still", ["ppocr"],                    "contain", None),
    ("PlantNet-300K",     "plant identification",  "pair",  ["plantnet_in", "plantnet_out"], "cover", None),
    ("Basic Pitch",       "audio to MIDI",         "still", ["basicpitch"],               "contain", None),
    ("Kokoro-82M",        "text-to-speech",        "tts",   ["kokoro"],                   "contain", None),
    ("Matcha-TTS",        "diffusion TTS",         "tts",   ["matcha"],                   "contain", None),
    ("CMGAN",             "speech enhancement",    "tts",   ["cmgan"],                    "contain", None),
    ("SmolVLM-256M",      "vision language model", "anim",  ["smolvlm"],                  "contain", 4.5),
    ("Qwen3-Embedding",   "semantic search",       "anim",  ["embed"],                    "contain", 4.0),
    ("RWKV-7",            "LLM on GPU",            "anim",  ["rwkv7"],                    "contain", 4.5),
]

TITLE_BIG = f"{len(REEL)} AI models"
TITLE_SUB = "one phone, LiteRT GPU"
OUTRO = [
    ("100% on-device / no cloud", 46, False, "FG"),
    ("github.com/john-rocky/LiteRT-Models", 46, True, "ACCENT"),
    ("", 20, False, "BG"),
    ("huggingface.co/mlboydaisuke", 34, False, "DIM"),
]


def font(px: int, bold: bool = True):
    names = ("Arial Bold.ttf", "Arial.ttf") if bold else ("Arial.ttf",)
    for n in names:
        try:
            return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{n}", px)
        except Exception:
            continue
    return ImageFont.load_default()


F_NAME, F_TASK, F_BIG, F_SMALL = font(46), font(32, False), font(84), font(38, False)


def parse_stem(stem: str) -> tuple[str, int | None, int | None]:
    if ":" not in stem:
        return stem, None, None
    base, spec = stem.split(":", 1)
    if spec == "L":
        return base, 0, 2
    if spec == "R":
        return base, 1, 2
    i, n = spec.split("/")
    return base, int(i), int(n)


def open_stem(stem: str, exts=(".png", ".jpg", ".jpeg", ".gif", ".mp4", ".wav")) -> Path:
    base = parse_stem(stem)[0]
    hits = [p for p in sorted(MEDIA.glob(f"{base}.*")) if p.suffix.lower() in exts]
    if not hits:
        sys.exit(f"missing media: {base}")
    return hits[0]


def still_image(stem: str) -> Image.Image:
    base, col, ncols = parse_stem(stem)
    im = Image.open(open_stem(stem))
    if getattr(im, "n_frames", 1) > 1:
        im.seek(im.n_frames // 3)
    im = im.convert("RGB")
    if col is not None:
        w, h = im.size
        im = im.crop((w * col // ncols, 0, w * (col + 1) // ncols, h))
    return im


def anim_frames(stem: str) -> list[Image.Image]:
    p = open_stem(stem)
    if p.suffix.lower() == ".mp4":
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-vf", f"fps={FPS//2}",
                            f"{td}/f%04d.png"], check=True)
            return [Image.open(f).convert("RGB").copy() for f in sorted(Path(td).glob("*.png"))]
    im = Image.open(p)
    out = []
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        out.append(im.convert("RGB").copy())
    return out


def fit(im: Image.Image, mode: str = "cover") -> Image.Image:
    """Fill the content box, upscaling small sources as needed."""
    canvas = Image.new("RGB", (SIZE, BOX), BG)
    w, h = im.size
    if mode == "cover":
        s = max(SIZE / w, BOX / h)
        c = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        x = (c.width - SIZE) // 2
        y = (c.height - BOX) // 2
        canvas.paste(c.crop((x, y, x + SIZE, y + BOX)), (0, 0))
    else:  # contain
        s = min(SIZE / w, BOX / h)
        c = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        canvas.paste(c, ((SIZE - c.width) // 2, (BOX - c.height) // 2))
    return canvas


def frame(content: Image.Image, name: str, task: str, tag: str | None = None) -> Image.Image:
    f = Image.new("RGB", (SIZE, SIZE), BG)
    f.paste(content, (0, 0))
    d = ImageDraw.Draw(f)
    d.text((36, BOX + 16), name, font=F_NAME, fill=FG)
    d.text((36, BOX + 64), task, font=F_TASK, fill=DIM)
    if tag:
        w = d.textlength(tag, font=F_SMALL)
        d.rectangle([SIZE - w - 60, 28, SIZE - 24, 84], fill=(0, 0, 0))
        d.text((SIZE - w - 42, 38), tag, font=F_SMALL, fill=ACCENT)
    return f


def card(lines, seconds: float):
    f = Image.new("RGB", (SIZE, SIZE), BG)
    d = ImageDraw.Draw(f)
    total = sum(fo.size + 26 for _, fo, _ in lines)
    y = (SIZE - total) // 2
    for text, fo, col in lines:
        w = d.textlength(text, font=fo)
        d.text(((SIZE - w) / 2, y), text, font=fo, fill=col)
        y += fo.size + 26
    for _ in range(int(seconds * FPS)):
        yield f


def seg_pair(name, task, stems, mode, dur):
    a = fit(still_image(stems[0]), mode)
    b = fit(still_image(stems[1]), mode)
    for _ in range(int(HOLD_A * FPS)):
        yield frame(a, name, task, "input")
    n = int(WIPE * FPS)
    for i in range(n):
        x = int(SIZE * (i + 1) / n)
        c = a.copy()
        c.paste(b.crop((0, 0, x, BOX)), (0, 0))
        d = ImageDraw.Draw(c)
        d.line([(x, 0), (x, BOX)], fill=ACCENT, width=4)
        yield frame(c, name, task, WIPE_TAG)
    for _ in range(int(HOLD_B * FPS)):
        yield frame(b, name, task, "output")


def seg_still(name, task, stems, mode, dur):
    im = fit(still_image(stems[0]), mode)
    for _ in range(int((dur or STILL) * FPS)):
        yield frame(im, name, task)


def seg_anim(name, task, stems, mode, dur):
    src = anim_frames(stems[0])
    if not src:
        return
    total = int((dur or MOTION) * FPS)
    # Always STRETCH the clip across the slot (v4 bug: short clips looped,
    # so 12 fps sequences played twice in their slot).
    idx = [min(len(src) - 1, i * len(src) // total) for i in range(total)]
    cache: dict[int, Image.Image] = {}
    for i in idx:
        if i not in cache:
            cache[i] = fit(src[i], mode)
        yield frame(cache[i], name, task)


def _wav_envelope(path: Path, bins: int) -> tuple[np.ndarray, float]:
    with wavemod.open(str(path), "rb") as w:
        n, sr, ch = w.getnframes(), w.getframerate(), w.getnchannels()
        raw = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)
    if ch > 1:
        raw = raw.reshape(-1, ch).mean(1)
    dur = len(raw) / sr
    chunk = max(1, len(raw) // bins)
    env = np.array([np.abs(raw[i * chunk:(i + 1) * chunk]).max() for i in range(bins)])
    env = env / (env.max() + 1e-6)
    return env, dur


def seg_tts(name, task, stems, mode, dur):
    """Sentence + waveform whose played portion lights up in sync."""
    wav = open_stem(stems[0], exts=(".wav",))
    BINS = 96
    env, wav_dur = _wav_envelope(wav, BINS)
    total = int(wav_dur * FPS) + int(0.3 * FPS)     # small tail hold
    text = TTS_TEXT.get(parse_stem(stems[0])[0], "")
    fo = font(44)
    # pre-wrap text to two centered lines
    words, lines, cur = text.split(), [], ""
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    for w_ in words:
        t = (cur + " " + w_).strip()
        if probe.textlength(t, font=fo) > SIZE - 160 and cur:
            lines.append(cur)
            cur = w_
        else:
            cur = t
    lines.append(cur)

    bar_w, gap = 7, 4
    total_w = BINS * (bar_w + gap) - gap
    x0 = (SIZE - total_w) // 2
    mid_y = BOX * 2 // 3
    accent = TTS_COLOR.get(parse_stem(stems[0])[0], ACCENT)
    for i in range(total):
        played = min(1.0, (i / FPS) / wav_dur)
        content = Image.new("RGB", (SIZE, BOX), BG)
        d = ImageDraw.Draw(content)
        ty = BOX // 4 - len(lines) * 30
        for ln in lines:
            lw = d.textlength(ln, font=fo)
            d.text(((SIZE - lw) / 2, ty), ln, font=fo, fill=FG)
            ty += 62
        for b in range(BINS):
            h = max(8, int(env[b] * 180))
            x = x0 + b * (bar_w + gap)
            col = accent if (b / BINS) <= played else (70, 70, 76)
            d.rectangle([x, mid_y - h, x + bar_w, mid_y + h], fill=col)
        yield frame(content, name, task)


SEG_FN = {"pair": seg_pair, "still": seg_still, "anim": seg_anim, "tts": seg_tts}


def seg_duration(kind, stems, dur) -> float:
    if kind == "pair":
        return HOLD_A + WIPE + HOLD_B
    if kind == "still":
        return dur or STILL
    if kind == "anim":
        return dur or MOTION
    if kind == "tts":
        return _wav_envelope(open_stem(stems[0], exts=(".wav",)), 8)[1] + 0.3
    raise ValueError(kind)


def build() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Segment start offsets -> audio events for the mux
    audio_events = []          # (start_seconds, wav_path)
    clock = 2.0                # title card
    for name, task, kind, stems, mode, dur in REEL:
        if kind == "tts":
            audio_events.append((clock, str(open_stem(stems[0], exts=(".wav",)))))
        elif kind == "anim":
            wav = MEDIA / f"{parse_stem(stems[0])[0]}.wav"
            if wav.exists():
                audio_events.append((clock, str(wav)))
        clock += seg_duration(kind, stems, dur)
    total_dur = clock + 2.6    # outro

    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}",
           "-r", str(FPS), "-i", "-"]
    for _, wav in audio_events:
        cmd += ["-i", wav]
    if audio_events:
        parts, mix = [], []
        for k, (start, _) in enumerate(audio_events):
            ms = int(start * 1000)
            parts.append(f"[{k+1}:a]aresample=44100,adelay={ms}:all=1[a{k}]")
            mix.append(f"[a{k}]")
        fc = ";".join(parts) + f";{''.join(mix)}amix=inputs={len(mix)}:normalize=0,apad[aout]"
        cmd += ["-filter_complex", fc, "-map", "0:v", "-map", "[aout]"]
    else:
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-map", "0:v", "-map", "1:a"]
    cmd += ["-t", f"{total_dur:.2f}", "-c:v", "libx264", "-preset", "medium", "-crf", "21",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "128k", str(OUT)]

    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def emit(g):
        for im in g:
            ff.stdin.write(im.tobytes())

    emit(card([(TITLE_BIG, F_BIG, FG),
               (TITLE_SUB, font(52, False), DIM)], 2.0))
    for name, task, kind, stems, mode, dur in REEL:
        print(f"  {name}", flush=True)
        emit(SEG_FN[kind](name, task, stems, mode, dur))
    colors = {"FG": FG, "ACCENT": ACCENT, "DIM": DIM, "BG": BG}
    emit(card([(t, font(px, b), colors[c]) for t, px, b, c in OUTRO], 2.6))

    ff.stdin.close()
    if ff.wait() != 0:
        sys.exit("ffmpeg failed")
    return OUT


if __name__ == "__main__":
    p = build()
    dur = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
        text=True).strip()
    print(f"{p}  {p.stat().st_size / 1e6:.1f} MB  {float(dur):.1f}s")
