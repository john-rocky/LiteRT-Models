"""Card hero: the app's view of a real diarization -- waveform on top, one lane per speaker below.

Runs nemotron3_diar_litert.py (low_latency streaming, 0.1 s pushes, the shipped files) on a 16 kHz wav and draws
the waveform (colored where exactly one speaker is active) over the per-speaker timeline (SPK 1..N in arrival order,
the app's TimelineView colors), with each speaker's talk time. Writes a PNG and a JSON of the segments.

Usage: .venv/bin/python make_hero_app.py --wav hero_16k.wav --model-dir <repo root> --out assets/hero.png
           [--json hero_segments.json] [--accelerator cpu] [--precision fp32]
"""

import argparse
import json

import numpy as np

import nemotron3_diar_litert as n3l

# TimelineView.COLORS (app), speaker k = arrival order k
COLORS = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#AB47BC", "#00ACC1", "#FF7043", "#9E9D24"]


def mmss(t, _=None):
  return f"{int(t // 60)}:{int(t % 60):02d}"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--wav", required=True)
  ap.add_argument("--model-dir", default=None)
  ap.add_argument("--out", required=True)
  ap.add_argument("--json", default=None)
  ap.add_argument("--accelerator", default="cpu")
  ap.add_argument("--precision", default="fp32")
  args = ap.parse_args()

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.ticker import FuncFormatter, MultipleLocator

  audio = n3l.load_wav(args.wav)
  d = n3l.Nemotron3Diarizer(args.model_dir, "low_latency", args.accelerator, args.precision)
  steps = []
  for i in range(0, audio.shape[0], 1600):
    steps += d.push(audio[i : i + 1600])
  steps += d.finish()
  logits = np.concatenate([s.logits for s in steps], 0)
  active = n3l.sigmoid(logits) > 0.5  # [frames, 8], one frame per 10 ms
  segs = n3l.speaker_segments(logits)
  speakers = sorted({s["Speaker"] for s in segs})
  dur = audio.shape[0] / n3l.SAMPLE_RATE
  talk = {k: active[:, k].sum() * n3l.FRAME_SECONDS for k in speakers}

  fig = plt.figure(figsize=(16, 2.3 + 0.62 * len(speakers)), dpi=100)
  gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 0.62 * len(speakers)], hspace=0.08,
                        left=0.075, right=0.915, top=0.86, bottom=0.14)
  ax_w = fig.add_subplot(gs[0])
  ax_t = fig.add_subplot(gs[1], sharex=ax_w)

  # waveform: min / max per 10 ms frame, gray, colored where exactly one speaker is active
  n = min(active.shape[0], audio.shape[0] // n3l.HOP)
  frames = audio[: n * n3l.HOP].reshape(n, n3l.HOP)
  lo, hi = frames.min(1), frames.max(1)
  t = (np.arange(n) + 0.5) * n3l.FRAME_SECONDS
  peak = max(np.abs(lo).max(), np.abs(hi).max(), 1e-6)
  ax_w.fill_between(t, lo / peak, hi / peak, color="#C9CDD2", linewidth=0)
  solo = active[:n].sum(1) == 1
  owner = active[:n].argmax(1)
  for k in speakers:
    m = solo & (owner == k)
    ax_w.fill_between(t, np.where(m, lo / peak, np.nan), np.where(m, hi / peak, np.nan), color=COLORS[k],
                      linewidth=0)
  ax_w.set_ylim(-1.05, 1.05)
  ax_w.set_yticks([])
  for side in ("top", "right", "left"):
    ax_w.spines[side].set_visible(False)
  ax_w.tick_params(axis="x", labelbottom=False, length=0)

  # speaker lanes
  lane_h = 0.62
  for row, k in enumerate(speakers):
    y = len(speakers) - 1 - row
    ax_t.add_patch(plt.Rectangle((0, y + (1 - lane_h) / 2), dur, lane_h, color="#F1F3F4", linewidth=0))
    for s in segs:
      if s["Speaker"] == k:
        ax_t.add_patch(plt.Rectangle((s["Start"], y + (1 - lane_h) / 2), s["End"] - s["Start"], lane_h,
                                     color=COLORS[k], linewidth=0))
    ax_t.text(-0.012 * dur, y + 0.5, f"SPK {k + 1}", color=COLORS[k], fontsize=15, fontweight="bold",
              ha="right", va="center")
    ax_t.text(dur * 1.01, y + 0.5, f"{talk[k]:.1f} s", color="#3C4043", fontsize=13, ha="left", va="center")
  ax_t.set_xlim(0, dur)
  ax_t.set_ylim(0, len(speakers))
  ax_t.set_yticks([])
  for side in ("top", "right", "left"):
    ax_t.spines[side].set_visible(False)
  ax_t.xaxis.set_major_locator(MultipleLocator(10))
  ax_t.xaxis.set_major_formatter(FuncFormatter(mmss))
  ax_t.tick_params(axis="x", labelsize=12, colors="#5F6368")
  ax_t.set_xlabel("time (m:ss)", fontsize=12, color="#5F6368")

  fig.text(0.075, 0.935, "Who spoke when", fontsize=19, fontweight="bold", color="#202124", ha="left")
  fig.text(0.915, 0.935,
           f"streaming, low_latency · {len(speakers)} speakers · {len(segs)} segments · {dur:.1f} s",
           fontsize=13, color="#5F6368", ha="right")
  fig.savefig(args.out)
  summary = dict(wav=args.wav, seconds=round(dur, 2), steps=len(steps), speakers=len(speakers),
                 segments=len(segs), talk_seconds={f"SPK {k + 1}": round(float(talk[k]), 2) for k in speakers},
                 accelerator=args.accelerator, precision=args.precision, segment_list=segs)
  if args.json:
    with open(args.json, "w") as f:
      json.dump(summary, f, indent=1)
  print(json.dumps({k: v for k, v in summary.items() if k != "segment_list"}))


if __name__ == "__main__":
  main()
