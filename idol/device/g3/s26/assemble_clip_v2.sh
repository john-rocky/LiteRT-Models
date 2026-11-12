#!/bin/bash
# Idol S26 clip v2 (2026-11-09): GUI takes only, no captions. The armchair's whole flow (start screen,
# photo picker, progress with the cutout, the last steps, one full turn of the splat at 1.2x), then four subjects each
# as the progress screen with the photo (0.8 s) and the turning splat (1.6x), ending on the turning axolotl.
set -euo pipefail
S=/Volumes/HD-SGDA/archive/Downloads/LiteRT-Models-idol-wt/idol/device/g3/s26
OUT=${1:?output mp4}
# take start end speed
SEGS=(
  "armchair_a 0.4 2.4 1.0"
  "armchair_a 2.4 10.6 4.0"
  "armchair_a 10.6 20.6 2.5"
  "armchair_b 24.5 27.4 2.0"
  "armchair_b 30.8 43.0 1.2"
  "backhoe_b 41.0 41.8 1.0"
  "backhoe_b 46.5 52.0 1.6"
  "puppyside_b 40.6 41.4 1.0"
  "puppyside_b 46.0 51.5 1.6"
  "sneaker_b 40.9 41.7 1.0"
  "sneaker_b 44.5 50.0 1.6"
  "axolotl_b 30.5 31.3 1.0"
  "axolotl_b 36.0 44.0 1.6"
)
ins=(); fc=''; cat=''
for i in "${!SEGS[@]}"; do
  read -r take a b sp <<< "${SEGS[$i]}"
  ins+=(-i "$S/idol_$take.mp4")
  # screenrecord writes a frame only when the screen changes: make the stream constant-rate first, or a trim
  # drops the frame still on screen (a static progress screen came out empty)
  fc+="[$i:v]fps=30,trim=start=$a:end=$b,setpts=(PTS-STARTPTS)/$sp,fps=30,format=yuv420p,setsar=1[v$i];"
  cat+="[v$i]"
done
fc+="${cat}concat=n=${#SEGS[@]}:v=1:a=0[out]"
nice -n 10 ffmpeg -v error -y "${ins[@]}" -filter_complex "$fc" -map '[out]' -c:v libx264 -preset slow -crf 20 \
  -pix_fmt yuv420p -movflags +faststart -an "$OUT"
