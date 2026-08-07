#!/bin/bash
# showcase_assemble.sh — assemble recorded clips into one showcase video.
#
# Pipeline: title card -> [each clip: trim, scale/pad to 1080x1920, label
# overlay, audio normalize] -> end card -> concat -> showcase.mp4.
# Text is rendered to PNGs by showcase_overlay.py (Pillow) because the
# local ffmpeg build has no drawtext filter; ffmpeg burns them in with the
# core `overlay` filter. Everything lives OUTSIDE the repo in
# ~/Downloads/showcase-video/.
#
# Usage:
#   tools/showcase_assemble.sh manifest
#       (Re)generate a template manifest.tsv from clips/*.mp4 — then edit
#       order, labels, and optional trims by hand.
#   tools/showcase_assemble.sh
#       Build showcase.mp4 from manifest.tsv.
#
# manifest.tsv format (tab-separated, # comments allowed):
#   <clip filename>  <overlay label>  [start_sec]  [duration_sec]
# Example:
#   edsr.mp4    EDSR x4 Super-Resolution    2    7

set -u

ROOT="${SHOWCASE_DIR:-$HOME/Downloads/showcase-video}"
CLIPS="$ROOT/clips"
PARTS="$ROOT/parts"
MANIFEST="$ROOT/manifest.tsv"
OUT="$ROOT/showcase.mp4"
TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
OVERLAY_PY="$TOOLS_DIR/showcase_overlay.py"

W=1080; H=1920; FPS=30
TAGLINE="${TAGLINE:-100% on-device | LiteRT GPU}"
TITLE_LINE1="${TITLE_LINE1:-One phone. Every model on-device.}"
TITLE_LINE2="${TITLE_LINE2:-LiteRT GPU showcase}"
END_LINE1="${END_LINE1:-All models + Android samples: open source}"
END_LINE2="${END_LINE2:-github.com/john-rocky/LiteRT-Models}"
CARD_SECONDS=2

die() { echo "ERROR: $*" >&2; exit 1; }

command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg not found"
[[ -f "$OVERLAY_PY" ]] || die "missing $OVERLAY_PY"

gen_manifest() {
  [[ -d "$CLIPS" ]] || die "no clips dir: $CLIPS"
  { echo "# <clip>\t<label>\t[start_sec]\t[duration_sec] — reorder lines to set clip order"
    for f in "$CLIPS"/*.mp4; do
      [[ -e "$f" ]] || continue
      printf '%s\t%s\n' "$(basename "$f")" "$(basename "$f" .mp4 | tr '_-' ' ')"
    done
  } > "$MANIFEST"
  echo "wrote $MANIFEST — edit labels/order, then rerun without arguments"
}

# Shared encoder settings so the final concat can stream-copy.
ENCODE=(-r "$FPS" -pix_fmt yuv420p -c:v libx264 -preset medium -crf 20
        -c:a aac -ar 48000 -ac 2 -b:a 128k)

# card_part <output.mp4> <line1> <line2>
card_part() {
  local out="$1" png="${1%.mp4}.png"
  python3 "$OVERLAY_PY" card "$png" "$2" "$3" || die "overlay render failed"
  ffmpeg -y -v error -loop 1 -t "$CARD_SECONDS" -i "$png" \
    -f lavfi -i "anullsrc=r=48000:cl=stereo" \
    "${ENCODE[@]}" -shortest "$out" || die "encoding failed for $out"
}

build() {
  [[ -f "$MANIFEST" ]] || die "no manifest — run: $0 manifest"
  rm -rf "$PARTS"; mkdir -p "$PARTS"
  local list="$PARTS/concat.txt"; : > "$list"
  local n=0

  # Title card (first 3 seconds must say what the video is).
  card_part "$PARTS/000_title.mp4" "$TITLE_LINE1" "$TITLE_LINE2"
  echo "file '$PARTS/000_title.mp4'" >> "$list"

  while IFS=$'\t' read -r file label start dur; do
    [[ -z "$file" || "$file" == \#* ]] && continue
    local src="$CLIPS/$file"
    [[ -f "$src" ]] || { echo "SKIP (missing): $file" >&2; continue; }
    n=$((n+1))
    local part; part=$(printf '%s/%03d.mp4' "$PARTS" "$n")
    local banner="${part%.mp4}_banner.png"
    python3 "$OVERLAY_PY" banner "$banner" "$label" "$TAGLINE" \
      || die "overlay render failed for $label"

    local -a in=()
    [[ -n "${start:-}" ]] && in+=(-ss "$start")
    [[ -n "${dur:-}" ]] && in+=(-t "$dur")
    in+=(-i "$src")
    # Clips without an audio stream get silence so concat stays uniform.
    # Input order: 0 = clip, [1 = anullsrc if needed], last = banner PNG.
    local aidx=0 extra=()
    if ! ffprobe -v error -select_streams a -show_entries stream=codec_type \
        -of csv=p=0 "$src" | grep -q audio; then
      in+=(-f lavfi -i "anullsrc=r=48000:cl=stereo")
      aidx=1; extra=(-shortest)
    fi
    local bidx=$((aidx + 1))
    ffmpeg -y -v error "${in[@]}" -i "$banner" -filter_complex \
      "[0:v]scale=${W}:${H}:force_original_aspect_ratio=decrease,pad=${W}:${H}:(ow-iw)/2:(oh-ih)/2:black[base];[base][${bidx}:v]overlay=0:0:format=auto[v]" \
      -map "[v]" -map "${aidx}:a" "${ENCODE[@]}" ${extra[@]+"${extra[@]}"} "$part" \
      || die "encoding failed for $file"
    echo "file '$part'" >> "$list"
    echo "  [$n] $file — $label"
  done < "$MANIFEST"

  [[ "$n" -gt 0 ]] || die "no usable clips in manifest"

  card_part "$PARTS/999_end.mp4" "$END_LINE1" "$END_LINE2"
  echo "file '$PARTS/999_end.mp4'" >> "$list"

  ffmpeg -y -v error -f concat -safe 0 -i "$list" -c copy "$OUT" || die "concat failed"
  echo
  echo "== $OUT =="
  ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT" \
    | awk '{printf "duration: %.1fs (target 60-120s)\n", $1}'
  echo "$n clips + title/end cards. Review with: open '$OUT'"
}

case "${1:-build}" in
  manifest) gen_manifest ;;
  build) build ;;
  *) die "usage: $0 [manifest|build]" ;;
esac
