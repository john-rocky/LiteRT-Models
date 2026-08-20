#!/usr/bin/env bash
# Gather the SAM3 model files into one folder for a single drag&drop into the
# app's Documents (Finder -> iPhone -> Files -> SAM3), renaming the tokenizer
# files the way the app expects.
#
#   ./stage_models.sh [dest]          image-side files only (default dest: ./stage)
#   ./stage_models.sh tracker [dest]  also stage the tracker payload (drag the
#                                     `tracker` FOLDER into Documents as well;
#                                     its presence switches the app into the
#                                     tracker autotest on next launch)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../models/out"
PRE="$HERE/../models/tracker_precheck"
DEVTRK="$HERE/../models/device_tracker"
TRACKER=0
DEST=""
for a in "$@"; do
  if [ "$a" = "tracker" ]; then TRACKER=1; else DEST="$a"; fi
done
DEST="${DEST:-$HERE/stage}"
mkdir -p "$DEST"
for f in sam3_vision.tflite sam3_text.tflite sam3_head.tflite sam3_token_embed.bin; do
  cp -c "$OUT/$f" "$DEST/$f" 2>/dev/null || cp "$OUT/$f" "$DEST/$f"
done
cp "$OUT/sam3_tokenizer/vocab.json" "$DEST/vocab.json"
cp "$OUT/sam3_tokenizer/merges.txt" "$DEST/merges.txt"
if [ "$TRACKER" = 1 ]; then
  cp -c "$OUT/sam3_vision_tri.tflite" "$DEST/sam3_vision_tri.tflite" 2>/dev/null \
    || cp "$OUT/sam3_vision_tri.tflite" "$DEST/sam3_vision_tri.tflite"
  mkdir -p "$DEST/tracker/graphs"
  for g in trk_memattn_n7.tflite trk_maskdec.tflite trk_memenc.tflite trk_initdec.tflite; do
    cp "$PRE/$g" "$DEST/tracker/graphs/$g"
  done
  cp -RL "$DEVTRK/consts" "$DEST/tracker/consts"
  cp -RL "$DEVTRK/clip" "$DEST/tracker/clip"
  cp -RL "$DEVTRK/expected" "$DEST/tracker/expected"
  cp "$DEVTRK/flags.json" "$DEST/tracker/flags.json"
fi
du -sh "$DEST"
echo "Drag the contents of $DEST into Finder -> iPhone -> Files -> SAM3."
