#!/usr/bin/env bash
# Push the SAM 3.1 TRACKER graphs + host assets + autotest fixtures into the app's
# private filesDir. Layout expected by Sam3Tracker/TrackerAutotest:
#   files/sam3_vision_tri.tflite  files/sam3_text.tflite  files/sam3_head.tflite
#   files/sam3_token_embed.bin    files/vocab.json        files/merges.txt
#   files/tracker/graphs/trk_{memattn_n7,maskdec,memenc,initdec}.tflite
#   files/tracker/{consts/,flags.json,clip/,expected/}    (from models/device_tracker)
# While files/tracker/expected/manifest.json exists the app boots into tracker
# autotest mode; `adb shell run-as com.sam3 rm -r files/tracker` restores image mode.
#
#   ./scripts/install_tracker_to_device.sh
set -e
PKG=com.sam3
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../models/out"
PRE="$HERE/../models/tracker_precheck"
DEV="$HERE/../models/device_tracker"

for f in "$OUT/sam3_vision_tri.tflite" "$PRE/trk_memattn_n7.tflite" \
         "$DEV/expected/manifest.json"; do
  [ -f "$f" ] || { echo "missing $f (run build_sam3_tri / tracker_precheck / dump_tracker_device_assets first)"; exit 1; }
done

adb shell run-as $PKG mkdir -p files/tracker/graphs 2>/dev/null || true

push() {  # push <src> <dest-under-files/>
  adb push "$1" "/data/local/tmp/$(basename "$1")" >/dev/null
  adb shell run-as $PKG cp "/data/local/tmp/$(basename "$1")" "files/$2"
  adb shell rm "/data/local/tmp/$(basename "$1")"
  echo "  files/$2"
}

# image-side deps the tracker shares (skip the ones the image install already put there)
for M in sam3_text.tflite sam3_head.tflite sam3_token_embed.bin; do
  if ! adb shell run-as $PKG test -f "files/$M" 2>/dev/null; then
    push "$OUT/$M" "$M"
  fi
done
if ! adb shell run-as $PKG test -f files/vocab.json 2>/dev/null; then
  push "$OUT/sam3_tokenizer/vocab.json" vocab.json
  push "$OUT/sam3_tokenizer/merges.txt" merges.txt
fi
# the tracker trunk (vision_tri REPLACES sam3_vision for tracking; both may coexist)
if ! adb shell run-as $PKG test -f files/sam3_vision_tri.tflite 2>/dev/null; then
  push "$OUT/sam3_vision_tri.tflite" sam3_vision_tri.tflite
fi
for M in trk_memattn_n7.tflite trk_maskdec.tflite trk_memenc.tflite trk_initdec.tflite; do
  push "$PRE/$M" "tracker/graphs/$M"
done

# host assets + clip + fixtures (directory push, then copy the tree over)
adb push "$DEV" /data/local/tmp/sam3_tracker >/dev/null
for D in consts clip expected; do
  adb shell run-as $PKG rm -r "files/tracker/$D" 2>/dev/null || true
  adb shell run-as $PKG cp -r "/data/local/tmp/sam3_tracker/$D" "files/tracker/$D"
done
adb shell run-as $PKG cp /data/local/tmp/sam3_tracker/flags.json files/tracker/flags.json
adb shell rm -r /data/local/tmp/sam3_tracker

adb shell run-as $PKG ls files/tracker files/tracker/graphs
echo "done — launch the SAM3 app (it boots into tracker autotest mode);"
echo "watch: adb logcat -s SAM3TRK; result: adb shell run-as $PKG cat files/tracker_result.txt"
