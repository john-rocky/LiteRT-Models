#!/usr/bin/env bash
# Push the RAM++ hybrid graphs into the app's private filesDir.
#   ./scripts/install_to_device.sh <dir-with-artifacts>   (default: ~/Downloads/meeting/ramplus-work/out)
# Graphs: G1 Swin stages0-2 (GPU), C2 stage-3 tail (CPU), R reweight (CPU), B tag head (GPU).
set -e
PKG=com.ram
DIR="${1:-$HOME/Downloads/meeting/ramplus-work/out}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for M in ram_swin_s012_fp16.tflite ram_stage3_tail_fp16.tflite ram_reweight_fp16.tflite ram_taghead_fp16.tflite; do
  echo "push $M"
  adb push "$DIR/$M" "/data/local/tmp/$M"
  adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
  adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the RAM++ app."
