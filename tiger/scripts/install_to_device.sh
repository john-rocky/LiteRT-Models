#!/usr/bin/env bash
# Push the three TIGER-DnR tflite models into the app's private filesDir (models are not committed).
#   ./scripts/install_to_device.sh <dir-with-tiger_*_fp16.tflite>   (default: current dir)
set -e
PKG=com.tiger; DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for M in tiger_dialog_fp16.tflite tiger_effect_fp16.tflite tiger_music_fp16.tflite; do
  adb push "$DIR/$M" "/data/local/tmp/$M"
  adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
  adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the TIGER app."
