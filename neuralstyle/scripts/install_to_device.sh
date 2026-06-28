#!/usr/bin/env bash
# Push the 4 style tflites into the app's private filesDir.
#   ./scripts/install_to_device.sh <dir-with-style_*_fp16.tflite>   (default: current dir)
set -e
PKG=com.neuralstyle; DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for S in candy mosaic rain_princess udnie; do
  M=style_${S}_fp16.tflite
  adb push "$DIR/$M" "/data/local/tmp/$M"
  adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
  adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the Neural Style app."
