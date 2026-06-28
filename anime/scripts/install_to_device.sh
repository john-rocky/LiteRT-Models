#!/usr/bin/env bash
# Push the AnimeGANv2 style tflites into the app's private filesDir.
#   ./scripts/install_to_device.sh <dir-with-anime_*_fp16.tflite>   (default: current dir)
set -e
PKG=com.anime; DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for S in paprika face_paint_512_v2; do
  M=anime_${S}_fp16.tflite
  adb push "$DIR/$M" "/data/local/tmp/$M"
  adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
  adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the AnimeGAN app."
