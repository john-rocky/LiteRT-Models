#!/usr/bin/env bash
# Push the MobileNetV4 tflite into the app's private filesDir.
# Build with scripts/build_mnv3.py (produces mnv4_fp16.tflite), then:
#   ./scripts/install_to_device.sh <dir-with-the-tflite>   (default: current dir)
set -e
PKG=com.mnv4
DIR="${1:-.}"
M=mnv4_fp16.tflite
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing $M ..."
adb push "$DIR/$M" "/data/local/tmp/$M"
adb shell chmod 644 "/data/local/tmp/$M"
adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
adb shell rm "/data/local/tmp/$M"
adb shell run-as $PKG ls -la files/
echo "done — launch the MobileNetV4 app."
