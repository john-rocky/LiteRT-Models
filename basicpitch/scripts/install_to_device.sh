#!/usr/bin/env bash
# Push the Basic Pitch model into the app's private filesDir.
#   ./scripts/install_to_device.sh <dir-with-basicpitch.tflite>   (default: current dir)
set -e
PKG=com.basicpitch; DIR="${1:-.}"; M=basicpitch.tflite
adb shell run-as $PKG mkdir files 2>/dev/null || true
adb push "$DIR/$M" "/data/local/tmp/$M"
adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
adb shell rm "/data/local/tmp/$M"
adb shell run-as $PKG ls -la files/
echo "done — launch the Basic Pitch app."
