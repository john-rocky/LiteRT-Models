#!/usr/bin/env bash
# Push the RTMPose-Animal tflite into the app's private filesDir.
#   ./scripts/install_to_device.sh <dir-with-rtm_animal_fp16.tflite>   (default: current dir)
set -e
PKG=com.rtmanimal
DIR="${1:-.}"
M=rtm_animal_fp16.tflite
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing $M ..."
adb push "$DIR/$M" "/data/local/tmp/$M"
adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
adb shell rm "/data/local/tmp/$M"
adb shell run-as $PKG ls -la files/
echo "done — launch the RTMPose-Animal app."
