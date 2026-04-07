#!/usr/bin/env bash
# Move staged Kokoro model and voices into the app's private files dir.
#
# Run this AFTER `com.kokoro` is installed on the device (via Android Studio
# build & run). The first APK launch will throw a "model not found" error;
# that's expected — run this script and relaunch the app.
#
# The script assumes you already ran `python convert_kokoro.py` and pushed
# the artifacts via:
#   adb push output/model_fp16.onnx /data/local/tmp/
#   adb push output/voices/. /data/local/tmp/kokoro_voices/

set -euo pipefail

PKG="com.kokoro"

if ! adb shell "pm list packages $PKG" | grep -q "$PKG"; then
    echo "Error: $PKG is not installed. Build & install the APK from Android Studio first." >&2
    exit 1
fi

if ! adb shell "[ -f /data/local/tmp/model_fp16.onnx ]"; then
    echo "Error: /data/local/tmp/model_fp16.onnx missing. Run 'adb push output/model_fp16.onnx /data/local/tmp/' first." >&2
    exit 1
fi

echo "Moving model_fp16.onnx into $PKG files dir..."
adb shell "run-as $PKG cp /data/local/tmp/model_fp16.onnx /data/data/$PKG/files/model_fp16.onnx"

echo "Creating voices dir..."
adb shell "run-as $PKG mkdir -p /data/data/$PKG/files/voices"

echo "Moving voice .bin files..."
adb shell "run-as $PKG sh -c 'cp /data/local/tmp/kokoro_voices/*.bin /data/data/$PKG/files/voices/'"

echo
echo "Verifying contents:"
adb shell "run-as $PKG ls -la /data/data/$PKG/files/ /data/data/$PKG/files/voices/"

echo
echo "Done. Relaunch the Kokoro app."
