#!/usr/bin/env bash
set -euo pipefail
PKG="com.moge"
MODEL="moge.tflite"

if ! adb shell "pm list packages $PKG" | grep -q "$PKG"; then
    echo "Error: $PKG not installed." >&2; exit 1
fi
if ! adb shell "test -f /data/local/tmp/$MODEL" 2>/dev/null; then
    echo "Push first: adb push moge/scripts/output/$MODEL /data/local/tmp/"; exit 1
fi
echo "Installing $MODEL (~136 MB)..."
adb shell "run-as $PKG cp /data/local/tmp/$MODEL /data/data/$PKG/files/$MODEL"
echo "Done. Relaunch the app."
