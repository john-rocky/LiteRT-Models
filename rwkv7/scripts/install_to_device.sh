#!/usr/bin/env bash
# Move the staged RWKV-7 model and embedding table into the app's private
# files dir.
#
# Run this AFTER `com.rwkv7` is installed on the device (via Android Studio
# build & run). The first APK launch shows "model files missing" — that's
# expected; run this script and relaunch the app.
#
# The script assumes you already produced the artifacts (see scripts/README.md)
# and pushed them via:
#   adb push rwkv7_step_fp16.tflite /data/local/tmp/
#   adb push rwkv7_emb_fp16.bin    /data/local/tmp/

set -euo pipefail

PKG="com.rwkv7"
FILES=(rwkv7_step_fp16.tflite rwkv7_emb_fp16.bin)

if ! adb shell "pm list packages $PKG" | grep -q "$PKG"; then
    echo "Error: $PKG is not installed. Build & install the APK from Android Studio first." >&2
    exit 1
fi

for f in "${FILES[@]}"; do
    if ! adb shell "[ -f /data/local/tmp/$f ]"; then
        echo "Error: /data/local/tmp/$f missing. Run 'adb push $f /data/local/tmp/' first." >&2
        exit 1
    fi
done

for f in "${FILES[@]}"; do
    echo "Moving $f into $PKG files dir..."
    adb shell "run-as $PKG cp /data/local/tmp/$f /data/data/$PKG/files/$f"
done

echo
echo "Verifying contents:"
adb shell "run-as $PKG ls -la /data/data/$PKG/files/"

echo
echo "Done. Relaunch the RWKV-7 app."
