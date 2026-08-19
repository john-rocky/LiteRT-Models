#!/usr/bin/env bash
# Stage the TIPSv2 fp16 model (318 MB, too large for APK assets) into the app's private filesDir.
#   ./scripts/install_to_device.sh [path/to/tipsv2_b14_dpt_fp16.tflite]
# Install the app (./gradlew :app:installDebug) first so run-as resolves the package.
set -e
PKG=com.tipsv2
MODEL="${1:-$(dirname "$0")/tipsv2_b14_dpt_fp16.tflite}"
NAME=$(basename "$MODEL")
adb shell run-as $PKG mkdir -p files 2>/dev/null || true
adb push "$MODEL" "/data/local/tmp/$NAME"
adb shell run-as $PKG cp "/data/local/tmp/$NAME" "files/$NAME"
adb shell rm "/data/local/tmp/$NAME"
adb shell run-as $PKG ls -la files/
echo "done — launch the TIPSv2 app."
