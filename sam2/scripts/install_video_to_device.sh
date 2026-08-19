#!/usr/bin/env bash
# Stage the SAM 2.1 video-path graphs + constants into the app's private files dir.
#
# The video graphs are too large to bundle in assets (the encoder alone is ~80 MB fp16), so the
# tracker loads them from filesDir via the file-path CompiledModel overload. Run this AFTER
# `com.sam2` is installed (build & install the APK from Android Studio). Open "Video tracking →"
# once first; it will report "not in filesDir" until this script has run, then relaunch.
#
# Produce the files with:
#   SAM2_OUT=sam2/scripts/output/video python sam2/scripts/convert_sam2_video.py
#
# Usage:
#   sam2/scripts/install_video_to_device.sh [path/to/output/video]   (default: scripts/output/video)

set -euo pipefail

PKG="com.sam2"
SRC="${1:-$(dirname "$0")/output/video}"

FILES=(
    "sam2v_encode.tflite"
    "sam2v_memcond7.tflite"
    "sam2v_memcond2.tflite"
    "sam2v_decode.tflite"
    "sam2v_memorize.tflite"
    "sam2v_prompt.bin"
    "sam2v_track_sparse.bin"
    "sam2v_mtpe.bin"
    "sam2v_no_obj_ptr.bin"
    "sam2v_tpos_proj.bin"
)

if ! adb shell "pm list packages $PKG" | grep -q "$PKG"; then
    echo "Error: $PKG is not installed. Build & install the APK from Android Studio first." >&2
    exit 1
fi

for f in "${FILES[@]}"; do
    if [ ! -f "$SRC/$f" ]; then
        echo "Error: $SRC/$f missing. Run convert_sam2_video.py first." >&2
        exit 1
    fi
done

echo "Pushing to /data/local/tmp/ ..."
for f in "${FILES[@]}"; do
    echo "  $f"
    adb push "$SRC/$f" "/data/local/tmp/$f" >/dev/null
done

echo "Moving into $PKG files dir..."
for f in "${FILES[@]}"; do
    adb shell "run-as $PKG cp /data/local/tmp/$f /data/data/$PKG/files/$f"
    adb shell "rm -f /data/local/tmp/$f"
done

echo
echo "Verifying contents:"
adb shell "run-as $PKG ls -la /data/data/$PKG/files/"

echo
echo "Done. Relaunch com.sam2, tap 'Video tracking →', pick a video, tap the object in frame 0."
