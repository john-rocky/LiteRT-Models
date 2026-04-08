#!/usr/bin/env bash
# Move all staged voice assistant model files into the app's private files dir.
#
# Run this AFTER `com.voiceassistant` is installed via Android Studio. The first
# launch will fail with a "model not found" error; that is expected — run this
# script and relaunch the app.
#
# Required staged files in /data/local/tmp/:
#   whisper_encoder.tflite     (~32 MB)
#   whisper_decoder.onnx       (~199 MB)
#   smolvlm_decoder.onnx       (~540 MB)
#   embed_tokens.bin           (~114 MB)
#   model_fp16.onnx            (~163 MB)  Kokoro
#   silero_vad.onnx            (~2.3 MB)  Voice activity detection
#   kokoro_voices/*.bin        (5 voices, ~2.5 MB)

set -euo pipefail

PKG="com.voiceassistant"

if ! adb shell "pm list packages $PKG" | grep -q "$PKG"; then
    echo "Error: $PKG is not installed. Build & install the APK from Android Studio first." >&2
    exit 1
fi

REQUIRED=(
    "whisper_encoder.tflite"
    "whisper_decoder.onnx"
    "smolvlm_decoder.onnx"
    "embed_tokens.bin"
    "model_fp16.onnx"
    "silero_vad.onnx"
)
for f in "${REQUIRED[@]}"; do
    if ! adb shell "[ -f /data/local/tmp/$f ]"; then
        echo "Error: /data/local/tmp/$f missing. Push it first." >&2
        exit 1
    fi
done

echo "Moving model files into $PKG files dir..."
for f in "${REQUIRED[@]}"; do
    echo "  $f"
    adb shell "run-as $PKG cp /data/local/tmp/$f /data/data/$PKG/files/$f"
done

echo "Creating voices dir..."
adb shell "run-as $PKG mkdir -p /data/data/$PKG/files/voices"

echo "Moving Kokoro voice .bin files..."
adb shell "run-as $PKG sh -c 'cp /data/local/tmp/kokoro_voices/*.bin /data/data/$PKG/files/voices/'"

echo
echo "Verifying contents:"
adb shell "run-as $PKG ls -la /data/data/$PKG/files/ /data/data/$PKG/files/voices/"

echo
echo "Done. Relaunch the VoiceAssistant app."
