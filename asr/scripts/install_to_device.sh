#!/usr/bin/env bash
# Push the wav2vec2-CTC tflite into the app's private filesDir (large model, not bundled).
#   ./scripts/install_to_device.sh <dir-with-w2v2_ctc_fp16.tflite>   (default: current dir)
set -e
PKG=com.asr; DIR="${1:-.}"; M=w2v2_ctc_fp16.tflite
adb shell run-as $PKG mkdir files 2>/dev/null || true
adb push "$DIR/$M" "/data/local/tmp/$M"
adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
adb shell rm "/data/local/tmp/$M"
adb shell run-as $PKG ls -la files/
echo "done — launch the ASR app."
