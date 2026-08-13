#!/usr/bin/env bash
# Push the Zipformer model into the app's private filesDir (too big to bundle in the APK).
# Get it from Hugging Face (litert-community/Zipformer-medium-CR-CTC-LiteRT), then:
#   ./scripts/install_to_device.sh <dir-with-the-tflite>   (default: current dir)
set -e
PKG=com.zipformer
DIR="${1:-.}"
M=zipformer_ctc_fp16.tflite
adb shell run-as $PKG mkdir files 2>/dev/null || true
echo "pushing $M ..."
adb push "$DIR/$M" "/data/local/tmp/$M"
adb shell chmod 644 "/data/local/tmp/$M"
adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
adb shell rm "/data/local/tmp/$M"
adb shell run-as $PKG ls -la files/
echo "done — launch the Zipformer ASR app."
