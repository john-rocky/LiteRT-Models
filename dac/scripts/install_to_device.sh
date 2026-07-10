#!/usr/bin/env bash
# Push the two DAC tflites into the app's private filesDir (they are too big to bundle).
# Get them from Hugging Face (mlboydaisuke/DAC-16kHz-LiteRT) or build with
# scripts/convert_dac_encoder.py + scripts/convert_dac_deconly.py, then:
#   ./scripts/install_to_device.sh <dir-with-the-tflites>   (default: current dir)
set -e
PKG=com.dac
DIR="${1:-.}"
for M in dac_16khz_encoder_fp16.tflite dac_16khz_deconly_zs_fp16.tflite; do
    echo "pushing $M ..."
    adb push "$DIR/$M" "/data/local/tmp/$M"
    adb shell chmod 644 "/data/local/tmp/$M"
    adb shell run-as $PKG mkdir files 2>/dev/null || true
    adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the DAC Codec app."
