#!/usr/bin/env bash
# Push the four EMA Lightning LiteRT graphs into the app's private filesDir (the app can also download them itself).
# Get them from Hugging Face (litert-community/EMA-Lightning-LiteRT), then:
#   ./scripts/install_to_device.sh <dir-with-the-tflites>   (default: current dir)
set -e
PKG=com.emalightning
DIR="${1:-.}"
MODELS=(
    ema_text_multi_fp32.tflite
    ema_align_multi_fp32.tflite
    ema_step_multi_fp32.tflite
    ema_decoder_multi_fp16.tflite
)
for M in "${MODELS[@]}"; do
    echo "pushing $M ..."
    adb push "$DIR/$M" "/data/local/tmp/$M"
    adb shell chmod 644 "/data/local/tmp/$M"
    adb shell run-as $PKG mkdir files 2>/dev/null || true
    adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
    adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the EMA Lightning TTS app."
