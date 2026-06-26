#!/usr/bin/env bash
# Push the 4 Mimi tflites + RVQ weights into the app's private filesDir (too big to bundle).
# Build them with build_hybrid_graphs.py + mimi_rvq_validate_export.py, or get from Hugging Face, then:
#   ./scripts/install_to_device.sh <dir-with-the-files>   (default: current dir)
set -e
PKG=com.mimi
DIR="${1:-.}"
FILES="mimi_enc_conv_fp16.tflite mimi_enc_tx_fp16.tflite mimi_dec_tx_fp16.tflite mimi_deconly_fp16.tflite mimi_rvq.bin"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for M in $FILES; do
    echo "pushing $M ..."
    adb push "$DIR/$M" "/data/local/tmp/$M"
    adb shell chmod 644 "/data/local/tmp/$M"
    adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
    adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the Mimi Codec app."
