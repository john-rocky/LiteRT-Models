#!/usr/bin/env bash
# Push the CLIPSeg models + host assets into the app's private filesDir.
#   ./scripts/install_to_device.sh <dir-with-artifacts>   (default: current dir)
# Needs: clipseg_vision_fp16.tflite clipseg_decoder.tflite, clipseg_text_fp16.tflite,
#        token_embedding_f16.bin, text_projection_f16.bin, vocab.json, merges.txt
set -e
PKG=com.clipseg; DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for M in clipseg_vision_fp16.tflite clipseg_decoder.tflite clipseg_text_fp16.tflite token_embedding_f16.bin text_projection_f16.bin vocab.json merges.txt; do
  adb push "$DIR/$M" "/data/local/tmp/$M"
  adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
  adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the CLIPSeg app."
