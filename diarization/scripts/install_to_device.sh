#!/usr/bin/env bash
# Push the diarization models into the app's private filesDir (models are not committed).
#   ./scripts/install_to_device.sh <dir-with-models>   (default: current dir)
# Needs: wespeaker_emb_fp16.tflite (embedding, GPU) + pyannote_seg30.onnx (segmentation, CPU)
set -e
PKG=com.diarization; DIR="${1:-.}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for M in wespeaker_emb_fp16.tflite pyannote_seg30.onnx; do
  adb push "$DIR/$M" "/data/local/tmp/$M"
  adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
  adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the Diarization app."
