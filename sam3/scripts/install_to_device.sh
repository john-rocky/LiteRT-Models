#!/usr/bin/env bash
# Push the SAM 3.1 image-side models + host assets into the app's private filesDir.
#   ./scripts/install_to_device.sh [dir-with-artifacts]   (default: models/out)
# Needs: sam3_vision.tflite sam3_text.tflite sam3_head.tflite sam3_token_embed.bin
#        sam3_tokenizer/vocab.json sam3_tokenizer/merges.txt
set -e
PKG=com.sam3
DIR="${1:-$(dirname "$0")/../models/out}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
push() {
  adb push "$1" "/data/local/tmp/$2"
  adb shell run-as $PKG cp "/data/local/tmp/$2" "files/$2"
  adb shell rm "/data/local/tmp/$2"
}
for M in sam3_vision.tflite sam3_text.tflite sam3_head.tflite sam3_token_embed.bin; do
  push "$DIR/$M" "$M"
done
push "$DIR/sam3_tokenizer/vocab.json" vocab.json
push "$DIR/sam3_tokenizer/merges.txt" merges.txt
# optional headless check: the app runs these prompts on autotest.jpg at startup and
# writes files/probe_sam3.txt + files/overlay_out.png (see MainActivity.autotest)
if [ -f "$DIR/autotest/autotest.jpg" ]; then
  push "$DIR/autotest/autotest.jpg" autotest.jpg
  push "$DIR/autotest/autotest_prompts.txt" autotest_prompts.txt
fi
adb shell run-as $PKG ls -la files/
echo "done — launch the SAM3 app."
