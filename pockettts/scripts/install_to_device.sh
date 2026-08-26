#!/bin/bash
# Push the Pocket TTS graphs + host assets into the app's external files dir.
# Build them first:  python scripts/build_pockettts.py all   (writes scripts/out/)
# Usage:             ./scripts/install_to_device.sh [dir-with-files]
set -e
SRC="${1:-$(dirname "$0")/out}"
DST="/sdcard/Android/data/com.pockettts/files"

FILES=(
  pt_flowlm_fused_fp16.tflite
  pt_mimi_dec_tx_fp16.tflite
  pt_mimi_deconly_fp16.tflite
  pt_embed_f16.bin
  pt_input_linear_f32.bin
  pt_bos_input_f32.bin
  pt_neutral_latent_f32.bin
  pt_tokenizer.tsv
  pt_voice_alba.bin
  pt_voice_marius.bin
  pt_voice_javert.bin
  pt_voice_charles.bin
  pt_voice_mary.bin
  pt_voice_eve.bin
)

# The app must have run once so Android creates its external files dir.
adb shell mkdir -p "$DST"
for f in "${FILES[@]}"; do
  echo "push $f"
  adb push "$SRC/$f" "$DST/$f" >/dev/null
done
echo "done: $(adb shell ls "$DST" | wc -l | tr -d ' ') files in $DST"
