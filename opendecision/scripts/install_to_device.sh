#!/usr/bin/env bash
# Download first (the four files the app needs):
# hf download litert-community/Open-Decision-DeBERTa-v3-Large-LiteRT \
#   deberta_v3_large_decision_s256_wfp16.tflite deberta_v3_large_decision_s512_wfp16.tflite \
#   word_embeddings_fp16.bin tokenizer.json --local-dir "$HOME/Downloads/Open-Decision-DeBERTa-v3-Large-LiteRT"
# Usage: ./scripts/install_to_device.sh [dir-with-downloaded-HF-repo]
# Set ANDROID_SERIAL to select a device when more than one is connected.
set -euo pipefail

if [[ $# -gt 1 ]]; then
    printf 'Usage: %s [dir-with-downloaded-HF-repo]\n' "$0" >&2
    exit 2
fi
SOURCE_DIR="${1:-$HOME/Downloads/Open-Decision-DeBERTa-v3-Large-LiteRT}"
PACKAGE=com.opendecision
TEMP_DIR=/data/local/tmp/opendecision
FILES=(
    deberta_v3_large_decision_s256_wfp16.tflite
    deberta_v3_large_decision_s512_wfp16.tflite
    word_embeddings_fp16.bin
    tokenizer.json
)

# Check every source before the first device command to avoid a partial installation.
for name in "${FILES[@]}"; do
    if [[ ! -f "$SOURCE_DIR/$name" ]]; then
        printf 'Missing source file: %s\n' "$SOURCE_DIR/$name" >&2
        exit 1
    fi
done
if [[ "$(wc -c < "$SOURCE_DIR/word_embeddings_fp16.bin" | tr -d ' ')" != 262348800 ]]; then
    printf 'word_embeddings_fp16.bin is not the [128100,1024] float16 table\n' >&2
    exit 1
fi

pending=''
cleanup() {
    if [[ -n "$pending" ]]; then
        adb shell rm -f "$pending"
    fi
}
trap cleanup EXIT

adb shell mkdir -p "$TEMP_DIR"
adb shell run-as "$PACKAGE" mkdir -p files
for name in "${FILES[@]}"; do
    pending="$TEMP_DIR/$$-$name"
    printf 'Staging %s\n' "$name"
    adb push "$SOURCE_DIR/$name" "$pending"
    adb shell run-as "$PACKAGE" cp "$pending" "files/$name"
    adb shell rm "$pending"
    pending=''
done
adb shell rmdir "$TEMP_DIR"
adb shell run-as "$PACKAGE" ls -la files/
printf 'Assets installed. Launch Open Decision.\n'
