#!/usr/bin/env bash
# Download first (the five files the app needs):
# hf download litert-community/GLiNER2.5-Decide-LiteRT gliner25_decide_s128_wfp16.tflite \
#   gliner25_decide_s256_wfp16.tflite gliner25_decide_s512_wfp16.tflite \
#   host_assets/word_embeddings_fp16.bin host_assets/tokenizer.json \
#   --local-dir "$HOME/Downloads/GLiNER2.5-Decide-LiteRT"
# Usage: ./scripts/install_to_device.sh [dir-with-downloaded-HF-repo]
# The directory holds the three *_wfp16.tflite graphs and host_assets/ (word_embeddings_fp16.bin,
# tokenizer.json). A directory without host_assets/ is read flat (all five files side by side).
# TOKENIZER_JSON=/path/to/tokenizer.json overrides where tokenizer.json is read from (for a
# conversion run's exports/, whose tokenizer.json stays in the pinned HF snapshot).
# Set ANDROID_SERIAL to select a device when more than one is connected.
set -euo pipefail

if [[ $# -gt 1 ]]; then
    printf 'Usage: %s [dir-with-downloaded-HF-repo]\n' "$0" >&2
    exit 2
fi
SOURCE_DIR="${1:-$HOME/Downloads/GLiNER2.5-Decide-LiteRT}"
PACKAGE=com.gliner25decide
TEMP_DIR=/data/local/tmp/gliner25decide
FILES=(
    gliner25_decide_s128_wfp16.tflite
    gliner25_decide_s256_wfp16.tflite
    gliner25_decide_s512_wfp16.tflite
    word_embeddings_fp16.bin
    tokenizer.json
)

source_file() {
    case "$1" in
        *.tflite) printf '%s/%s\n' "$SOURCE_DIR" "$1" ;;
        tokenizer.json) if [[ -n "${TOKENIZER_JSON:-}" ]]; then printf '%s\n' "$TOKENIZER_JSON"; else host_asset "$1"; fi ;;
        *) host_asset "$1" ;;
    esac
}

host_asset() {
    if [[ -d "$SOURCE_DIR/host_assets" ]]; then
        printf '%s/host_assets/%s\n' "$SOURCE_DIR" "$1"
    else
        printf '%s/%s\n' "$SOURCE_DIR" "$1"
    fi
}

# Check every source before the first device command to avoid a partial installation.
for name in "${FILES[@]}"; do
    source="$(source_file "$name")"
    if [[ ! -f "$source" ]]; then
        printf 'Missing source file: %s\n' "$source" >&2
        exit 1
    fi
done
table="$(source_file word_embeddings_fp16.bin)"
if [[ "$(wc -c < "$table" | tr -d ' ')" != 262166528 ]]; then
    printf 'word_embeddings_fp16.bin is not the [128011,1024] float16 table: %s\n' "$table" >&2
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
    adb push "$(source_file "$name")" "$pending"
    adb shell run-as "$PACKAGE" cp "$pending" "files/$name"
    adb shell rm "$pending"
    pending=''
done
adb shell rmdir "$TEMP_DIR"
adb shell run-as "$PACKAGE" ls -la files/
printf 'Assets installed. Launch GLiNER2.5 Decide.\n'
