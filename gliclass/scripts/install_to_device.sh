#!/usr/bin/env bash
# Download first (the four files the app needs):
# hf download litert-community/GLiClass-Edge-v3.0-LiteRT gliclass_edge_v3_s128_fp32.tflite \
#   gliclass_edge_v3_s256_fp32.tflite host_assets/tok_embeddings_fp16.bin host_assets/tokenizer.json \
#   --local-dir "$HOME/Downloads/GLiClass-Edge-v3.0-LiteRT"
# Usage: ./scripts/install_to_device.sh [dir-with-downloaded-HF-repo]
# The directory holds the two *_fp32.tflite graphs and host_assets/ (tok_embeddings_fp16.bin,
# tokenizer.json). A directory without host_assets/ is read flat (all four files side by side).
# Set ANDROID_SERIAL to select a device when more than one is connected.
set -euo pipefail

if [[ $# -gt 1 ]]; then
    printf 'Usage: %s [dir-with-downloaded-HF-repo]\n' "$0" >&2
    exit 2
fi
SOURCE_DIR="${1:-$HOME/Downloads/GLiClass-Edge-v3.0-LiteRT}"
PACKAGE=com.gliclass
TEMP_DIR=/data/local/tmp/gliclass
FILES=(
    gliclass_edge_v3_s128_fp32.tflite
    gliclass_edge_v3_s256_fp32.tflite
    tok_embeddings_fp16.bin
    tokenizer.json
)

source_file() {
    case "$1" in
        *.tflite) printf '%s/%s\n' "$SOURCE_DIR" "$1" ;;
        *)
            if [[ -d "$SOURCE_DIR/host_assets" ]]; then
                printf '%s/host_assets/%s\n' "$SOURCE_DIR" "$1"
            else
                printf '%s/%s\n' "$SOURCE_DIR" "$1"
            fi
            ;;
    esac
}

# Check every source before the first device command to avoid a partial installation.
for name in "${FILES[@]}"; do
    source="$(source_file "$name")"
    if [[ ! -f "$source" ]]; then
        printf 'Missing source file: %s\n' "$source" >&2
        exit 1
    fi
done
table="$(source_file tok_embeddings_fp16.bin)"
if [[ "$(wc -c < "$table" | tr -d ' ')" != 38684160 ]]; then
    printf 'tok_embeddings_fp16.bin is not the [50370,384] float16 table: %s\n' "$table" >&2
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
printf 'Assets installed. Launch GLiClass Edge.\n'
