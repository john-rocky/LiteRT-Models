#!/usr/bin/env bash
# Download first:
# hf download litert-community/GLiNER2.5-Small-LiteRT --local-dir "$HOME/Downloads/GLiNER2.5-Small-LiteRT"
# Usage: ./scripts/install_to_device.sh [dir-with-downloaded-HF-repo]
# Set ANDROID_SERIAL to select a device when more than one is connected.
set -euo pipefail

if [[ $# -gt 1 ]]; then
    printf 'Usage: %s [dir-with-downloaded-HF-repo]\n' "$0" >&2
    exit 2
fi
SOURCE_DIR="${1:-$HOME/Downloads/GLiNER2.5-Small-LiteRT}"
PACKAGE=com.gliner25
TEMP_DIR=/data/local/tmp/gliner25
FILES=(
    gliner25_small_s128_wfp16.tflite
    gliner25_small_s256_wfp16.tflite
    gliner25_small_s512_wfp16.tflite
    word_embeddings_fp32.bin
    tokenizer.json
    sparse_decoder_fp32.safetensors
    graph_contract_s128.json
    graph_contract_s256.json
    graph_contract_s512.json
)

source_file() {
    case "$1" in
        *.tflite) printf '%s/%s\n' "$SOURCE_DIR" "$1" ;;
        *) printf '%s/host_assets/%s\n' "$SOURCE_DIR" "$1" ;;
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
printf 'Assets installed. Launch GLiNER2.5 Small.\n'
