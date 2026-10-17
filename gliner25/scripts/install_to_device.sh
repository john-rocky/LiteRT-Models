#!/usr/bin/env bash
# Download first:
# hf download litert-community/GLiNER2.5-Small-LiteRT --local-dir "$HOME/Downloads/GLiNER2.5-Small-LiteRT"
# For --multi, see README.md (s128 graph, contracts and host_assets/ without the fp32 table).
# Usage: ./scripts/install_to_device.sh [--multi] [dir-with-downloaded-HF-repo]
# Set ANDROID_SERIAL to select a device when more than one is connected.
set -euo pipefail

MULTI=0
if [[ "${1:-}" == --multi ]]; then
    MULTI=1
    shift
fi
if [[ $# -gt 1 ]]; then
    printf 'Usage: %s [--multi] [dir-with-downloaded-HF-repo]\n' "$0" >&2
    exit 2
fi
PACKAGE=com.gliner25
TEMP_DIR=/data/local/tmp/gliner25
if [[ $MULTI -eq 1 ]]; then
    SOURCE_DIR="${1:-$HOME/Downloads/GLiNER2.5-Multi-LiteRT}"
    TARGET_DIR=files/multi
    FILES=(
        gliner25_multi_s128_wfp16.tflite
        word_embeddings_fp16.bin
        tokenizer.json
        sparse_decoder_fp32.safetensors
        config.json
        graph_contract_s128.json
        graph_contract_s256.json
        graph_contract_s512.json
    )
    # The s256/s512 graphs are optional downloads for texts beyond the s128 window.
    for optional in gliner25_multi_s256_wfp16.tflite gliner25_multi_s512_wfp16.tflite; do
        if [[ -f "$SOURCE_DIR/$optional" ]]; then
            FILES+=("$optional")
        fi
    done
    LAUNCH='GLiNER2.5 and choose Multi'
else
    SOURCE_DIR="${1:-$HOME/Downloads/GLiNER2.5-Small-LiteRT}"
    TARGET_DIR=files
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
    LAUNCH='GLiNER2.5 Small'
fi

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
adb shell run-as "$PACKAGE" mkdir -p "$TARGET_DIR"
for name in "${FILES[@]}"; do
    pending="$TEMP_DIR/$$-$name"
    printf 'Staging %s\n' "$name"
    adb push "$(source_file "$name")" "$pending"
    adb shell run-as "$PACKAGE" cp "$pending" "$TARGET_DIR/$name"
    adb shell rm "$pending"
    pending=''
done
adb shell rmdir "$TEMP_DIR"
adb shell run-as "$PACKAGE" ls -la "$TARGET_DIR/"
printf 'Assets installed. Launch %s.\n' "$LAUNCH"
