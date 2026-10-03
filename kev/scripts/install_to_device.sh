#!/usr/bin/env bash
# Download the files (the L512 graph, the head and the tokenizer; the L1024 / L2048 graphs are optional):
# hf download litert-community/Kev-0.8B-LiteRT kev-0.8b_rowprefill_L512_fp16fc_i8emb.tflite \
#   head/kev_0.8b_pointer_head.safetensors tokenizer/tokenizer.json \
#   --local-dir "$HOME/Downloads/Kev-0.8B-LiteRT"
# Usage: ./scripts/install_to_device.sh [dir-with-downloaded-HF-repo]
# The directory holds the graphs at its top level, head/ and tokenizer/. A directory without
# head/ and tokenizer/ is read flat (all files side by side). The L1024 and L2048 graphs are
# installed when present; WINDOWS="512" (or "512 1024") limits the graphs to those windows. Every
# file must have the published size; nothing is copied otherwise.
# Install the debug APK before running this script: run-as needs a debuggable package.
# Set ANDROID_SERIAL to select a device when more than one is connected.
set -euo pipefail

if [[ $# -gt 1 ]]; then
    printf 'Usage: %s [dir-with-downloaded-HF-repo]\n' "$0" >&2
    exit 2
fi
SOURCE_DIR="${1:-$HOME/Downloads/Kev-0.8B-LiteRT}"
PACKAGE=com.kev
TEMP_DIR=/data/local/tmp/kev
TOKENIZER=tokenizer.json
HEAD=kev_0.8b_pointer_head.safetensors
graph() { printf 'kev-0.8b_rowprefill_L%s_fp16fc_i8emb.tflite\n' "$1"; }

# Published sizes in bytes.
size_of() {
    case "$1" in
        "$TOKENIZER") echo 19989325 ;;
        "$HEAD") echo 2099632 ;;
        "$(graph 512)") echo 1264068368 ;;
        "$(graph 1024)") echo 1269023216 ;;
        "$(graph 2048)") echo 1285227888 ;;
    esac
}

source_file() {
    case "$1" in
        "$TOKENIZER") sub=tokenizer ;;
        "$HEAD") sub=head ;;
        *) sub='' ;;
    esac
    if [[ -n "$sub" && -d "$SOURCE_DIR/$sub" ]]; then
        printf '%s/%s/%s\n' "$SOURCE_DIR" "$sub" "$1"
    else
        printf '%s/%s\n' "$SOURCE_DIR" "$1"
    fi
}

FILES=("$TOKENIZER" "$HEAD" "$(graph 512)")
for window in ${WINDOWS:-512 1024 2048}; do
    case "$window" in
        512) ;;
        1024 | 2048)
            if [[ -f "$(source_file "$(graph "$window")")" ]]; then
                FILES+=("$(graph "$window")")
            fi
            ;;
        *)
            printf 'WINDOWS holds %s; the graphs are 512, 1024 and 2048\n' "$window" >&2
            exit 2
            ;;
    esac
done

# Check every source before any device command to avoid a partial installation.
for name in "${FILES[@]}"; do
    source="$(source_file "$name")"
    if [[ ! -f "$source" ]]; then
        printf 'Missing source file: %s\n' "$source" >&2
        exit 1
    fi
    actual="$(wc -c < "$source" | tr -d ' ')"
    if [[ "$actual" != "$(size_of "$name")" ]]; then
        printf '%s is %s bytes, expected %s: %s\n' "$name" "$actual" "$(size_of "$name")" "$source" >&2
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
    copied="$(adb shell run-as "$PACKAGE" stat -c %s "files/$name" | tr -d '\r')"
    if [[ "$copied" != "$(size_of "$name")" ]]; then
        printf 'files/%s on the device is %s bytes, expected %s\n' "$name" "$copied" "$(size_of "$name")" >&2
        exit 1
    fi
done
adb shell rmdir "$TEMP_DIR" 2>/dev/null || true
adb shell run-as "$PACKAGE" ls -la files/
printf 'Files installed. Launch Kev Decide.\n'
