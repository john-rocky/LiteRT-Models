#!/usr/bin/env bash
# Download the files (the L128 and L256 graphs, the Ls128 shared-state pair, the head and the
# tokenizer; the L64, L512, L1024 and L2048 graphs and the Ls256 pair are optional):
# hf download litert-community/Kev-0.8B-LiteRT kev-0.8b_rowprefill_L128_fp16fc_i8emb.tflite \
#   kev-0.8b_rowprefill_L256_fp16fc_i8emb.tflite \
#   kev-0.8b_sharedstate_Ls128_Lq64_fp16fc_i8emb.tflite \
#   head/kev_0.8b_pointer_head.safetensors tokenizer/tokenizer.json \
#   --local-dir "$HOME/Downloads/Kev-0.8B-LiteRT"
# Usage: ./scripts/install_to_device.sh [dir-with-downloaded-HF-repo]
# The directory holds the graphs at its top level, head/ and tokenizer/. A directory without
# head/ and tokenizer/ is read flat (all files side by side). WINDOWS selects the graphs: "128 256"
# by default; WINDOWS="64 128 256 512 1024 2048" installs all six, WINDOWS="" none. PAIR selects the
# shared-state pairs by their state length: the Ls128 pair by default (PAIR=1 or PAIR=128,
# kev-0.8b_sharedstate_Ls128_Lq64_fp16fc_i8emb.tflite), PAIR="128 256" both, PAIR=0 none. Every
# file must be in the directory with the published size; nothing is copied otherwise.
# CHECK_SIZES=0 skips the published sizes (files you converted yourself) and checks each copy
# against its source instead.
# Graphs already on the device that WINDOWS and PAIR do not name stay there.
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
pair() { printf 'kev-0.8b_sharedstate_Ls%s_Lq64_fp16fc_i8emb.tflite\n' "$1"; }

# Published sizes in bytes (the files staged for the next upload; to be confirmed at upload).
size_of() {
    case "$1" in
        "$TOKENIZER") echo 19989325 ;;
        "$HEAD") echo 2099632 ;;
        "$(graph 64)") echo 1258031552 ;;
        "$(graph 128)") echo 1258444912 ;;
        "$(graph 256)") echo 1259246704 ;;
        "$(graph 512)") echo 1261233328 ;;
        "$(graph 1024)") echo 1266799568 ;;
        "$(graph 2048)") echo 1284223520 ;;
        "$(pair 128)") echo 1261368160 ;;
        "$(pair 256)") echo 1261918016 ;;
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

FILES=("$TOKENIZER" "$HEAD")
for window in ${WINDOWS-128 256}; do
    case "$window" in
        64 | 128 | 256 | 512 | 1024 | 2048) FILES+=("$(graph "$window")") ;;
        *)
            printf 'WINDOWS holds %s; the graphs are 64, 128, 256, 512, 1024 and 2048\n' "$window" >&2
            exit 2
            ;;
    esac
done
for length in ${PAIR-1}; do
    case "$length" in
        0) ;;
        1 | 128) FILES+=("$(pair 128)") ;;
        256) FILES+=("$(pair 256)") ;;
        *)
            printf 'PAIR holds %s; use PAIR=0, PAIR=1 or a list of state lengths, 128 and 256\n' "$length" >&2
            exit 2
            ;;
    esac
done
case "${CHECK_SIZES:-1}" in
    0 | 1) ;;
    *)
        printf 'CHECK_SIZES is %s; use CHECK_SIZES=0 to skip the published sizes\n' "$CHECK_SIZES" >&2
        exit 2
        ;;
esac
if [[ ${#FILES[@]} -eq 2 ]]; then
    printf 'WINDOWS and PAIR name no graph\n' >&2
    exit 2
fi

# The size a file must have: the published one, or the source's own with CHECK_SIZES=0.
expected_size() {
    if [[ "${CHECK_SIZES:-1}" == 0 ]]; then
        wc -c < "$(source_file "$1")" | tr -d ' '
    else
        size_of "$1"
    fi
}

# Check every source before any device command to avoid a partial installation.
for name in "${FILES[@]}"; do
    source="$(source_file "$name")"
    if [[ ! -f "$source" ]]; then
        printf 'Missing source file: %s\n' "$source" >&2
        exit 1
    fi
    actual="$(wc -c < "$source" | tr -d ' ')"
    if [[ "$actual" != "$(expected_size "$name")" ]]; then
        printf '%s is %s bytes, expected %s: %s\n' "$name" "$actual" "$(expected_size "$name")" "$source" >&2
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
    if [[ "$copied" != "$(expected_size "$name")" ]]; then
        printf 'files/%s on the device is %s bytes, expected %s\n' "$name" "$copied" "$(expected_size "$name")" >&2
        exit 1
    fi
done
adb shell rmdir "$TEMP_DIR" 2>/dev/null || true
adb shell run-as "$PACKAGE" ls -la files/
printf 'Files installed. Launch Kev Decide.\n'
