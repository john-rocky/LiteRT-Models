#!/usr/bin/env bash
# Download the files (contract.json, tokenizer.json and the decision graphs for 128 and 256 positions;
# the other buckets are optional):
# hf download litert-community/d1-omni-600M-LiteRT contract.json tokenizer.json \
#   d1-omni-600M_decide_L128_fp16.tflite d1-omni-600M_decide_L256_fp16.tflite \
#   --local-dir "$HOME/Downloads/d1-omni-600M-LiteRT"
# Usage: ./scripts/install_to_device.sh [dir-with-downloaded-HF-repo]
# BUCKETS selects the decision graphs: "128 256" by default; BUCKETS="128 256 512 1024 2048" adds the
# longer ones (L4096 does not fit the GPU of a 12 GB phone). AUDIO selects the audio graphs by their
# bucket T: none by default; AUDIO="1001" adds d1-omni-600M_audio_T1001_fp16.tflite (clips up to
# 10 s; 501 / 2001 / 3001 hold 5 / 20 / 30 s). Every file must be in the directory with the size
# contract.json gives; nothing is copied otherwise. A file already in files/ with the contract's
# size and sha256 is left as it is; each other file goes through /data/local/tmp/d1omni/ into the
# app's private files/ with run-as com.d1omni, the temporary copy is removed, and the copy's sha256
# on the phone (toybox sha256sum) is checked against contract.json. Files already on the device that
# BUCKETS and AUDIO do not name stay there.
# The app needs the default BUCKETS with AUDIO="1001" VISION=1 (eight files).
# Install the debug APK before running this script: run-as needs a debuggable package.
# Set ANDROID_SERIAL to select a device when more than one is connected.
set -euo pipefail

if [[ $# -gt 1 ]]; then
    printf 'Usage: %s [dir-with-downloaded-HF-repo]\n' "$0" >&2
    exit 2
fi
SOURCE_DIR="${1:-$HOME/Downloads/d1-omni-600M-LiteRT}"
PACKAGE=com.d1omni
TEMP_DIR=/data/local/tmp/d1omni
CONTRACT="$SOURCE_DIR/contract.json"
graph() { printf 'd1-omni-600M_decide_L%s_fp16.tflite\n' "$1"; }
audio_graph() { printf 'd1-omni-600M_audio_T%s_fp16.tflite\n' "$1"; }

if [[ ! -f "$CONTRACT" ]]; then
    printf 'Missing %s: download contract.json with the graphs\n' "$CONTRACT" >&2
    exit 1
fi
command -v python3 > /dev/null || { printf 'python3 is needed to read contract.json\n' >&2; exit 1; }

# "<bytes> <sha256>" of a file from contract.json (the tokenizer's from its own entry).
expected() {
    python3 -I -c '
import json, sys
c = json.load(open(sys.argv[1]))
name = sys.argv[2]
for f in c["files"]:
    if f["name"] == name:
        print(f["bytes"], f["sha256"]); break
else:
    if name == c["tokenizer"]["file"]:
        print("-", c["tokenizer"]["sha256"])
    else:
        sys.exit("contract.json does not list " + name)' "$CONTRACT" "$1"
}

FILES=(tokenizer.json)
for bucket in ${BUCKETS-128 256}; do
    case "$bucket" in
        128 | 256 | 512 | 1024 | 2048) FILES+=("$(graph "$bucket")") ;;
        4096)
            printf 'BUCKETS holds 4096: that graph does not fit the GPU of a 12 GB phone; use 128 to 2048\n' >&2
            exit 2
            ;;
        *)
            printf 'BUCKETS holds %s; the buckets are 128, 256, 512, 1024 and 2048\n' "$bucket" >&2
            exit 2
            ;;
    esac
done
for bucket in ${AUDIO-}; do
    case "$bucket" in
        501 | 1001 | 2001 | 3001) FILES+=("$(audio_graph "$bucket")") ;;
        *)
            printf 'AUDIO holds %s; the audio buckets are 501, 1001, 2001 and 3001\n' "$bucket" >&2
            exit 2
            ;;
    esac
done

# Check every source before any device command to avoid a partial installation.
for name in "${FILES[@]}"; do
    source="$SOURCE_DIR/$name"
    if [[ ! -f "$source" ]]; then
        printf 'Missing source file: %s\n' "$source" >&2
        exit 1
    fi
    read -r bytes sha <<< "$(expected "$name")"
    actual="$(wc -c < "$source" | tr -d ' ')"
    if [[ "$bytes" != "-" && "$actual" != "$bytes" ]]; then
        printf '%s is %s bytes, contract.json says %s: %s\n' "$name" "$actual" "$bytes" "$source" >&2
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

phone_sha() {  # name: the sha256 of files/<name> on the phone, empty when it is not there
    adb shell run-as "$PACKAGE" toybox sha256sum "files/$1" 2>/dev/null | cut -d' ' -f1 | tr -d '\r'
}

copy() {  # name source: unless files/ holds it already, stage, copy into files/, remove the stage, check
    local name=$1 source=$2 want_bytes want_sha copied sum
    want_bytes="$(wc -c < "$source" | tr -d ' ')"
    if [[ "$name" == contract.json ]]; then
        want_sha="$(shasum -a 256 "$source" | cut -d' ' -f1)"
    else
        read -r _ want_sha <<< "$(expected "$name")"
    fi
    # A missing file makes stat fail: under set -e / pipefail that must not end the script.
    copied="$(adb shell run-as "$PACKAGE" stat -c %s "files/$name" 2>/dev/null | tr -d '\r' || true)"
    if [[ "$copied" == "$want_bytes" && -n "$want_sha" && "$(phone_sha "$name")" == "$want_sha" ]]; then
        printf 'Already installed: %s\n' "$name"
        return
    fi
    pending="$TEMP_DIR/$$-$name"
    printf 'Staging %s\n' "$name"
    adb push "$source" "$pending"
    adb shell run-as "$PACKAGE" cp "$pending" "files/$name"
    adb shell rm "$pending"
    pending=''
    copied="$(adb shell run-as "$PACKAGE" stat -c %s "files/$name" | tr -d '\r')"
    if [[ "$copied" != "$want_bytes" ]]; then
        printf 'files/%s on the device is %s bytes, expected %s\n' "$name" "$copied" "$want_bytes" >&2
        exit 1
    fi
    sum="$(phone_sha "$name")"
    if [[ -z "$sum" || "$sum" != "$want_sha" ]]; then
        printf 'files/%s on the device has sha256 %s, expected %s\n' "$name" "${sum:-<none>}" "$want_sha" >&2
        exit 1
    fi
}

# vision (round 3): VISION=1 also installs the picture path: the vision tower, the projector and the
# position table (host/vision_position_table.npy in the download, files/host/vision_position_table.npy
# on the phone). Their sizes are checked before any device command; each copy's sha256 on the phone
# is checked against contract.json like the files above. Download them with:
# hf download litert-community/d1-omni-600M-LiteRT d1-omni-600M_vision_tower_fp16.tflite \
#   d1-omni-600M_projector_fp16.tflite host/vision_position_table.npy --local-dir "$SOURCE_DIR"
if [[ "${VISION:-0}" == 1 ]]; then
    VISION_FILES=(d1-omni-600M_vision_tower_fp16.tflite d1-omni-600M_projector_fp16.tflite
        host/vision_position_table.npy)
    for name in "${VISION_FILES[@]}"; do
        source="$SOURCE_DIR/$name"
        [[ -f "$source" ]] || { printf 'Missing source file: %s\n' "$source" >&2; exit 1; }
        read -r bytes sha <<< "$(expected "$name")"
        actual="$(wc -c < "$source" | tr -d ' ')"
        if [[ "$actual" != "$bytes" ]]; then
            printf '%s is %s bytes, contract.json says %s: %s\n' "$name" "$actual" "$bytes" "$source" >&2
            exit 1
        fi
    done
    adb shell mkdir -p "$TEMP_DIR"
    adb shell run-as "$PACKAGE" mkdir -p files/host
    for name in "${VISION_FILES[@]}"; do
        # stage under a flat name: the copy() above would make a host/ dir in the stage
        source="$SOURCE_DIR/$name"
        read -r bytes sha <<< "$(expected "$name")"
        pending="$TEMP_DIR/$$-${name//\//_}"
        printf 'Staging %s\n' "$name"
        adb push "$source" "$pending"
        adb shell run-as "$PACKAGE" cp "$pending" "files/$name"
        adb shell rm "$pending"
        pending=''
        copied="$(adb shell run-as "$PACKAGE" stat -c %s "files/$name" | tr -d '\r')"
        sum="$(adb shell run-as "$PACKAGE" toybox sha256sum "files/$name" | cut -d' ' -f1 | tr -d '\r')"
        if [[ "$copied" != "$bytes" || -z "$sum" || "$sum" != "$sha" ]]; then
            printf 'files/%s on the device is %s bytes / sha256 %s, expected %s / %s\n' \
                "$name" "$copied" "${sum:-<none>}" "$bytes" "$sha" >&2
            exit 1
        fi
    done
fi
# end vision (round 3)

adb shell mkdir -p "$TEMP_DIR"
adb shell run-as "$PACKAGE" mkdir -p files
copy contract.json "$CONTRACT"
for name in "${FILES[@]}"; do
    copy "$name" "$SOURCE_DIR/$name"
done
adb shell rmdir "$TEMP_DIR" 2>/dev/null || true
adb shell run-as "$PACKAGE" ls -la files/
printf 'Files installed. Launch d1-omni Decide.\n'
