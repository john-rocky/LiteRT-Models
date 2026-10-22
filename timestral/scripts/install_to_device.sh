#!/usr/bin/env bash
# Copies the Timestral model files into the app's private files directory.
#
# The files are the graph files and host tables that app/src/main/assets/timestral_forms.json names
# (the float16 graphs for the GPU and NPU, the int8 graphs for the CPU, forecast_query.f32 and the
# two RoPE tables), taken from one directory, for example the downloaded model repository.
# Usage: ./scripts/install_to_device.sh [dir-with-the-model-files]
# FILES="a.tflite b.f32" installs only the named files. Every file is checked before any device
# command (no partial installation), and each copy's size on the phone is compared with its source.
# Install the debug APK first: run-as needs a debuggable package. Set ANDROID_SERIAL to select a
# device when more than one is connected.
set -euo pipefail

if [[ $# -gt 1 ]]; then
    printf 'Usage: %s [dir-with-the-model-files]\n' "$0" >&2
    exit 2
fi
MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${1:-$HOME/Downloads/Timestral-209M-LiteRT}"
CONFIG="$MODULE_DIR/app/src/main/assets/timestral_forms.json"
PACKAGE=com.timestral
TEMP_DIR=/data/local/tmp/timestral

if [[ -n "${FILES:-}" ]]; then
    read -r -a NAMES <<< "$FILES"
else
    # Every quoted *.tflite and *.f32 name of the configuration, each once.
    NAMES=()
    while IFS= read -r name; do
        NAMES+=("$name")
    done < <(grep -oE '"[^"]+\.(tflite|f32)"' "$CONFIG" | tr -d '"' | awk '!seen[$0]++')
fi
if [[ ${#NAMES[@]} -eq 0 ]]; then
    printf 'No file names in %s\n' "$CONFIG" >&2
    exit 2
fi

for name in "${NAMES[@]}"; do
    if [[ ! -s "$SOURCE_DIR/$name" ]]; then
        printf 'Missing source file: %s/%s\n' "$SOURCE_DIR" "$name" >&2
        exit 1
    fi
done
if ! adb shell run-as "$PACKAGE" true 2>/dev/null; then
    printf '%s is not installed as a debuggable app: install the debug APK first.\n' "$PACKAGE" >&2
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
for name in "${NAMES[@]}"; do
    expected="$(wc -c < "$SOURCE_DIR/$name" | tr -d ' ')"
    pending="$TEMP_DIR/$$-$name"
    printf 'Staging %s (%s bytes)\n' "$name" "$expected"
    adb push "$SOURCE_DIR/$name" "$pending"
    adb shell run-as "$PACKAGE" cp "$pending" "files/$name"
    adb shell rm "$pending"
    pending=''
    copied="$(adb shell run-as "$PACKAGE" stat -c %s "files/$name" | tr -d '\r')"
    if [[ "$copied" != "$expected" ]]; then
        printf 'files/%s on the device is %s bytes, expected %s\n' "$name" "$copied" "$expected" >&2
        exit 1
    fi
done
adb shell rmdir "$TEMP_DIR" 2>/dev/null || true
adb shell run-as "$PACKAGE" ls -la files/
printf 'Files installed. Open Timestral Forecast.\n'
