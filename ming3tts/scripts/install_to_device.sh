#!/bin/bash
# Copy the Ming-3.0-tts graphs + host files into the app's private files dir (files/models).
#
# Download the model files (hf download litert-community/Ming-3.0-tts-0.5B --local-dir <dir>), install the
# debug APK (it must be debuggable: the copy uses run-as), then run:
#   ./scripts/install_to_device.sh [<dir>]        (default: the folder above scripts/)
# Each file goes through /data/local/tmp (adb push), is copied in with run-as and the staged copy is
# deleted right away, so the phone needs room for one file twice, not for all of them. The same four
# graphs serve the default placement and its GPU-only fallback; the two lm2_epsdiv_* files are the
# language model of the faster mode that runs every graph on the NPU. The NPU part needs the Qualcomm
# vendor libraries in app/src/main/jniLibs/arm64-v8a/ before the build (see app/build.gradle.kts);
# without them the app runs the flow on the GPU. With several phones connected, set ANDROID_SERIAL.
set -euo pipefail
PKG=com.ming3tts
SRC="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
STAGE=/data/local/tmp/ming3tts_stage

FILES=(
  lm2_wfp16.tflite
  flow_r4b_s10_wfp16.tflite
  agg_r4b_wfp16.tflite
  dec_c16_wfp16.tflite
  embed_tokens_f16.bin
  temb_epss10.bin
  istft_window.bin
  stop_head.bin
  char_classes.bin
  tokenizer_vocab.txt
  tokenizer_merges.txt
  tokenizer_added.txt
  lm2_epsdiv_prefill_wfp16.tflite
  lm2_epsdiv_step_wfp16.tflite
)

if ! adb shell pm list packages "$PKG" | grep -q "package:$PKG\$"; then
  echo "$PKG is not installed: build and install the debug APK first (./gradlew :app:installDebug)." >&2
  exit 1
fi
for f in "${FILES[@]}"; do
  [ -f "$SRC/$f" ] || { echo "missing $SRC/$f: pass the folder of the downloaded model files" >&2; exit 1; }
done
adb shell mkdir -p "$STAGE"
adb shell run-as "$PKG" mkdir -p files/models
for f in "${FILES[@]}"; do
  size=$(stat -L -f %z "$SRC/$f" 2>/dev/null || stat -L -c %s "$SRC/$f")
  echo "install $f ($size bytes)"
  adb push "$SRC/$f" "$STAGE/$f" >/dev/null
  adb shell run-as "$PKG" cp "$STAGE/$f" "files/models/$f"
  adb shell rm -f "$STAGE/$f"
  got=$(adb shell run-as "$PKG" stat -c %s "files/models/$f" | tr -d '\r')
  [ "$got" = "$size" ] || { echo "size mismatch for $f: $got != $size" >&2; exit 1; }
done
adb shell rmdir "$STAGE" 2>/dev/null || true
echo "done: ${#FILES[@]} files in /data/data/$PKG/files/models. Start the app."
