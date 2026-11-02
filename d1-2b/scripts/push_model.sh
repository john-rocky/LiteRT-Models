#!/usr/bin/env bash
# Copies a downloaded d1-2B-LiteRT folder to the app's model folder on the phone and checks
# every file's size. Open the app once before (it creates its folder). Usage:
#   scripts/push_model.sh /path/to/d1-2B-LiteRT
# Set ANDROID_SERIAL when more than one phone is connected.
set -euo pipefail

SOURCE="${1:?usage: $0 /path/to/d1-2B-LiteRT}"
TARGET=/sdcard/Android/data/com.d1_2b/files/model

if ! adb shell "test -d /sdcard/Android/data/com.d1_2b/files"; then
  echo "Open the app once first: it creates its folder on the phone." >&2
  exit 1
fi

cd "$SOURCE"
FILES=()
for f in provider/tokenizer.json provider/config.json provider/calibration.json tables/embed_table.npy; do
  [ -f "$f" ] || { echo "Missing $SOURCE/$f" >&2; exit 1; }
  FILES+=("$f")
done
shopt -s nullglob
GRAPHS=(d1-2b_*.tflite)
[ ${#GRAPHS[@]} -gt 0 ] || { echo "No d1-2b_*.tflite graph in $SOURCE" >&2; exit 1; }
FILES+=("${GRAPHS[@]}")

for f in "${FILES[@]}"; do
  adb shell "mkdir -p '$TARGET/$(dirname "$f")'"
  echo "pushing $f"
  adb push "$f" "$TARGET/$f" >/dev/null
  local_size=$(wc -c < "$f" | tr -d ' ')
  phone_size=$(adb shell "stat -c %s '$TARGET/$f'" | tr -d '\r')
  if [ "$local_size" != "$phone_size" ]; then
    echo "Size differs for $f: $local_size here, $phone_size on the phone" >&2
    exit 1
  fi
done
echo "Done: ${#FILES[@]} files in $TARGET. Restart the app."
