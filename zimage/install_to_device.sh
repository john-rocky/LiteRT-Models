#!/usr/bin/env bash
# Stage the Z-Image int8 LiteRT graphs to the app's files dir.
# Usage: ./install_to_device.sh <graphs_dir>   (dir with qwen_enc.tflite, zdit.tflite, zvae.tflite)
# The graphs are large (~10 GB total) and never committed — convert with scripts/build_zimage.py.
set -euo pipefail

GRAPHS="${1:?usage: install_to_device.sh <graphs_dir>}"
PKG="com.zimage"
DEST="/sdcard/Android/data/${PKG}/files"

for f in qwen_enc.tflite zdit.tflite zvae.tflite; do
  [ -f "${GRAPHS}/${f}" ] || { echo "missing ${GRAPHS}/${f}"; exit 1; }
done

adb shell mkdir -p "${DEST}"
for f in qwen_enc.tflite zdit.tflite zvae.tflite; do
  echo "pushing ${f} ($(du -h "${GRAPHS}/${f}" | cut -f1)) ..."
  adb push "${GRAPHS}/${f}" "${DEST}/${f}"
done
echo "done -> ${DEST}"
