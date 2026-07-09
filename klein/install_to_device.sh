#!/usr/bin/env bash
# Stage the FLUX.2-klein int8 LiteRT graphs and the host .bin tensors to the device.
#
# Usage: ./install_to_device.sh <graphs_dir> <bins_dir>
#   graphs_dir  holds the twelve .tflite chunks (scripts/chunked_export_klein.py,
#               scripts/build_klein_enc.py, scripts/vae_deploy_klein.py)
#   bins_dir    holds klein_bins/ written by scripts/gen_prep_klein.py
#
# The graphs total ~6.2 GB and are never committed. The app reads them from its
# external files dir with the file-path CompiledModel.create overload.
set -euo pipefail

GRAPHS="${1:?usage: install_to_device.sh <graphs_dir> <bins_dir>}"
BINS="${2:?usage: install_to_device.sh <graphs_dir> <bins_dir>}"
PKG="com.klein"
DEST="/sdcard/Android/data/${PKG}/files"

GRAPH_FILES=(
  ke_enc0.tflite ke_enc1.tflite ke_enc2.tflite
  kc_prep.tflite kc_double0.tflite kc_double1.tflite
  kc_single0.tflite kc_single1.tflite kc_single2.tflite kc_single3.tflite
  kc_final.tflite kv_vae.tflite
)

for f in "${GRAPH_FILES[@]}"; do
  [ -f "${GRAPHS}/${f}" ] || { echo "missing ${GRAPHS}/${f}"; exit 1; }
done
[ -d "${BINS}" ] || { echo "missing ${BINS}"; exit 1; }

adb shell mkdir -p "${DEST}/klein_bins"
for f in "${GRAPH_FILES[@]}"; do
  echo "pushing ${f} ($(du -h "${GRAPHS}/${f}" | cut -f1)) ..."
  adb push "${GRAPHS}/${f}" "${DEST}/${f}"
done
adb push "${BINS}/." "${DEST}/klein_bins/"

# adb-created dirs are group-owned by shell; make them app-traversable.
adb shell chmod 777 "${DEST}/klein_bins"
adb shell 'chmod 666 '"${DEST}"'/klein_bins/*.bin'
echo "done -> ${DEST}"
