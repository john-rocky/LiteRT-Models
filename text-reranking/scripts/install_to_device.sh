#!/usr/bin/env bash
# Stage the Qwen3-Embedding graph + token embedding table into the app's private filesDir.
#   ./scripts/install_to_device.sh [dir-with-artifacts]   (default: ~/Downloads/meeting/qwen3rerank-work)
# Artifacts (produced by conversion/build_qwen3emb.py + scripts/export_embeddings.py):
#   qwen3rerank_gpu_fp16.tflite  (881 MB)   embeddings_fp16.bin  (310 MB)
set -e
PKG=com.textreranking
DIR="${1:-$HOME/Downloads/meeting/qwen3rerank-work}"
adb shell run-as $PKG mkdir files 2>/dev/null || true
for M in qwen3rerank_gpu_fp16.tflite embeddings_fp16.bin; do
  echo "push $M"
  adb push "$DIR/$M" "/data/local/tmp/$M"
  adb shell run-as $PKG cp "/data/local/tmp/$M" "files/$M"
  adb shell rm "/data/local/tmp/$M"
done
adb shell run-as $PKG ls -la files/
echo "done — launch the Reranker app."
