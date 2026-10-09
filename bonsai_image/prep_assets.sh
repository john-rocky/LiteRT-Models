#!/bin/bash
# Copies the app's bundled assets (tokenizer tables + pipeline meta) into
# app/src/main/assets/. assets/ is not committed — run this before building.
#
#   ./prep_assets.sh <tokenizer dir> [pipeline_meta.json]
#
# <tokenizer dir> holds vocab.json + merges.txt, or is a model download whose
# tokenizer/ subfolder does. pipeline_meta.json defaults to the one next to it;
# pass it explicitly when that copy predates the 256x256 variant, e.g.
#   ./prep_assets.sh ~/models/bonsai-image-256/hub512 ~/models/bonsai-image-256/upload/pipeline_meta.json
set -euo pipefail
if [ $# -lt 1 ]; then
  sed -n '5,10p' "$0" >&2
  exit 2
fi
TOK="$1"
[ -f "$TOK/vocab.json" ] || TOK="$1/tokenizer"
META="${2:-$1/pipeline_meta.json}"
for f in "$TOK/vocab.json" "$TOK/merges.txt" "$META"; do
  [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
grep -q '"variants"' "$META" ||
  echo "warning: $META has no variants — the app will offer 512x512 only" >&2
DST="$(dirname "$0")/app/src/main/assets"
mkdir -p "$DST"
cp "$TOK/vocab.json" "$TOK/merges.txt" "$DST/"
cp "$META" "$DST/pipeline_meta.json"
ls -la "$DST"
shasum -a 256 "$DST"/*
