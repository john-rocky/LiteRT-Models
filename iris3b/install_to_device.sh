#!/usr/bin/env bash
# Install the iris-3b model files into the app's private storage on a connected phone.
#
#   ./gradlew :app:installDebug            # the app first (a debug build: run-as needs it)
#   ./install_to_device.sh <model_dir>     # then the 24 model files (13.7 GB)
#   CONFIG=iris3b_config_dit_int8_fp16.json ./install_to_device.sh <model_dir>
#                                          # or the int8 transformer setting (24 files, 11.3 GB)
#
# <model_dir> holds the 20 LiteRT graphs (fp16 weights, fp32 compute), the bf16 embedding table, the
# host tensors, the tokenizer and iris3b_config.json, under the names listed in FILES below, and the
# int8 transformer setting: iris3b_config_dit_int8_fp16.json and its six int8 trunk graphs
# (iris_dit_*_i8w_dit.tflite). CONFIG picks the config (default iris3b_config.json, every graph at
# FP32); it is copied as iris3b_config.json, the name the app reads, with the graphs it names.
#
# Each file is streamed straight into /data/data/com.iris3b/files (adb exec-in + run-as), so nothing
# is staged on the phone: it needs room for the files plus MARGIN_GB and no more. The free space is
# read before the first file and the script stops if it is short. Every file is checked by size (and
# by sha256 when <model_dir>/SHA256SUMS exists); a file already on the phone with the right size is
# skipped, so a re-run resumes an interrupted install. The app's files are removed with the app
# (adb uninstall com.iris3b).
#
# Env: ANDROID_SERIAL picks the phone when several are attached; ADB = the adb binary (default adb);
#      MARGIN_GB = free space to leave on /data (default 3); CONFIG = the config to install (above).
set -euo pipefail

PKG=com.iris3b
ADB=${ADB:-adb}
MARGIN_GB=${MARGIN_GB:-3}
DIR=${1:?usage: ./install_to_device.sh <model_dir>}
CONFIG=${CONFIG:-iris3b_config.json}
case "$CONFIG" in
  iris3b_config.json) W=f16w ;;
  iris3b_config_dit_int8_fp16.json) W=i8w ;;
  *) echo "CONFIG is iris3b_config.json or iris3b_config_dit_int8_fp16.json (got $CONFIG)"; exit 1 ;;
esac

# Largest first; the config last, so a partial install reads as "not installed" in the app.
FILES=(
  iris_dit_dual0_${W}_dit.tflite iris_dit_dual1_${W}_dit.tflite
  iris_dit_single0_${W}_dit.tflite iris_dit_single1_${W}_dit.tflite
  iris_embed_bf16.f32
  iris_te_c01_f16w.tflite iris_te_c02_f16w.tflite iris_te_c03_f16w.tflite iris_te_c04_f16w.tflite
  iris_te_c05_f16w.tflite iris_te_c06_f16w.tflite iris_te_c07_f16w.tflite iris_te_c08_f16w.tflite
  iris_te_c09_f16w.tflite iris_te_c10_f16w.tflite iris_te_c11_f16w.tflite
  iris_dit_dual2_${W}_dit.tflite iris_dit_single2_${W}_dit.tflite
  iris_dit_adapter_f16w_dit.tflite iris_dit_pixel_f16w_dit.tflite iris_te_c00_f16w.tflite
  iris_host.f32 iris_tokenizer.json
  "$CONFIG"
)
# Every file keeps its name on the phone except the config, which the app reads as iris3b_config.json.
target() { if [ "$1" = "$CONFIG" ]; then echo iris3b_config.json; else echo "$1"; fi; }

size_of() { stat -L -c %s "$1" 2>/dev/null || stat -L -f %z "$1"; }
phone() { "$ADB" shell "$@" | tr -d '\r'; }
phone_size() { phone run-as "$PKG" stat -c %s "files/$1" 2>/dev/null || true; }
local_sha() { { sha256sum "$1" 2>/dev/null || shasum -a 256 "$1"; } | awk '{ print $1 }'; }
on_phone() {  # file -> 0 when the phone has it: by size, and the config also by sha256 (the two configs are both ~19 kB)
  local t
  t=$(target "$1")
  [ "$(phone_size "$t")" = "$(size_of "$DIR/$1")" ] || return 1
  [ "$1" != "$CONFIG" ] && return 0
  [ "$(phone run-as "$PKG" sha256sum "files/$t" 2>/dev/null | awk '{ print $1 }')" = "$(local_sha "$DIR/$1")" ]
}
gb() { awk -v b="$1" 'BEGIN { printf "%.2f GB", b / 1e9 }'; }

for f in "${FILES[@]}"; do
  [ -f "$DIR/$f" ] || { echo "missing $DIR/$f"; exit 1; }
done
"$ADB" get-state > /dev/null || { echo "no phone on adb"; exit 1; }
packages=$(phone pm list packages "$PKG")
if ! grep -qx "package:$PKG" <<< "$packages"; then
  echo "$PKG is not installed: run ./gradlew :app:installDebug first"; exit 1
fi
phone run-as "$PKG" mkdir -p files > /dev/null || { echo "run-as $PKG failed: install a debug build of the app"; exit 1; }

todo=()
need=0
for f in "${FILES[@]}"; do
  want=$(size_of "$DIR/$f")
  if on_phone "$f"; then
    echo "on the phone already: $(target "$f")"
  else
    todo+=("$f")
    need=$((need + want))
  fi
done
if [ "${#todo[@]}" = 0 ]; then
  echo "all ${#FILES[@]} files are on the phone; open the app"
  exit 0
fi
free_kb=$(phone df -k /data | awk 'NR == 2 { print $4 }')
[ -n "$free_kb" ] || { echo "cannot read the free space of /data"; exit 1; }
margin=$(( MARGIN_GB * 1000000000 ))
echo "to copy: ${#todo[@]} files, $(gb "$need"); free on /data: $(gb $(( free_kb * 1024 ))); margin $(gb "$margin")"
if [ $(( free_kb * 1024 )) -lt $(( need + margin )) ]; then
  echo "not enough free space on the phone: need $(gb $(( need + margin ))) (files + margin), free $(gb $(( free_kb * 1024 )))"
  exit 3
fi

sums=""
[ -f "$DIR/SHA256SUMS" ] && sums="$DIR/SHA256SUMS"
t_all=$(date +%s)
i=0
for f in "${todo[@]}"; do
  i=$((i + 1))
  want=$(size_of "$DIR/$f")
  t=$(date +%s)
  t_name=$(target "$f")
  echo "[$i/${#todo[@]}] $f ($(gb "$want"))$([ "$t_name" = "$f" ] || echo " as $t_name")"
  "$ADB" exec-in "run-as $PKG sh -c 'cat > files/$t_name'" < "$DIR/$f"
  # exec-in can return before the phone has written everything: wait for the size, give up after
  # 30 s without growth.
  got=$(phone_size "$t_name")
  last=""
  still=0
  while [ "$got" != "$want" ] && [ "$still" -lt 30 ]; do
    sleep 1
    last=$got
    got=$(phone_size "$t_name")
    if [ "$got" = "$last" ]; then still=$((still + 1)); else still=0; fi
  done
  if [ "$got" != "$want" ]; then
    phone run-as "$PKG" rm -f "files/$t_name" > /dev/null
    echo "copy failed: $t_name has ${got:-no} bytes on the phone, $want here (removed it; run again to resume)"
    exit 1
  fi
  if [ -n "$sums" ]; then
    expect=$(awk -v n="$f" '$2 == n || $2 == "*" n { print $1 }' "$sums")
    actual=$(phone run-as "$PKG" sha256sum "files/$t_name" | awk '{ print $1 }')
    if [ -n "$expect" ] && [ "$actual" != "$expect" ]; then
      phone run-as "$PKG" rm -f "files/$t_name" > /dev/null
      echo "sha256 mismatch: $t_name (removed it; run again)"
      exit 1
    fi
  fi
  s=$(( $(date +%s) - t ))
  echo "  done in ${s}s"
done
echo "installed ${#todo[@]} files in $(( $(date +%s) - t_all ))s; free on /data now: $(gb $(( $(phone df -k /data | awk 'NR == 2 { print $4 }') * 1024 )))"
echo "open the app (or tap Check again)"
