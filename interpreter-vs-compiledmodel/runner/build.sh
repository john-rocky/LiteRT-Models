#!/bin/bash
# Build ftfl_bench for Android arm64 with the NDK's clang (no Gradle, no CMake). Headers only, nothing linked from
# LiteRT: the TFLite C API headers of the LiteRT 1.4.0 AARs (tfl_path.cc) and the LiteRT v2.3.0 litert_cc_sdk.zip C
# headers (lrt_path.cc; the v2.3.2 C API is the same). The runtime libraries are dlopened on the phone.
#   NDK=<Android NDK dir> X=<headers dir> bash build.sh        (from anywhere)
# X holds litert-1.4.0/headers and litert-gpu-1.4.0/headers (the two 1.4.0 AARs unpacked) and
# litert_cc_sdk/litert_cc_sdk (the v2.3.0 litert_cc_sdk.zip unpacked); manifest.json lists their URLs and sha256.
# Writes build/ftfl_bench next to this script, its sha256 (build/ftfl_bench.sha256) and the compiler's banner.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
NDK=${NDK:-$HOME/Library/Android/sdk/ndk/29.0.13113456}
X=${X:-$(cd "$HERE/../.." && pwd)/aar/x}
API=${API:-28}
CXX=$(ls "$NDK"/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android${API}-clang++ | head -n 1)
[ -x "$CXX" ] || { echo "no clang++ for API $API under $NDK" >&2; exit 1; }
for d in "$X/litert-1.4.0/headers/tflite/c" "$X/litert-gpu-1.4.0/headers/tflite/delegates/gpu" "$X/litert_cc_sdk/litert_cc_sdk/litert/c"; do
  [ -d "$d" ] || { echo "missing $d (set X to the dir with the unpacked AARs and litert_cc_sdk-v2.3.0.zip)" >&2; exit 1; }
done
SDK=$X/litert_cc_sdk/litert_cc_sdk
# build_config.h is generated from build_config.h.in by CMake; GPU and NPU stay enabled (each toggle = 0)
if [ ! -f "$SDK/litert/build_common/build_config.h" ]; then
  sed -E 's/^#cmakedefine01 ([A-Z_]+)/#define \1 0/' "$SDK/litert/build_common/build_config.h.in" > "$SDK/litert/build_common/build_config.h"
fi
mkdir -p "$HERE/build"
OUT=$HERE/build/ftfl_bench
COMMON=(-std=c++17 -O2 -Wall -fno-exceptions -static-libstdc++ -I "$HERE")
"$CXX" "${COMMON[@]}" -c "$HERE/tfl_path.cc" -o "$HERE/build/tfl_path.o" \
  -I "$X/litert-1.4.0/headers" -I "$X/litert-1.4.0/headers/external/org_tensorflow" -I "$X/litert-gpu-1.4.0/headers"
"$CXX" "${COMMON[@]}" -c "$HERE/lrt_path.cc" -o "$HERE/build/lrt_path.o" -I "$SDK"
"$CXX" "${COMMON[@]}" -c "$HERE/ftfl_bench.cc" -o "$HERE/build/ftfl_bench.o"
"$CXX" -static-libstdc++ "$HERE/build/ftfl_bench.o" "$HERE/build/tfl_path.o" "$HERE/build/lrt_path.o" -ldl -o "$OUT"
rm -f "$HERE/build/"*.o
SHA=$(shasum -a 256 "$OUT" | cut -d' ' -f1)
echo "$SHA" > "$OUT.sha256"
"$CXX" --version | head -1 > "$HERE/build/compiler.txt"
echo "ftfl_bench $SHA $(stat -f %z "$OUT") B ($(cat "$HERE/build/compiler.txt"))"
