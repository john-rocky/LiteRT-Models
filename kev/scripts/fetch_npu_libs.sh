#!/usr/bin/env bash
# Copies the Qualcomm NPU runtime for JIT compilation into app/src/main/jniLibs/arm64-v8a, so that
# the next build packages it and the NPU choice of the app works.
#
# The libraries are not in git: the QAIRT license allows them inside an app (APK) only. Collect them
# into one directory first:
#   libLiteRtDispatch_Qualcomm.so          litert_npu_runtime_libraries.zip (LiteRT GitHub Release)
#   libLiteRtCompilerPlugin_Qualcomm.so    litert_npu_runtime_libraries_jit.zip (separate release asset)
#   libQnnHtp.so libQnnSystem.so libQnnHtpPrepare.so libQnnIr.so libQnnSaver.so
#   libQnnHtpV<NN>Stub.so libQnnHtpV<NN>CalculatorStub.so   QAIRT lib/aarch64-android/
#   libQnnHtpV<NN>Skel.so                  QAIRT lib/hexagon-v<NN>/unsigned/
# <NN> follows the SoC: SM8550 v73, SM8650 v75, SM8750 v79, SM8850 v81 (Galaxy S26, measured). The
# root README section "Running on the NPU" lists the same files.
#
# usage: scripts/fetch_npu_libs.sh <directory-with-the-libraries> [hexagon-version, default v81]
set -euo pipefail

SOURCE=${1:?usage: scripts/fetch_npu_libs.sh <directory-with-the-libraries> [v81]}
HEXAGON=${2:-v81}
MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$MODULE_DIR/app/src/main/jniLibs/arm64-v8a"
UPPER=$(printf '%s' "$HEXAGON" | tr 'v' 'V')

FILES=(
  libLiteRtDispatch_Qualcomm.so
  libLiteRtCompilerPlugin_Qualcomm.so
  libQnnHtp.so
  libQnnSystem.so
  libQnnHtpPrepare.so
  libQnnIr.so
  libQnnSaver.so
  "libQnnHtp${UPPER}Stub.so"
  "libQnnHtp${UPPER}CalculatorStub.so"
  "libQnnHtp${UPPER}Skel.so"
)

missing=0
for name in "${FILES[@]}"; do
  if [ ! -f "$SOURCE/$name" ]; then
    echo "missing: $SOURCE/$name" >&2
    missing=1
  fi
done
[ "$missing" -eq 0 ] || exit 2

mkdir -p "$TARGET"
for name in "${FILES[@]}"; do
  cp "$SOURCE/$name" "$TARGET/$name"
done
echo "Copied ${#FILES[@]} libraries for Hexagon $HEXAGON into $TARGET"
