#!/usr/bin/env bash
# --validate-only checks the files and the APK and never starts adb.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" -B "$SCRIPT_DIR/install_assets.py" "$@"
