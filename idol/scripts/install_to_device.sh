#!/usr/bin/env bash
# Copy Idol's graphs, side files and manifest into the app's private files/models/, one file at a time
# (adb push -> run-as cp -> rm), and with FIXTURES=1 also the test fixtures (device/fx/<tag>/ -> files/fx/<tag>/) and
# the autorun plans (plans/*.json -> files/plans/).
#
# The file list, sizes and SHA-256 come from manifest.json in IDOL_DIR. Sources:
#   IDOL_DIR  manifest.json, every graph except dino_* and dis, the side and JSON files
#   DINO_DIR  dino_0..3.tflite (and dino_rope_cos/sin.f32 when they are there, else IDOL_DIR's)
#   DIS_DIR   dis.tflite (optional: only the "Background removal on the GPU" setting uses it)
# Every source is checked by size before any device command; after each copy the SHA-256 is checked on the phone
# (the manifest's value for a graph, the local file's for the rest). A file whose copy on the phone already has the
# expected SHA-256 is not copied again. MODELS=0 skips the models (fixtures and plans only, with FIXTURES=1).
# Install the debug APK first: run-as needs a debuggable package. ANDROID_SERIAL selects the device. Needs python3
# (reads the manifest).
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Defaults for local testing: this worktree's export (models/v2 holds every graph except dis.tflite).
# For the published release point them at the downloaded repos (see README), e.g.
#   IDOL_DIR=~/Downloads/Idol DINO_DIR=~/Downloads/DINOv3-ViT-H16plus-Idol-LiteRT DIS_DIR=~/Downloads/DIS-ISNet-LiteRT
#   (IDOL_DIR=~/Downloads/Idol/262k for the first release's form: 262,144 Gaussians, same DINO_DIR)
IDOL_DIR="${IDOL_DIR:-$MODULE_DIR/models/v2}"
DINO_DIR="${DINO_DIR:-$IDOL_DIR}"
DIS_DIR="${DIS_DIR:-$HOME/Downloads/DIS-ISNet-LiteRT}"
MODELS="${MODELS:-1}"
FIXTURES="${FIXTURES:-0}"
FX_TAGS="${FX_TAGS:-armchair axolotl succulent black}"
PACKAGE=com.idol
TEMP_DIR=/data/local/tmp/idol_install

if [[ $# -gt 0 ]]; then
    printf 'Usage: [IDOL_DIR=...] [DINO_DIR=...] [DIS_DIR=...] [MODELS=0] [FIXTURES=1] %s\n' "$0" >&2
    exit 2
fi
MANIFEST="$IDOL_DIR/manifest.json"
if [[ ! -f "$MANIFEST" ]]; then
    printf 'Missing %s: set IDOL_DIR (and DINO_DIR, DIS_DIR) to the downloaded model folders (see README)\n' \
        "$MANIFEST" >&2
    exit 1
fi

sha_of() {
    if command -v sha256sum > /dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# "kind name file bytes sha" per line; bytes and sha are "-" when the manifest has none
ENTRIES="$(python3 -I - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for name, g in m['graphs'].items():
    print('graph', name, g['file'], g['bytes'], g.get('sha256') or '-')
for f, s in m.get('side_files', {}).items():
    n = 4
    for d in s['shape']:
        n *= d
    print('side', f, f, n, '-')
for f in m.get('json_files', []):
    print('json', f, f, '-', '-')
PY
)"

# The source of each file.
source_of() {
    local kind="$1" name="$2" file="$3"
    case "$kind:$name" in
        graph:dino_*) printf '%s/%s' "$DINO_DIR" "$file" ;;
        graph:dis) printf '%s/%s' "$DIS_DIR" "$file" ;;
        side:dino_rope_*)
            if [[ -f "$DINO_DIR/$file" ]]; then printf '%s/%s' "$DINO_DIR" "$file"; else printf '%s/%s' "$IDOL_DIR" "$file"; fi
            ;;
        *) printf '%s/%s' "$IDOL_DIR" "$file" ;;
    esac
}

# Check every source before any device command to avoid a partial installation.
PLAN=()
if [[ "$MODELS" == 0 ]]; then ENTRIES=''; fi
while read -r kind name file bytes sha; do
    [[ -z "$kind" ]] && continue
    src="$(source_of "$kind" "$name" "$file")"
    if [[ ! -f "$src" ]]; then
        if [[ "$kind:$name" == graph:dis ]]; then
            printf 'Skipping dis.tflite (not in %s): the GPU background removal setting will not work\n' "$DIS_DIR"
            continue
        fi
        printf 'Missing source file: %s\n' "$src" >&2
        exit 1
    fi
    if [[ "$bytes" != - ]]; then
        actual="$(wc -c < "$src" | tr -d ' ')"
        if [[ "$actual" != "$bytes" ]]; then
            printf '%s is %s bytes, expected %s\n' "$src" "$actual" "$bytes" >&2
            exit 1
        fi
    fi
    # files without a manifest SHA-256 (side and JSON files) are checked against the local copy's
    if [[ "$sha" == - ]]; then sha="$(sha_of "$src")"; fi
    PLAN+=("$src|models/$file|$sha")
done <<< "$ENTRIES"
if [[ "$MODELS" != 0 ]]; then
    PLAN+=("$MANIFEST|models/manifest.json|$(sha_of "$MANIFEST")")
fi

if [[ "$FIXTURES" == 1 ]]; then
    for tag in $FX_TAGS; do
        dir="$MODULE_DIR/device/fx/$tag"
        if [[ ! -d "$dir" ]]; then
            printf 'Missing fixture folder: %s\n' "$dir" >&2
            exit 1
        fi
        for f in "$dir"/*; do
            [[ -f "$f" ]] || continue
            PLAN+=("$f|fx/$tag/$(basename "$f")|$(sha_of "$f")")
        done
    done
    for f in "$MODULE_DIR"/plans/*.json; do
        [[ -f "$f" ]] || continue
        PLAN+=("$f|plans/$(basename "$f")|$(sha_of "$f")")
    done
fi

pending=''
cleanup() {
    if [[ -n "$pending" ]]; then
        adb shell rm -f "$pending" < /dev/null
    fi
}
trap cleanup EXIT

# Copies $1 (a local file) to files/$2 of the app and checks the SHA-256 $3 on the phone ("-": no check).
install_file() {
    local src="$1" dst="$2" sha="$3"
    pending="$TEMP_DIR/$$-$(basename "$dst")"
    adb push "$src" "$pending" < /dev/null
    adb shell run-as "$PACKAGE" mkdir -p "files/$(dirname "$dst")" < /dev/null
    adb shell run-as "$PACKAGE" cp "$pending" "files/$dst" < /dev/null
    adb shell rm "$pending" < /dev/null
    pending=''
    if [[ "$sha" != - ]]; then
        local got
        got="$(adb shell run-as "$PACKAGE" sha256sum "files/$dst" < /dev/null | cut -d' ' -f1 | tr -d '\r')"
        if [[ "$got" != "$sha" ]]; then
            printf 'files/%s on the device has SHA-256 %s, expected %s\n' "$dst" "$got" "$sha" >&2
            exit 1
        fi
    fi
}

adb shell mkdir -p "$TEMP_DIR" < /dev/null
adb shell run-as "$PACKAGE" mkdir -p files/models < /dev/null
for entry in "${PLAN[@]}"; do
    IFS='|' read -r src dst sha <<< "$entry"
    if [[ "$sha" != - ]]; then
        have="$(adb shell run-as "$PACKAGE" sha256sum "files/$dst" < /dev/null 2> /dev/null | cut -d' ' -f1 | tr -d '\r' || true)"
        if [[ "$have" == "$sha" ]]; then
            printf 'Up to date: %s\n' "$dst"
            continue
        fi
    fi
    printf 'Installing %s (%s bytes)\n' "$dst" "$(wc -c < "$src" | tr -d ' ')"
    install_file "$src" "$dst" "$sha"
done
adb shell rmdir "$TEMP_DIR" < /dev/null 2> /dev/null || true
adb shell run-as "$PACKAGE" ls -la files/models < /dev/null
printf 'Files installed. Open Idol.\n'
