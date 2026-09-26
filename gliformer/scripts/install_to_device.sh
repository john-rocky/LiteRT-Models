#!/usr/bin/env bash
# Install the debug APK first; run-as copies external models into com.gliformer's private files.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR=''
SERIAL=''
FIXTURES=''
WITH_S256=false
WITH_FP32_TABLE=false
PACKAGE=com.gliformer
TEMP_DIR="/data/local/tmp/gliformer_assets_$$"
LOCAL_TEMP="$SCRIPT_DIR/../.local/install-$$"

usage() {
    printf 'Usage: %s --serial SERIAL [--with-s256] [--fp32-table] [--fixtures DATA] [--staging-dir DIR] MODEL_DIR\n' "$0"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --serial|--fixtures|--staging-dir)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            case "$1" in
                --serial) SERIAL="$2" ;;
                --fixtures) FIXTURES="$2" ;;
                --staging-dir) TEMP_DIR="$2" ;;
            esac
            shift 2 ;;
        --with-s256) WITH_S256=true; shift ;;
        --fp32-table) WITH_FP32_TABLE=true; shift ;;
        --help|-h) usage; exit 0 ;;
        --*) usage >&2; exit 2 ;;
        *)
            [[ -z "$SOURCE_DIR" ]] || { usage >&2; exit 2; }
            SOURCE_DIR="$1"; shift ;;
    esac
done
[[ -n "$SERIAL" && -n "$SOURCE_DIR" ]] || { usage >&2; exit 2; }
[[ "$TEMP_DIR" =~ ^/data/local/tmp/gliformer_[A-Za-z0-9_-]+$ ]] || {
    printf 'Staging directory must be a dedicated /data/local/tmp/gliformer_NAME directory.\n' >&2
    exit 2
}
FILES=(
    gliformer_large_ner_s128_wfp16.tflite
    word_embeddings_fp16.bin
    tokenizer.json
    tokenizer_config.json
    gliner_config.json
    graph_contract_s128.json
)
if "$WITH_S256"; then
    FILES+=(
        gliformer_large_ner_s256_encoder_wfp16.tflite
        gliformer_large_ner_s256_head_wfp16.tflite
        graph_contract_s256.json
    )
fi
if "$WITH_FP32_TABLE"; then FILES+=(word_embeddings_fp32.bin); fi
source_file() {
    case "$1" in
        *.tflite) printf '%s/%s\n' "$SOURCE_DIR" "$1" ;;
        *) printf '%s/host_assets/%s\n' "$SOURCE_DIR" "$1" ;;
    esac
}
# Validate all inputs before the first device command.
for name in "${FILES[@]}"; do
    source="$(source_file "$name")"
    [[ -f "$source" ]] || { printf 'Missing source file: %s\n' "$source" >&2; exit 1; }
done

ADB=(adb -s "$SERIAL")
staging_created=false
pending_name=''
cleanup() {
    local result=$?
    if [[ -n "$pending_name" ]]; then
        "${ADB[@]}" shell run-as "$PACKAGE" rm -f "files/$pending_name.pending" || true
    fi
    if "$staging_created"; then "${ADB[@]}" shell rm -rf "$TEMP_DIR" || true; fi
    if [[ -d "$LOCAL_TEMP" ]]; then rm -rf "$LOCAL_TEMP"; fi
    return "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ -n "$FIXTURES" ]]; then
    mkdir -p "$LOCAL_TEMP"
    python3 "$SCRIPT_DIR/package_gate_fixtures.py" "$FIXTURES" "$LOCAL_TEMP/gate-fixtures.tar"
fi
"${ADB[@]}" shell run-as "$PACKAGE" mkdir -p files
# A pre-existing directory is an error: cleanup only owns the directory created here.
"${ADB[@]}" shell mkdir "$TEMP_DIR"
staging_created=true
for name in "${FILES[@]}"; do
    pending_name="$name"
    printf 'Staging %s\n' "$name"
    "${ADB[@]}" push "$(source_file "$name")" "$TEMP_DIR/$name"
    "${ADB[@]}" shell run-as "$PACKAGE" cp "$TEMP_DIR/$name" "files/$name.pending"
    "${ADB[@]}" shell run-as "$PACKAGE" mv "files/$name.pending" "files/$name"
    "${ADB[@]}" shell rm "$TEMP_DIR/$name"
    pending_name=''
done
if [[ -n "$FIXTURES" ]]; then
    "${ADB[@]}" push "$LOCAL_TEMP/gate-fixtures.tar" "$TEMP_DIR/gate-fixtures.tar"
    "${ADB[@]}" shell run-as "$PACKAGE" tar -xf "$TEMP_DIR/gate-fixtures.tar" -C files
fi
"${ADB[@]}" shell run-as "$PACKAGE" ls -la files/
printf 'GLiFormer files installed. Launch com.gliformer/.MainActivity.\n'
