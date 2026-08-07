#!/bin/bash
# showcase_record.sh — record one showcase clip from the connected device.
#
# Clips are written OUTSIDE the repo to ~/Downloads/showcase-video/clips/
# (videos must never be committed).
#
# Usage:
#   tools/showcase_record.sh check
#       Preflight: device connection, which candidate apps are installed,
#       and the current foreground activity.
#   tools/showcase_record.sh <name> [seconds]
#       Video-only recording via `adb screenrecord` (no audio, max 180 s).
#   tools/showcase_record.sh <name> [seconds] --audio
#       Video + device audio via `scrcpy --record` (required for TTS demos).
#
# After each recording the clip is pulled/verified and a one-line ffprobe
# summary is printed. Re-record by running the same command again.

set -u

ADB="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
SCRCPY="${SCRCPY:-scrcpy}"
CLIPS_DIR="${CLIPS_DIR:-$HOME/Downloads/showcase-video/clips}"
DEFAULT_SECONDS=12
BIT_RATE=12000000

# Candidate packages for the showcase lineup (preflight `check` lists which
# of these are already installed on the device).
CANDIDATE_PACKAGES=(
  com.edsr com.xfeat com.gfpgan com.lama com.dehaze com.da3
  com.edgetam com.edgetamvideo com.yolo com.yolopose com.yolotracking
  com.kokoro com.whisper com.smolvlm com.tddfa com.moge
)

die() { echo "ERROR: $*" >&2; exit 1; }

require_device() {
  "$ADB" get-state >/dev/null 2>&1 || die "no device connected (adb get-state failed)"
}

probe_summary() {
  local f="$1"
  if command -v ffprobe >/dev/null 2>&1; then
    local dur streams
    dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null)
    streams=$(ffprobe -v error -show_entries stream=codec_type -of csv=p=0 "$f" 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    echo "  -> $(basename "$f"): ${dur%.*}s, streams=[$streams], $(du -h "$f" | cut -f1 | tr -d ' ')"
    if [[ "${WANT_AUDIO:-0}" == 1 ]] && ! echo "$streams" | grep -q audio; then
      echo "  !! WARNING: audio was requested but the clip has NO audio stream" >&2
    fi
  else
    ls -lh "$f"
  fi
}

cmd_check() {
  echo "== adb devices =="
  "$ADB" devices -l
  "$ADB" get-state >/dev/null 2>&1 || { echo "-- no device: connect the Pixel and rerun --"; exit 1; }
  echo
  echo "== installed showcase candidates =="
  local installed
  installed=$("$ADB" shell pm list packages 2>/dev/null)
  for p in "${CANDIDATE_PACKAGES[@]}"; do
    if echo "$installed" | grep -q "package:$p$"; then
      echo "  [x] $p"
    else
      echo "  [ ] $p   (needs install)"
    fi
  done
  echo
  echo "== current foreground activity =="
  "$ADB" shell dumpsys activity activities 2>/dev/null | grep topResumedActivity | head -1
  echo
  echo "== scrcpy =="
  command -v "$SCRCPY" >/dev/null 2>&1 && "$SCRCPY" --version | head -1 || echo "scrcpy NOT FOUND"
}

main() {
  [[ $# -ge 1 ]] || die "usage: $0 check | <name> [seconds] [--audio]"
  if [[ "$1" == "check" ]]; then cmd_check; exit 0; fi

  local name="$1"; shift
  local seconds="$DEFAULT_SECONDS"
  WANT_AUDIO=0
  for a in "$@"; do
    case "$a" in
      --audio) WANT_AUDIO=1 ;;
      ''|*[!0-9]*) die "unknown argument: $a" ;;
      *) seconds="$a" ;;
    esac
  done

  require_device
  mkdir -p "$CLIPS_DIR"
  local out="$CLIPS_DIR/${name}.mp4"

  # Show taps on screen while recording (restored afterwards).
  "$ADB" shell settings put system show_touches 1

  if [[ "$WANT_AUDIO" == 1 ]]; then
    command -v "$SCRCPY" >/dev/null 2>&1 || die "scrcpy not found (needed for --audio)"
    echo "Recording ${seconds}s WITH device audio -> $out  (Ctrl+C to stop early)"
    "$SCRCPY" --record="$out" --time-limit="$seconds" --no-playback \
      --video-bit-rate="$BIT_RATE"
  else
    [[ "$seconds" -le 180 ]] || die "adb screenrecord caps at 180 s"
    local remote="/data/local/tmp/showcase_${name}.mp4"
    echo "Recording ${seconds}s video-only -> $out"
    "$ADB" shell screenrecord --time-limit "$seconds" --bit-rate "$BIT_RATE" "$remote"
    "$ADB" pull "$remote" "$out" >/dev/null
    "$ADB" shell rm -f "$remote"
  fi

  "$ADB" shell settings put system show_touches 0

  [[ -s "$out" ]] || die "recording failed: $out is missing or empty"
  probe_summary "$out"
}

main "$@"
