#!/bin/zsh
# take.sh <tag> [scenes M,D]: one recorded take of the app's autoplay on this Mac's display, then its checks.
#   zsh scripts/take.sh t1              (a new tag per take; a tag that has files in out/ is refused)
#   UI_ONLY=1 zsh scripts/take.sh u1    (the app's --ui-only: no model; for the windows, the settings and the recorder)
# The recording shows the main display at 1920 x 1080 with only these apps in it: the app (its window and a plain
# backdrop it lays over the desktop, behind every window), the TV app, Calculator, TextEdit (the two drafts), and the
# menu bar's system items (Control Center, the input menu, Spotlight, SystemUIServer). Every other app that is running
# when the recording starts is left out by ScreenCaptureKit's filter (scripts/record_display.swift): its windows, menu
# bar items and notifications are not in the picture, and nothing of it is closed or changed.
# Order:
#   1. the state before (load, swap, disk); the recorder built from record_display.swift when it is older;
#   2. the Mac's devices and the TV app's preferences saved (out/<tag>.mac_original.json, out/<tag>.tv_prefs.plist);
#      everything below is put back from them when this script ends, whatever happens;
#   3. the TV app's window placed for the recording, its sidebar folded (TV_FRAME; the sidebar shows the account's
#      name);
#   4. the scene's opening set on the Mac (app/actions.py scene: the TV app and Calculator running, both drafts open,
#      the volume 25, the music playing; the music's loop has this script as its owner, MAC_MUSIC_OWNER, so it plays
#      until the Mac is put back), then Calculator's and the drafts' windows placed (WINDOWS; System Events,
#      which needs the Accessibility permission of the terminal; without it they stay where macOS put them);
#   5. when MAC_WAIT_CMD is set, wait until the machine is quiet (the app's compile and warm-up are GPU work);
#   6. the app (app/mac_demo.py --scenes ... --stay --backdrop): compile, warm-up, READY; swap and memory logged every
#      5 s while it loads (swap that grows by SWAP_RISE_STOP_MB, default 5,120 MB, within 60 s stops the app and the
#      take, exit 7: another process needs the memory);
#   7. take_inner.sh <tag> <scenes> (a picture of the display through the recording's filter = <tag>_display.png,
#      with GO_WAIT_S > 0 a wait for <tag>.GO_REC (go) or <tag>.NOGO (exit 11), then recorder on, the scenes one after
#      the other, 2 s, recorder off), inside a measurement window named mac-demo-<tag> when MAC_HOLD_CMD is set;
#   8. the run JSON, then <tag>.QUIT (the app closes its window, puts back the Mac it found, and exits); then the
#      Mac's devices and the TV app's preferences put back from step 2;
#   9. (not with UI_ONLY) cpu_reference.py --from-run on the scenes' own requests, check_run.py, check_frames.py,
#      ffprobe of the video.
# Optional measurement lock, for a Mac that runs other GPU work (unset: nothing waits, nothing is written):
#   MAC_WAIT_CMD  a command prefix run as `$MAC_WAIT_CMD -- <command>`; it returns when the machine is quiet;
#   MAC_HOLD_CMD  a command prefix run as `$MAC_HOLD_CMD <label> -- <command>`; it holds the lock while <command> runs;
#   MAC_LOCK_FILE the lock file, logged here and recorded by the app at each Decide.
# Log: out/<tag>.take.log (+ <tag>.app.log, the app's own output). Exit 0 when every step and check passed, else the
# first failing step's code.
set -u
zmodload zsh/datetime
HERE=${0:A:h}
D=${HERE:h}
OUT=$D/out
TAG=$1
SCENES=${2:-M,D}
PY=$D/venv-demo/bin/python
QW=(${=MAC_WAIT_CMD:-})
QH=(${=MAC_HOLD_CMD:-})
LOCK=${MAC_LOCK_FILE:-}
UI=${UI_ONLY:-0}
GO_WAIT_S=${GO_WAIT_S:-0}
READY_LIMIT_S=${READY_LIMIT_S:-900}
SWAP_RISE_STOP_MB=${SWAP_RISE_STOP_MB:-5120}
# the TV app's window: x y width height of its frame from the screen's bottom left, in points (its smallest size is
# 980 x 580), lower right of the screen, behind the app's window where they meet
TV_FRAME=${TV_FRAME:-"920 20 980 580"}
# Calculator's and the drafts' windows: "<device> x y [w h]" from the screen's top left, in points, right of the app's
# window above the TV app's window
WINDOWS=${WINDOWS:-"calculator 1112 52;draft_1 1372 52 250 230;draft_2 1636 52 250 230"}
KEEP_BUNDLES=com.apple.TV,com.apple.calculator,com.apple.TextEdit,com.apple.controlcenter,com.apple.systemuiserver,com.apple.TextInputMenuAgent,com.apple.Spotlight
LOG=$OUT/$TAG.take.log
mkdir -p $OUT
setopt NULL_GLOB
existing=($OUT/$TAG.* $OUT/${TAG}_*)
unsetopt NULL_GLOB
(( ${#existing} )) && { print "tag $TAG has files already (${existing[1]:t} ...): pick a new tag"; exit 2 }

log() { print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a $LOG }
lock() { [[ -n $LOCK ]] && cut -c1-80 $LOCK 2>/dev/null }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}' }
tv_running() { pgrep -x TV > /dev/null }

log "take $TAG scenes $SCENES ui_only=$UI go_wait=${GO_WAIT_S}s"
log "uptime:$(uptime | sed 's/.*up/ up/')"
log "swap: $(sysctl -n vm.swapusage)"
log "disk: $(df -h $D | tail -1 | awk '{print $4 " free of " $2}')"
log "measurement lock: ${LOCK:-none} '$(lock)'"
if [[ ! -x $OUT/record_display || $HERE/record_display.swift -nt $OUT/record_display ]]; then
  xcrun swiftc -O -o $OUT/record_display $HERE/record_display.swift 2> $OUT/record_display.build.log \
    || { log "record_display did not build ($OUT/record_display.build.log)"; exit 1 }
  log "record_display built"
fi

# 2. what is put back at the end, whatever happens
$PY $D/app/actions.py read > $OUT/$TAG.mac_original.json 2>> $LOG || { log "could not read the Mac"; exit 1 }
defaults export com.apple.TV $OUT/$TAG.tv_prefs.plist || { log "could not save the TV app's preferences"; exit 1 }
log "saved: $(tr -d '\n ' < $OUT/$TAG.mac_original.json | cut -c1-300); TV preferences $(stat -f %z $OUT/$TAG.tv_prefs.plist) B"
APP=
put_back() {
  [[ -n $APP ]] && kill -0 $APP 2>/dev/null && { log "stopping the app (pid $APP)"; kill $APP; wait $APP }
  log "devices put back: $($PY $D/app/actions.py restore $OUT/$TAG.mac_original.json 2>&1 | cut -c1-500)"
  defaults import com.apple.TV $OUT/$TAG.tv_prefs.plist && log "TV preferences put back"
}
trap 'put_back' EXIT

# 3. the TV app's window for the recording
tv_running && { log "the TV app is running: quit it first"; exit 1 }
read -r tx ty tw th <<< "$TV_FRAME"
SCREEN=$(/usr/bin/python3 -c 'import json, re, subprocess as s; o = s.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True).stdout; m = re.search(r"UI Looks like: (\d+) x (\d+)", o); print(m.group(1), m.group(2))')
read -r sw sh <<< "$SCREEN"
defaults write com.apple.TV "NSWindow Frame VUIWindowFrameAutosaveName" "$tx $ty $tw $th 0 0 $sw $(( sh - 30 )) "
defaults write com.apple.TV "NSSplitView Subview Frames videosui.restorationId:rootSplitViewController" -array \
  "0.000000, 0.000000, 220.000000, $th.000000, YES, NO" "0.000000, 0.000000, $tw.000000, $th.000000, NO, NO"
log "TV window for the recording: frame $TV_FRAME on a $SCREEN screen, sidebar folded"

# 4. the scene's opening, and the windows placed
log "opening: $(MAC_MUSIC_OWNER=$$ $PY $D/app/actions.py scene 25 2>&1 | cut -c1-600)"
sleep 2
zsh $HERE/place_windows.sh "$WINDOWS" 2>&1 | while read -r l; do log "$l"; done
log "windows: $($PY -c 'import sys, json; sys.path.insert(0, "'$D/app'"); import actions as A; print(json.dumps(A.device_windows()))' 2>&1 | cut -c1-600)"
# between scene M and scene D (take_inner.sh runs it before it readies scene D): both drafts open again, placed
DRAFT_WINDOWS=$(print -l ${(s:;:)WINDOWS} | grep '^draft_' | tr '\n' ';')
BETWEEN_SCENES="$PY $D/app/actions.py set draft_1.open=true draft_2.open=true > /dev/null; zsh $HERE/place_windows.sh '$DRAFT_WINDOWS'"

if [[ $UI != 1 && ${#QW} -gt 0 ]]; then
  $QW -- /usr/bin/true 2>&1 | tee -a $LOG
fi
ARGS=(--scenes $SCENES --tag $TAG --trigger-dir $OUT --stay --backdrop)
[[ $UI == 1 ]] && ARGS+=(--ui-only)
PYTHONDONTWRITEBYTECODE=1 $PY $D/app/mac_demo.py $ARGS > $OUT/$TAG.app.log 2>&1 &
APP=$!
log "app pid $APP: mac_demo.py $ARGS"

t0=$EPOCHREALTIME
max_swap=0
next_log=0
typeset -a swaps   # swap used (MB) at each 5 s log line
while [[ ! -e $OUT/$TAG.READY ]]; do
  if [[ -e $OUT/$TAG.FAILED ]] || ! kill -0 $APP 2>/dev/null; then
    log "the app stopped before READY:"; tail -30 $OUT/$TAG.app.log | tee -a $LOG; exit 3
  fi
  el=$(( EPOCHREALTIME - t0 ))
  if (( el > next_log )); then
    s=$(swap_mb); (( s > max_swap )) && max_swap=$s
    swaps+=($s)
    log "loading $(printf %.0f $el) s: swap ${s} MB, lock '$(lock)', memory free level $(sysctl -n kern.memorystatus_level)%"
    if (( ${#swaps} > 12 && s - swaps[-13] >= SWAP_RISE_STOP_MB )); then
      log "STOP: swap grew by $(( s - swaps[-13] )) MB within 60 s while loading"; exit 7
    fi
    next_log=$(( next_log + 5 ))
  fi
  (( el > READY_LIMIT_S )) && { log "no READY within $READY_LIMIT_S s"; exit 3 }
  sleep 0.2
done
log "READY after $(printf %.1f $(( EPOCHREALTIME - t0 ))) s, max swap while loading ${max_swap} MB"
rc=0

log "scenes $SCENES: lock before '$(lock)'"
if [[ $UI == 1 || ${#QH} -eq 0 ]]; then
  KEEP_BUNDLES=$KEEP_BUNDLES GO_WAIT_S=$GO_WAIT_S BETWEEN_SCENES=$BETWEEN_SCENES zsh $HERE/take_inner.sh $TAG $SCENES 2>&1 \
    | tee -a $LOG
  irc=${pipestatus[1]}
else
  KEEP_BUNDLES=$KEEP_BUNDLES GO_WAIT_S=$GO_WAIT_S BETWEEN_SCENES=$BETWEEN_SCENES $QH mac-demo-$TAG -- \
    zsh $HERE/take_inner.sh $TAG $SCENES 2>&1 | tee -a $LOG
  irc=${pipestatus[1]}
fi
log "take_inner exit $irc"
(( irc == 11 )) && { log "NOGO: not recorded"; exit 11 }
(( irc != 0 )) && rc=5

for i in {1..300}; do
  [[ -e $OUT/$TAG.run.json ]] && break
  kill -0 $APP 2>/dev/null || break
  sleep 0.1
done
[[ -e $OUT/$TAG.run.json ]] || { log "no run JSON"; (( rc == 0 )) && rc=6 }
touch $OUT/$TAG.QUIT
wait $APP
log "app exit $?: $(grep -E '^RESTORED' $OUT/$TAG.app.log | cut -c1-400)"
APP=
put_back
trap - EXIT
log "uptime after:$(uptime | sed 's/.*up/ up/')"

if [[ $UI != 1 && -e $OUT/$TAG.run.json ]]; then
  PYTHONDONTWRITEBYTECODE=1 $PY -B $HERE/cpu_reference.py --from-run $OUT/$TAG.run.json 2>&1 | tail -3 | tee -a $LOG
  $PY -B $HERE/check_run.py $OUT/$TAG.run.json 2>&1 | tee -a $LOG
  crc=${pipestatus[1]}
  (( crc != 0 && rc == 0 )) && rc=8
fi
if [[ -e $OUT/${TAG}_take.mp4 ]]; then
  log "ffprobe ${TAG}_take.mp4: $(ffprobe -v error -select_streams v:0 -count_frames -show_entries \
    stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_frames:format=duration,nb_streams \
    -of compact=p=0 $OUT/${TAG}_take.mp4 | tr '\n' ' ')"
  log "audio streams in ${TAG}_take.mp4: $(ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 $OUT/${TAG}_take.mp4 | wc -l | tr -d ' ')"
  $PY -B $HERE/check_frames.py $OUT/$TAG.run.json 2>&1 | tee -a $LOG
  frc=${pipestatus[1]}
  (( frc != 0 && rc == 0 )) && rc=9
fi
log "take $TAG done: exit $rc"
exit $rc
