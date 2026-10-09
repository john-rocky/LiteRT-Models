#!/bin/zsh
# take.sh <tag> [scenes A,B]: one recorded take of the app's autoplay on this Mac (window only), then its checks.
#   zsh scripts/take.sh t1              (a new tag per take; a tag that has files in out/ is refused)
#   UI_ONLY=1 zsh scripts/take.sh u1    (the app's --ui-only: no model; for the window and the recorder)
# Order:
#   1. the state before (load, swap, disk); the recorder built from record_window.swift when it is older;
#   2. when D1_WAIT_CMD is set, wait until the machine is quiet (the app's compile and warm-up are GPU work);
#   3. the app (app/d1_demo.py --scenes ... --stay): compile, warm-up, READY; swap and memory logged every 5 s while
#      it loads (swap above 10 GB stops the app and the take, exit 7);
#   4. a screenshot of the window (screencapture -x -o -l <window id>: the window's layer only) = <tag>_ready.png;
#   5. per scene: take_inner.sh <tag> <scene> (recorder on, the scene, 3 s, recorder off), inside a measurement
#      window named d1-demo-<tag>-<scene> when D1_HOLD_CMD is set; then <tag>_<scene>_done.png;
#   6. the run JSON, then <tag>.QUIT (the app closes its window and exits);
#   7. check_run.py (not with UI_ONLY), check_frames.py per scene, ffprobe of each video, sips of each screenshot.
# Optional measurement lock, for a Mac that runs other GPU work (unset: nothing waits, nothing is written):
#   D1_WAIT_CMD  a command prefix run as `$D1_WAIT_CMD -- <command>`; it returns when the machine is quiet;
#   D1_HOLD_CMD  a command prefix run as `$D1_HOLD_CMD <label> -- <command>`; it holds the lock while <command> runs;
#   D1_LOCK_FILE   the lock file, logged here and recorded by the app at each Decide.
# Log: out/<tag>.take.log (+ <tag>.app.log, the app's own output). Exit 0 when every step and check passed, else the
# first failing step's code.
set -u
zmodload zsh/datetime
HERE=${0:A:h}
D=${HERE:h}
OUT=$D/out
TAG=$1
SCENES=${2:-A,B}
PY=$D/venv-demo/bin/python
QW=(${=D1_WAIT_CMD:-})
QH=(${=D1_HOLD_CMD:-})
LOCK=${D1_LOCK_FILE:-}
UI=${UI_ONLY:-0}
READY_LIMIT_S=${READY_LIMIT_S:-900}
SWAP_STOP_MB=10240
LOG=$OUT/$TAG.take.log
mkdir -p $OUT
setopt NULL_GLOB
existing=($OUT/$TAG.* $OUT/${TAG}_*)
unsetopt NULL_GLOB
(( ${#existing} )) && { print "tag $TAG has files already (${existing[1]:t} ...): pick a new tag"; exit 2 }

log() { print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a $LOG }
lock() { [[ -n $LOCK ]] && cut -c1-80 $LOCK 2>/dev/null }
swap_mb() { sysctl -n vm.swapusage | awk '{for (i = 1; i <= NF; i++) if ($i == "used") {v = $(i + 2); sub("M", "", v); print int(v)}}' }
shot() {   # shot <window id> <png>: the window's layer, no shadow; prints its pixel size
  screencapture -x -o -l $1 $2 || { log "screencapture failed for $2"; return 1 }
  local wh=$(sips -g pixelWidth -g pixelHeight $2 | awk '/pixelWidth/ {w = $2} /pixelHeight/ {h = $2} END {print w "x" h}')
  log "screenshot ${2:t}: $wh"
  [[ $wh == 1080x1920 ]] || { log "screenshot ${2:t} is $wh, expected 1080x1920"; return 1 }
}

log "take $TAG scenes $SCENES ui_only=$UI"
log "uptime:$(uptime | sed 's/.*up/ up/')"
log "swap: $(sysctl -n vm.swapusage)"
log "disk: $(df -h $D | tail -1 | awk '{print $4 " free of " $2}')"
log "measurement lock: ${LOCK:-none} '$(lock)'"
if [[ ! -x $OUT/record_window || $HERE/record_window.swift -nt $OUT/record_window ]]; then
  xcrun swiftc -O -o $OUT/record_window $HERE/record_window.swift 2> $OUT/record_window.build.log \
    || { log "record_window did not build ($OUT/record_window.build.log)"; exit 1 }
  log "record_window built"
fi

if [[ $UI != 1 && ${#QW} -gt 0 ]]; then
  $QW -- /usr/bin/true 2>&1 | tee -a $LOG
fi
ARGS=(--scenes $SCENES --tag $TAG --trigger-dir $OUT --stay)
[[ $UI == 1 ]] && ARGS+=(--ui-only)
PYTHONDONTWRITEBYTECODE=1 $PY $D/app/d1_demo.py $ARGS > $OUT/$TAG.app.log 2>&1 &
APP=$!
log "app pid $APP: d1_demo.py $ARGS"

t0=$EPOCHREALTIME
max_swap=0
next_log=0
while [[ ! -e $OUT/$TAG.READY ]]; do
  if [[ -e $OUT/$TAG.FAILED ]] || ! kill -0 $APP 2>/dev/null; then
    log "the app stopped before READY:"; tail -30 $OUT/$TAG.app.log | tee -a $LOG; exit 3
  fi
  el=$(( EPOCHREALTIME - t0 ))
  if (( el > next_log )); then
    s=$(swap_mb); (( s > max_swap )) && max_swap=$s
    log "loading $(printf %.0f $el) s: swap ${s} MB, lock '$(lock)', memory free level $(sysctl -n kern.memorystatus_level)%"
    if (( s > SWAP_STOP_MB )); then
      log "STOP: swap ${s} MB > ${SWAP_STOP_MB} MB while loading"; kill $APP; wait $APP; exit 7
    fi
    next_log=$(( next_log + 5 ))
  fi
  (( el > READY_LIMIT_S )) && { log "no READY within $READY_LIMIT_S s"; kill $APP; exit 3 }
  sleep 0.2
done
WID=$(/usr/bin/python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["window"]["id"])' $OUT/$TAG.READY)
log "READY after $(printf %.1f $(( EPOCHREALTIME - t0 ))) s, window $WID, max swap while loading ${max_swap} MB"
sleep 0.5
rc=0
shot $WID $OUT/${TAG}_ready.png || rc=4

for S in ${(s:,:)SCENES}; do
  log "scene $S: lock before '$(lock)'"
  if [[ $UI == 1 || ${#QH} -eq 0 ]]; then
    zsh $HERE/take_inner.sh $TAG $S 2>&1 | tee -a $LOG
    irc=${pipestatus[1]}
  else
    $QH d1-demo-$TAG-$S -- zsh $HERE/take_inner.sh $TAG $S 2>&1 | tee -a $LOG
    irc=${pipestatus[1]}
  fi
  log "scene $S: take_inner exit $irc"
  if (( irc != 0 )); then
    (( rc == 0 )) && rc=5
    if [[ ! -e $OUT/$TAG.GO_$S ]]; then   # the app still waits for this scene: nothing more can run
      log "scene $S never started (no GO): stopping the app"; kill $APP; wait $APP; exit $rc
    fi
  fi
  shot $WID $OUT/${TAG}_${S}_done.png || { (( rc == 0 )) && rc=4 }
done

for i in {1..300}; do
  [[ -e $OUT/$TAG.run.json ]] && break
  kill -0 $APP 2>/dev/null || break
  sleep 0.1
done
[[ -e $OUT/$TAG.run.json ]] || { log "no run JSON"; (( rc == 0 )) && rc=6 }
touch $OUT/$TAG.QUIT
wait $APP
log "app exit $?"
log "uptime after:$(uptime | sed 's/.*up/ up/')"

if [[ $UI != 1 && -e $OUT/$TAG.run.json ]]; then
  $PY -B $HERE/check_run.py $OUT/$TAG.run.json 2>&1 | tee -a $LOG
  crc=${pipestatus[1]}
  (( crc != 0 && rc == 0 )) && rc=8
fi
for S in ${(s:,:)SCENES}; do
  [[ -e $OUT/${TAG}_$S.mp4 ]] || continue
  log "ffprobe ${TAG}_$S.mp4: $(ffprobe -v error -select_streams v:0 -count_frames -show_entries \
    stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_frames:format=duration,nb_streams \
    -of compact=p=0 $OUT/${TAG}_$S.mp4 | tr '\n' ' ')"
  log "audio streams in ${TAG}_$S.mp4: $(ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 $OUT/${TAG}_$S.mp4 | wc -l | tr -d ' ')"
  $PY -B $HERE/check_frames.py $OUT/$TAG.run.json $S 2>&1 | tee -a $LOG
  frc=${pipestatus[1]}
  (( frc != 0 && rc == 0 )) && rc=9
done
log "take $TAG done: exit $rc"
exit $rc
