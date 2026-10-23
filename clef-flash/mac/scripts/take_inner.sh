#!/bin/zsh
# take_inner.sh <tag> <scenes>: the recorded part of a take (take.sh runs it, inside a measurement window when one is
# set): the first scene readied (<tag>.ARM_<s> -> <tag>.ARMED_<s>), a picture of the display through the recording's
# filter (<tag>_display.png); with GO_WAIT_S > 0 a wait for <tag>.GO_REC (go) or <tag>.NOGO (exit 11); then the
# display recorder starts, and 0.5 s after it has started the scenes play one after the other in the same recording:
# per scene <tag>.ARM_<s> -> <tag>.ARMED_<s> (the first was readied before the recorder; before each later one the
# command BETWEEN_SCENES runs, take.sh: both drafts opened again and placed), <tag>.GO_<s>, the scene,
# <tag>.DONE_<s>; 0.8 s after the last DONE the recorder stops.
# The recorder keeps the app (by its pid) and the apps of KEEP_BUNDLES (take.sh sets them); every other app running
# at its start is left out.
# Bounded: the recorder has 10 s to start, each scene 60 s from GO to DONE, the recorder stops itself at 55 s.
# Files (out/): <tag>_take.mp4, .frames.jsonl, .rec.log; <tag>_display.png, .shot.log; <tag>.GO_<s> (the epoch it
# was written).
# Exit 0, or 3 (the recorder did not start), 4 (no DONE in time), 5 (the recorder failed), 6 (no picture), 11 (no go).
set -u
zmodload zsh/datetime
HERE=${0:A:h}
D=${HERE:h}
OUT=$D/out
TAG=$1
SCENES=(${(s:,:)2})
REC=$OUT/record_display
STEM=$OUT/${TAG}_take
KEEP=$(/usr/bin/python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["pid"])' $OUT/$TAG.READY) || exit 3
BUNDLES=${KEEP_BUNDLES:-com.apple.TV}
GO_WAIT_S=${GO_WAIT_S:-0}
[[ -e $STEM.mp4 ]] && { print "take_inner: files of $TAG exist"; exit 3 }

arm() {   # arm <scene>: the app readies the scene's first screen
  print -r -- $EPOCHREALTIME > $OUT/$TAG.ARM_$1.tmp && mv $OUT/$TAG.ARM_$1.tmp $OUT/$TAG.ARM_$1
  for i in {1..300}; do
    [[ -e $OUT/$TAG.ARMED_$1 || -e $OUT/$TAG.FAILED ]] && break
    sleep 0.05
  done
  [[ -e $OUT/$TAG.ARMED_$1 ]]
}

arm ${SCENES[1]} || { print "take_inner: no $TAG.ARMED_${SCENES[1]}"; exit 3 }
sleep 1
$REC --keep $KEEP --keep-bundles $BUNDLES --shot $OUT/${TAG}_display.png > $OUT/${TAG}_display.shot.log 2>&1 \
  || { print "take_inner: no picture of the display"; cat $OUT/${TAG}_display.shot.log; exit 6 }
print "take_inner: display picture $OUT/${TAG}_display.png $(grep -o '"kept":\[[^]]*\]' $OUT/${TAG}_display.shot.log)"
if (( GO_WAIT_S > 0 )); then
  print "take_inner: WAITING FOR GO ($TAG.GO_REC or $TAG.NOGO, up to $GO_WAIT_S s)"
  t0=$EPOCHREALTIME
  while [[ ! -e $OUT/$TAG.GO_REC && ! -e $OUT/$TAG.NOGO ]]; do
    (( EPOCHREALTIME - t0 > GO_WAIT_S )) && break
    sleep 0.5
  done
  [[ -e $OUT/$TAG.GO_REC ]] || { print "take_inner: no go ($([[ -e $OUT/$TAG.NOGO ]] && print nogo || print timeout))"; exit 11 }
  print "take_inner: go after $(printf %.0f $(( EPOCHREALTIME - t0 ))) s"
fi
$REC --keep $KEEP --keep-bundles $BUNDLES $STEM.mp4 $STEM.frames.jsonl $STEM.STOP_REC 30 55 > $STEM.rec.log 2>&1 &
RP=$!
for i in {1..200}; do
  grep -q '^RECORDING' $STEM.rec.log && break
  kill -0 $RP 2>/dev/null || break
  sleep 0.05
done
if ! grep -q '^RECORDING' $STEM.rec.log; then
  print "take_inner: the recorder did not start"; cat $STEM.rec.log
  kill $RP 2>/dev/null; wait $RP 2>/dev/null
  exit 3
fi
sleep 0.5
rc=0
first=1
for S in $SCENES; do
  if (( ! first )); then
    if [[ -n ${BETWEEN_SCENES:-} ]]; then   # take.sh: both drafts open again, placed, before the scene is readied
      zsh -c "$BETWEEN_SCENES" | sed 's/^/take_inner: between scenes: /'
    fi
    arm $S || { print "take_inner: no $TAG.ARMED_$S"; rc=4; break }
  fi
  first=0
  print -r -- $EPOCHREALTIME > $OUT/$TAG.GO_$S.tmp && mv $OUT/$TAG.GO_$S.tmp $OUT/$TAG.GO_$S
  for i in {1..1200}; do
    [[ -e $OUT/$TAG.DONE_$S || -e $OUT/$TAG.FAILED ]] && break
    sleep 0.05
  done
  [[ -e $OUT/$TAG.DONE_$S ]] || { print "take_inner: no $TAG.DONE_$S"; rc=4; break }
done
sleep 0.8
touch $STEM.STOP_REC
wait $RP
rrc=$?
rm -f $STEM.STOP_REC
grep '^SUMMARY' $STEM.rec.log
(( rrc != 0 )) && { print "take_inner: the recorder exited $rrc"; cat $STEM.rec.log; (( rc == 0 )) && rc=5 }
exit $rc
