#!/bin/zsh
# take_inner.sh <tag> <scene>: one scene of a take (take.sh runs it, inside the scene's measurement window when one is
# set): <tag>.ARM_<scene> readies the app (scene A: the editor as it opens; scene B: A's answers stay on screen),
# <tag>.ARMED_<scene> comes back, the window recorder starts on the app's window, 1 s after its first frame
# <tag>.GO_<scene> is written, the app plays the scene (typing, taps, Decide, the answers), and 3 s after
# <tag>.DONE_<scene> appears the recorder stops. Scene A also gets a screenshot while the ticket is being typed
# (<tag>_A_typing.png, 1.6 s after GO, before Decide). Bounded: the recorder has 10 s to start, the app 40 s from GO
# to DONE, the recorder stops itself at 55 s; the whole step stays under 60 s.
# Files (out/): <tag>_<scene>.mp4, .frames.jsonl, .rec.log; <tag>.GO_<scene> (the epoch it was written).
# Exit 0, or 3 (the recorder did not start), 4 (no DONE in time), 5 (the recorder failed).
set -u
zmodload zsh/datetime
HERE=${0:A:h}
D=${HERE:h}
OUT=$D/out
TAG=$1
S=$2
REC=$OUT/record_window
STEM=$OUT/${TAG}_$S
WID=$(/usr/bin/python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["window"]["id"])' $OUT/$TAG.READY) || exit 3
[[ -e $STEM.mp4 || -e $OUT/$TAG.GO_$S || -e $OUT/$TAG.ARM_$S ]] && { print "take_inner: files of $TAG scene $S exist"; exit 3 }

# the app readied for the scene first, so the video opens on the scene's first screen
print -r -- $EPOCHREALTIME > $OUT/$TAG.ARM_$S.tmp && mv $OUT/$TAG.ARM_$S.tmp $OUT/$TAG.ARM_$S
for i in {1..200}; do
  [[ -e $OUT/$TAG.ARMED_$S || -e $OUT/$TAG.FAILED ]] && break
  sleep 0.05
done
[[ -e $OUT/$TAG.ARMED_$S ]] || { print "take_inner: no $TAG.ARMED_$S"; exit 3 }
$REC $WID $STEM.mp4 $STEM.frames.jsonl $STEM.STOP_REC 30 55 > $STEM.rec.log 2>&1 &
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
sleep 1
print -r -- $EPOCHREALTIME > $OUT/$TAG.GO_$S.tmp && mv $OUT/$TAG.GO_$S.tmp $OUT/$TAG.GO_$S
if [[ $S == A ]]; then
  (sleep 1.6; screencapture -x -o -l $WID $OUT/${TAG}_A_typing.png) &
fi
rc=0
for i in {1..800}; do
  [[ -e $OUT/$TAG.DONE_$S || -e $OUT/$TAG.FAILED ]] && break
  sleep 0.05
done
[[ -e $OUT/$TAG.DONE_$S ]] || { print "take_inner: no $TAG.DONE_$S"; rc=4 }
sleep 3
touch $STEM.STOP_REC
wait $RP
rrc=$?
rm -f $STEM.STOP_REC
grep '^SUMMARY' $STEM.rec.log
(( rrc != 0 )) && { print "take_inner: the recorder exited $rrc"; cat $STEM.rec.log; (( rc == 0 )) && rc=5 }
exit $rc
