#!/bin/bash
# keeper.sh: the hold keeper of one chain run: the named waiting line (queue_cli.py) and an owned hold (device_hold.py
# through hold_cli.py). chain.sh starts it from a copy in the round dir (one keeper per hold), so an edit of this file
# never reaches a running keeper.
#   usage: keeper.sh <signal dir> <hold file> <script name>        env: CA (the hold tools' dir), SERIAL
# 1. `queue_cli.py wait <hold file> <script name> <this shell's pid>`: in the line behind the entries whose keepers are
#    waiting, acquires the owned hold with THIS shell's pid when the hold is free; never forces, never clears another
#    session's hold (a dead holder's stale hold is cleared by device_hold.acquire itself);
# 2. writes <signal dir>/HOLD_ACQUIRED ("<serial>", "pid <pid> since <time>");
# 3. keeps the hold until <signal dir>/RELEASE_HOLD appears or HOLD_LIMIT seconds (default 1,800 = the 30-minute window)
#    after acquiring, then `hold_cli.py release <hold file> <pid>` (removes the hold only when it is this pid's) and writes
#    <signal dir>/HOLD_RELEASED. Gives up waiting after WAIT_LIMIT seconds (default 4 h; queue_cli exit 3 -> GAVE_UP).
# 4. At HOLD_LIMIT (no RELEASE_HOLD: the chain was killed, hung or stopped, so its own cleanup did not run), before the
#    release: <signal dir>/HOLD_TIMED_OUT first (lib.sh's adb calls stop when they see it), then <signal dir>/ON_RELEASE.sh
#    when the chain wrote one (its own processes killed, its app force-stopped, its stage removed), for at most
#    ON_RELEASE_MAX seconds (default 120; then killed), its output in <signal dir>/on_release.txt and here. Without
#    ON_RELEASE.sh the release is as before. After the release the time-out line goes before the first line of
#    <signal dir>/keeper.log (the chains' log of this keeper) and into HOLD_TIMED_OUT, so a reader sees it first
#    (2026-10-28: a stopped tab's hold went back at its 40 minutes with its app on the GPU, 3.2 GB, in front of the phone,
#    and the next three lanes' GPU legs could not start).
set -u
SIG=${1:?signal dir}; LOCK=${2:?hold file}; NAME=${3:?script name}
CA=${CA:?CA = the directory of queue_cli.py and hold_cli.py}
SERIAL=${SERIAL:?SERIAL = the phone}
HOLD_LIMIT=${HOLD_LIMIT:-1800}
WAIT_LIMIT=${WAIT_LIMIT:-14400}
ON_RELEASE_MAX=${ON_RELEASE_MAX:-120}
mkdir -p "$SIG"; rm -f "$SIG/HOLD_ACQUIRED" "$SIG/RELEASE_HOLD" "$SIG/HOLD_RELEASED" "$SIG/GAVE_UP" "$SIG/HOLD_TIMED_OUT"
echo "keeper pid $$ started $(date '+%F %T') hold=$LOCK name=$NAME hold_limit=${HOLD_LIMIT}s wait_limit=${WAIT_LIMIT}s on_release_max=${ON_RELEASE_MAX}s"
echo "line at start:"; python3 "$CA/queue_cli.py" list "$LOCK" | sed 's/^/  /'
if ! python3 "$CA/queue_cli.py" wait "$LOCK" "$NAME" $$ --timeout "$WAIT_LIMIT" --poll 2; then
  echo "gave up waiting $(date '+%F %T')"; date '+%F %T' > "$SIG/GAVE_UP"; exit 3
fi
ACQ=$(date +%s)
END=$(( ACQ + HOLD_LIMIT ))
{ echo "$SERIAL"; echo "pid $$ since $(date -r "$ACQ" '+%Y-%m-%d %H:%M:%S')"; } > "$SIG/HOLD_ACQUIRED"
echo "HOLD ACQUIRED by pid $$ at $(date -r "$ACQ" '+%F %T'); hold file: $(cat "$LOCK")"
echo "auto release at $(date -r $END '+%T')"

tree_pids() {  # pid -> that pid and every pid under it (an ON_RELEASE.sh's adb clients run inside $(...) subshells: a
  # kill of its direct children alone leaves them running)
  local c
  echo "$1"
  for c in $(pgrep -P "$1" 2>/dev/null); do tree_pids "$c"; done
}

on_release() {  # at the limit, before the release: the chain's ON_RELEASE.sh for at most ON_RELEASE_MAX s -> OR_NOTE
  local f=$SIG/ON_RELEASE.sh hp t0 rc pids
  if [ ! -f "$f" ]; then OR_NOTE="none in the signal dir (released as before)"; echo "ON_RELEASE.sh: $OR_NOTE"; return 0; fi
  echo "ON_RELEASE.sh (sha256 $(shasum -a 256 "$f" | cut -c1-16)) start $(date '+%F %T'), at most ${ON_RELEASE_MAX}s"
  t0=$(date +%s)
  bash "$f" > "$SIG/on_release.txt" 2>&1 < /dev/null &
  hp=$!
  while kill -0 "$hp" 2>/dev/null && [ $(( $(date +%s) - t0 )) -lt "$ON_RELEASE_MAX" ]; do sleep 1; done
  if kill -0 "$hp" 2>/dev/null; then
    # the whole tree stopped first (no new child after the list), listed again, then killed
    kill -STOP $(tree_pids "$hp") 2>/dev/null; pids=$(tree_pids "$hp")
    kill -STOP $pids 2>/dev/null; kill -9 $pids 2>/dev/null; wait "$hp" 2>/dev/null
    OR_NOTE="killed after ${ON_RELEASE_MAX}s"
  else
    wait "$hp"; rc=$?
    OR_NOTE="exit $rc after $(( $(date +%s) - t0 ))s"
  fi
  echo "ON_RELEASE.sh $OR_NOTE $(date '+%F %T'):"; sed 's/^/  /' "$SIG/on_release.txt"
}

first_line() {  # line file: the line goes before the file's first line, rewritten in place (this keeper's stdout may be
  # the same file: it is the keeper's last write, so neither an appending nor a positioned stdout writes after it)
  [ -f "$2" ] || return 0
  { printf '%s\n' "$1"; cat "$2"; } > "$2.first.$$" && cat "$2.first.$$" > "$2"
  rm -f "$2.first.$$"
}

TIMED_OUT=0
while [ ! -f "$SIG/RELEASE_HOLD" ]; do
  if [ "$(date +%s)" -ge "$END" ]; then TIMED_OUT=1; echo "hold limit reached $(date '+%F %T')"; break; fi
  sleep 2
done
if [ "$TIMED_OUT" = 1 ]; then
  echo "HOLD TIMED OUT $(date '+%F %T'): HOLD_LIMIT ${HOLD_LIMIT}s, no RELEASE_HOLD from the chain; keeper pid $$ ($NAME)" > "$SIG/HOLD_TIMED_OUT"
  on_release
fi
python3 "$CA/hold_cli.py" release "$LOCK" $$
rm -f "$SIG/HOLD_ACQUIRED"
date '+%F %T' > "$SIG/HOLD_RELEASED"
echo "HOLD RELEASED at $(date '+%F %T'); hold file now: $(cat "$LOCK" 2>/dev/null || echo '<absent>')"
if [ "$TIMED_OUT" = 1 ]; then
  LINE="HOLD TIMED OUT: keeper pid $$ ($NAME, held since $(date -r "$ACQ" '+%F %T')) released the hold at its limit (HOLD_LIMIT ${HOLD_LIMIT}s) at $(cat "$SIG/HOLD_RELEASED"), not on the chain's RELEASE_HOLD; ON_RELEASE.sh: $OR_NOTE"
  echo "$LINE" > "$SIG/HOLD_TIMED_OUT"
  first_line "$LINE" "$SIG/keeper.log"
fi
