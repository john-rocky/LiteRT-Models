#!/bin/zsh
# place_windows.sh "<device> x y [w h];...": move (and size) the window of each device's process (app/actions.py: the
# TV app, Calculator, a draft's TextEdit) to x y from the screen's top left, in points, through System Events (it needs
# the Accessibility permission of the terminal; without it the windows stay where macOS put them). One System Events
# script places them all; it is tried again every 0.1 s, for up to 3 s, while a process that has just started has no
# window yet. For a recording (take.sh); prints one line per window.
set -u
HERE=${0:A:h}
D=${HERE:h}
PY=$D/venv-demo/bin/python
specs=(${(s:;:)1})
# each device's process, one line per spec (empty when it has none)
pids=("${(@f)$($PY -c 'import sys; sys.path.insert(0, sys.argv[1]); import actions as A
for d in sys.argv[2:]:
    p = A.app_pids(d) if d in A.APPS else [A.draft_pids().get(d)]
    print(p[0] if p and p[0] else "")' $D/app ${specs%% *})}")
script=(-e 'tell application "System Events"')
lines=()
for i in {1..${#specs}}; do
  parts=(${=specs[i]})
  pid=${pids[i]:-}
  if [[ -z $pid ]]; then
    lines+=("window ${parts[1]}: no process")
    continue
  fi
  # position and size are set one at a time (System Events refuses a TextEdit window both as one list)
  script+=(-e "tell (first process whose unix id is $pid)" -e "set position of window 1 to {${parts[2]}, ${parts[3]}}")
  (( ${#parts} >= 5 )) && script+=(-e "set size of window 1 to {${parts[4]}, ${parts[5]}}")
  script+=(-e "end tell")
  lines+=("window ${parts[1]} (pid $pid): ${parts[2,-1]}")
done
script+=(-e "end tell")
r=
for i in {1..30}; do
  r=$(osascript $script 2>&1) && { r=; break }
  sleep 0.1
done
for l in $lines; do
  print -r -- "$l${r:+ (not placed: $r)}"
done
