#!/bin/zsh
# One-shot device verification for the demo: launch the app, wait for the run
# to finish, pull the fresh log. Requires the phone plugged in and unlocked.
#
#   ./verify-device-run.sh            # launch + wait + pull
#   ./verify-device-run.sh --pull     # skip launch (app was started by hand)
#
# The launch step tolerates the CoreDevice hang seen on 2026-08-18: if launch
# does not return in 20s, tap the app icon on the phone instead — the poll
# below keeps working either way.
set -u
DEVICE=A6F3E849-1947-5202-9AD1-9C881CA58EEF
BUNDLE=com.lfmtools.app
DEST=${TMPDIR:-/tmp}

list_logs() {
  xcrun devicectl device info files --device "$DEVICE" \
    --domain-type appDataContainer --domain-identifier "$BUNDLE" \
    --subdirectory Documents 2>/dev/null | grep -oE "run-[0-9]+\.log" | sort
}

baseline=$(list_logs | tail -1)
echo "baseline log: ${baseline:-none}"

if [[ "${1:-}" != "--pull" ]]; then
  # --autorun is required: without it the app opens the chat screen and the
  # stage demo never starts (and no run log is ever written).
  echo "launching $BUNDLE --autorun (20s timeout; on hang, tap the icon on the phone)..."
  timeout 20 xcrun devicectl device process launch --terminate-existing \
    --device "$DEVICE" "$BUNDLE" --autorun >/dev/null 2>&1 || echo "launch did not return - tap the icon if the app is not on screen"
fi

echo "waiting for a new run log (demo takes ~2-3 min)..."
for i in {1..40}; do
  newest=$(list_logs | tail -1)
  if [[ -n "$newest" && "$newest" != "$baseline" ]]; then
    # Wait for the run to finish: size stable across two polls.
    sleep 20
    xcrun devicectl device copy from --device "$DEVICE" \
      --domain-type appDataContainer --domain-identifier "$BUNDLE" \
      --source "Documents/$newest" --destination "$DEST/$newest" >/dev/null 2>&1
    echo "pulled: $DEST/$newest"
    cat "$DEST/$newest"
    exit 0
  fi
  sleep 10
done
echo "no new run log appeared - did the app start?"
exit 1
