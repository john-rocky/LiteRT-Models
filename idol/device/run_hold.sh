#!/usr/bin/env bash
# One device hold of the Idol lane: wait in the phone's queue, then install / run AutoRun plans / pull results.
# The keeper is this script: queue_cli.py `wait` takes the hold under this script's pid, and every step checks that
# the hold file still names it before any adb call. Never write the hold file by hand.
#
#   DEVICE=s26|p11p STEPS="apk,models,fixtures,plan:e2e_s26,pull:e2e_s26" device/run_hold.sh
#
# Steps (comma list, run in order):
#   apk             adb install -r the debug APK
#   models          scripts/install_to_device.sh (graphs + side files, one file at a time)
#   fixtures        FIXTURES=1 MODELS=0 scripts/install_to_device.sh (device/fx/<tag>/ and plans/*.json)
#   plan:<name>     run files/plans/<name>.json into files/runs/<name>, sampling memory every 0.5 s; waits for DONE.json
#   pull:<name>     copy files/runs/<name> to device/results/<device>/<name>
#   shot:<name>     screenshot to device/results/<device>/<name>.png (only while our activity is in front)
#   stop            force-stop com.idol
#   state           log the screen and keyguard state and the activity in front
#   wait:<s>        keep the hold for GUI work done by hand, until device/results/<device>/CONTINUE exists or s seconds
# The sampler's cpu_ticks (utime + stime, 1/100 s) and threads show whether the CPU graph (BiRefNet) runs on several
# threads: the C-API runner's CPU options do not reach LiteRT 2.3.0 (r3_rt_runner d815-1), the app's must.
# DRY=1: no queue, no hold; adb on PATH must be the fake (device/fake_adb/adb), checked before anything runs.
set -euo pipefail

K=$(cd "$(dirname "$0")/.." && pwd)
CA=$HOME/code/litertlm-convert/community_accel_work
PKG=com.idol
case "${DEVICE:-s26}" in
    s26) SERIAL=RFGL80R6A6H; HOLD=$CA/s2_npu_sweep/.device_hold ;;
    p11p) SERIAL=67170DLKX00C9J; HOLD=$CA/s2_npu_sweep/.device_hold.pixel11pro ;;
    *) echo "DEVICE must be s26 or p11p" >&2; exit 2 ;;
esac
export ANDROID_SERIAL=$SERIAL
NAME=${NAME:-idol-${DEVICE:-s26}}
STEPS=${STEPS:?STEPS is required}
OUT=$K/device/results/${DEVICE:-s26}
APK=$K/app/build/outputs/apk/debug/app-debug.apk
mkdir -p "$OUT"
log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/hold.log"; }

if [[ "${DRY:-0}" == 1 ]]; then
    [[ "$(command -v adb)" == "$K/device/fake_adb/adb" ]] || { echo "DRY=1 but adb is $(command -v adb)" >&2; exit 2; }
    check_hold() { :; }
else
    log "queue: waiting as $NAME (pid $$) on $HOLD"
    python3 "$CA/queue_cli.py" enqueue "$HOLD" "$NAME" idol-lane 30
    python3 "$CA/queue_cli.py" wait "$HOLD" "$NAME" $$ --timeout "${QUEUE_TIMEOUT:-14400}"
    trap 'python3 "$CA/hold_cli.py" release "$HOLD" $$ >/dev/null 2>&1 || true; log "hold released"' EXIT
    check_hold() {
        python3 "$CA/hold_cli.py" read "$HOLD" | grep -qE "'pid': $$[,}]" || { log "hold file does not name pid $$: stop"; exit 3; }
    }
fi
check_hold
log "hold: $(python3 "$CA/hold_cli.py" read "$HOLD" 2>/dev/null | head -1)"
adb get-state >/dev/null
# Every step ends inside the hold: 28 of its 30 minutes by default. A step that does not fit in what is left ends the
# hold there (the rest is carried to the next hold); a plan still running at the budget gets STOP, then force-stop.
HOLD_T0=$SECONDS
HOLD_BUDGET_S=${HOLD_BUDGET_S:-1680}
left() { echo $(( HOLD_BUDGET_S - (SECONDS - HOLD_T0) )); }

app_pid() { adb shell pidof "$PKG" 2>/dev/null | tr -d '\r' || true; }

sample_memory() {  # $1 = csv; runs until the file $1.stop exists
    local csv=$1
    echo "t,app_pid,vmhwm_kb,vmrss_kb,kgsl_page_alloc,mem_available_kb,cpu_ticks,threads" > "$csv"
    while [[ ! -e "$csv.stop" ]]; do
        local pid; pid=$(app_pid)
        local line
        line=$(adb shell "cat /proc/${pid:-0}/status 2>/dev/null | grep -E '^Vm(HWM|RSS)' | tr -s ' ' | cut -d' ' -f2 | tr '\n' ','; \
               cat /sys/class/kgsl/kgsl/page_alloc 2>/dev/null; echo -n ,; grep MemAvailable /proc/meminfo | tr -s ' ' | cut -d' ' -f2; \
               echo -n ,; awk '{print \$14 + \$15}' /proc/${pid:-0}/stat 2>/dev/null; echo -n ,; ls /proc/${pid:-0}/task 2>/dev/null | wc -l" \
               2>/dev/null | tr -d '\r\n' || true)
        echo "$(date +%s.%N | cut -c1-14),${pid:-},$line" >> "$csv"
        sleep 0.5
    done
}

run_plan() {  # $1 = plan name
    local plan=$1 csv=$OUT/$1.mem.csv
    [[ -n "$plan" ]] || { log "empty plan name"; exit 2; }
    adb shell run-as "$PKG" rm -r -f "files/runs/$plan"
    adb logcat -c
    rm -f "$csv.stop"
    sample_memory "$csv" &
    local sampler=$!
    log "plan $plan: start"
    adb shell am start -W -n "$PKG/.MainActivity" --es autorun "plans/$plan.json" --es out "runs/$plan" >/dev/null
    local t0=$SECONDS limit=${PLAN_TIMEOUT:-1500}
    local budget=$(( $(left) - 90 ))   # room for STOP (20 s) and the pull after it
    (( budget < limit )) && limit=$budget
    while ! adb shell run-as "$PKG" test -f "files/runs/$plan/DONE.json" 2>/dev/null; do
        check_hold
        if [[ -z "$(app_pid)" ]]; then log "plan $plan: app process gone"; break; fi
        if (( SECONDS - t0 > limit )); then
            log "plan $plan: limit $limit s reached (hold budget or PLAN_TIMEOUT): STOP, then force-stop"
            adb shell run-as "$PKG" touch "files/runs/$plan/STOP" || true
            local t1=$SECONDS
            while (( SECONDS - t1 < 20 )) && ! adb shell run-as "$PKG" test -f "files/runs/$plan/DONE.json" 2>/dev/null; do
                sleep 2
            done
            adb shell am force-stop "$PKG"
            break
        fi
        sleep 5
    done
    touch "$csv.stop"; wait "$sampler" || true
    adb logcat -d -v time -s IdolAutoRun:V AndroidRuntime:E libc:F DEBUG:F lowmemorykiller:I lmkd:I > "$OUT/$plan.logcat.txt" || true
    log "plan $plan: end after $((SECONDS - t0)) s; $(grep -c ROW "$OUT/$plan.logcat.txt" || true) ROW lines"
}

need_s() {  # the least time a step needs to finish inside the hold
    case "$1" in
        models) echo "${MODELS_NEED_S:-600}" ;;
        fixtures) echo 150 ;;
        plan:*) echo "${PLAN_NEED_S:-240}" ;;
        pull:*) echo 20 ;;
        *) echo 30 ;;
    esac
}

IFS=',' read -r -a steps <<< "$STEPS"
for i in "${!steps[@]}"; do
    step=${steps[$i]}
    check_hold
    if (( $(left) < $(need_s "$step") )); then
        log "carry: $(left) s left in the hold budget, not enough for $step; not run: ${steps[*]:$i}"
        break
    fi
    case "$step" in
        apk) log "apk: install"; adb install -r "$APK" | tail -1 ;;
        models) log "models: install"; ANDROID_SERIAL=$SERIAL "$K/scripts/install_to_device.sh" ;;
        fixtures) log "fixtures: install"; FIXTURES=1 MODELS=0 ANDROID_SERIAL=$SERIAL "$K/scripts/install_to_device.sh" ;;
        plan:*) run_plan "${step#plan:}" ;;
        pull:*)
            name=${step#pull:}; mkdir -p "$OUT/$name"
            adb exec-out run-as "$PKG" tar -cf - "files/runs/$name" | tar -xf - -C "$OUT/$name" --strip-components 3
            log "pull $name: $(ls "$OUT/$name" | wc -l | tr -d ' ') files" ;;
        shot:*)
            if adb shell dumpsys activity activities | grep -m1 topResumedActivity | grep -q "$PKG"; then
                adb exec-out screencap -p > "$OUT/${step#shot:}.png"; log "shot ${step#shot:}"
            else
                log "shot ${step#shot:}: $PKG is not in front, skipped"
            fi ;;
        stop) adb shell am force-stop "$PKG"; log "force-stop $PKG" ;;
        state)  # screen and lock state (GUI work and recordings need the screen on and no keyguard)
            log "state: $(adb shell dumpsys power | grep -m1 -oE 'mWakefulness=[A-Za-z]+' | tr -d '\r');" \
                "$(adb shell dumpsys window | grep -m1 -oE 'isKeyguardShowing=[a-z]+|mShowingLockscreen=[a-z]+' | tr -d '\r');" \
                "top $(adb shell dumpsys activity activities | grep -m1 topResumedActivity | grep -oE '[a-z0-9.]+/[A-Za-z0-9.]+' | head -1 | tr -d '\r')" ;;
        wait:*)  # hold the phone for GUI work done by hand: until $OUT/CONTINUE exists or the seconds pass
            secs=${step#wait:}; rm -f "$OUT/CONTINUE"; log "wait: up to $secs s for $OUT/CONTINUE"
            t0=$SECONDS
            while [[ ! -e "$OUT/CONTINUE" ]] && (( SECONDS - t0 < secs && $(left) > 120 )); do check_hold; sleep 3; done
            log "wait: over after $((SECONDS - t0)) s" ;;
        *) log "unknown step $step"; exit 2 ;;
    esac
done
log "end of the hold's steps (asked: $STEPS)"
