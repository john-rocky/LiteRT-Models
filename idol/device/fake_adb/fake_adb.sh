#!/usr/bin/env bash
# Dry-run stand-in for adb: records every call, answers the few queries run_hold.sh makes, never touches a phone.
echo "$(date +%H:%M:%S) adb $*" >> "${FAKE_ADB_LOG:-/tmp/fake_adb_calls.log}"
case "$*" in
    "get-state") echo device ;;
    *"pidof"*) echo 4242 ;;
    *"DONE.json"*) [[ -n "${FAKE_NO_DONE:-}" ]] && exit 1; exit 0 ;;
    *"dumpsys activity activities"*) echo "  topResumedActivity=ActivityRecord{1 u0 com.idol/.MainActivity t1}" ;;
    *) : ;;
esac
exit 0
