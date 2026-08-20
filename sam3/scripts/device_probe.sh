#!/usr/bin/env bash
# Run one .tflite on the Pixel through com.litertprobe (CompiledModel GPU / ML Drift).
#   device_probe.sh <graph.tflite> <input.bin> <out.bin> [cpu|gpu]
set -e
TFL="$1"; INP="$2"; OUTB="$3"; ACC="${4:-gpu}"
PKG=com.litertprobe
adb shell run-as $PKG sh -c 'rm -f files/probe_out_*.bin files/probe_accel.txt' 2>/dev/null || true
adb push "$TFL" /data/local/tmp/probe.tflite >/dev/null
adb push "$INP" /data/local/tmp/probe_input.bin >/dev/null
adb shell run-as $PKG cp /data/local/tmp/probe.tflite files/probe.tflite
adb shell run-as $PKG cp /data/local/tmp/probe_input.bin files/probe_input.bin
if [ "$ACC" = "cpu" ]; then adb shell "echo cpu > /data/local/tmp/pa.txt"; adb shell run-as $PKG cp /data/local/tmp/pa.txt files/probe_accel.txt; fi
adb shell am force-stop $PKG
adb logcat -c
adb shell am start -n $PKG/.MainActivity >/dev/null
for i in $(seq 1 120); do
  sleep 1
  if adb shell run-as $PKG sh -c "test -f files/probe_out_0.bin && echo y" 2>/dev/null | grep -q y; then sleep 1; break; fi
done
adb logcat -d 2>/dev/null | grep -iE "LRTPROBE|Replacing|not supported|GPU FAIL" | tail -6 | cut -c1-200
adb shell run-as $PKG cat files/probe_out_0.bin > "$OUTB"
adb shell run-as $PKG sh -c 'rm -f files/probe_accel.txt' 2>/dev/null || true
echo "pulled $OUTB ($(wc -c < "$OUTB") bytes)"
