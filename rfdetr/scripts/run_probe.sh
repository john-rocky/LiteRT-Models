#!/bin/bash
# Drive com.litertprobe on the connected device for one graph.
#   usage: run_probe.sh <tflite> <input.bin> <num_outputs> <out_prefix>
set -e
TFL="$1"; INP="$2"; NOUT="$3"; PFX="$4"
D=~/Downloads/meeting/rfdetr-work
cd "$D"
adb shell run-as com.litertprobe sh -c 'rm -f files/probe_out_*.bin' 2>/dev/null || true
adb push "$TFL" /data/local/tmp/probe.tflite >/dev/null
adb push "$INP" /data/local/tmp/probe_input.bin >/dev/null
adb shell run-as com.litertprobe cp /data/local/tmp/probe.tflite    files/probe.tflite
adb shell run-as com.litertprobe cp /data/local/tmp/probe_input.bin files/probe_input.bin
adb shell am force-stop com.litertprobe
adb logcat -c
adb shell am start -n com.litertprobe/.MainActivity >/dev/null
# wait for the last output file to appear (bounded)
for i in $(seq 1 40); do
  sleep 1
  if adb shell run-as com.litertprobe sh -c "test -f files/probe_out_$((NOUT-1)).bin && echo y" 2>/dev/null | grep -q y; then
    sleep 1; break
  fi
done
echo "===== LOGCAT ($PFX) ====="
adb logcat -d 2>/dev/null | grep -iE "LRTPROBE|LITERT_CL|Replacing|partition|delegate|InferenceCalculator|GPU FAIL|TfLiteGpu" | tail -40
echo "===== OUTPUTS ($PFX) ====="
for j in $(seq 0 $((NOUT-1))); do
  adb shell run-as com.litertprobe cat files/probe_out_$j.bin > "${PFX}_out_$j.bin" 2>/dev/null
  echo "pulled ${PFX}_out_$j.bin ($(wc -c < ${PFX}_out_$j.bin) bytes)"
done
