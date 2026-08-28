#!/bin/zsh
# Phase 2: thermally-gated measurement. Each arm waits for Thermal Status 0 (NONE),
# max 15 min, then runs 50 iters in a fresh process. Interleaved ctrl/fold + replicates.
export PATH="$HOME/Library/Android/sdk/platform-tools:$PATH"
SP=/private/tmp/claude-501/-Users-majimadaisuke-Downloads-depthanything-android/f824af90-c3fe-40f4-88a7-7ad1c2808e51/scratchpad/an
C=com.litertzoo.npubench.NpuBenchmarkTest
R=com.litertzoo.npubench.test/androidx.test.runner.AndroidJUnitRunner
LOG=$SP/phase2.log
: > $LOG

wait_cool() {
  adb shell input keyevent KEYCODE_SLEEP >/dev/null 2>&1
  for i in $(seq 1 240); do
    st=$(adb shell dumpsys thermalservice 2>/dev/null | grep -m1 "Thermal Status:" | grep -o '[0-9]*')
    [ "${st:-9}" -eq 0 ] && { echo "$st"; return 0; }
    /bin/sleep 10
  done
  echo "${st:-9}"
}

arm() {  # label name accel
  st=$(wait_cool)
  adb shell am force-stop com.litertzoo.npubench >/dev/null 2>&1
  /bin/sleep 3
  adb logcat -c
  timeout 900 adb shell am instrument -w -e class "$C#sweep" -e model "/data/local/tmp/npubench/$2.tflite" -e accel "$3" "$R" >/dev/null 2>&1
  line=$(adb logcat -d | grep -E "NpuBench.*(SWEEP|FAILED)" | tail -1 | sed 's/^.*NpuBench: //')
  echo "$1|gate_status=$st|$line" >> $LOG
}

# dinov2 family, interleaved, 2 replicates of the key arms
arm d2_ctrl_npu_a   dinov2_ctrl   npu
arm d2_fold_npu_a   dinov2_fold   npu
arm d2_repack_npu   dinov2_repack npu
arm d2_ctrl_gpu     dinov2_ctrl   gpu
arm d2_fold_gpu     dinov2_fold   gpu
arm d2_ctrl_npu_b   dinov2_ctrl   npu
arm d2_fold_npu_b   dinov2_fold   npu

# zipformer family
arm zf_ctrl_npu_a   zipf_ctrl npu
arm zf_fold_npu_a   zipf_fold npu
arm zf_ctrl_gpu     zipf_ctrl gpu
arm zf_fold_gpu     zipf_fold gpu
arm zf_ctrl_npu_b   zipf_ctrl npu
arm zf_fold_npu_b   zipf_fold npu

# seq-length probes, NPU then GPU
for t in t512 t1024 t1025 t1500 t2048; do arm pr_${t}_npu probe_$t npu; done
arm pr_w16_npu probe_w16 npu
for t in t512 t1024 t1025 t1500 t2048; do arm pr_${t}_gpu probe_$t gpu; done
arm pr_w16_gpu probe_w16 gpu

echo "PHASE2_DONE" >> $LOG
