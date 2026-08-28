#!/bin/zsh
# Phase 9: batches 4-6 (prefill-T sweep, RMSNorm flavors, fusion breakers,
# granularity curve, GEMV dtype). Same protocol as phase2/5/8:
#   push everything -> compile pass (1 iter, no gate, heats the device) ->
#   gated measure pass (50 iters, thermal NONE per arm).
# Usage: nohup ./phase9_chain.sh <probe_dir> > phase9_nohup.out 2>&1 &
export PATH="$HOME/Library/Android/sdk/platform-tools:$PATH"
export ANDROID_SERIAL=RFGL80R6A6H   # S26; a second device (Pixel 8a) may be attached
DIR=${1:?probe dir}
C=com.litertzoo.npubench.NpuBenchmarkTest
R=com.litertzoo.npubench.test/androidx.test.runner.AndroidJUnitRunner
LOG=$DIR/phase9.log
: > $LOG

PF_ARMS=(pf1151 pf1152 pf1153 pf1279 pf1280 pf1281 pf1407 pf1408 pf1409 pf1500 \
         pf1535 pf1536 pf1537 pf1663 pf1664 pf1665 pfrmssafe1536 pfrmsmax1536)
FX_ARMS=(fx_tanh fx_sig fx_exp fx_abs fx_max fx_relu fx_rsqrt fx_sqrt fx_pow fx_erf)
GR_ARMS=(gr10 gr30 gr100 gr300 gr1000 gr3000 grmono)
GEMV=(gemv_fp32 gemv_fp16 gemv_int8 gemv_int4)
WH_ARMS=(wh_ctrl wh_pad1536)   # probe_wh_ctrl = shipped whisper_encoder (same-day control)
MEM_ARMS=(mem_k100 mem_k140 mem_k186 mem_k240 mem_k310 mem_k380 mem_k440 mem_k530 mem_full)
GPU_ARMS=(pf1536 pf1500 gemv_fp32 gemv_int8 gr300 grmono wh_ctrl wh_pad1536)

wait_cool() {
  adb shell input keyevent KEYCODE_SLEEP >/dev/null 2>&1
  for i in $(seq 1 240); do
    st=$(adb shell dumpsys thermalservice 2>/dev/null | grep -m1 "Thermal Status:" | grep -o '[0-9]*')
    [ "${st:-9}" -eq 0 ] && { echo "$st"; return 0; }
    /bin/sleep 10
  done
  echo "${st:-9}"
}

run_arm() {  # label arm accel iters gate  (file on device: probe_<arm>.tflite)
  if [ "$5" = "gate" ]; then st=$(wait_cool); else st=skip; fi
  adb shell am force-stop com.litertzoo.npubench >/dev/null 2>&1
  /bin/sleep 3
  adb logcat -c
  timeout 1800 adb shell am instrument -w -e class "$C#sweep" \
    -e model "/data/local/tmp/npubench/probe_$2.tflite" -e accel "$3" -e iters "$4" "$R" >/dev/null 2>&1
  line=$(adb logcat -d | grep -E "NpuBench.*(SWEEP|FAILED)" | tail -1 | sed 's/^.*NpuBench: //')
  echo "$1|gate_status=$st|$line" >> $LOG
}

echo "== push ==" >> $LOG
for g in $GEMV; do  # gemv files are built unprefixed; normalize the names
  [ -f "$DIR/probe_$g.tflite" ] || cp "$DIR/$g.tflite" "$DIR/probe_$g.tflite"
done
for f in $PF_ARMS $FX_ARMS $GR_ARMS $GEMV $WH_ARMS $MEM_ARMS; do
  adb push "$DIR/probe_$f.tflite" /data/local/tmp/npubench/ >/dev/null 2>&1 \
    && echo "pushed $f" >> $LOG || echo "PUSH_FAIL $f" >> $LOG
done

echo "== compile pass (npu) ==" >> $LOG
for f in $PF_ARMS $FX_ARMS $GR_ARMS $GEMV $WH_ARMS $MEM_ARMS; do
  run_arm c_${f}_npu $f npu 1 nogate
done
echo "== compile pass (gpu refs) ==" >> $LOG
for f in $GPU_ARMS; do
  run_arm c_${f}_gpu $f gpu 1 nogate
done

echo "== measure pass (npu) ==" >> $LOG
for f in $PF_ARMS $FX_ARMS $GR_ARMS $GEMV $WH_ARMS $MEM_ARMS; do
  run_arm m_${f}_npu $f npu 50 gate
done
echo "== measure pass (gpu refs) ==" >> $LOG
for f in $GPU_ARMS; do
  run_arm m_${f}_gpu $f gpu 50 gate
done

echo "PHASE9_DONE" >> $LOG
