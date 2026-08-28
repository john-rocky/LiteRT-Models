#!/bin/zsh
# Mac-side build of every phase-9 probe (batches 4-6, whisper pad pair,
# memorize prefix cuts, GEMV dtype family). Device measurement is
# phase9_chain.sh once the S26 is free; this script only writes .tflite files
# (~2 GB) into $1 plus a build9.log.
#   nohup zsh build_phase9_probes.sh <out_dir> > build9_nohup.out 2>&1 &
set -u
DIR=${1:?out dir}
F=${0:A:h}
MEM=$F/../../edgetam-video/app/src/main/assets/memorize.tflite
source ~/venvs/ltconv040dev/bin/activate
mkdir -p $DIR && cd $DIR
LOG=$DIR/build9.log
: > $LOG

step() {
  local label=$1; shift
  echo "== $label ==" >> $LOG
  "$@" >> $LOG 2>&1 && echo "-- $label ok" >> $LOG || echo "-- $label FAIL rc=$?" >> $LOG
}

step verify_batch4 python $F/probe_batch4.py verify
step batch4_pf python $F/probe_batch4.py pf1151 pf1152 pf1153 pf1279 pf1280 pf1281 \
  pf1407 pf1408 pf1409 pf1500 pf1535 pf1536 pf1537 pf1663 pf1664 pf1665
step batch4_rms python $F/probe_batch4.py pfrmssafe1536 pfrmsmax1536
step batch6_fx python $F/probe_batch6.py fx_tanh fx_sig fx_exp fx_abs fx_max \
  fx_relu fx_rsqrt fx_sqrt fx_pow fx_erf
step batch6_gr python $F/probe_batch6.py gr10 gr30 gr100 gr300 gr1000 gr3000 grmono
step whisper_pad python $F/build_whisper_pad.py
[ -f whisper_enc_pad1536.tflite ] && mv whisper_enc_pad1536.tflite probe_wh_pad1536.tflite
step whisper_ctrl python $F/build_whisper_ctrl.py
step gemv_fp32 python $F/probe_batch5.py
step gemv_fp16 python $F/make_fp16.py gemv_fp32.tflite gemv_fp16.tflite
step gemv_int8 python $F/make_int8.py gemv_fp32.tflite gemv_int8.tflite
step gemv_int4 python $F/make_int8.py gemv_fp32.tflite gemv_int4.tflite --int4
step mem_full cp $MEM probe_mem_full.tflite
for k in 100 140 186 240 310 380 440 530; do
  step mem_k$k python $F/cut_prefix.py probe_mem_full.tflite $k probe_mem_k$k.tflite
done
ls -la $DIR/*.tflite >> $LOG
echo "BUILD9_DONE" >> $LOG
