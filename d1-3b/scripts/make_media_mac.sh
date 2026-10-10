#!/bin/zsh
# make_media_mac.sh <tag>: the videos of one take, cut from its two window recordings (take.sh) without touching
# what the screen shows:
#   out/<tag>_x.mp4     scene A, then scene B, then the end card (2.0 s)
#   out/<tag>_A_x.mp4   scene A, then the end card
#   out/<tag>_B_x.mp4   scene B, then the end card
#   out/<tag>_endcard.png, out/<tag>_<scene>.cuts.json (media_cuts.py: where each cut starts and ends, and why)
# Each scene's cut (media_cuts.py): 1.0 s of the scene's first screen before anything on it changes (all of it when
# shorter), the scene as recorded, and the final screen (the answers, fully drawn) for 3.0 s from the frame it first
# appears. The recordings are constant 30 fps (record_window: one frame per tick, ticks_skipped 0 in rec.log), so the
# cut selects whole frames; for the end, tpad clones the last frame, fps=30:round=up, and the trim stays on the
# 30 fps grid. No caption, no speed change, no frame reordered or edited; the end card is the only frame added.
# The end card is converted to the recordings' colour (BT.709, limited range) so its background matches theirs.
# Encoding: libx264 -preset slow -crf 18 -profile:v high -pix_fmt yuv420p -movflags +faststart, BT.709 tags, no
# audio stream, 30 fps CFR. Check: check_media.py <tag>.
set -eu
HERE=${0:A:h}
D=${HERE:h}
OUT=$D/out
TAG=$1
PY=$D/venv-demo/bin/python
A=$OUT/${TAG}_A.mp4
B=$OUT/${TAG}_B.mp4
CARD=$OUT/${TAG}_endcard.png
CARD_FRAMES=60
for f in $A $B; do [[ -f $f ]] || { echo "no $f"; exit 1 }; done
for s in A B; do
  skipped=$(grep '^SUMMARY' $OUT/${TAG}_$s.rec.log | sed 's/.*"ticks_skipped":\([0-9]*\).*/\1/')
  [[ $skipped == 0 ]] || { echo "${TAG}_$s.mp4: $skipped ticks skipped, not constant rate"; exit 1 }
done

read SA EA NA <<< $($PY -B $HERE/media_cuts.py $TAG A)
read SB EB NB <<< $($PY -B $HERE/media_cuts.py $TAG B)
echo "cut A: frames $SA..$((EA - 1)) of $NA ($((EA - SA)) frames); cut B: frames $SB..$((EB - 1)) of $NB ($((EB - SB)) frames)"
$PY -B $HERE/make_end_card.py $CARD

ENC=(-c:v libx264 -preset slow -crf 18 -profile:v high -pix_fmt yuv420p -movflags +faststart
     -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv -fps_mode cfr -r 30 -an)
seg() {   # seg <input> <start frame> <end frame, exclusive> <label>
  print -rn -- "[$1:v]trim=start_frame=$2,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=4,fps=30:round=up,"
  print -rn -- "trim=end_frame=$(( $3 - $2 )),setpts=PTS-STARTPTS,setsar=1[$4];"
}
card() {  # card <input> <label>
  print -rn -- "[$1:v]scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,fps=30,trim=end_frame=$CARD_FRAMES,"
  print -rn -- "setpts=PTS-STARTPTS,setsar=1[$2];"
}
ffmpeg -v error -y -i $A -i $B -loop 1 -framerate 30 -i $CARD \
  -filter_complex "$(seg 0 $SA $EA a)$(seg 1 $SB $EB b)$(card 2 e)[a][b][e]concat=n=3:v=1:a=0[v]" -map "[v]" $ENC \
  $OUT/${TAG}_x.mp4
ffmpeg -v error -y -i $A -loop 1 -framerate 30 -i $CARD \
  -filter_complex "$(seg 0 $SA $EA a)$(card 1 e)[a][e]concat=n=2:v=1:a=0[v]" -map "[v]" $ENC $OUT/${TAG}_A_x.mp4
ffmpeg -v error -y -i $B -loop 1 -framerate 30 -i $CARD \
  -filter_complex "$(seg 0 $SB $EB b)$(card 1 e)[b][e]concat=n=2:v=1:a=0[v]" -map "[v]" $ENC $OUT/${TAG}_B_x.mp4
for f in $OUT/${TAG}_x.mp4 $OUT/${TAG}_A_x.mp4 $OUT/${TAG}_B_x.mp4; do
  echo "${f:t} $(stat -f %z $f) B $(ffprobe -v error -select_streams v:0 -show_entries \
    stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration -of csv=p=0 $f | tr '\n' ' ') sha256 \
$(shasum -a 256 $f | cut -d' ' -f1)"
done
