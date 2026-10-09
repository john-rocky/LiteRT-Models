#!/bin/zsh
# fetch_hub.sh [--all]: the files of the public model repository litert-community/d1-3B-LiteRT at a pinned revision,
# into hub/, then every downloaded file checked with `shasum -a 256 -c` against the repository's SHA256SUMS.
#   zsh scripts/fetch_hub.sh          the default set, 27 files, 16.1 GB: what the bundled examples and
#                                     examples/run_example.py --check load (the 64-token pair, the 512- and
#                                     4,096-token row graphs, the picture tower and projector, the tables, the
#                                     tokenizer, the host, the examples, the fixtures)
#   zsh scripts/fetch_hub.sh --all    every file of the repository, 80 files, 45.3 GB: every row graph (128 to 4,096
#                                     tokens) and every pair, so each request runs on the smallest graph that holds it
#   REV=<commit> pins another revision.
# A file already there at the repository's size is kept, so a second run only checks the files. One file at a time
# over HTTPS (curl). A transfer can end early with exit 0, so each file is resumed with `curl -C -` until its size is
# the repository's (the tree listing at REV), then moved into place. FETCH_IDLE_IF=<interface> (en0, for example)
# waits before each large file until that interface moves less than 400 KB/s over 8 s; unset, nothing waits.
# The 4,096-token graph is in the default set for run_example.py --check: its refused request expects "longer than
# the largest graph (4096)", and the host names the largest row graph present.
# Log: out/fetch_hub.log. Exit 1 on a size or sha256 mismatch.
set -u
HERE=${0:A:h}
D=${HERE:h}
HUB=$D/hub
OUT=$D/out
LOG=$OUT/fetch_hub.log
REPO=litert-community/d1-3B-LiteRT
REV=${REV:-c7a76154b0eac65a86775d0bdd683d61ca78e14f}
IF=${FETCH_IDLE_IF:-}
IDLE_BPS=$((400 * 1024))
BIG=$((100 * 1000 * 1000))
ALL=0
[[ ${1:-} == --all ]] && ALL=1
mkdir -p $HUB $OUT

DEFAULT_FILES=(
  SHA256SUMS README.md LICENSE NOTICE contract.json
  host/d1_litert.py host/d1_shared_state.py host/d1_vision.py host/requirements-host.txt
  examples/run_example.py examples/run_example.expected.json
  fixtures/LICENSE-SemIf-MIT.txt fixtures/README.md fixtures/rebuild_requests.py fixtures/red_arms_public.json
  fixtures/reference_probs.json fixtures/requests_public.json fixtures/token_probes.json
  tokenizer/tokenizer.json
  tables/readout_table.safetensors tables/vision_position_table.safetensors tables/embed_table.safetensors
  d1-3b_projector_fp16fc.tflite
  d1-3b_vision_tower_fp16fc.tflite
  d1-3b_sharedstate_embeds_Ls64_Lq64_fp16fc.tflite
  d1-3b_rowprefill_embeds_L512_fp16fc.tflite
  d1-3b_rowprefill_embeds_L4096_fp16fc.tflite
)

log() { print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

if_rate() {   # bytes per second on $IF (in + out) over 8 s
  local a b
  a=$(netstat -ib -I $IF | awk 'NR==2 {print $7 + $10}')
  sleep 8
  b=$(netstat -ib -I $IF | awk 'NR==2 {print $7 + $10}')
  print $(( (b - a) / 8 ))
}

wait_for_idle() {
  [[ -n $IF ]] || return 0
  local r
  while true; do
    r=$(if_rate)
    if (( r < IDLE_BPS )); then
      log "$IF quiet ($((r / 1024)) KB/s)"
      return
    fi
    log "$IF busy ($((r / 1024)) KB/s), next look in 60 s"
    sleep 60
  done
}

# The repository's size of every file at REV (the tree listing), as "<size> <path>" lines.
TREE=$OUT/fetch_hub.tree.json
curl -sS --fail -o $TREE "https://huggingface.co/api/models/$REPO/tree/$REV?recursive=true" || { log "no tree listing"; exit 1; }
typeset -A SIZE
while read -r sz p; do SIZE[$p]=$sz; done < <(/usr/bin/python3 -c '
import json, sys
for x in json.load(open(sys.argv[1])):
    if x["type"] == "file":
        print(x["size"], x["path"])' $TREE)
if (( ALL )); then
  FILES=(SHA256SUMS ${(o)${(k)SIZE}:#SHA256SUMS})
else
  FILES=($DEFAULT_FILES)
fi
log "start: $REPO at $REV, ${#FILES} files$( (( ALL )) && print ' (--all)')"

gated=0
for f in $FILES; do
  want=${SIZE[$f]:-}
  [[ -n $want ]] || { log "MISSING in the repository at $REV: $f"; exit 1; }
  dst=$HUB/$f
  if [[ -f $dst && $(stat -f %z $dst) == $want ]]; then
    log "present $f ($want B)"
    continue
  fi
  mkdir -p ${dst:h}
  if (( want >= BIG || gated == 0 )); then wait_for_idle; gated=1; fi
  t0=$(date +%s)
  attempt=0
  while true; do
    cur=0; [[ -f $dst.part ]] && cur=$(stat -f %z $dst.part)
    (( cur >= want )) && break
    (( cur > 0 )) && log "resume $f at $cur B"
    curl -L -sS --fail -C - --speed-limit 30720 --speed-time 180 -o $dst.part \
      "https://huggingface.co/$REPO/resolve/$REV/$f" && continue
    attempt=$((attempt + 1))
    log "attempt $attempt on $f ended at $([[ -f $dst.part ]] && stat -f %z $dst.part || print 0) B"
    (( attempt >= 30 )) && { log "GIVING UP on $f"; exit 1; }
    sleep 20
    (( want >= BIG )) && wait_for_idle
  done
  got=$(stat -f %z $dst.part)
  [[ $got == $want ]] || { log "SIZE MISMATCH $f: $got != $want"; exit 1; }
  mv $dst.part $dst
  dt=$(( $(date +%s) - t0 ))
  log "fetched $f ($want B in ${dt} s$( (( dt > 0 )) && print ", $(( want / dt / 1000 / 1000 )) MB/s"))"
done

# Every file of the set against SHA256SUMS, whose paths read "./<path>". SHA256SUMS does not list itself or the
# repository's .gitattributes.
LIST=$OUT/fetch_hub.sha256sums
: > $LIST
n=0; bytes=0
for f in $FILES; do
  line=$(grep -F "  ./$f" $HUB/SHA256SUMS | awk -v p="./$f" '$2 == p')
  if [[ -z $line ]]; then
    [[ $f == SHA256SUMS || $f == .gitattributes ]] && continue
    log "NO SHA256SUMS LINE for $f"; exit 1
  fi
  print -r -- $line >> $LIST
  n=$((n + 1)); bytes=$((bytes + $(stat -f %z $HUB/$f)))
done
log "sha256 check of $n files ($bytes B) against SHA256SUMS"
if (cd $HUB && shasum -a 256 -c $LIST) >> $LOG 2>&1; then
  log "SHA256 OK: $n files, $bytes B (+ SHA256SUMS itself, $(stat -f %z $HUB/SHA256SUMS) B)"
else
  log "SHA256 MISMATCH (lines above)"; exit 1
fi
