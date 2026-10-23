#!/bin/zsh
# fetch_hub.sh [--length N ...] [--list]: the files this app needs from the public model repository
# litert-community/clef-flash-LiteRT at a pinned revision, into ../hub/ (the same folder and revision as the clef-flash
# sample's own scripts/fetch_hub.sh, so a file either script has downloaded is kept), then every one of them checked
# with `shasum -a 256 -c` against the repository's SHA256SUMS.
#   zsh scripts/fetch_hub.sh                 the 2,048-token row graph, the tables, the joint head, the tokenizer, the
#                                            host, README, LICENSE, NOTICE, SHA256SUMS: 23 files, 18.2 GB
#   zsh scripts/fetch_hub.sh --length 4096   and the 4,096-token row graph as well (--length takes 256, 512, 1024, 2048
#                                            or 4096 and can be given more than once)
#   zsh scripts/fetch_hub.sh --list          print the files and their sizes, download nothing
#   REV=<commit> pins another revision.
# No picture tower and no picture is downloaded: this app sends text only. A file already there at the repository's
# size is kept, so a second run only checks the files. One file at a time over HTTPS (curl). A transfer can end early
# with exit 0, so each file is resumed with `curl -C -` until its size is the repository's (the tree listing at REV),
# then moved into place.
# Log: out/fetch_hub.log. Exit 1 on a size or sha256 mismatch (the line names the file), 2 on a bad argument.
set -u
HERE=${0:A:h}
H=${HERE:h}
HUB=${H:h}/hub
OUT=$H/out
LOG=$OUT/fetch_hub.log
REPO=litert-community/clef-flash-LiteRT
REV=${REV:-419ecb59c3d65bcf4f9e3dfab004fdddf805da86}
LENGTHS=(2048)
LIST_ONLY=0
while (( $# )); do
  case $1 in
    --length)
      [[ ${2:-} == (256|512|1024|2048|4096) ]] || { print -u2 "--length takes 256, 512, 1024, 2048 or 4096"; exit 2 }
      LENGTHS+=($2); shift 2 ;;
    --list) LIST_ONLY=1; shift ;;
    *) print -u2 "usage: zsh scripts/fetch_hub.sh [--length N ...] [--list]"; exit 2 ;;
  esac
done
LENGTHS=(${(u)LENGTHS})
mkdir -p $HUB $OUT

TEXT_FILES=(
  SHA256SUMS README.md LICENSE NOTICE contract.json
  host/clef_litert/__init__.py host/clef_litert/_erf.py host/clef_litert/errors.py host/clef_litert/head.py
  host/clef_litert/host.py host/clef_litert/pixels.py host/clef_litert/positions.py host/clef_litert/request.py
  host/clef_litert/response.py host/clef_litert/runtime.py host/clef_litert/tables.py host/requirements-host.txt
  tokenizer/tokenizer.json
  head/joint_head.safetensors head/joint_head_config.json
  tables/embed_tokens.safetensors tables/lm_head.safetensors
)

log() { print -r -- "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

# The repository's size of every file at REV (the tree listing), as "<size> <path>" lines.
TREE=$OUT/fetch_hub.tree.json
curl -sS --fail -o $TREE "https://huggingface.co/api/models/$REPO/tree/$REV?recursive=true" || { log "no tree listing"; exit 1; }
PY=$(command -v python3.14 || command -v python3) || { log "no python3 to read the tree listing"; exit 1; }
typeset -A SIZE
while read -r sz p; do SIZE[$p]=$sz; done < <($PY -c '
import json, sys
for x in json.load(open(sys.argv[1])):
    if x["type"] == "file":
        print(x["size"], x["path"])' $TREE)
FILES=($TEXT_FILES)
for L in $LENGTHS; do FILES+=(clef-flash_rowprefill_embeds_L${L}_fp16fc.tflite); done
if (( LIST_ONLY )); then
  total=0
  for f in $FILES; do print -r -- "${SIZE[$f]:-MISSING} $f"; total=$((total + ${SIZE[$f]:-0})); done
  print -r -- "${#FILES} files, $total B"
  exit 0
fi
log "start: $REPO at $REV, ${#FILES} files into $HUB (row graphs: ${(j:, :)LENGTHS} tokens)"

for f in $FILES; do
  want=${SIZE[$f]:-}
  [[ -n $want ]] || { log "MISSING in the repository at $REV: $f"; exit 1; }
  dst=$HUB/$f
  if [[ -f $dst && $(stat -f %z $dst) == $want ]]; then
    log "present $f ($want B)"
    continue
  fi
  mkdir -p ${dst:h}
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
  done
  got=$(stat -f %z $dst.part)
  [[ $got == $want ]] || { log "SIZE MISMATCH $f: $got != $want"; exit 1; }
  mv $dst.part $dst
  dt=$(( $(date +%s) - t0 ))
  log "fetched $f ($want B in ${dt} s$( (( dt > 0 )) && print ", $(( want / dt / 1000 / 1000 )) MB/s"))"
done

# Every file of the set against SHA256SUMS, whose paths read "./<path>". SHA256SUMS does not list itself.
LIST=$OUT/fetch_hub.sha256sums
: > $LIST
n=0; bytes=0
for f in $FILES; do
  line=$(grep -F "  ./$f" $HUB/SHA256SUMS | awk -v p="./$f" '$2 == p')
  if [[ -z $line ]]; then
    [[ $f == SHA256SUMS ]] && continue
    log "NO SHA256SUMS LINE for $f"; exit 1
  fi
  print -r -- $line >> $LIST
  n=$((n + 1)); bytes=$((bytes + $(stat -f %z $HUB/$f)))
done
log "sha256 check of $n files ($bytes B) against SHA256SUMS"
CHECK=$OUT/fetch_hub.sha256check
(cd $HUB && shasum -a 256 -c $LIST) > $CHECK 2>&1
rc=$?
cat $CHECK >> $LOG
if (( rc == 0 )); then
  log "SHA256 OK: $n files, $bytes B (+ SHA256SUMS itself, $(stat -f %z $HUB/SHA256SUMS) B)"
else
  bad=(${(f)"$(sed -n 's/^\.\///; s/: FAILED.*$//p' $CHECK)"})
  log "SHA256 MISMATCH: ${(j:, :)bad} (delete it from ../hub/ and run this script again)"; exit 1
fi
