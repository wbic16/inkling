#!/bin/bash
# Full base-checkpoint download ON plain-view -> local disk, sha-verified, then copied to NFS.
set -u
SRC=/mnt/mirrorborn/shared/models/inkling-2-mini-hf
DEST=$HOME/inkling-hf-base
mkdir -p "$DEST"; cd "$DEST"
LOG=$DEST/pv-download.log
echo "=== pv download start $(date -Is) ===" >> "$LOG"
cp "$SRC/manifest.tsv" . || { echo "NO MANIFEST" >> "$LOG"; exit 1; }
FN=model-00001-of-00001.safetensors
SIZE=19526019168
fail=0
# small files first
while IFS=$'\t' read -r fn size sha; do
  case "$fn" in \#*|""|"$FN") continue;; esac
  mkdir -p "$(dirname "$fn")"
  actual=$(stat -c%s "$fn" 2>/dev/null || echo 0)
  if [ "$actual" != "$size" ]; then
    curl -sL --retry 5 --retry-delay 5 -o "$fn" "https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/$fn" >> "$LOG" 2>&1
    actual=$(stat -c%s "$fn" 2>/dev/null || echo 0)
  fi
  if [ "$actual" != "$size" ]; then echo "SIZE FAIL $fn got=$actual want=$size" >> "$LOG"; fail=1; continue; fi
  if [ -n "$sha" ]; then
    got=$(sha256sum "$fn" | cut -d' ' -f1)
    [ "$got" != "$sha" ] && { echo "SHA FAIL $fn" >> "$LOG"; fail=1; continue; }
  fi
  echo "OK $fn" >> "$LOG"
done < manifest.tsv
# big file: 8 parallel ranges
SHA=$(grep "^$FN" manifest.tsv | cut -f3)
if [ -f "$FN" ] && [ "$(sha256sum "$FN" | cut -d' ' -f1)" = "$SHA" ]; then
  echo "big file already verified" >> "$LOG"
else
  N=8; CHUNK=$(( (SIZE + N - 1) / N ))
  for i in $(seq 0 $((N-1))); do
    S=$(( i * CHUNK )); E=$(( S + CHUNK - 1 )); [ $E -ge $SIZE ] && E=$((SIZE-1))
    ( curl -sL --retry 5 --retry-delay 5 -r $S-$E -o "$FN.part$i" "https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/$FN" >> "$LOG" 2>&1 ) &
  done
  wait
  echo "--- parts done $(date -Is)" >> "$LOG"
  ok=1
  for i in $(seq 0 $((N-1))); do
    S=$(( i * CHUNK )); E=$(( S + CHUNK - 1 )); [ $E -ge $SIZE ] && E=$((SIZE-1))
    want=$(( E - S + 1 )); got=$(stat -c%s "$FN.part$i" 2>/dev/null || echo 0)
    [ "$got" != "$want" ] && { echo "PART $i SIZE FAIL got=$got want=$want" >> "$LOG"; ok=0; }
  done
  [ $ok = 0 ] && { echo "PV FAIL parts" >> "$LOG"; touch PV_FAIL; exit 1; }
  cat "$FN".part0 "$FN".part1 "$FN".part2 "$FN".part3 "$FN".part4 "$FN".part5 "$FN".part6 "$FN".part7 > "$FN"
  rm -f "$FN".part*
  GOT=$(sha256sum "$FN" | cut -d' ' -f1)
  echo "big file sha256=$GOT expect=$SHA" >> "$LOG"
  [ "$GOT" != "$SHA" ] && { echo "PV FAIL sha" >> "$LOG"; touch PV_FAIL; exit 1; }
fi
echo "=== local set complete fail=$fail $(date -Is) ===" >> "$LOG"
# copy to NFS for the fleet record
cp -a . "$SRC.tmp-copy" 2>>"$LOG" && mv "$SRC.tmp-copy"/* "$SRC/" 2>>"$LOG" && rmdir "$SRC.tmp-copy" 2>/dev/null
NSHA=$(sha256sum "$SRC/$FN" | cut -d' ' -f1)
echo "NFS big file sha256=$NSHA" >> "$LOG"
[ "$NSHA" = "$SHA" ] && echo "NFS COPY VERIFIED" >> "$LOG" || echo "NFS COPY MISMATCH" >> "$LOG"
echo "=== pv download done $(date -Is) ===" >> "$LOG"
touch PV_DOWNLOAD_DONE
