#!/bin/bash
# Assembly: build the big safetensors from parts (full) or part-prefix+sub, verify, copy to NFS.
set -u
DEST=$HOME/inkling-hf-base; cd "$DEST"
LOG=$DEST/pv-download.log
FN=model-00001-of-00001.safetensors
SIZE=19526019168; N=8; CHUNK=$(( (SIZE + N - 1) / N ))
SHA=$(grep "^$FN" manifest.tsv | cut -f3)
pkill -f "run-pv-download.sh" 2>/dev/null  # take over completion from the first script
echo "=== assembly watcher start $(date -Is), expect sha $SHA ===" >> "$LOG"
seg_ready() {
  local i=$1 S=$2 E=$3 LEN=$4 HALF=$5
  local psz=$(stat -c%s "$FN.part$i" 2>/dev/null || echo 0)
  local ssz=$(stat -c%s "$FN.sub$i" 2>/dev/null || echo 0)
  if [ "$ssz" = "$((LEN-HALF))" ] && [ "$psz" -ge "$HALF" ]; then return 0; fi
  if [ "$psz" = "$LEN" ]; then return 0; fi
  return 1
}
while true; do
  all=1
  for i in $(seq 0 $((N-1))); do
    S=$(( i * CHUNK )); E=$(( S + CHUNK - 1 )); [ $E -ge $SIZE ] && E=$((SIZE-1))
    LEN=$(( E - S + 1 )); HALF=$(( LEN / 2 ))
    seg_ready $i $S $E $LEN $HALF || { all=0; break; }
  done
  [ $all = 1 ] && break
  sleep 60
done
echo "--- all segments satisfiable $(date -Is)" >> "$LOG"
pkill -f "safetensors.part" 2>/dev/null; sleep 2
for i in $(seq 0 $((N-1))); do
  S=$(( i * CHUNK )); E=$(( S + CHUNK - 1 )); [ $E -ge $SIZE ] && E=$((SIZE-1))
  LEN=$(( E - S + 1 )); HALF=$(( LEN / 2 ))
  ssz=$(stat -c%s "$FN.sub$i" 2>/dev/null || echo 0)
  if [ "$ssz" = "$((LEN-HALF))" ]; then
    head -c $HALF "$FN.part$i" > "$FN.seg$i"; cat "$FN.sub$i" >> "$FN.seg$i"
  else
    cp "$FN.part$i" "$FN.seg$i"
  fi
  gotsz=$(stat -c%s "$FN.seg$i")
  [ "$gotsz" != "$LEN" ] && { echo "SEG $i SIZE FAIL $gotsz != $LEN" >> "$LOG"; touch PV_FAIL; exit 1; }
done
cat "$FN".seg0 "$FN".seg1 "$FN".seg2 "$FN".seg3 "$FN".seg4 "$FN".seg5 "$FN".seg6 "$FN".seg7 > "$FN"
rm -f "$FN".seg* "$FN".part* "$FN".sub*
GOT=$(sha256sum "$FN" | cut -d' ' -f1)
echo "assembled sha256=$GOT" >> "$LOG"
[ "$GOT" != "$SHA" ] && { echo "ASSEMBLY SHA FAIL" >> "$LOG"; touch PV_FAIL; exit 1; }
echo "ASSEMBLY VERIFIED $(date -Is)" >> "$LOG"
SRC=/mnt/mirrorborn/shared/models/inkling-2-mini-hf
cp -a . "$SRC.tmp-copy" && for f in "$SRC.tmp-copy"/*; do mv "$f" "$SRC/"; done; rmdir "$SRC.tmp-copy" 2>/dev/null
NSHA=$(sha256sum "$SRC/$FN" | cut -d' ' -f1)
echo "NFS sha256=$NSHA" >> "$LOG"
[ "$NSHA" = "$SHA" ] && echo "NFS COPY VERIFIED" >> "$LOG" || { echo "NFS MISMATCH" >> "$LOG"; touch PV_FAIL; exit 1; }
echo "=== download stage complete $(date -Is) ===" >> "$LOG"
touch PV_DOWNLOAD_DONE
