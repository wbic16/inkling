#!/bin/bash
# Parallel range download of the 19.53 GB safetensors (per-connection throttle workaround).
set -u
DEST=/mnt/mirrorborn/shared/models/inkling-2-mini-hf
FN=model-00001-of-00001.safetensors
SIZE=19526019168
N=8
LOG=$DEST/parallel.log
cd "$DEST"
echo "=== parallel start $(date -Is) ===" >> "$LOG"
SHA=$(grep "^$FN" manifest.tsv | cut -f3)
echo "expected sha256=$SHA" >> "$LOG"
CHUNK=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  S=$(( i * CHUNK )); E=$(( S + CHUNK - 1 )); [ $E -ge $SIZE ] && E=$((SIZE-1))
  ( curl -sL --retry 5 --retry-delay 5 -r $S-$E -o "$FN.part$i" "https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/$FN" >> "$LOG" 2>&1 ) &
done
wait
echo "--- parts done $(date -Is)" >> "$LOG"
ls -la "$FN".part* >> "$LOG"
# verify part sizes
ok=1
for i in $(seq 0 $((N-1))); do
  S=$(( i * CHUNK )); E=$(( S + CHUNK - 1 )); [ $E -ge $SIZE ] && E=$((SIZE-1))
  want=$(( E - S + 1 )); got=$(stat -c%s "$FN.part$i" 2>/dev/null || echo 0)
  [ "$got" != "$want" ] && { echo "PART $i SIZE MISMATCH got=$got want=$want" >> "$LOG"; ok=0; }
done
[ $ok = 0 ] && { echo "PARALLEL FAIL (sizes)" >> "$LOG"; touch PARALLEL_FAIL; exit 1; }
cat "$FN".part0 "$FN".part1 "$FN".part2 "$FN".part3 "$FN".part4 "$FN".part5 "$FN".part6 "$FN".part7 > "$FN"
rm -f "$FN".part*
GOT=$(sha256sum "$FN" | cut -d' ' -f1)
echo "assembled sha256=$GOT" >> "$LOG"
if [ "$GOT" = "$SHA" ]; then echo "PARALLEL OK $(date -Is)" >> "$LOG"; touch PARALLEL_DONE; else echo "PARALLEL FAIL sha" >> "$LOG"; touch PARALLEL_FAIL; fi
