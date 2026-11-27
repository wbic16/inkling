#!/bin/bash
# Second wave: download the SECOND HALF of each part's range as .sub$i files.
set -u
DEST=$HOME/inkling-hf-base; cd "$DEST"
FN=model-00001-of-00001.safetensors
SIZE=19526019168; N=8; CHUNK=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  S=$(( i * CHUNK )); E=$(( S + CHUNK - 1 )); [ $E -ge $SIZE ] && E=$((SIZE-1))
  LEN=$(( E - S + 1 )); HALF=$(( LEN / 2 )); MS=$(( S + HALF ))
  if [ ! -f "$FN.sub$i" ]; then
    ( curl -sL --retry 5 --retry-delay 5 -r $MS-$E -o "$FN.sub$i" "https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/$FN" >> pv-download.log 2>&1 ) &
  fi
done
wait
echo "--- sub wave done $(date -Is)" >> pv-download.log
ls -la "$FN".sub* >> pv-download.log 2>/dev/null
touch SUB_DONE
