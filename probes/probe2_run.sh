#!/bin/bash
# Probe v2 on plain-view: fleet llama-server vs our GGUF — load log + greedy gen,
# default vs --override-kv inkling2.num_loops=int:4. Stage-checkpointed in ~/rse-probe/.
set -u
P=$HOME/rse-probe; mkdir -p $P
M=/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/Inkling-2-Mini-Q4_K_M.gguf
SRV=""
for c in $HOME/llama-bin/cpu/llama-server $HOME/inkling-stage/cpu/llama-server; do
  [ -x "$c" ] && SRV=$c && break
done
if [ -z "$SRV" ]; then
  mkdir -p $HOME/rse-runtime && tar xzf /mnt/mirrorborn/shared/models/inkling-2-mini-gguf/runtime/llama-runtime.tar.gz -C $HOME/rse-runtime
  SRV=$HOME/rse-runtime/cpu/llama-server
fi
echo "server=$SRV host=$(hostname)" > $P/probe2.log
sensors 2>/dev/null | grep -iE "tctl|edge" >> $P/probe2.log
date >> $P/probe2.log

leg () {
  name=$1; shift
  echo "=== LEG $name extra_args: $@" >> $P/probe2.log
  "$SRV" -m "$M" --host 127.0.0.1 --port 18099 -c 2048 -t 16 "$@" > $P/$name.server.log 2>&1 &
  pid=$!
  for i in $(seq 1 60); do
    sleep 5
    grep -q "listening" $P/$name.server.log && break
    kill -0 $pid 2>/dev/null || break
  done
  grep -iE "arch|unused|missing|loop|n_layer|tensor|error|fail" $P/$name.server.log | head -30 >> $P/probe2.log
  curl -s -m 240 http://127.0.0.1:18099/completion -H 'Content-Type: application/json' \
    -d '{"prompt":"What is 17 + 25? Give the number, then keep counting upward by 7 each step.","n_predict":48,"temperature":0,"cache_prompt":false}' \
    > $P/$name.json 2>> $P/probe2.log
  echo "--- completion ($name) ---" >> $P/probe2.log
  python3 -c "import json;d=json.load(open('$P/$name.json'));print(repr(d.get('content')));print('timings:',d.get('timings'))" >> $P/probe2.log 2>&1
  kill $pid 2>/dev/null; sleep 3; kill -9 $pid 2>/dev/null
  sensors 2>/dev/null | grep -iE "tctl|edge" >> $P/probe2.log
  date >> $P/probe2.log
}

leg base
leg loops4 --override-kv inkling2.num_loops=int:4
echo "PROBE2 DONE" >> $P/probe2.log
