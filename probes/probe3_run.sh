#!/bin/bash
# Probe v3 on plain-view: the fleet's real Inkling — Inkling-2-Mini-AD-Q4_K.gguf (inkling2_loop).
# Legs: default (loop.count=3) vs --override-kv inkling2_loop.loop.count=int:4.
set -u
P=$HOME/rse-probe; mkdir -p $P
M=/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/Inkling-2-Mini-AD-Q4_K.gguf
SRV=$HOME/llama-bin/cpu/llama-server
[ -x "$SRV" ] || SRV=$HOME/rse-runtime/cpu/llama-server
echo "server=$SRV host=$(hostname) model=$M" > $P/probe3.log
sha256sum "$M" >> $P/probe3.log
sensors 2>/dev/null | grep -iE "tctl|edge" >> $P/probe3.log
date >> $P/probe3.log

leg () {
  name=$1; shift
  echo "=== LEG $name extra_args: $@" >> $P/probe3.log
  "$SRV" -m "$M" --host 127.0.0.1 --port 18099 -c 2048 -t 16 "$@" > $P/$name.server.log 2>&1 &
  pid=$!
  ok=0
  for i in $(seq 1 72); do
    sleep 5
    grep -q "listening" $P/$name.server.log && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break
  done
  echo "listening=$ok" >> $P/probe3.log
  grep -iE "arch|unused|missing|loop|n_layer|tensor|error|fail" $P/$name.server.log | head -20 >> $P/probe3.log
  if [ $ok = 1 ]; then
    curl -s -m 600 http://127.0.0.1:18099/completion -H 'Content-Type: application/json' \
      -d '{"prompt":"What is 17 + 25? Give the number, then keep counting upward by 7 each step.","n_predict":48,"temperature":0,"cache_prompt":false}' \
      > $P/$name.json 2>> $P/probe3.log
    echo "--- completion ($name) ---" >> $P/probe3.log
    python3 -c "import json;d=json.load(open('$P/$name.json'));print(repr(d.get('content')));print('timings:',d.get('timings'))" >> $P/probe3.log 2>&1
    curl -s -m 600 http://127.0.0.1:18099/completion -H 'Content-Type: application/json' \
      -d '{"prompt":"Janet has 3 baskets with 14 apples each. She gives away 9 apples. How many apples does she have left? Think step by step, then give the final number.","n_predict":96,"temperature":0,"cache_prompt":false}' \
      > $P/$name-math.json 2>> $P/probe3.log
    echo "--- math completion ($name) ---" >> $P/probe3.log
    python3 -c "import json;d=json.load(open('$P/$name-math.json'));print(repr(d.get('content')))" >> $P/probe3.log 2>&1
  fi
  kill $pid 2>/dev/null; sleep 3; kill -9 $pid 2>/dev/null
  sensors 2>/dev/null | grep -iE "tctl|edge" >> $P/probe3.log
  date >> $P/probe3.log
}

leg adbase
leg adloops4 --override-kv inkling2_loop.loop.count=int:4
echo "PROBE3 DONE" >> $P/probe3.log
