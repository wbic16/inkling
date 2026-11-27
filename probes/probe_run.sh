#!/bin/bash
# Decisive runtime probe: what does llama.cpp do with our fleet GGUF?
# Stage-checkpointed: logs land in ~/rse-probe/ as each leg finishes.
set -u
P=/home/wbic16/rse-probe
mkdir -p $P
M=/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/Inkling-2-Mini-Q4_K_M.gguf
CLI=/home/wbic16/llama.cpp/build-cpu/bin/llama-cli
[ -x "$CLI" ] || CLI=/home/wbic16/llama.cpp/build-vulkan/bin/llama-cli
echo "cli=$CLI" > $P/probe.log
sensors 2>/dev/null | grep -i "tctl" >> $P/probe.log || cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -3 >> $P/probe.log
date >> $P/probe.log

run_leg () {
  name=$1; shift
  echo "=== LEG $name : $@" >> $P/probe.log
  timeout 900 "$CLI" -m "$M" -p "What is 17 + 25? Give the number, then keep counting upward by 7 each step." \
    -n 32 --temp 0 -t 16 "$@" > $P/$name.out 2> $P/$name.err
  echo "leg $name exit=$?" >> $P/probe.log
  grep -iE "arch|unused|missing|loop|n_layer|error|warn" $P/$name.err | head -25 >> $P/probe.log
  echo "--- generated ($name) ---" >> $P/probe.log
  cat $P/$name.out >> $P/probe.log
  sensors 2>/dev/null | grep -i "tctl" >> $P/probe.log
  date >> $P/probe.log
}

run_leg base
run_leg loops4 --override-kv inkling2.num_loops=int:4
echo "PROBE DONE" >> $P/probe.log
