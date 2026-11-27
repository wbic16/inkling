#!/bin/bash
# Measure: current main file growth + a parallel range stream growth over 45 s.
F=/mnt/mirrorborn/shared/models/inkling-2-mini-hf/model-00001-of-00001.safetensors
T=/tmp/range-test.bin
rm -f "$T"
curl -sL -r 12000000000-13000000000 -o "$T" https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/model-00001-of-00001.safetensors &
CPID=$!
A=$(stat -c%s "$F"); sleep 45
B=$(stat -c%s "$F"); S=$(stat -c%s "$T" 2>/dev/null || echo 0)
kill $CPID 2>/dev/null; rm -f "$T"
echo "main: $(( (B-A)/45/1024 )) KB/s   parallel-stream: $(( S/45/1024 )) KB/s"
