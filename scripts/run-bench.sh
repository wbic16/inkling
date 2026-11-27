#!/bin/bash
set -u
cd ~/rse-fpsolve
B=/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/Inkling-2-Mini-AD-Q4_K.gguf
F=/mnt/mirrorborn/shared/models/inkling-2-mini-gguf/fpsolve-staging/Inkling-2-Mini-FPSOLVE-Q4_K.gguf
sensors 2>/dev/null | grep -i tctl
./venv/bin/python bench.py "fpsolve4:$F:4,base4:$B:4,fpsolve3:$F:3,base3:$B:3" 32 > bench/bench-run.log 2>&1
echo "bench rc=$?"
cat bench/summary.json 2>/dev/null
