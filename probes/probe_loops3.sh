#!/bin/bash
cd /home/wbic16/llama.cpp || exit 1
echo "=== LOOP KV strings ==="
grep -n "LOOP_START\|LOOP_END\|LOOP_COUNT\|LOOP_GATE_RANK\|LOOP_VARYING\|LOOP_MSG\|LOOP_Q_CTRL\|LOOP_FFN_CTRL" src/llama-arch.cpp | head -20
echo "=== inkling2.cpp: all 'pass' mentions ==="
grep -n -i "pass" src/models/inkling2.cpp | head -40
echo "=== inkling2.cpp graph build section (lines 255-330) ==="
sed -n '255,330p' src/models/inkling2.cpp
echo "=== models.h inkling2 structs ==="
grep -n "inkling2" src/models/models.h | head
sed -n '/struct llama_model_inkling2 /,/^};/p' src/models/models.h | head -60
