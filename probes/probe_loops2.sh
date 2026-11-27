#!/bin/bash
cd /home/wbic16/llama.cpp || exit 1
echo "=== inkling2-loop.cpp lines 1-70 ==="
sed -n '1,70p' src/models/inkling2-loop.cpp
echo "=== which arch name maps to which class ==="
grep -n "inkling2" src/llama-model.cpp | head; grep -rn "INKLING2" src/llama-arch.cpp | head
echo "=== inkling2.cpp: pass loop + kv keys ==="
grep -n "get_key\|n_pass\|pass\b\|for.*pass\|num_loops\|LLM_KV" src/models/inkling2.cpp | head -30
echo "=== LLM_KV names for loops ==="
grep -rn "num_loops\|NUM_LOOPS" src/llama-kv.h include/ src/llama-arch.* 2>/dev/null | head -10
grep -rn "\"%s.num_loops\"\|num_loops" src/*.cpp src/*.h 2>/dev/null | head -10
