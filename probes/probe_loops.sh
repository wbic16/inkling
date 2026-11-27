#!/bin/bash
cd /home/wbic16/llama.cpp || exit 1
echo "=== git state ==="
git log --oneline -3 2>/dev/null; git branch --show-current 2>/dev/null
echo "=== num_loops / n_loop references ==="
grep -rn "num_loops\|n_loop\|num_passes\|n_pass" src/ include/ ggml/ common/ tools/ conversion/ 2>/dev/null | grep -iv "loop_count\|loopback" | head -40
echo "=== inkling2 hparams in llama-hparams / model loader ==="
grep -rn "inkling2" src/llama-hparams.* src/llama-model.cpp 2>/dev/null | head -20
grep -rn "loop" src/models/inkling2.cpp | head -30
echo "=== inkling2-loop.cpp head ==="
wc -l src/models/inkling2.cpp src/models/inkling2-loop.cpp
grep -n "loop" src/models/inkling2-loop.cpp | head -30
