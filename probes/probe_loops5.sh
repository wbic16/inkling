#!/bin/bash
cd /home/wbic16/llama.cpp || exit 1
echo "=== llama-hparams.h inkling2 section ==="
sed -n '245,285p' src/llama-hparams.h
echo "=== who reads n_loop / num_loops hparam ==="
grep -rn "n_loop\b" src/ include/ | head -20
echo "=== base inkling2 graph constructor ==="
awk '/llama_model_inkling2::graph::graph/,/^}/' src/models/inkling2.cpp | head -80
echo "=== base load_arch_tensors: loop tensors? ==="
grep -n "loop\|LOOP" src/models/inkling2.cpp | head -20
