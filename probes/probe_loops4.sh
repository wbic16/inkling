#!/bin/bash
cd /home/wbic16/llama.cpp || exit 1
echo "=== conversion/inkling2.py: arch + tensor names + metadata keys ==="
grep -n "inkling2_loop\|set_gguf_parameters\|num_loops\|loop\|architecture" conversion/inkling2.py | head -30
echo "=== git log for inkling2 model files ==="
git log --oneline -5 -- src/models/inkling2.cpp src/models/inkling2-loop.cpp conversion/inkling2.py 2>/dev/null
echo "=== inkling-stage dir ==="
ls -la /home/wbic16/inkling-stage/ 2>/dev/null | head -20
ls /home/wbic16/inkling-stage/cpu 2>/dev/null | head
echo "=== llama-bin dirs ==="
ls /home/wbic16/llama-bin/ /home/wbic16/llama-bin/cpu 2>/dev/null | head -20
echo "=== strings: does fleet llama-server know loop_attn_down / num_loops? ==="
strings /home/wbic16/llama-bin/cpu/libllama.so 2>/dev/null | grep -iE "loop_attn|num_loops|loop\.count|loop_q_ctrl" | sort -u | head -20
echo "=== strings: hub build-vulkan libllama ==="
strings /home/wbic16/llama.cpp/build-vulkan/bin/libllama.so 2>/dev/null | grep -iE "loop_attn|num_loops|loop\.count|loop_q_ctrl" | sort -u | head -20
ls /home/wbic16/llama.cpp/build-vulkan/bin/ 2>/dev/null | head
