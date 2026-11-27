#!/bin/bash
M=~/rse-fpsolve/venv/lib/python3.12/site-packages/transformers/models/inkling2_loop/modeling_inkling2_loop.py
echo "===== SlidingLoopDecoderLayer.forward (line 937) ====="
sed -n '937,1000p' "$M"
echo "===== SlidingDecoderLayer.forward (line 876) ====="
sed -n '876,913p' "$M"
