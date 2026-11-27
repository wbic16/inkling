#!/bin/bash
M=~/rse-fpsolve/venv/lib/python3.12/site-packages/transformers/models/inkling2_loop/modeling_inkling2_loop.py
echo "=== decoder layer forwards ==="
grep -n "class Inkling2Loop.*DecoderLayer\|class Inkling2LoopTextModel\|class Inkling2LoopModel" "$M"
awk '/class Inkling2LoopSlidingDecoderLayer/,/def forward/' "$M" | head -5
grep -n "    def forward" "$M" | head -40
