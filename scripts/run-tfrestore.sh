#!/bin/bash
set -u
cd ~/rse-fpsolve
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
./venv/bin/pip install "transformers==5.24.1" > tf-restore.log 2>&1; echo "pip rc=$?"
M=./venv/lib/python3.12/site-packages/transformers/models/inkling2_loop/modeling_inkling2_loop.py
grep -nE "^class |Cache" "$M" | head -30
echo "=== cache_utils inkling ==="
grep -rn "inkling" ./venv/lib/python3.12/site-packages/transformers/cache_utils.py | head -5
grep -rln "Inkling2LoopCache" ./venv/lib/python3.12/site-packages/transformers/ | head -5
