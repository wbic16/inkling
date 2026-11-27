#!/bin/bash
cd /home/wbic16/rse-looper-src || exit 1
git log --oneline -1
echo "=== pyproject ==="
cat pyproject.toml
echo "=== module sizes ==="
wc -l rse_looper/*.py tests/*.py
echo "=== preflight: cuda/tf32/causal_conv checks ==="
grep -rn "cuda\|tf32\|causal_conv\|rocm\|hip\|device" rse_looper/preflight.py | head -25
echo "=== fpsolve: tensor selection ==="
grep -n "def \|name\|gate\|router\|conv\|output_proj\|layers\|range(" rse_looper/fpsolve.py | head -50
