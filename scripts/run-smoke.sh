#!/bin/bash
cd ~/rse-fpsolve || exit 1
W=$HOME/rse-fpsolve/work; mkdir -p "$W"
sensors 2>/dev/null | grep -i tctl
./venv/bin/python driver_fpsolve.py --base "$HOME/inkling-hf-base" --work "$W" --mode smoke > "$W/smoke-driver.log" 2>&1
echo "smoke rc=$?"
tail -20 "$W/driver-smoke.log" 2>/dev/null
sensors 2>/dev/null | grep -i tctl
