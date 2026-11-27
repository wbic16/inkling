#!/bin/bash
echo "=== solve-driver.log (stderr) ==="
cat ~/rse-fpsolve/work/solve-driver.log 2>/dev/null | tail -20
echo "=== driver-solve.log ==="
cat ~/rse-fpsolve/work/driver-solve.log 2>/dev/null
echo "=== dmesg oom ==="
dmesg 2>/dev/null | tail -30 | grep -iE "oom|killed process|memory" | tail -10
sudo dmesg 2>/dev/null | grep -iE "oom|killed process" | tail -5
echo "=== mem/swap ==="
free -g; swapon --show 2>/dev/null
echo "=== thermal tail ==="
tail -3 ~/rse-fpsolve/work/thermal.log 2>/dev/null
